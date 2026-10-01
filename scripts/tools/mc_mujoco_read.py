"""Read an mc_mujoco bench log in the same terms as flip_probe.py / delay_probe.py.

System python3 (mc_log_ui is not in the uv env).

  python3 scripts/tools/mc_mujoco_read.py flip|walk|stand <log.bin>

flip: windows after the command flip (velCmd_x leaves 0). Pelvis = |root angular
velocity|, upper body = max |joint velocity| over the 18 non-leg joints, read from
the policy's own observation (V3 joint_vel block, mjlab order: chest, head, arms
first) so the numbers compare with flip_probe.py.
walk / stand: fall, advance and target agitation after the first inference.
"""
import sys

import mc_log_ui
import numpy as np

mode, path = sys.argv[1], sys.argv[2]
d = mc_log_ui.read_log(path)


def g(b):
  ks = sorted((k for k in d if k.startswith(b + "_") and k[len(b) + 1:].isdigit()),
              key=lambda k: int(k.rsplit("_", 1)[1]))
  return np.stack([np.asarray(d[k]) for k in ks], -1)


t = np.asarray(d["t"])
x = np.asarray(d["FloatingBase_position_x"])
z = np.asarray(d["FloatingBase_position_z"])
rq = g("NewRLQPController_RL_q")
act = g("NewRLQPController_RL_currentAction")
i0 = int(np.argmax(np.abs(act).sum(1) > 0))
fell = (z[i0:] < 0.6).any()

if mode == "flip":
  vx = np.asarray(d["NewRLQPController_velCmd_x"])
  w = np.degrees(np.sqrt(sum(np.asarray(d["FloatingBase_angularVelocity_" + c]) ** 2 for c in "xyz")))
  obs = g("NewRLQPController_RL_currentObservation")
  qu = np.degrees(np.abs(obs[:, 51:69]).max(1))
  lf = np.asarray(d["LeftFootForceSensor_fz"])
  rf = np.asarray(d["RightFootForceSensor_fz"])
  i = int(np.argmax(np.abs(vx) > 1e-6))
  tf = t[i]
  for a, b in ((0.0, 0.3), (0.3, 1.0), (1.0, 2.3)):
    s = (t >= tf + a) & (t < tf + b)
    print(f"  {a:.1f}-{b:.1f} s  bassin {w[s].max():5.1f} / {w[s].max():5.1f} deg/s"
          f"   haut du corps {qu[s].max():5.1f} / {qu[s].max():5.1f} deg/s"
          f"   Fz min {min(lf[s].min(), rf[s].min()):5.0f} N")
  print(f"  avance 5 s apres la bascule {x[min(i + 1000, len(x) - 1)] - x[i]:5.2f} m")
  print(f"  chute {'oui' if fell else 'non'}")
else:
  n = int(2.0 / np.median(np.diff(t)))
  late = rq[-n:]
  rms = np.degrees(np.sqrt((np.diff(late, axis=0) ** 2).mean()))
  print(f"  chute {'oui' if fell else 'non'}  avance {x[-1] - x[i0]:5.2f} m en {t[-1] - t[i0]:.0f} s"
        f"  agitation finale {rms:.3f} deg/pas")
