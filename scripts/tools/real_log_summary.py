"""Real-robot NewRLQPController log: commands, balance and left/right signature.

  python3 scripts/tools/real_log_summary.py <log.bin> [<log.bin> ...]

System python (mc_log_ui). Joints are read from the policy observation
(21:51 + default pose, policy order); NewRLQPController_RL_q is in the
controller's order. Per command segment: tilt, roll, longest single support
per foot and peak foot force. Then the whole armed run: standing load,
force percentiles per foot, ankle currents at stand, steps and their length.
"""

import sys

import mc_log_ui
import numpy as np

STAND_N = 20.0  # foot counted unloaded below this (sensor offsets reach ~35 N)
SS_N = 100.0  # single support: the other foot below this


def segments(cmd, t):
  s = np.sign(np.round(cmd * (np.abs(cmd) > 0.5)))
  ch = np.r_[0, np.where(np.any(np.diff(s, axis=0) != 0, axis=1))[0] + 1, len(t)]
  return [(a, b, s[a].astype(int)) for a, b in zip(ch[:-1], ch[1:]) if t[b - 1] - t[a] >= 0.3]


def runs(mask):
  e = np.diff(np.r_[0, mask.astype(int), 0])
  return np.where(e == 1)[0], np.where(e == -1)[0]


for f in sys.argv[1:]:
  d = mc_log_ui.read_log(f)
  t = np.asarray(d["t"])
  dt = float(np.median(np.diff(t)))
  ob = np.stack([np.asarray(d[f"NewRLQPController_RL_currentObservation_{i}"]) for i in range(126)], -1)
  armed = np.abs(ob[:, 20]) > 1e-6
  if not armed.any():
    print(f"== {f}: policy never armed")
    continue
  ia = np.argmax(armed)
  cmd, g = ob[:, 123:126], ob[:, 18:21]
  tilt = np.degrees(np.arccos(np.clip(-g[:, 2] / np.linalg.norm(g, axis=1).clip(1e-6), -1, 1)))
  roll = np.degrees(np.arcsin(np.clip(-g[:, 1], -1, 1)))  # + = leaning right
  lf, rf = np.asarray(d["LeftFootForceSensor_fz"]), np.asarray(d["RightFootForceSensor_fz"])
  print(f"== {f.split('/')[-1]}  armed t {t[ia]:.1f}, {t[armed][-1] - t[ia]:.0f} s armed")
  for a, b, s in segments(cmd, t):
    a = max(a, ia)
    if b <= a + 20:
      continue
    seg = slice(a, b)
    ssl = (rf[seg] < SS_N) & (lf[seg] > SS_N)  # on the left foot only
    ssr = (lf[seg] < SS_N) & (rf[seg] > SS_N)
    lon = lambda m: max(((e - s0) * dt for s0, e in zip(*runs(m))), default=0.0)
    print(f"  {t[a]:6.1f}-{t[b-1]:6.1f} cmd {s!s:11s} tilt max {tilt[seg].max():4.1f}  roll {roll[seg].min():+5.1f}..{roll[seg].max():+5.1f}"
          f"  appui seul max G {lon(ssl):.2f} D {lon(ssr):.2f} s  Fz max G {lf[seg].max():5.0f} D {rf[seg].max():5.0f}")
  w = slice(ia + 100, ia + 300)
  cur = lambda j: (np.asarray(d.get(f"current_L_{j}", [np.nan] * len(t)))[w].mean(),
                   np.asarray(d.get(f"current_R_{j}", [np.nan] * len(t)))[w].mean())
  ca = cur("ANKLE_P")
  print(f"  debout G/D {lf[w].mean():.0f}/{rf[w].mean():.0f} N, courant cheville G {ca[0]:+.2f} D {ca[1]:+.2f} A")
  A = armed & (tilt < 25)
  print(f"  Fz G p50/p99/max {np.percentile(lf[A],50):.0f}/{np.percentile(lf[A],99):.0f}/{lf[A].max():.0f}  "
        f"D {np.percentile(rf[A],50):.0f}/{np.percentile(rf[A],99):.0f}/{rf[A].max():.0f} N")
  for nm, x in (("G", lf), ("D", rf)):
    s0, e = runs(armed & (x < STAND_N))
    sw = (e - s0) * dt
    sw = sw[sw > 0.1]
    print(f"  pied {nm}: {len(sw)} pas, vol median {np.median(sw) if len(sw) else 0:.2f} s, p90 {np.percentile(sw, 90) if len(sw) else 0:.2f} s")
  big = np.where(armed & (tilt > 20))[0]
  print(f"  chute (inclinaison > 20 deg) : " + (f"oui a t {t[big[0]]:.1f}, cmd {np.round(cmd[big[0]], 2)}" if len(big) else "non"))
