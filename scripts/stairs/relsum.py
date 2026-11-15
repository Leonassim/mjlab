"""relsum.py <label> <log>...: one line per mc_mujoco stairs run (cx 1.64, tread 0.27) + a summary line."""
import sys, os, numpy as np
from mc_log_ui import read_log
lab = sys.argv[1]; rows = []
for f in sys.argv[2:]:
    d = read_log(f); t = np.array(d['t'])
    px, pz = np.array(d['FloatingBase_position_x']), np.array(d['FloatingBase_position_z'])
    act = np.abs(np.array(d['NewRLQPController_RL_currentAction_0'])) > 0
    t0 = t[np.argmax(act)] if act.any() else 0.0
    fall = np.where(pz < 0.55)[0]; tf = t[fall[0]] if len(fall) else None
    end = fall[0] if len(fall) else len(t)
    xm = px[:end].max(); plat = xm > 1.04; down = xm > 3.5
    m = act.copy(); m[end:] = False
    kn = np.maximum(np.abs(np.array(d['torque_L_KNEE_P'])), np.abs(np.array(d['torque_R_KNEE_P'])))
    v = np.stack([np.array(d[f'NewRLQPController_RL_currentObservation_{i}']) for i in range(69, 81)], 1)
    rows.append((tf is None, plat, down, tf, xm, 100 * (kn[m] >= 105).mean(), np.abs(v[m]).max()))
    print(f"  {os.path.basename(f)[-23:-4]} fall {('%.1f' % (tf - t0)) + ' s after arm' if tf else 'none':18s} max x {xm:4.2f} platform {int(plat)} down {int(down)} | knee>=105 {rows[-1][5]:4.1f}% | leg vmax {rows[-1][6]:.1f} rad/s")
n = len(rows)
print(f"RELBENCH {lab}: no-fall {sum(r[0] for r in rows)}/{n}, platform {sum(r[1] for r in rows)}/{n}, down {sum(r[2] for r in rows)}/{n} | knee>=105 {np.mean([r[5] for r in rows]):.1f}% | leg vmax {max(r[6] for r in rows):.1f}")
