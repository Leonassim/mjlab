import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tags = sys.argv[2].split(",")
for run in sys.argv[1].split(","):
  ea = EventAccumulator(glob.glob(f"logs/rsl_rl/rhps1_tracking/{run}/events*")[0], size_guidance={"scalars": 0}); ea.Reload()
  for t in tags:
    if t not in ea.Tags()["scalars"]:
      print(run, t, "missing"); continue
    s = ea.Scalars(t); st = np.array([x.step for x in s]); v = np.array([x.value for x in s])
    edges = np.arange(st.min(), st.max() + 1, 250)
    out = []
    for a in edges:
      m = (st >= a) & (st < a + 250)
      if m.any(): out.append(f"{a}:{np.mean(v[m]):.3g}")
    print(run, t, " ".join(out))
