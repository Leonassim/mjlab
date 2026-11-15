import os, inspect
os.environ.setdefault("RHPS1_STAIRS", "1")
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start")
print("decimation", cfg.decimation, "timestep", cfg.sim.mujoco.timestep)
for k, t in cfg.rewards.items():
  f = t.func
  name = getattr(f, "__name__", type(f).__name__)
  mod = getattr(f, "__module__", type(f).__module__)
  print(f"R {k:32s} w {t.weight:+9.4g} {mod.split('.')[-1]}.{name} {dict(t.params) if t.params else ''}")
for k, t in cfg.terminations.items():
  f = t.func; print(f"T {k:32s} {getattr(f,'__module__','?').split('.')[-1]}.{getattr(f,'__name__',type(f).__name__)} {dict(t.params) if t.params else ''}")
for k, t in cfg.events.items():
  print(f"E {k:32s} {t.mode} {getattr(t.func,'__name__',type(t.func).__name__)}")
print("actor obs", {k: (v.history_length if hasattr(v,'history_length') else None) for k, v in cfg.observations['actor'].terms.items()} if hasattr(cfg.observations['actor'],'terms') else '')
