"""parity.py <run> <ckpt> <onnx>: ONNX actions vs the torch actor (scan encoder + normaliser) on random obs."""
import os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, onnxruntime as ort, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
TASK = os.environ["MJLAB_TASK"]
run, ck, onx = sys.argv[1:4]
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path(os.environ["MJLAB_LOG_ROOT"]) / run / ck), map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
sess = ort.InferenceSession(onx)
torch.manual_seed(0)
obs = env.get_observations()
worst = 0.0
for i in range(8):
  o = {k: v.clone() for k, v in obs.items()}
  x = o["actor"] if "actor" in o else o["policy"]
  x += torch.randn_like(x) * 0.3
  if i % 2: x[:, 126:] = torch.rand(1, 325) * 0.4 - 0.2  # stair-like scan values
  with torch.no_grad():
    a_t = pol(o).numpy()
  feed = {"obs": x.numpy().astype(np.float32)}
  if any(i.name == "time_step" for i in sess.get_inputs()): feed["time_step"] = np.zeros((1, 1), np.float32)
  a_o = sess.run([sess.get_outputs()[0].name], feed)[0]
  worst = max(worst, float(np.abs(a_t - a_o).max()))
print(f"PARITY obs dim {x.shape[1]} max |a_torch - a_onnx| over 8 draws = {worst:.2e}")
