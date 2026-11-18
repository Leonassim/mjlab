"""depth_rl.py: PPO fine-tuning of the depth-image student (depth_student.DepthPPOModel).

DAgger copies a teacher that sees more than the student (exact scan, no latency, under the feet), and the copy
falls twice as often on high steps (depthscan, the same pipeline on the exact scan, matches the teacher). PPO on
the student's own inputs lets it find the behaviour its information supports. Starts from a DAgger student
(actor) and from the teacher's critic (same privileged critic group), then trains the critic alone for WARMUP
iterations so the first actor updates see sane advantages.

  RHPS1_DEPTH=1 RHPS1_HEAD_HOLD=1 RHPS1_MIRROR=0 RHPS1_BOUND_W=0 + recipe (depth_rl.sh):
  python depth_rl.py --student <dagger.pt> --teacher <teacher.pt> --tag rl1 [--envs 2048] [--iters 3000]
  python depth_rl.py --export <run>/model_<it>.pt   ->  <run>/student_<it>.pt + policy_<it>.onnx (depth_eval format)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("RHPS1_STAIRS", "1")

import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--student", default="")
ap.add_argument("--teacher", default="logs/rsl_rl/rhps1_tracking/2026-11-15_10-23-12/model_15000.pt")
ap.add_argument("--tag", default="rl")
ap.add_argument("--envs", type=int, default=2048)
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--std", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--mb", type=int, default=16)
ap.add_argument("--export", default="")
args = ap.parse_args()

if args.export:
  from mjlab.tasks.tracking.config.rhps1 import depth_student as D  # noqa: E402

  ck = torch.load(args.export, map_location="cpu")
  sd = ck["actor_state_dict"]
  n_prop = sd["mean"].shape[-1]
  n_act = sd["mlp.4.bias"].shape[0]
  m = D.DepthStudent(n_prop, n_act, sd["mean"], sd["std"])
  m.cnn.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("cnn.")})
  m.cnn_fc.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("cnn_fc.")})
  m.gru.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("rnn.rnn.")})
  m.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
  p = Path(args.export)
  it = p.stem.split("_")[-1]
  torch.save({"student": m.state_dict(), "n_prop": n_prop, "n_act": n_act, "iter": int(it)}, p.parent / f"student_{it}.pt")
  m.eval()
  dummy = (torch.zeros(1, n_prop), torch.zeros(1, D.H * D.W), torch.zeros(1, 1, m.gru_dim))
  torch.onnx.export(D.DepthStudentOnnx(m), dummy, str(p.parent / f"policy_{it}.onnx"),
                    input_names=["obs", "depth", "h_in"], output_names=["actions", "h_out"], dynamo=False,
                    opset_version=17)
  print(f"[depth_rl] exported {p.parent / f'student_{it}.pt'} and policy_{it}.onnx")
  raise SystemExit(0)

os.environ["RHPS1_DEPTH_INIT"] = args.student
import mjlab.tasks  # noqa: E402, F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import depth_student as D  # noqa: E402
from mjlab.rl.runner import MjlabOnPolicyRunner  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

assert D.DEPTH and os.environ.get("RHPS1_MIRROR", "0") == "0", "RHPS1_DEPTH=1 and RHPS1_MIRROR=0 required"
TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
dev = "cuda:0"
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.scene.num_envs = args.envs
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)

rl = load_rl_cfg(TASK)
rl.obs_groups = {"actor": ("proprio", "depth"), "critic": ("critic",)}
rl.actor.class_name = "mjlab.tasks.tracking.config.rhps1.depth_student:DepthPPOModel"
rl.actor.hidden_dims = (256, 128)
rl.actor.obs_normalization = False
rl.actor.distribution_cfg = {"class_name": "GaussianDistribution", "init_std": args.std, "std_type": "scalar"}
rl.algorithm.entropy_coef = 0.0
rl.algorithm.gamma, rl.algorithm.lam = 0.99, 0.95
rl.algorithm.learning_rate = args.lr
rl.algorithm.num_mini_batches = args.mb  # recurrent batches are padded per env: small ones fit next to the sim
rl.experiment_name = "rhps1_depth_rl"
rl.run_name = args.tag
rl.save_interval = 250
rl.max_iterations = args.iters
rl.clip_actions = 12.0
log_dir = Path("logs/rsl_rl/rhps1_depth_rl") / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.tag}"
d = asdict(rl)


class _Runner(MjlabOnPolicyRunner):
  """mjlab's config handling and wandb, rsl_rl's plain save (the ONNX export is done by --export)."""

  def save(self, path, infos=None):
    OnPolicyRunner.save(self, path, infos)


runner = _Runner(env, d, str(log_dir), device=dev)
tch = torch.load(args.teacher, map_location=dev)
runner.alg.critic.load_state_dict(tch["critic_state_dict"])
log_dir.mkdir(parents=True, exist_ok=True)
(log_dir / "env.txt").write_text(
  "\n".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith(("RHPS1_", "MJLAB_"))) + "\n")
print(f"[depth_rl] actor from {args.student}, critic from {args.teacher} -> {log_dir}", flush=True)

actor_params = [p for p in runner.alg.actor.parameters()]
for p in actor_params:
  p.requires_grad_(False)
# Fixed LR while the actor is frozen: its KL is 0, and the adaptive schedule would push the LR to its 1e-2 ceiling
# (x100), which wrecked the student at the first actor update (rl1, episode length 813 -> 29).
runner.alg.schedule = "fixed"
runner.alg.learning_rate = 1e-3  # critic only (actor frozen)
for g in runner.alg.optimizer.param_groups:
  g["lr"] = 1e-3
runner.learn(args.warmup)
for p in actor_params:
  p.requires_grad_(True)
runner.alg.schedule = "adaptive"
runner.alg.learning_rate = args.lr
for g in runner.alg.optimizer.param_groups:
  g["lr"] = args.lr
runner.learn(args.iters - args.warmup)
