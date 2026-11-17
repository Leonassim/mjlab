"""depth_distill.py: DAgger from the exact-scan stairs teacher to the depth-image student (depth_student.py).

The student reads what the robot has: the teacher's proprioceptive inputs (the actor group minus the height scan)
and the delayed, degraded head depth image. It is labelled at every step with the teacher's action on the exact
scan. Rollouts mix teacher and student actions per step with probability beta (1 -> 0 over BETA_ITERS), so the
student learns on the states it visits itself. Truncated BPTT over each T-step rollout, hidden state carried.

  RHPS1_DEPTH=1 RHPS1_HEAD_HOLD=1 + the stairs recipe (run it through depth_distill.sh):
  python depth_distill.py --teacher <run>/<ckpt> --tag depth1 [--envs 2048] [--iters 6000] [--resume <pt>]
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("RHPS1_STAIRS", "1")

import torch  # noqa: E402

import mjlab.tasks  # noqa: E402, F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import depth_student as D  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True, help="logs/rsl_rl/rhps1_tracking/<run>/<ckpt>.pt")
ap.add_argument("--tag", required=True)
ap.add_argument("--envs", type=int, default=2048)
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--beta-iters", type=int, default=1000)
ap.add_argument("--mb", type=int, default=8, help="minibatches (over envs) per update")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--save", type=int, default=250)
ap.add_argument("--resume", default="")
ap.add_argument("--wandb", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
assert D.DEPTH, "RHPS1_DEPTH=1 is required"
dev = args.device

cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.scene.num_envs = args.envs
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
u = env.unwrapped

runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(args.teacher, load_cfg={"actor": True}, strict=True, map_location=dev)
teacher = runner.get_inference_policy(device=dev)
norm = runner.alg.actor.obs_normalizer
obs = env.get_observations()
n_act = u.action_manager.total_action_dim
n_prop = obs["actor"].shape[-1] - ST.scan_dim()
mean = norm._mean[0, :n_prop].detach()
std = norm._std[0, :n_prop].detach() + norm.eps
student = D.DepthStudent(n_prop, n_act, mean, std).to(dev)
opt = torch.optim.Adam(student.parameters(), lr=args.lr)
it0 = 0
out = Path("logs/rsl_rl/rhps1_depth") / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.tag}"
if args.resume:
  ck = torch.load(args.resume, map_location=dev)
  student.load_state_dict(ck["student"])
  opt.load_state_dict(ck["opt"])
  it0 = ck["iter"] + 1
  out = Path(args.resume).parent
out.mkdir(parents=True, exist_ok=True)
rhps1_env = {k: v for k, v in os.environ.items() if k.startswith(("RHPS1_", "MJLAB_"))}
(out / "env.txt").write_text("\n".join(f"{k}={v}" for k, v in sorted(rhps1_env.items())) + "\n")
print(f"[depth] envs {args.envs} prop {n_prop} act {n_act} image {D.H}x{D.W} -> {out}")

wb = None
if args.wandb:
  import wandb

  wb = wandb.init(project="mjlab", name=f"{out.name}", config={**vars(args), **rhps1_env, "n_prop": n_prop})


def split(o):
  return o["actor"][:, :n_prop], o["depth"]


def export(path: Path) -> None:
  m = D.DepthStudentOnnx(student).to("cpu").eval()
  m.m = m.m.to("cpu")
  dummy = (torch.zeros(1, n_prop), torch.zeros(1, D.H * D.W), torch.zeros(1, 1, student.gru_dim))
  torch.onnx.export(m, dummy, str(path), input_names=["obs", "depth", "h_in"], output_names=["actions", "h_out"],
                    dynamo=False, opset_version=17)
  student.to(dev).train()  # .eval() above: cuDNN refuses the GRU backward outside training mode


h = torch.zeros(1, args.envs, student.gru_dim, device=dev)
T = args.steps
for it in range(it0, it0 + args.iters):
  t0 = time.time()
  beta = max(0.0, 1.0 - it / max(args.beta_iters, 1))
  P = torch.zeros(T, args.envs, n_prop, device=dev)
  I = torch.zeros(T, args.envs, D.H * D.W, device=dev, dtype=torch.float16)  # half: 96-step rollouts
  A = torch.zeros(T, args.envs, n_act, device=dev)
  Dn = torch.zeros(T, args.envs, device=dev)
  h0 = h.detach().clone()
  logs = defaultdict(list)
  with torch.no_grad():
    for t in range(T):
      prop, img = split(obs)
      a_t = teacher(obs).clamp(-12, 12)
      a_s, h = student.step(prop, img, h)
      use_t = (torch.rand(args.envs, device=dev) < beta)[:, None]
      obs, _, dones, extras = env.step(torch.where(use_t, a_t, a_s))
      P[t], I[t], A[t], Dn[t] = prop, img, a_t, dones.float()
      h = h * (1 - dones.float())[None, :, None]
      for k, v in extras.get("log", {}).items():
        if k.startswith(("Curriculum/stairs_levels", "Episode_Termination/stairs_fall",
                         "Episode_Termination/time_out", "Episode_Termination/stairs_saturation")):
          logs[k].append(float(v))
  t_roll = time.time() - t0
  # Truncated BPTT over the rollout, minibatches over envs.
  tot, nb = 0.0, 0
  for _ in range(args.epochs):
    for idx in torch.randperm(args.envs, device=dev).chunk(args.mb):
      pred, _ = student.seq(P[:, idx], I[:, idx].float(), h0[:, idx], Dn[:, idx])
      loss = torch.nn.functional.mse_loss(pred, A[:, idx])
      opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
      opt.step()
      tot += float(loss)
      nb += 1
  rec = {"Loss/behavior": tot / nb, "Policy/beta": beta, "Perf/rollout_s": t_roll, "Perf/iter_s": time.time() - t0,
         **{k: sum(v) / len(v) for k, v in logs.items()}}
  if it % 10 == 0:
    print(f"it {it} " + " ".join(f"{k.split('/')[-1]} {v:.4f}" for k, v in rec.items()), flush=True)
  if wb is not None:
    wb.log(rec, step=it)
  if (it + 1) % args.save == 0 or it == it0 + args.iters - 1:
    p = out / f"model_{it + 1}.pt"
    torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "iter": it, "n_prop": n_prop,
                "n_act": n_act, "teacher": args.teacher}, p)
    export(out / f"policy_{it + 1}.onnx")
    print(f"[depth] saved {p}", flush=True)
