"""evprobe.py <run> <ckpt>: critic explained variance on stochastic rollouts in the training env (training reward config)."""
import os, sys
from dataclasses import asdict
from pathlib import Path
import torch
os.environ.setdefault("RHPS1_STAIRS", "1")
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
N = int(os.environ.get("EV_N", "64")); T = int(os.environ.get("EV_T", "2400")); dev = os.environ.get("EV_DEV", "cpu")
G, L = 0.9975, 0.987
torch.manual_seed(int(os.environ.get("EV_SEED", "0")))
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.scene.num_envs = N
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), map_location=dev)
runner.alg.actor.eval(); runner.alg.critic.eval()
alg = runner.alg
obs = env.get_observations()
R, D, V, TO = [], [], [], []
for t in range(T):
  with torch.no_grad():
    a = alg.act(obs)
    v = alg.transition.values.squeeze(-1).clone()
    obs, rew, dones, extras = env.step(a)
  to = extras.get("time_outs", torch.zeros_like(dones)).float()
  R.append(rew + G * v * to); D.append(dones.float()); V.append(v); TO.append(to)
with torch.no_grad():
  vT = alg.critic(obs).squeeze(-1)
R, D, V = torch.stack(R), torch.stack(D), torch.stack(V)
V1 = torch.cat([V[1:], vT[None]])
def targets(chunk):
  A = torch.zeros_like(R); ret = torch.zeros_like(R)
  for t in reversed(range(T)):
    last = (t + 1) % chunk == 0 or t == T - 1
    nv = V1[t]
    delta = R[t] + G * nv * (1 - D[t]) - V[t]
    A[t] = delta + (0 if last else G * L * (1 - D[t]) * A[t + 1])
  return A + V
def mc():
  g = torch.zeros_like(R); nxt = vT
  for t in reversed(range(T)):
    nxt = R[t] + G * (1 - D[t]) * nxt; g[t] = nxt
  return g
keep = slice(0, T - 1200)
for name, tg in (("MC", mc()), ("lambda full", targets(10**9)), ("lambda 24-step (PPO)", targets(24))):
  y, v = tg[keep].flatten(), V[keep].flatten()
  print(f"{name:22s}: target std {y.std():.2f} mean {y.mean():+.2f} | V mean {v.mean():+.2f} | mse {((y - v) ** 2).mean():.1f} | EV {1 - (y - v).var() / y.var():.3f}")
print(f"falls/resets in rollout: dones {int(D.sum())} timeouts {int(torch.stack(TO).sum())} over {N} envs x {T * 0.005:.0f} s; reward/step mean {R.mean():.4f}")
