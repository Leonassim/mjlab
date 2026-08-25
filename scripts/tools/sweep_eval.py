"""Deterministic acceptance sweep: one rollout per command, stats per command.

Training metrics are not an acceptance test. On 2026-08-25 a run reported 3.3%
falls for a policy that fell in mc_mujoco above 0.16 m/s -- because those
numbers are averaged over a command distribution that is mostly slow, and "must
never fall" is a statement about a RANGE, not about a mean.

So: pin every environment to one command, roll out, read the metrics, repeat
for the next command. Each row is then a real answer about that command rather
than an average over conditions the robot handles unevenly.

  uv run python scripts/tools/sweep_eval.py <run> <model_xxx.pt> [--steps 600]

RHPS1_ABLATION and the other env knobs must match the run being evaluated, the
same way export_onnx.py needs them.
"""

import argparse
import statistics
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict

TASK = "Mjlab-Velocity-Flat-RHPS1"
# vx, vy, yaw. Zero first: a robot that drifts while asked to stand still fails
# the criterion just as surely as one that falls while walking.
GRID = [
  (0.0, 0.0, 0.0),
  (0.10, 0.0, 0.0), (0.20, 0.0, 0.0), (0.30, 0.0, 0.0),
  (-0.20, 0.0, 0.0),
  (0.0, 0.20, 0.0), (0.0, -0.20, 0.0),
  (0.20, 0.20, 0.0),
  (0.0, 0.0, 0.30), (0.0, 0.0, -0.30),
  (0.20, 0.0, 0.30),
]
# falls is counted from the step return, not read from the log: the
# Episode_Termination tags are written by the training loop at episode end, so
# they simply are not there during a rollout -- they read nan, which is exactly
# the silent hole an acceptance test must not have.
WATCH = [
  ("clear", "Metrics/sole_clearance_p90"),
  ("impact", "Metrics/pre_contact_peak_vel_mean"),
  ("satleg", "Metrics/torque_saturated_frac_legs"),
  ("flat", "Metrics/flat_support_contacts_mean"),
  ("period", "Metrics/step_period_mean"),
  ("len", "Metrics/step_length_mean"),
  # The hip-yaw ceiling, deterministically. CROTCH_Y read 0.82 of its torque
  # limit in training against the knee's 0.60, and the whole "yaw is the wall"
  # theory rests on that number -- which, like falls and clipping, may simply be
  # domain randomisation. yaw_err says whether the robot rotates when nobody
  # asked it to, which is what would load the joint in the first place.
  ("crotchY", "TorqueRatio/L_CROTCH_Y"),
  ("kneeP", "TorqueRatio/L_KNEE_P"),
  ("yaw_err", "Metrics/twist/error_vel_yaw"),
]


def pin(env, cmd):
  """Freeze every env on one command.

  Overriding _resample_command rather than writing vel_command_b once: the term
  resamples on its own timer, so a value poked in from outside is silently
  replaced a second later. That trap has cost this project a measurement before.
  """
  term = env.unwrapped.command_manager.get_term("twist")
  v = torch.tensor(cmd, device=env.unwrapped.device, dtype=torch.float32)

  def _fixed(env_ids):
    term.vel_command_b[env_ids] = v
    term.is_standing_env[env_ids] = False
    term.is_heading_env[env_ids] = False

  term._resample_command = _fixed
  _fixed(slice(None))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run")
  p.add_argument("checkpoint")
  p.add_argument("--steps", type=int, default=1200)  # 6 s at 5 ms, ~14 foulees
  p.add_argument("--envs", type=int, default=512)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  a = p.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = a.envs
  # Standing envs are a training device; here the zero command is its own row.
  cur = getattr(env_cfg, "curriculum", {})
  cur.pop("standing_envs", None)
  cur.pop("command_vel", None)

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=load_rl_cfg(TASK).clip_actions,
  )
  agent_cfg = load_rl_cfg(TASK)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
  runner.load(str(Path(a.root) / a.run / a.checkpoint), load_cfg={"actor": True},
              strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  hdr = f"{'vx':>6s}{'vy':>6s}{'yaw':>6s} {'falls':>8s} " + " ".join(f"{n:>8s}" for n, _ in WATCH)
  print(hdr)
  print("-" * len(hdr))
  worst = {}
  for cmd in GRID:
    obs, _ = env.reset()
    pin(env, cmd)
    acc = {n: [] for n, _ in WATCH}
    fell = torch.zeros(a.envs, dtype=torch.bool, device=env.unwrapped.device)
    # no_grad, not inference_mode: the actuators keep state across resets, and
    # inference tensors created in one rollout cannot be written by the next
    # env.reset().
    with torch.no_grad():
      for i in range(a.steps):
        obs, _, dones, extras = env.step(policy(obs))
        # A time-out is the episode ending on the clock, not the robot failing.
        timeout = extras.get("time_outs")
        real = dones.bool() if timeout is None else (dones.bool() & ~timeout.bool())
        fell |= real
        log = extras.get("log", {})
        if i > a.steps // 4:  # let the reset transient pass
          for n, tag in WATCH:
            if tag in log:
              acc[n].append(float(log[tag]))
    row = {n: (statistics.fmean(v) if v else float("nan")) for n, v in acc.items()}
    row["falls"] = float(fell.float().mean())
    print(f"{cmd[0]:6.2f}{cmd[1]:6.2f}{cmd[2]:6.2f} {row['falls']:8.4f} "
          + " ".join(f"{row[n]:8.4f}" for n, _ in WATCH))
    for n in ("falls", "impact", "satleg"):
      worst[n] = max(worst.get(n, 0.0), row[n] if row[n] == row[n] else 0.0)
  print("\npire cas sur la grille : " + "  ".join(f"{k}={v:.4f}" for k, v in worst.items()))


if __name__ == "__main__":
  main()
