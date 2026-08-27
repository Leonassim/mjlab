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
  ("impact", "Metrics/landing_vel_mean"),
  ("peakVel", "Metrics/pre_contact_peak_vel_mean"),
  ("satleg", "Metrics/torque_saturated_frac_legs"),
  # Le haut du corps a son propre critere : deux tiers de l'ecretage y vivent,
  # et le sortir de l'agregat pour isoler les jambes l'avait laisse sans mesure.
  ("satup", "Metrics/torque_saturated_frac_upper"),
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
  # Flatness as an angle, not as a count of patches the solver happened to see.
  ("tiltGnd", "Metrics/sole_tilt_loaded"),
  ("tiltTD", "Metrics/sole_tilt_touchdown"),
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
  # 1024, not 256: three config-identical probes measured falls at 0.0156,
  # 0.0195 and 0.0352 -- a 2.3x spread on the criterion that outranks every
  # other one. Quadrupling the sample halves that band.
  p.add_argument("--envs", type=int, default=1024)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  # Robustness is not optional in the acceptance test. play=True drops domain
  # randomisation and the pushes, and that blind spot let this sweep certify
  # model_15600 at under 0.8% falls -- a checkpoint that collapses to 86-127%
  # the moment it is resumed under randomisation. A policy that only survives
  # calm conditions is exactly what must not reach the robot.
  p.add_argument("--rand", action="store_true", help="garder la randomisation")
  a = p.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK, play=not a.rand)
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
          # yaw_err is computed here and not read from the log. The command
          # manager only publishes error_vel_yaw through reset(), so on a 1200
          # step rollout that never resets the key never reaches extras["log"]
          # and the column read nan on EVERY policy -- a permanent hole, not a
          # result. Same definition as _update_metrics: emitted command minus
          # measured yaw rate, in the base frame.
          cm = env.unwrapped.command_manager.get_term("twist")
          acc["yaw_err"].append(
            float(
              torch.abs(
                cm.vel_command_out[:, 2]
                - cm.robot.data.root_link_ang_vel_b[:, 2]
              ).mean()
            )
          )
    row = {n: (statistics.fmean(v) if v else float("nan")) for n, v in acc.items()}
    row["falls"] = float(fell.float().mean())
    print(f"{cmd[0]:6.2f}{cmd[1]:6.2f}{cmd[2]:6.2f} {row['falls']:8.4f} "
          + " ".join(f"{row[n]:8.4f}" for n, _ in WATCH))
    # nan means the metric was never emitted -- the term that logs it is not in
    # this config. Coercing it to 0.0 made the policy 0 calibration print
    # "ECHEC lever de pied +100%" about a foot lift nobody had measured. A
    # criterion with no measurement has to say so, not fail.
    for n in ("falls", "impact", "satleg", "satup", "tiltGnd", "peakVel"):
      if row[n] == row[n]:
        worst[n] = max(worst.get(n, float("-inf")), row[n])
    # Minima, for the criteria where more is better -- and only on the moving
    # commands: a standing robot legitimately has no clearance, and folding that
    # zero into the worst case would fail every policy forever.
    if any(abs(c) > 0.05 for c in cmd):
      for n in ("clear", "flat"):
        if row[n] != row[n]:
          continue
        k = n + "_min"
        worst[k] = row[n] if k not in worst else min(worst[k], row[n])
  # Verdict. The point of a sweep is to end in a decision, and the decision has
  # to name the single worst-off criterion -- one change per iteration is what
  # keeps a result attributable. Ranked by how far past its threshold each one
  # sits, in units of the threshold, so criteria of different scales compare.
  print()
  rows = []
  for name, worst_v, thr, want_low in [
    ("ne jamais tomber", worst.get("falls"), 0.01, True),
    ("couples faisables", worst.get("satleg"), 0.25, True),
    # Meme seuil que les jambes : un actionneur sature est un actionneur que la
    # politique commande sans le sentir, quel que soit le membre.
    ("couples haut du corps", worst.get("satup"), 0.25, True),
    # landing_vel_mean, not pre_contact_peak_vel_mean. 0.16 was written for the
    # touchdown speed -- "just above policy 0's 0.158, the gait that landed
    # softly enough on hardware" -- but the criterion read the PEAK over the
    # pre-contact window, a strictly larger quantity that a walking swing leg
    # cannot get under. That mismatch is why this criterion has been named the
    # next target by every sweep since it was written, at a flat +107%.
    ("impact faible", worst.get("impact"), 0.16, True),
    ("lever de pied", worst.get("clear_min"), 0.030, False),
    # Tilt, not the corner count. flat_support_contacts_mean reports how many
    # of four patches the solver calls loaded, which is a function of its
    # contact threshold: a sole parallel to the ground within 16 um scored
    # 2.15 of 4, so the criterion failed at "+33% of threshold" on a foot that
    # was already flat. Tilt is the quantity meant by "flat foot" and it is
    # solver-independent. 0.05 rad is 2.9 deg, just under the 0.066 measured.
    ("pieds a plat", worst.get("tiltGnd"), 0.05, True),
  ]:
    if worst_v is None:
      rows.append((-1.0, name, float("nan"), thr, None))
      continue
    ok = (worst_v <= thr) if want_low else (worst_v >= thr)
    miss = (worst_v / thr - 1.0) if want_low else (1.0 - worst_v / thr)
    rows.append((0.0 if ok else miss, name, worst_v, thr, ok))
  rows.sort(reverse=True)
  for miss, name, v, thr, ok in rows:
    mark = "?    " if ok is None else ("OK  " if ok else "ECHEC")
    extra = "  NON MESURE" if ok is None else ("" if ok else f"   ({miss * 100:+.0f}% du seuil)")
    print(f"  {mark} {name:20s} {v:8.4f}  seuil {thr:6.3f}{extra}")
  # `is False`, not `not r[4]`: an unmeasured criterion is None, and `not None`
  # is True -- it would come back as the next target, which is the same mistake
  # the nan coercion made one layer down.
  bad = [r for r in rows if r[4] is False]
  unmeasured = [r[1] for r in rows if r[4] is None]
  if unmeasured:
    print("  (non mesure : " + ", ".join(unmeasured) + ")")
  print("\n=> " + ("tous les criteres mesures passent" if not bad
                   else f"prochaine cible : {bad[0][1]}"))


if __name__ == "__main__":
  main()
