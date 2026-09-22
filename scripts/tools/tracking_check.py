"""Le robot leve-t-il le pied, et tient-il debout ? Sur la tache tracking.

Les metriques de la tache tracking ne publient ni hauteur de pied ni chutes --
elles mesurent l'ecart a la reference. Or c'est le lever de pied qui n'a jamais
bouge en dix jours, donc c'est lui qu'il faut lire directement.

  uv run python scripts/tools/tracking_check.py <run> <model.pt> [--secs 20]
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1"
STEP_DT = 0.005


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--secs", type=float, default=20.0)
  p.add_argument("--envs", type=int, default=16)
  p.add_argument("--motion", default="docs/bwc_motion.npz")
  p.add_argument("--root", default="logs/rsl_rl/rhps1_tracking")
  p.add_argument("--task", default=TASK)
  p.add_argument("--student", default=None,
                 help="checkpoint de l ELEVE distille ; sinon l experte")
  a = p.parse_args()
  steps = int(a.secs / STEP_DT)

  task = a.task
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = a.envs
  cfg.commands["motion"].motion_file = a.motion
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device="cpu"),
    clip_actions=load_rl_cfg(task).clip_actions,
  )
  if a.student:
    # L'eleve est un reseau autonome, pas une politique rsl_rl : il lit le groupe
    # "student" (99 dims, deployable) la ou l'experte lit "actor" (165 dims).
    from scripts.tools.distill import Student

    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    net = Student(ck["n_in"], ck["n_out"])
    net.load_state_dict(ck["model"])
    net.eval()

    def policy(obs):
      with torch.no_grad():
        return net(obs["student"])
  else:
    runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device="cpu")
    runner.load(str(Path(a.root) / a.run / a.checkpoint),
                load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

  robot = env.unwrapped.scene["robot"]
  bodies = list(robot.body_names)
  feet = [bodies.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
  # Saturation de couple, articulation par articulation. C'est le critere de
  # Leo : la politique ne doit pas commander au-dela des limites. Lue sur le
  # couple APPLIQUE rapporte a effort_limit, sur les actionneurs de jambe.
  # qfrc_actuator : le couple en espace ARTICULAIRE, 30 colonnes dans l'ordre de
  # robot.joint_names. actuator_force en a 60 -- le XML porte ses propres
  # actionneurs en plus de ceux de mjlab -- et ses colonnes sont permutees par
  # ctrl_ids, ce qui a donne trois lectures fausses dont une a 0.000 partout.
  # Une correspondance par NOM sur un canal a une colonne par articulation ne
  # peut pas se decaler.
  joints = list(robot.joint_names)
  lim_by_joint = {}
  for x in robot.actuators:
    if not hasattr(x, "force_limit"):
      continue
    for jj, n in enumerate(x._target_names):
      lim_by_joint[n] = float(x.force_limit[0, jj])
  leg_names = [n for n in joints
               if any(t in n for t in ("CROTCH", "KNEE", "ANKLE")) and n in lim_by_joint]
  leg_cols = [joints.index(n) for n in leg_names]
  leg_lim = torch.tensor([lim_by_joint[n] for n in leg_names])
  sat = []

  obs, _ = env.reset()
  alive = torch.ones(a.envs, dtype=torch.bool)
  life = torch.full((a.envs,), float(steps))
  zs = []
  with torch.no_grad():
    for i in range(steps):
      obs, _, dones, _ = env.step(policy(obs))
      newly = alive & dones.bool()
      life[newly] = float(i)
      alive &= ~dones.bool()
      # TOUS les environnements, pas seulement les vivants : le nombre de
      # survivants change quand un env se termine, et empiler des tableaux de
      # tailles differentes echoue. On masque ensuite avec `alive`.
      z_all = robot.data.body_link_pos_w[:, feet, 2].cpu().numpy()
      m = alive.cpu().numpy()
      zs.append(np.where(m[:, None], z_all, np.nan))
      tau = robot.data.qfrc_actuator[:, leg_cols].abs() / leg_lim
      sat.append(np.where(m[:, None], tau.cpu().numpy(), np.nan))
      if not alive.any():
        break

  z = np.concatenate([q.reshape(-1) for q in zs])
  z = z[~np.isnan(z)]
  ground = np.percentile(z, 2)
  dur = len(zs) * STEP_DT
  h = z - ground
  peaks = h[h > 0.005]
  fell = int((life < steps).sum())
  print(f"\n{a.run} {a.checkpoint}   {a.envs} envs   {a.secs:.0f} s")
  print(f"  survie mediane        {float(life.median()) * STEP_DT:6.1f} s   "
        f"termines {fell}/{a.envs}")
  print(f"  hauteur de pied       p50 {np.median(h) * 100:5.2f} cm   "
        f"p90 {np.percentile(h, 90) * 100:5.2f}   p99 {np.percentile(h, 99) * 100:5.2f}   "
        f"max {h.max() * 100:5.2f}")
  if peaks.size:
    print(f"  au-dessus de 5 mm     p50 {np.median(peaks) * 100:5.2f} cm   "
          f"fraction {peaks.size / h.size:5.3f}")
  # CADENCE. Comparer des percentiles de hauteur ne dit pas combien de pas sont
  # faits : une politique qui leve bien mais deux fois trop rarement a les memes
  # percentiles bas qu'une politique qui leve mal. La reference fait 27.9 % du
  # temps en l'air ; c'est ce nombre-la qu'il faut egaler, pas une hauteur.
  zz = np.stack(zs) if len(zs) else None
  if zz is not None:
    hh = zz.reshape(len(zs), -1) - ground
    air = np.nan_to_num(hh, nan=-1.0) > 0.005
    # phases de vol : transitions montantes, moyennees sur les colonnes
    tr = np.diff(air.astype(int), axis=0)
    n_flights = (tr == 1).sum() / max(air.shape[1], 1)
    print(f"  cadence               {n_flights / dur:5.2f} vols/s/pied   "
          f"reference 0.56   (fraction en l air {air.mean():5.3f} contre 0.279)")
  print(f"  reference BWC         pic median 5.4 cm, max 7.5")
  if sat:
    r = np.concatenate(sat, axis=0)
    r = r[~np.isnan(r).any(axis=1)]
    if r.max() < 1e-6:
      print("  couple jambes         LECTURE INVALIDE (0.000 partout) -- "
            "colonnes mal appariees, ne pas interpreter")
      raise SystemExit(1)
    print(f"  couple jambes         |tau|/limite  p50 {np.median(r):5.3f}  "
          f"p90 {np.percentile(r, 90):5.3f}  max {r.max():5.3f}   "
          f"fraction >= 1 : {(r >= 0.999).mean():5.3f}")
    worst = np.argsort(-r.mean(0))[:3]
    print("  pires articulations   " + "  ".join(
      f"{leg_names[k]} {r[:, k].mean():.2f}" for k in worst))


if __name__ == "__main__":
  main()
