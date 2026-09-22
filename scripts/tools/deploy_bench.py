"""L'EPREUVE DE DEPLOIEMENT : depart debout, direction constante, ou va-t-il ?

Les metriques internes a la tache tracking (pics, cadence, survie) ont laisse
passer une politique qui parcourait 0.099 m pour 2.25 m commandes -- elles
mesurent le suivi d'une reference dans laquelle le robot est teleporte, jamais
l'obeissance a une consigne depuis l'arret. Ce banc-ci mesure exactement ce que
mc_mujoco demande.
"""
import argparse, numpy as np, torch
from dataclasses import asdict
from pathlib import Path
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab.tasks.tracking.config.rhps1.env_cfgs as E

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("checkpoint")
p.add_argument("--task", default="Mjlab-Tracking-Flat-RHPS1-Copy-V3-Cadence")
p.add_argument("--secs", type=float, default=12.0)
p.add_argument("--envs", type=int, default=8)
a = p.parse_args()
steps = int(a.secs / 0.005)

MODES = [("immobile", (0,0,0)), ("avant", (1,0,0)), ("gauche", (0,1,0)),
         ("droite", (0,-1,0)), ("rotation +", (0,0,1))]
print(f"\n{a.run} {a.checkpoint}   {a.secs:.0f} s par mode, depart DEBOUT")
print(f"{'mode':>12s} {'avance':>9s} {'lateral':>9s} {'rotation':>10s} {'debout':>8s} "
      f"{'articul.':>8s} {'lever':>8s} {'duree':>7s}")
print("  (une vraie marche : amplitude articulaire >15 deg, lever de pied >5 cm)")
for nom, c in MODES:
    CMD = torch.tensor(c, dtype=torch.float32)
    cfg = load_env_cfg(a.task, play=True); cfg.scene.num_envs = a.envs
    # Standing start with a held command: the reference is irrelevant, so its
    # terminations only teleport envs. Falls are read from pelvis height.
    for _k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
      cfg.terminations.pop(_k, None)
    import os as _os
    if _os.environ.get("MCMUJOCO_PLANT"):
      # Reproduire le plant de mc_mujoco : Euler explicite a 1 kHz au lieu de
      # implicitfast a 400 Hz. decimation ajuste pour garder 200 Hz de politique.
      cfg.sim.mujoco.timestep = 0.001
      cfg.sim.mujoco.integrator = "euler"
      cfg.decimation = 5
    # Remplacer la fonction DANS LE TERME de la config chargee. Remplacer
    # l'attribut du module ne sert a rien : les configs de taches sont
    # construites A L'IMPORT (__init__.py), donc le terme garde une reference
    # vers la fonction d'origine. Mesure du 2026-09-15 : les cinq modes rendaient
    # un resultat identique au centimetre, parce qu'ils recevaient tous la
    # vitesse de la reference au lieu de ma consigne.
    for grp in ("actor", "student"):
      if grp in cfg.observations and "velocity_command" in cfg.observations[grp].terms:
        cfg.observations[grp].terms["velocity_command"].func = (
          lambda env, command_name, _c=CMD: _c.to(env.device).repeat(env.num_envs, 1)
        )
    cfg.commands["motion"].motion_file = "/home/lmoussafir/mjlab-rhps1/docs/bwc_motion_v2.npz"
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
    r = load_runner_cls(a.task)(env, asdict(load_rl_cfg(a.task)), device="cpu")
    r.load(str(Path("logs/rsl_rl/rhps1_tracking")/a.run/a.checkpoint),
           load_cfg={"actor": True}, strict=True, map_location="cpu")
    pol = r.get_inference_policy(device="cpu")
    robot = env.unwrapped.scene["robot"]
    obs, _ = env.reset()
    q0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.unwrapped.scene.env_origins
    robot.write_root_state_to_sim(root)
    p0 = robot.data.root_link_pose_w.clone()
    bodies = list(robot.body_names)
    feet = [bodies.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
    QJ, FZ = [], []
    # S'ARRETER A LA PREMIERE TERMINAISON. Sinon la reinitialisation teleporte le
    # robot a une posture de reference aleatoire, et max-min sur la fenetre
    # additionne ces sauts : mesure du 2026-09-16, 188 deg d'amplitude et 21 cm
    # de lever pour un robot qui ne marchait pas, avec des episodes d'une seconde.
    n_used = 0
    with torch.no_grad():
        for _ in range(steps):
            obs, _, dones, _ = env.step(pol(obs))
            if bool(dones[0]):
                break
            QJ.append(robot.data.joint_pos[0].cpu().numpy())
            FZ.append((robot.data.body_link_pos_w[0, feet, 2]
                       - env.unwrapped.scene.env_origins[0, 2]).cpu().numpy())
            n_used += 1
    p1 = robot.data.root_link_pose_w
    amp = np.degrees((np.stack(QJ).max(0) - np.stack(QJ).min(0))) if QJ else np.zeros(1)
    fz = np.stack(FZ) if FZ else np.zeros((1, 1))
    d = (p1[:, :2] - p0[:, :2])
    def yaw(q):
        w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
        return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    th0 = yaw(p0[:, 3:7])
    fwd = ( d[:,0]*torch.cos(th0) + d[:,1]*torch.sin(th0)).median().item()
    lat = (-d[:,0]*torch.sin(th0) + d[:,1]*torch.cos(th0)).median().item()
    dyaw = torch.atan2(torch.sin(yaw(p1[:,3:7])-th0), torch.cos(yaw(p1[:,3:7])-th0))
    up = (p1[:, 2] > 0.5).sum().item()
    lift = (fz.max(0) - np.percentile(fz, 2, axis=0)).max() * 100
    print(f"  {nom:>10s} {fwd:8.2f}m {lat:8.2f}m {np.degrees(dyaw.median().item()):9.0f}° "
          f"{up:4d}/{a.envs} {np.percentile(amp,95):7.1f}° {lift:7.1f}cm"
          f" {n_used*0.005:6.1f}s   |lacet| med {np.degrees(dyaw.abs().median().item()):.0f}° "
          f"[{np.degrees(dyaw.min().item()):+.0f}, {np.degrees(dyaw.max().item()):+.0f}]")
    del env, r
