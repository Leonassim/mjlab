"""Le robot va-t-il DANS LE BON SENS, et tient-il en place a zero ?

Critere refait le 2026-09-10 sur la demande de Leo : suivre pile la vitesse
commandee n'importe pas. Ce qui importe est
  - avancer sur une commande avant, reculer sur une commande arriere
  - rester stable et neutre a commande nulle

L'erreur de vitesse (Metrics/twist/error_vel_xy) mesurait autre chose et
recalait des politiques qui vont pourtant dans le bon sens.
"""
import os, numpy as np, torch
from dataclasses import asdict
from pathlib import Path
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
T="Mjlab-Velocity-Flat-RHPS1"
RUN=os.environ["RUN"]; CK=os.environ["CK"]
dev="cuda:0"
cfg=load_env_cfg(T,play=True); cfg.scene.num_envs=int(os.environ.get("ENVS","128"))
env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device=dev),clip_actions=load_rl_cfg(T).clip_actions)
r=(load_runner_cls(T) or MjlabOnPolicyRunner)(env,asdict(load_rl_cfg(T)),device=dev)
r.load(str(Path("logs/rsl_rl/rhps1_velocity")/RUN/CK),load_cfg={"actor":True},strict=True,map_location=dev)
pol=r.get_inference_policy(device=dev)
term=env.unwrapped.command_manager.get_term("twist"); rob=env.unwrapped.scene["robot"]
print(f"\n{RUN} {CK}")
print(f"{'commande':>10s} {'avance mediane':>15s} {'bon sens':>10s} {'derive laterale':>16s} {'chutes':>8s}")
for cx in (0.30, 0.20, -0.20, 0.0):
    v=torch.tensor([cx,0.,0.],device=dev)
    def fix(i, v=v):
        term.vel_command_b[i]=v; term.is_standing_env[i]=False; term.is_heading_env[i]=False
    term._resample_command=fix
    obs,_=env.reset(); fix(slice(None))
    V=[];L=[];alive=torch.ones(cfg.scene.num_envs,dtype=torch.bool,device=dev)
    with torch.no_grad():
        for i in range(400):
            obs,_,dones,_=env.step(pol(obs)); alive&=~dones.bool()
            if i>100:
                V.append(rob.data.root_link_lin_vel_b[:,0].cpu())
                L.append(rob.data.root_link_lin_vel_b[:,1].abs().cpu())
    V=torch.stack(V); L=torch.stack(L)
    med=float(V.median())
    ok = "--" if cx==0 else ("OUI" if (med>0.05 if cx>0 else med<-0.05) else "NON")
    fall=1.0-float(alive.float().mean())
    print(f"{cx:10.2f} {med:15.4f} {ok:>10s} {float(L.median()):16.4f} {fall:8.3f}")
