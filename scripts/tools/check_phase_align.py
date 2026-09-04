"""Balayer les 50 decalages de phase possibles entre la politique et le profil.

Si l'alignement est bon, l'erreur est minimale au decalage 0. Si le minimum est
ailleurs, le mapping phase -> indice est faux et la reference tire la politique
vers le mauvais moment du cycle -- ce qui explique une erreur PIRE que celle
d'une politique immobile.
"""
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK="Mjlab-Velocity-Flat-RHPS1"
LEGS=("L_CROTCH_Y","L_CROTCH_R","L_CROTCH_P","L_KNEE_P","L_ANKLE_R","L_ANKLE_P",
      "R_CROTCH_Y","R_CROTCH_R","R_CROTCH_P","R_KNEE_P","R_ANKLE_R","R_ANKLE_P")

def main():
  p=argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--steps",type=int,default=400); p.add_argument("--envs",type=int,default=256)
  p.add_argument("--root",default="logs/rsl_rl/rhps1_velocity"); a=p.parse_args()
  dev="cuda:0" if torch.cuda.is_available() else "cpu"
  cfg=load_env_cfg(TASK,play=True); cfg.scene.num_envs=a.envs
  env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device=dev),
                         clip_actions=load_rl_cfg(TASK).clip_actions)
  r=(load_runner_cls(TASK) or MjlabOnPolicyRunner)(env,asdict(load_rl_cfg(TASK)),device=dev)
  r.load(str(Path(a.root)/a.run/a.checkpoint),load_cfg={"actor":True},strict=True,map_location=dev)
  pol=r.get_inference_policy(device=dev)
  term=env.unwrapped.command_manager.get_term("twist")
  v=torch.tensor([0.2,0.0,0.0],device=dev)
  def fix(ids):
    term.vel_command_b[ids]=v; term.is_standing_env[ids]=False; term.is_heading_env[ids]=False
  term._resample_command=fix
  prof=json.load(open("docs/bwc_gait_profile.json"))
  N=prof["phase_bins"]
  P=torch.tensor([prof["joints"][n] for n in LEGS],dtype=torch.float,device=dev).t()  # [N,J]
  rob=env.unwrapped.scene["robot"]; names=list(rob.joint_names)
  ids=[names.index(n) for n in LEGS]
  gait=env.unwrapped.reward_manager.get_term_cfg("gait_phase").func
  sw=float(env.unwrapped.reward_manager.get_term_cfg("gait_phase").params["swing_duration"])
  per=float(env.unwrapped.reward_manager.get_term_cfg("gait_phase").params["period_slow"])
  sr=sw/per
  errs=torch.zeros(N,device=dev); n=0
  obs,_=env.reset(); fix(slice(None))
  with torch.no_grad():
    for i in range(a.steps):
      obs,_,_,_=env.step(pol(obs))
      ph=getattr(gait,"phase",None)
      if ph is None or i<=a.steps//4: continue
      q=rob.data.joint_pos[:,ids]
      base=((ph-sr)%1.0)*N
      for k in range(N):
        idx=((base+k)%N).long().clamp(0,N-1)
        errs[k]+=torch.mean(torch.square(q-P[idx]))
      n+=1
  e=(errs/max(n,1)).sqrt().cpu().numpy()
  best=int(np.argmin(e))
  print(f"\nerreur RMS par decalage (rad), {n} pas")
  for k in range(0,N,2):
    m=" <-- MINIMUM" if k==best else (" (mon alignement)" if k==0 else "")
    print(f"  decalage {k:2d} ({k/N*100:5.1f} % du cycle) : {e[k]:.4f}{m}")
  print(f"\nmon alignement (0) : {e[0]:.4f}")
  print(f"meilleur ({best}, {best/N*100:.0f} % du cycle) : {e[best]:.4f}")
  print(f"gain possible : {(e[0]-e[best])/e[0]*100:.0f} %")

if __name__=="__main__": main()
