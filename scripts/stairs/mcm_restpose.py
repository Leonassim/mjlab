import sys, mc_log_ui, numpy as np
l=mc_log_ui.read_log(sys.argv[1]); t=np.asarray(l['t']); g=lambda k: np.asarray(l[k])
R=["L_CROTCH_Y","L_CROTCH_R","L_CROTCH_P","L_KNEE_P","L_ANKLE_R","L_ANKLE_P","CHEST_Y","CHEST_P","R_CROTCH_Y","R_CROTCH_R","R_CROTCH_P","R_KNEE_P","R_ANKLE_R","R_ANKLE_P"]
z=np.stack([g(f'NewRLQPController_RL_qZero_{i}') for i in range(14)],1)
q=np.stack([g(f'qIn_{i}') for i in range(14)],1)
act=np.stack([g(f'NewRLQPController_RL_currentAction_{i}') for i in range(30)],1); armed=np.abs(act).sum(1)>0
ta=t[armed][0]
vc=g('NewRLQPController_velCmd_x'); on=np.where(np.abs(vc)>1e-3)[0]; t0,t1=t[on[0]],t[on[-1]]
w,x,y,zz=[g('Observers_Pipeline_MCWaiko_FloatingBase_world_pose_q'+a) for a in 'wxyz']
pitch=np.degrees(np.arcsin(np.clip(2*(w*(-y)-(-zz)*(-x)),-1,1)))
print(sys.argv[2], "qZero (deg):", " ".join(f"{n} {v:+.1f}" for n,v in zip(R,np.degrees(z[armed][0])) if n.endswith(("CROTCH_P","KNEE_P","ANKLE_P","CHEST_P"))))
for name,m in (("before arming",(t>ta-0.5)&(t<ta)),("pre-walk",(t>t0-1.5)&(t<t0-0.2)),("rest after",(t>t1+3)&(t<t1+6))):
  d=np.degrees((q[m]-z[m]).mean(0))
  print(f"  {name:14s} pelvis pitch {pitch[m].mean():+.1f} | q - q0: "+" ".join(f"{n} {v:+.1f}" for n,v in zip(R,d) if abs(v)>=0.5))
