import sys, mc_log_ui, numpy as np
l=mc_log_ui.read_log(sys.argv[1]); t=np.asarray(l['t']); g=lambda k: np.asarray(l[k])
ax,ay,az=[g('Accelerometer_linearAcceleration_'+a) for a in 'xyz']
k=40; f=lambda x: np.convolve(x,np.ones(k)/k,mode='same'); ax,az=f(ax),f(az)
pitch=np.degrees(np.arctan2(-ax,az)); z=g('FloatingBase_position_z')
a=np.stack([g(f'NewRLQPController_RL_currentAction_{i}') for i in range(30)],1); ta=t[np.argmax(np.abs(a).sum(1)>0)]
vc=g('NewRLQPController_velCmd_x'); run=t>ta
fall=np.where(run&((z<0.6)|(pitch<-20)|(pitch>20)))[0]
msg=f"FELL at +{t[fall[0]]-ta:.1f} s after arm (cmd vx {vc[fall[0]]:+.2f})" if len(fall) else "no fall"
back=run&(vc<-0.01)
# 1 Hz pitch oscillation amplitude while standing after the forward walk
st=run&(t>ta+2+3+1)&(t<ta+2+3+5)
print(f"{sys.argv[2]:22s} {msg:38s} | pitch min {pitch[run].min():+6.1f} | in back {pitch[back].min() if back.any() else float('nan'):+6.1f} | stand p2p {np.ptp(pitch[st]) if st.any() else float('nan'):4.1f}")
