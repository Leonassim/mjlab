import os, sys, numpy as np, torch
from dataclasses import asdict
from pathlib import Path
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
run, ck, label = sys.argv[1], sys.argv[2], sys.argv[3]
TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"; N = 16; SECS = 20.0
torch.manual_seed(1)
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = N
for _k in ("anchor_pos", "anchor_ori", "ee_body_pos"): cfg.terminations.pop(_k, None)
if "noise" in label: cfg.observations["actor"].enable_corruption = True
for _only in ("base_lin_vel", "base_ang_vel", "joint_vel", "joint_pos"):
    if f"only_{_only}" in label:
        for _k, _t in cfg.observations["actor"].terms.items():
            if _k != _only: _t.noise = None
if "plant" in label:
    cfg.sim.mujoco.timestep = 0.001; cfg.sim.mujoco.integrator = "euler"; cfg.decimation = 5
if "fric1" in label:
    import dataclasses
    art = cfg.scene.entities["robot"]
    art.collisions = tuple(dataclasses.replace(c, friction={k: (1.0,) for k in c.friction} if isinstance(c.friction, dict) else (1.0,)) for c in art.collisions)
CMD = torch.tensor([1., 0., 0.])
cfg.observations["actor"].terms["velocity_command"].func = lambda env, command_name, _c=CMD: _c.to(env.device).repeat(env.num_envs, 1)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v2.npz"
if hasattr(cfg.commands["motion"], "standing_start_prob"): cfg.commands["motion"].standing_start_prob = 0.0
rl = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=rl.clip_actions)
r = load_runner_cls(TASK)(env, asdict(rl), device="cpu")
r.load(str(Path("logs/rsl_rl/rhps1_tracking")/run/ck), load_cfg={"actor": True}, strict=True, map_location="cpu")
pol = r.get_inference_policy(device="cpu")
robot = env.unwrapped.scene["robot"]
obs, _ = env.reset()
q0 = robot.data.default_joint_pos.clone(); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
root = robot.data.default_root_state.clone(); root[:, :3] += env.unwrapped.scene.env_origins; robot.write_root_state_to_sim(root)
def yaw(q): w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]; return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
if "settle" in label:
    with torch.no_grad():
        for _ in range(int(3.0 / env.unwrapped.step_dt)):
            obs, _, _, _ = env.step(torch.zeros(N, env.num_actions))
p0 = robot.data.root_link_pose_w.clone(); th = yaw(p0[:,3:7]); prev = th.clone(); acc = torch.zeros(N)
alive = torch.ones(N, dtype=torch.bool); out = {}
import sys as _sys; _sys.path.insert(0, "scripts/tools")
from gait_stats import SOLE, gait_stats
from mjlab.utils.lab_api.math import quat_apply
_bn = list(robot.body_names); _fb = [_bn.index("L_ANKLE_P_LINK"), _bn.index("R_ANKLE_P_LINK")]
_sole = torch.tensor(SOLE, dtype=torch.float32); SOLES = []; FORCES = []
steps = int(SECS / env.unwrapped.step_dt)
with torch.no_grad():
    for i in range(steps):
        obs, _, dones, _ = env.step(pol(obs))
        alive &= ~dones.bool() & (robot.data.root_link_pos_w[:, 2] > 0.5)
        _p = robot.data.body_link_pos_w[:, _fb]; _q = robot.data.body_link_quat_w[:, _fb]
        SOLES.append((_p + quat_apply(_q, _sole.expand_as(_p))).numpy())
        _f = env.unwrapped.scene["feet_ground_contact"].data.force
        FORCES.append(_f.norm(dim=-1).squeeze(-1).numpy())
        y = yaw(robot.data.root_link_pose_w[:,3:7]); dy = torch.atan2(torch.sin(y-prev), torch.cos(y-prev)); acc += torch.where(alive, dy, 0*dy); prev = y
        if i+1 in (int(12/env.unwrapped.step_dt), steps):
            d = (robot.data.root_link_pose_w[:, :2] - p0[:, :2]).norm(dim=1)
            out[round((i+1)*env.unwrapped.step_dt)] = (d.clone(), torch.rad2deg(acc).clone(), alive.clone())
for t,(d,a,al) in out.items():
    print(f"{label:>12s} {t:2d}s debout {int(al.sum())}/{N}  dist med {d[al].median():.2f} m  lacet med {a[al].median():+.0f}  |lacet| med {a[al].abs().median():.0f}  min {a[al].min() if al.any() else 0:+.0f} max {a[al].max() if al.any() else 0:+.0f}")
_F = np.stack(SOLES)
_ok = [e for e in range(N) if bool(alive[e])][:4]
for e in _ok[:1]:
    _Fe = _F[:, e]; _st = _Fe[:, :, 2] < np.percentile(_Fe[:, :, 2], 30, axis=0) + 0.003
    _co = np.stack(FORCES)[:, e] > 5.0
    print(f"{label:>12s} pieds env{e} : " + gait_stats(_Fe, _st, env.unwrapped.step_dt, contact=_co))
