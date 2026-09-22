"""Re-export the ONNX of an arbitrary checkpoint.

Training only exports the ONNX beside the checkpoint it just saved, so a run's
`<run>.onnx` always corresponds to the LAST save. Deploying an earlier checkpoint
needs its own export -- swapping weights inside the existing graph is not it, the
observation normaliser is baked in too.

  uv run python scripts/tools/export_onnx.py <run> <model_XXXX.pt> [out.onnx]

Une tache autre que la velocity : MJLAB_TASK et MJLAB_LOG_ROOT dans
l'environnement. Le groupe `actor` de la tache decide du format d'observation
exporte, il doit donc etre exactement celui de l'entrainement.
"""

import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = os.environ.get("MJLAB_TASK", "Mjlab-Velocity-Flat-RHPS1")
LOG_ROOT = Path(os.environ.get("MJLAB_LOG_ROOT", "logs/rsl_rl/rhps1_velocity"))


def main() -> None:
  run, ckpt = sys.argv[1], sys.argv[2]
  out = Path(sys.argv[3]) if len(sys.argv) > 3 else LOG_ROOT / run / f"{Path(ckpt).stem}.onnx"
  ckpt_path = LOG_ROOT / run / ckpt
  if not ckpt_path.is_file():
    raise SystemExit(f"no such checkpoint: {ckpt_path}")

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # play=True and a single env: nothing is stepped, the env exists only so the
  # runner can size the networks and so get_base_metadata can read the joint
  # order, default pose and action scales that the deployment side depends on.
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1
  # Une tache de tracking exige son .npz : rien n'est joue ici, mais le
  # MotionCommand se construit au chargement.
  motion = os.environ.get("MJLAB_MOTION_FILE")
  if motion:
    env_cfg.commands["motion"].motion_file = motion
  agent_cfg = load_rl_cfg(TASK)

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device))
  runner_cls = load_runner_cls(TASK)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  # map_location: the checkpoint was saved from CUDA tensors, and this export
  # runs fine on CPU -- without it torch.load refuses on a machine with no GPU.
  runner.load(str(ckpt_path), map_location=device)

  # export_policy_to_onnx names the file after the directory, so export into a
  # scratch dir and move it -- otherwise it overwrites the run's own ONNX.
  tmp_dir = Path("scratch/.onnx_export")
  tmp_dir.mkdir(parents=True, exist_ok=True)
  runner.export_policy_to_onnx(str(tmp_dir), "policy.onnx")
  attach_metadata_to_onnx(str(tmp_dir / "policy.onnx"), get_base_metadata(env.unwrapped, run))

  out.parent.mkdir(parents=True, exist_ok=True)
  shutil.move(str(tmp_dir / "policy.onnx"), out)
  _strip_to_actions(out)
  print(f"wrote {out}")


def _strip_to_actions(path: Path) -> None:
  """Ne garder que obs -> actions.

  L'exportateur de la tache tracking ajoute une entree `time_step` et six
  sorties qui rejouent la reference du mouvement a cet instant (joint_pos,
  body_pos_w, ...). Le controleur n'en a aucun usage et RLPolicyInterface
  refuse tout modele a plus d'une entree : "Expected 1 input, got 2", puis
  obs=0 act=0 et retour a l'index 0. `actions` ne depend que de `obs`, le
  sous-graphe est donc exact -- ce n'est pas une troncature du calcul.
  """
  import onnx

  m = onnx.load(str(path))
  init = {i.name for i in m.graph.initializer}
  real_in = [i.name for i in m.graph.input if i.name not in init]
  if real_in == ["obs"] and [o.name for o in m.graph.output] == ["actions"]:
    return
  meta = {p.key: p.value for p in m.metadata_props}
  onnx.utils.extract_model(str(path), str(path), ["obs"], ["actions"])
  m = onnx.load(str(path))
  del m.metadata_props[:]
  for k, v in meta.items():
    e = m.metadata_props.add()
    e.key, e.value = k, v
  onnx.save(m, str(path))


if __name__ == "__main__":
  main()
