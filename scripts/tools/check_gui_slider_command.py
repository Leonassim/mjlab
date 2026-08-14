"""Le curseur du viewer atteint-il vraiment la policy ?

Le terme de commande zero les environnements tires "immobiles" (rel_standing_envs
= 0.4, conserve en mode play) A L'INTERIEUR de _update_command, juste avant
_emit_command. L'override du curseur viser, lui, tourne APRES super().compute(),
donc il ecrit dans vel_command_b une valeur que le compute suivant remet a zero
avant de la propager vers vel_command_out -- qui est ce que la policy observe.

Le chemin manette est different : get_gamepad_command() est appele DANS
_update_command, apres le zerotage, donc lui passe.

Ce script reproduit les deux chemins sur l'env 0 force en "immobile".

  uv run python scripts/tools/check_gui_slider_command.py
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-RHPS1"


def main() -> int:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 4
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg, device=device)
  cmd = env.command_manager.get_term("twist")

  env.reset()
  print(f"rel_standing_envs = {cmd.cfg.rel_standing_envs}")
  print(f"vel_ramp_rate     = {cmd.cfg.vel_ramp_rate}")
  print(
    f"env 0 : heading={bool(cmd.is_heading_env[0])} "
    f"world={bool(cmd.is_world_env[0])}"
  )

  # Faux handles viser : meme interface que ce que create_gui() installe.
  class _Handle:
    def __init__(self, value: float) -> None:
      self.value = value

  cmd._joystick_enabled = _Handle(True)  # type: ignore[assignment]
  cmd._joystick_sliders = [_Handle(0.3), _Handle(0.0), _Handle(0.0)]
  cmd._joystick_get_env_idx = lambda: 0

  for standing in (False, True):
    cmd.is_standing_env[:] = standing
    for _ in range(20):
      env.command_manager.compute(dt=env.step_dt)
    seen = float(cmd.command[0, 0])
    print(
      f"  env 0 {'immobile' if standing else 'marche  '} : "
      f"curseur 0.30 -> la policy observe {seen:+.3f}"
    )

  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
