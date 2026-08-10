"""Evenements de randomisation propres a la tache velocity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_actuator_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stiffness_range: tuple[float, float] = (0.9, 1.1),
  damping_range: tuple[float, float] = (0.9, 1.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Multiplie kp et kd par un facteur tire par environnement et par joint.

  Pourquoi : les gains 20000/400 ne sont pas une mesure, ce sont les valeurs
  d'un servo de position emule -- le vrai robot cache sa boucle P/PI dans le
  drive et on ne connait pas les gains equivalents. Une politique entrainee sur
  une seule valeur apprend implicitement la reponse exacte de ce PD-la ; la
  randomiser l'oblige a rester correcte sur une famille de plants, ce qui est
  precisement le transfert qu'on cherche.

  Les facteurs sont tires INDEPENDAMMENT par articulation, pas un seul par
  robot : un desaccord entre articulations est le cas realiste (chaque drive a
  son propre reglage) et c'est aussi le plus dur, donc le plus utile.

  Le tirage part toujours de ``default_stiffness`` / ``default_damping``, jamais
  de la valeur courante -- sinon les facteurs se composent d'un episode au
  suivant et les gains derivent sans borne.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  for act in asset.actuators:
    stiffness = getattr(act, "stiffness", None)
    damping = getattr(act, "damping", None)
    default_k = getattr(act, "default_stiffness", None)
    default_d = getattr(act, "default_damping", None)
    if stiffness is None or damping is None:
      continue
    if default_k is None or default_d is None:
      # Un actionneur sans defaut memorise ne peut pas etre randomise sans
      # composer les facteurs. Le sauter en silence serait pire que de le dire.
      raise RuntimeError(
        f"randomize_actuator_gains: l'actionneur {type(act).__name__} n'expose "
        "pas default_stiffness/default_damping"
      )
    shape = (len(env_ids), stiffness.shape[1])
    fk = torch.empty(shape, device=stiffness.device).uniform_(*stiffness_range)
    fd = torch.empty(shape, device=damping.device).uniform_(*damping_range)
    stiffness[env_ids] = default_k[env_ids] * fk
    damping[env_ids] = default_d[env_ids] * fd
