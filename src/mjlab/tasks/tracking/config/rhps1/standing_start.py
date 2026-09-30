"""Depart ARRETE depuis la posture de deploiement.

La tache de tracking teleporte le robot dans l'etat de la reference a chaque
reinitialisation. Il apprend donc a CONTINUER un mouvement, jamais a en DEMARRER
un depuis la posture ou mc_rtc le laisse au demarrage.

Mesure du 2026-09-15 : la posture par defaut (= le q0 du controleur, verifie
identique sur les 30 articulations) differe de la posture debout du BWC de 9.3
degres aux genoux. Environ 13 % des episodes traversent bien la transition
debout -> marche, mais toujours depuis la posture de la REFERENCE, jamais depuis
celle du deploiement. Resultat au banc, sur trois lectures consecutives : la
distance parcourue sur commande avant tombe de 0.49 m a 0.13 puis 0.04, pendant
que recompense et survie montent. L'entrainement s'ameliorait en s'eloignant de
ce qu'on veut.

Sous-classe plutot que modification du code amont : ce fichier survit aux fusions.
"""

from dataclasses import dataclass

import os

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class StandingStartMotionCommand(MotionCommand):
  """MotionCommand avec une fraction de departs arretes en posture q0."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    p = float(getattr(self.cfg, "standing_start_prob", 0.0))
    if p <= 0.0 or len(env_ids) == 0:
      return

    pick = env_ids[torch.rand(len(env_ids), device=self.device) < p]
    if len(pick) == 0:
      return

    # Robot et reference SYNCHRONISES, tous deux debout juste avant le depart
    # de la marche du BWC.
    #
    # v1 (2026-09-15) : reference dans le prologue, robot en posture q0. La
    # commande y vaut zero -> j'ajoutais 30 % d'episodes qui recompensent
    # l'immobilite. Distance sur commande : 0.04 -> 0.01 m.
    #
    # v2 : reference au debut de la marche, robot toujours en q0. Le robot
    # demarrait donc DESYNCHRONISE -- 9.3 degres d'ecart aux genoux et un
    # decalage de phase. Ne pouvant pas rattraper la reference en marchant, il a
    # appris a se TRAINER vers elle : 1.84 m parcourus avec seulement 3 degres
    # d'amplitude articulaire, quand une vraie marche en demande des dizaines.
    #
    # v3 : le robot prend la posture de la REFERENCE a cette trame, donc ils
    # partent ensemble. La politique apprend la sequence de demarrage telle
    # qu'elle existe dans le log, au lieu de compenser un ecart initial.
    # v5 (2026-09-19) : l'attente d'armement se place DANS le prologue immobile
    # de la reference, qui dure 14 s. v4 la mettait au depart de la marche, donc
    # la reference s'eloignait de 42 cm pendant que le robot etait tenu, et la
    # politique reprenait la main en retard et dephasee : suivi 0.80 -> 0.16.
    w0 = int(self._standing_end())
    hmax = float(os.environ.get("RHPS1_ACTION_HOLD_S", "3.0"))
    hold_steps = torch.randint(
      0, max(int(hmax / self._env.step_dt), 1), (len(pick),), device=self.device
    )
    # Free stand: the policy holds the robot at command 0 for a random time
    # after the hold, then the command flips where the reference's does. The
    # operator's joystick push on the robot, which training never showed in
    # control (a jolt at the flip, 2026-09-30).
    free = [float(x) for x in os.environ.get("RHPS1_STAND_FREE_S", "0,0").split(",")]
    if free[1] > 0.0:
      lo, hi = int(free[0] / self._env.step_dt), int(free[1] / self._env.step_dt)
      free_steps = torch.randint(lo, max(hi, lo + 1), (len(pick),), device=self.device)
      self.time_steps[pick] = (self._command_flip() - free_steps - hold_steps).clamp(min=0)
    else:
      self.time_steps[pick] = (w0 - hold_steps).clamp(min=0)

    # Racine et posture prises sur la reference a la trame choisie, vitesses
    # nulles : le robot EST la reference a l'arret, juste avant son depart.
    root_pos = self.body_pos_w[pick, 0].clone()
    root_ori = self.body_quat_w[pick, 0].clone()
    zeros3 = torch.zeros(len(pick), 3, device=self.device)
    q = self.joint_pos[pick].clone()

    # v4 (2026-09-19) : une part des departs se fait depuis q0, la posture reelle
    # du robot au deploiement -- mesuree a 11.9 deg du genou droit de la
    # reference. A 100 % de posture de reference, la politique n'apprend a
    # demarrer que d'un etat qu'elle ne verra jamais sur le robot : model_3500
    # marchait dans la reference et restait fige sous mc_mujoco.
    import os as _os

    frac = float(_os.environ.get("RHPS1_STANDING_Q0_FRAC", "0.5"))
    if frac > 0.0:
      robot = self._env.scene["robot"]
      at_q0 = torch.rand(len(pick), device=self.device) < frac
      q = torch.where(at_q0[:, None], robot.data.default_joint_pos[pick], q)
      # La hauteur du bassin va avec la posture : q0 a les genoux plus flechis.
      root_pos[:, 2] = torch.where(
        at_q0, robot.data.default_root_state[pick, 2], root_pos[:, 2]
      )
    # Duree pendant laquelle les actions sont ignorees : le robot est tenu a sa
    # posture, comme avant l'armement dans mc_mujoco (0 a 3 s).
    hold = getattr(self._env, "_action_hold_steps", None)
    if hold is None:
      hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self._env._action_hold_steps = hold
    hold[env_ids] = 0  # only RSI envs resampled here keep no hold
    hold[pick] = hold_steps  # meme tirage que le recul du temps de reference

    self._write_reference_state_to_sim(
      pick, root_pos, root_ori, zeros3, zeros3, q, torch.zeros_like(q)
    )

  def _command_flip(self) -> int:
    """First frame of the command that the walk after the prologue carries."""
    if not hasattr(self, "_command_flip_cache"):
      from mjlab.tasks.tracking.config.rhps1.env_cfgs import _reference_velocity_command

      _reference_velocity_command(self._env, "motion")  # builds self._dir_cmd
      on = (self._dir_cmd != 0).any(dim=1)
      f = self._standing_end()
      while f > 0 and on[f - 1]:
        f -= 1
      self._command_flip_cache = f
    return self._command_flip_cache

  def _standing_end(self) -> int:
    """Derniere trame du prologue debout, calculee une fois sur le mouvement."""
    if not hasattr(self, "_standing_end_cache"):
      idx = [self.cfg.body_names.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
      z = self.motion.body_pos_w[:, idx, 2]
      moving = (z.max(dim=1).values > 0.11).nonzero()
      self._standing_end_cache = int(moving[0]) if len(moving) else z.shape[0] // 10
    return self._standing_end_cache


@dataclass(kw_only=True)
class StandingStartMotionCommandCfg(MotionCommandCfg):
  standing_start_prob: float = 0.0
  """Fraction des reinitialisations qui placent le robot ARRETE dans la posture
  de la reference, juste avant le depart de la marche du BWC."""

  def build(self, env) -> StandingStartMotionCommand:
    return StandingStartMotionCommand(self, env)
