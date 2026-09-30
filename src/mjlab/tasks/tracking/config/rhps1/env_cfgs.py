"""Configuration de suivi de mouvement (BeyondMimic) pour le RHPS1.

Calquee sur config/g1/env_cfgs.py. Ce qui change est la liste des corps, et le
pas de temps : la tache tracking tourne a 50 Hz par defaut (0.005 x 4) alors que
le RHPS1 est deploye a 200 Hz -- mc_rtc tourne a 5 ms et le controleur infere a
chaque tick. Un plant a 50 Hz ne se deploierait pas.

POURQUOI CETTE TACHE plutot que le terme bwc_ref fait maison : celui-ci compare
des ANGLES articulaires indexes par une phase, et sur 6000 iterations il a
stabilise le robot sans jamais faire lever le pied -- plat a 9 mm contre 54 pour
le BWC. La hauteur du pied est une petite difference entre de grands mouvements
(hanche, genou et cheville se compensent), donc suivre les angles ne la contraint
pas. Cette tache suit les POSES DE CORPS dans l'espace : un pied a la bonne
position ne peut pas rester au sol.
"""

import math
import os
from typing import cast

import torch

from mjlab.asset_zoo.robots.RHPS1.rhps1_constants import (
  RHPS1_ACTION_SCALE,
  get_rhps1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp

# Paires de proximite, reprises telles quelles de la tache velocity. Leo les a
# demandees explicitement comme protection MATERIELLE : le QP du robot traite ces
# memes paires (mc_rhps1/src/rhps1.cpp, Collision(body1, body2, iDist, sDist)) et
# une politique qui ne les a jamais vues se fait bloquer par le QP a une distance
# qu'elle ne connait pas. Les distances different par paire, comme cote robot.
#
# ANKLE_R et non ANKLE_P : le modele MuJoCo porte ses coques sur ANKLE_R_LINK.
_PROXIMITY = (
  ("crotch_proximity", r"^rhps1_collision_L_CROTCH_P_LINK$",
   r"^rhps1_collision_R_CROTCH_P_LINK$", 0.04),
  ("leg_proximity", r"^rhps1_collision_L_(KNEE_P|ANKLE_R)_LINK$",
   r"^rhps1_collision_R_(KNEE_P|ANKLE_R)_LINK$", 0.02),
  ("knee_proximity", r"^rhps1_collision_L_KNEE_P_LINK$",
   r"^rhps1_collision_R_KNEE_P_LINK$", 0.035),
  ("arm_torso_proximity", r"^rhps1_collision_[LR]_(ELBOW_Y|WRIST_Y)_LINK$",
   r"^rhps1_collision_(CHEST_P_LINK|BODY)$", 0.05),
  ("shoulder_chest_proximity", r"^rhps1_collision_[LR]_SHOULDER_Y_LINK$",
   r"^rhps1_collision_CHEST_P_LINK$", 0.02),
  ("shoulder_body_proximity", r"^rhps1_collision_[LR]_SHOULDER_Y_LINK$",
   r"^rhps1_collision_BODY$", 0.05),
  ("wrist_thigh_proximity", r"^rhps1_collision_[LR]_WRIST_Y_LINK$",
   r"^rhps1_collision_[LR]_CROTCH_P_LINK$", 0.05),
)


def _proximity_sensor(name: str, primary: str, secondary: str) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=primary, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern=secondary, entity="robot"),
    fields=("found", "dist"),
    reduce="mindist",
    num_slots=1,
  )

# Les quatorze corps suivis. Meme decoupage que le G1 : le flottant, les trois
# etages de chaque jambe, le torse, et les trois etages de chaque bras.
_TRACKED_BODIES = (
  "BODY",
  "L_CROTCH_R_LINK",
  "L_KNEE_P_LINK",
  "L_ANKLE_P_LINK",
  "R_CROTCH_R_LINK",
  "R_KNEE_P_LINK",
  "R_ANKLE_P_LINK",
  "CHEST_P_LINK",
  "L_SHOULDER_R_LINK",
  "L_ELBOW_P_LINK",
  "L_WRIST_Y_LINK",
  "R_SHOULDER_R_LINK",
  "R_ELBOW_P_LINK",
  "R_WRIST_Y_LINK",
)



def _feet_height_error_exp(
  env, command_name: str, std: float, body_names: tuple[str, ...]
) -> "torch.Tensor":
  """Suivre la HAUTEUR des pieds seule, relative a l'ancre.

  motion_feet_pos somme les carres sur les trois axes. Mesure sur la reference,
  mouvement du pied relatif au bassin :

      avant-arriere  ecart-type 4.71 cm
      lateral        ecart-type 3.34 cm
      vertical       ecart-type 1.88 cm

  Le vertical ne pese donc que ~9 % de la variance : en placant le pied en
  avant-arriere et en lateral, la politique satisfait 91 % du terme SANS avoir a
  le lever. C'est ce qui a maintenu p90 plat a 1.75 cm sur 6000 iterations
  pendant que le maximum montait a 9 cm -- quelques pas tres hauts, le pied au
  sol le reste du temps.

  std 0.02 : une erreur de hauteur de 2 cm coute la moitie du terme.
  """
  cmd = env.command_manager.get_term(command_name)
  idx = [cmd.cfg.body_names.index(n) for n in body_names]
  err = torch.square(
    cmd.body_pos_relative_w[:, idx, 2] - cmd.robot_body_pos_w[:, idx, 2]
  )
  return torch.exp(-err.mean(-1) / (std * std))



LIFT_CREDIT_FRACTION = float(os.environ.get("RHPS1_LIFT_CREDIT", "0.6"))


def _randomize_posture_stiffness(env, env_ids, k_range: tuple[float, float]) -> None:
  """Tirer le retard du filtre de posture a chaque reinitialisation.

  Le filtre de la PostureTask de mc_rtc ajoute un retard ~1/sqrt(K). Mesure du
  2026-09-16 : a K fixe, la politique GARDE sa demarche mais marche a l'envers
  (+1.69 m sans filtre, -1.70 m a K=8000, meme amplitude et meme lever). Et la
  reentrainer a K fixe l'effondre : le chemin de "marcher en arriere" vers
  "marcher en avant" passe par "ne plus marcher", qui est un optimum local
  confortable. Elle s'y installe -- constate a K=8000 comme a K=64000.

  En tirant K par episode sur toute la plage, aucune phase particuliere ne peut
  etre exploitee : la politique doit construire une demarche qui avance quel que
  soit le retard. C'est aussi ce qui la rendra robuste au robot reel, dont le
  filtre effectif ne sera pas exactement celui de la simulation.
  """
  lo, hi = k_range
  for act in env.scene["robot"].actuators:
    if not hasattr(act, "posture_stiffness"):
      continue
    n = len(env_ids)
    u = torch.rand(n, 1, device=act.posture_stiffness.device)
    k = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))  # log-uniforme
    act.posture_stiffness[env_ids] = k.expand_as(act.posture_stiffness[env_ids])


def _contact_schedule_match(
  env, command_name: str, sensor_name: str, body_names: tuple[str, ...],
  air_threshold: float = 0.01,
) -> "torch.Tensor":
  """Le pied est-il en l'air QUAND la reference l'est ?

  Le deficit qui restait apres la phase 1 n'etait pas la hauteur -- 7.96 cm de
  pic par cycle contre 8.69 pour la reference, 8 % d'ecart -- mais le NOMBRE de
  pas : 114 cycles contre 160 sur 20 s. Le robot reste en appui trop longtemps.

  Aucun terme de POSITION ne corrige ca. Un pas manque met la reference a 8.5 cm
  du pied du robot, ce que le noyau de motion_feet_pos paie deja zero : alourdir
  son poids n'ajoute rien la ou le terme rend deja zero, et le resserrer a
  supprime le gradient (voir le commentaire de motion_feet_pos).

  On compare donc les ETATS DE CONTACT, pas les positions. La reference dit a
  chaque instant quel pied doit etre en vol ; le capteur dit lequel l'est. Le
  terme rend 1 quand les deux coincident sur les deux pieds, 0 quand aucun. Il
  ne sature jamais et garde du gradient quelle que soit l'erreur de position,
  parce qu'il ne mesure pas une distance.
  """
  cmd = env.command_manager.get_term(command_name)
  idx = [cmd.cfg.body_names.index(n) for n in body_names]

  # L_ANKLE_P_LINK est l'ARTICULATION de cheville, pas la semelle : son z vaut
  # deja ~8 cm pied a plat. Sans cet offset, `z > seuil` est toujours vrai, la
  # reference est declaree en vol en permanence et le terme se reduit a la
  # fraction de temps ou le robot a les pieds en l'air -- il recompense le saut.
  # L'offset est le plancher de la reference elle-meme, calcule une fois sur
  # tout le mouvement et mis en cache.
  if not hasattr(cmd, "_sole_offset"):
    z_all = cmd.motion.body_pos_w[:, idx, 2]
    # Le plancher se prend sur la portion MARCHEE, pas sur tout le fichier : le
    # log BWC commence par 14 s debout (19 % des trames), ou les deux pieds sont
    # a plat. En marche la cheville du pied d'appui monte au deroule talon-pointe,
    # donc un plancher calcule sur le prologue declare le pied d'appui en vol.
    walking = z_all.max(dim=1).values > 0.11
    z_walk = z_all[walking] if bool(walking.any()) else z_all
    cmd._sole_offset = z_walk.min(dim=0).values
  ref_z = cmd.body_pos_w[:, idx, 2] - env.scene.env_origins[:, None, 2]
  ref_air = (ref_z - cmd._sole_offset > air_threshold).float()

  sensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    raise RuntimeError("_contact_schedule_match a besoin du champ 'found'.")
  in_contact = (found > 0).float()

  # APPUI (la reference a le pied au sol) : credit binaire sur le contact.
  # VOL (la reference a leve) : credit PROPORTIONNEL a la hauteur atteinte,
  # rapportee a celle que la reference demande au meme instant.
  #
  # La premiere version ne testait que l'absence de contact, donc un pied
  # decolle de 2 mm valait un pied a 8 cm : la politique satisfaisait le terme
  # en effleurant, et le pic median est tombe de 7.96 a 4.11 cm pendant que la
  # cadence montait. Le ratio de hauteur ne sature pas (il est borne a 1 par la
  # reference elle-meme) et ne s'annule pas (il decroit continument vers 0), donc
  # il garde du gradient la ou un noyau gaussien n'en a plus.
  robot = env.scene["robot"]
  b_idx = [robot.body_names.index(n) for n in body_names]
  robot_h = (
    robot.data.body_link_pos_w[:, b_idx, 2]
    - env.scene.env_origins[:, None, 2]
    - cmd._sole_offset
  )
  ref_h = ref_z - cmd._sole_offset
  # Credit PLEIN des 60 % de la hauteur de reference : au-dela, lever davantage
  # ne rapporte plus rien. Sans ce plafond le ratio est lineaire jusqu'a 100 %,
  # donc gagner un centimetre de hauteur reste toujours plus rentable que gagner
  # un pas -- mesure le 2026-09-15 : entre les iterations 2289 et 3573 le pic est
  # passe de 79 a 90 % de la reference pendant que le ratio de cycles restait
  # fige a 0.73. La politique reconstruisait la phase 1 au lieu de cadencer.
  # Avec le plafond, une politique qui leve deja bien n'a plus qu'un seul levier
  # pour augmenter ce terme : rater moins de vols.
  lift_ratio = torch.clamp(
    robot_h / (LIFT_CREDIT_FRACTION * ref_h).clamp(min=air_threshold), 0.0, 1.0
  )

  # SEULES les trames ou la reference est en vol comptent. La version
  # precedente creditait aussi l'appui, or l'appui represente 79 % des trames et
  # le robot y est pose de toute facon : le terme encaissait 0.79 gratuitement,
  # et tout l'enjeu de cadence se jouait sur les 8 % restants. Mesure du
  # 2026-09-15 a l'iteration 6000 : terme a 0.917 alors que le robot ratait 28 %
  # des pas et passait 40 % de temps en l'air en moins que la reference.
  #
  # Rapporte au seul vol, rater 40 % de l'air coute 40 % du terme. Le denominateur
  # est borne pour les instants ou la reference a les deux pieds au sol.
  # Trames de vol PONDEREES PAR LA HAUTEUR DE REFERENCE. Compter les trames a
  # poids egal sature le terme : la moitie du balancement se joue sous 1 cm de
  # hauteur de reference (mediane mesuree), ou le seuil 0.6*ref_h tombe sous
  # 6 mm et se satisfait sans lever. Preuve du 2026-09-15, iterations 11365 ->
  # 12665 : la politique a perdu 93 vols sur deux segments -- pres de la moitie
  # -- et le terme n'a bougé que de 0.9607 a 0.9563, soit 0.44 %. Il etait
  # aveugle a la cadence.
  #
  # Pondere par ref_h, ce sont les instants ou la reference est FRANCHEMENT en
  # l'air qui portent le terme -- ceux qu'un pas manque laisse a zero.
  w = (ref_air * ref_h.clamp(min=0.0)).sum(dim=-1)
  credited = (ref_air * ref_h.clamp(min=0.0) * lift_ratio).sum(dim=-1)
  swing = credited / w.clamp(min=1e-6)

  # DOUBLE APPUI de la reference : on rend 1.0, c'est-a-dire RIEN a gagner ni a
  # perdre sur ces trames. C'est bien un credit gratuit -- 27.6 % des trames en
  # marche avant, 67 % en lateral, 84 % en rotation -- et j'ai essaye de le
  # remplacer le 2026-09-15 par `in_contact.prod()`, en demandant au robot
  # d'avoir lui aussi les deux pieds au sol. RESULTAT : la marche avant s'est
  # effondree de 115 a 34 vols en 1000 iterations. Recompenser les deux pieds au
  # sol sur 27.6 % des trames paie la politique pour ne PAS lever, ce qui est
  # pire que neutre. Le credit gratuit reste donc, faute de mieux : il ne guide
  # pas, mais il ne pousse pas non plus dans le mauvais sens.
  return torch.where(w > 1e-6, swing, torch.ones_like(w))


HEADING_TAU = 3.0
HEADING_STD = math.radians(5.0)


LOAD_RATE_STD = 20000.0  # N/s, echelle de la penalite
LAND_N = 30.0            # seuil de pose
LAND_WINDOW = 0.05       # s : le BWC est a 30 % du poids a 50 ms, le robot a 90 %
BODY_WEIGHT_N = 540.0    # poids du RHPS1


def _soft_landing(env, sensor_name: str) -> "torch.Tensor":
  """Cout de la mise en charge, sur les 150 ms qui suivent chaque pose.

  Mesure 2026-09-19/20 : le robot atteint 90 % de son poids en 5 a 105 ms selon
  le checkpoint, le BWC en 75 ms -- et rien ne recompense cette douceur, d'ou le
  tirage au sort d'un checkpoint a l'autre. Moyenne sur TOUS les pas, un noyau
  rendait 0.99 (la montee ne dure que quelques pas sur 200) : la penalite est
  donc fenetree sur la pose, nulle ailleurs, et quadratique pour ne pas saturer.
  """
  force = env.scene[sensor_name].data.force.norm(dim=-1).squeeze(-1)
  prev = getattr(env, "_prev_foot_force", None)
  if prev is None or prev.shape != force.shape:
    prev = force.clone()
  rate = ((force - prev) / env.step_dt).clamp(min=0.0)
  since = getattr(env, "_since_touchdown", None)
  if since is None or since.shape != force.shape:
    since = torch.full_like(force, 1e4)
  landed = (force > LAND_N) & (prev <= LAND_N)
  since = torch.where(landed, torch.zeros_like(since), since + env.step_dt)
  env._since_touchdown = since
  env._prev_foot_force = force.clone()
  # Penalise la CHARGE DEJA PRISE juste apres la pose, pas la derivee : la
  # derivee se concentre sur un seul pas et sa moyenne ne distinguait pas
  # 5 ms de 160 ms (-0.102 contre -0.107). La force atteinte a 50 ms, elle,
  # separe les deux directement.
  del rate
  window = (since < LAND_WINDOW).float()
  return (window * (force / BODY_WEIGHT_N) ** 2).mean(dim=-1)


def _double_support_match(
  env, command_name: str, sensor_name: str, body_names: tuple[str, ...],
  air_threshold: float = 0.01,
) -> "torch.Tensor":
  """Both feet down when the reference has both feet down.

  contact_schedule scores the SWING feet and hands out a free 1.0 during the
  reference's double-support phases -- 54 % of the BWC cycle. Measured
  2026-09-18 in mc_mujoco: the policy is in double support 9.3 % of the time
  against the BWC's 54.4 %, so it dumps its whole weight on one foot at every
  step (peak 704 N against 549, pelvis roll 1.3 deg against 0.3) while the term
  still read 0.99.
  """
  cmd = env.command_manager.get_term(command_name)
  sensor = env.scene[sensor_name]
  idx = [cmd.cfg.body_names.index(n) for n in body_names]  # command order, not robot's
  # Contact de la reference sur les COINS DE LA SEMELLE, pas sur la cheville :
  # pendant le deroule du pied la cheville est remontee alors que le talon ou la
  # pointe touche encore. Segment avant du BWC : cheville a 1 cm -> 40 % de
  # double appui, coins a 2 cm -> 63 %, forces -> 72 %.
  from mjlab.utils.lab_api.math import matrix_from_quat

  corners = getattr(cmd, "_sole_corners", None)
  if corners is None:
    corners = torch.tensor(
      [[sx * 0.1145 + 0.015, sy * 0.065 + 0.01, -0.102]
       for sx in (1.0, -1.0) for sy in (1.0, -1.0)],
      device=cmd.body_pos_w.device, dtype=torch.float,
    )
    cmd._sole_corners = corners
  n_env, n_foot = cmd.body_quat_w.shape[0], len(idx)
  rot = matrix_from_quat(cmd.body_quat_w[:, idx].reshape(-1, 4)).reshape(n_env, n_foot, 3, 3)
  zc = (rot @ corners.T).transpose(-1, -2)[..., 2] + cmd.body_pos_w[:, idx, None, 2]
  ref_z = zc.min(dim=-1).values - env.scene.env_origins[:, None, 2]
  if not hasattr(cmd, "_sole_offset_ds"):
    cmd._sole_offset_ds = torch.quantile(ref_z, 0.05, dim=0)
  ref_down = (ref_z - cmd._sole_offset_ds <= 0.02).float()
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  both_ref = ref_down.prod(dim=-1)
  # Only while the reference WALKS: standing still with a standing reference
  # scored 1.47/1.08 for a robot doing nothing, and the motion stands for 30 %
  # of its frames.
  anchor = cmd.cfg.body_names.index(cmd.cfg.anchor_body_name)  # le bassin, pas un pied
  moving = (cmd.body_lin_vel_w[:, anchor, :2].norm(dim=-1) > 0.05).float()
  force = env.scene[sensor_name].data.force
  loaded = (force.norm(dim=-1).squeeze(-1) > DS_LOAD_N).float()
  both_robot = (loaded.sum(dim=-1) > 1.5).float()

  # Comparaison de FRACTIONS sur une fenetre glissante, pas d'instants. Mesure du
  # 2026-09-19 : le robot est en double appui 41 % du temps contre 33 % pour la
  # reference, mais seules 46 % des phases coincident -- le terme instantane
  # notait donc un dephasage, pas un defaut d'appui.
  # Cible = fraction MESUREE sur le log BWC en marche avant (27.7 %, forces avec
  # hysterese). La geometrie des semelles donnait 58 % -- un pied peut etre a 2 cm
  # du sol sans rien porter -- donc une cible inatteignable, que la politique a
  # laissee tomber : le terme baissait depuis son introduction.
  both_ref = torch.full_like(both_ref, DS_TARGET)
  a = env.step_dt / DS_WINDOW
  for key, val in (("_ds_ref", both_ref * moving), ("_ds_robot", both_robot * moving),
                   ("_ds_active", moving)):
    prev = getattr(env, key, None)
    if prev is None or prev.shape != val.shape:
      prev = torch.zeros_like(val)
    fresh = env.episode_length_buf <= 1
    setattr(env, key, torch.where(fresh, val, (1 - a) * prev + a * val))
  ref_frac = env._ds_ref
  robot_frac = env._ds_robot
  err = robot_frac - ref_frac
  return moving * torch.exp(-((err / DS_STD) ** 2))


DS_TARGET = 0.277  # fraction de double appui du BWC en marche avant (log, forces)
DS_LOAD_N = 30.0   # au-dela, le pied PORTE ; en-deca il effleure
DS_WINDOW = 2.0   # s, une foulee environ
DS_STD = 0.10     # 10 points de fraction

_SLIP_LEVER = 0.1  # m, half a foot: turns foot yaw rate into a rim speed


def _no_slip(env, sensor_name: str, body_names: tuple[str, ...], std: float) -> "torch.Tensor":
  """Foot planted while in contact, as the BWC does (0.1 mm/s, p90 2.5).

  The policy rips the foot at touchdown and lift-off: p90 30 mm/s in deployment,
  45-60 in training (2026-09-18). Yaw rate counts too -- a foot pivoting in place
  is what turns the robot off its heading.
  """
  robot = env.scene["robot"]
  sensor = env.scene[sensor_name]
  ids = getattr(env, "_slip_body_ids", None)
  if ids is None:
    names = list(robot.body_names)
    ids = torch.tensor([names.index(n) for n in body_names], device=robot.data.body_link_pos_w.device)
    env._slip_body_ids = ids
  assert sensor.data.found is not None
  contact = (sensor.data.found > 0).float()
  v = robot.data.body_link_lin_vel_w[:, ids, :2].norm(dim=-1)
  w = robot.data.body_link_ang_vel_w[:, ids, 2].abs()
  speed = v + _SLIP_LEVER * w
  k = torch.exp(-((speed / std) ** 2))
  num = (k * contact).sum(dim=-1)
  den = contact.sum(dim=-1)
  return torch.where(den > 0, num / den.clamp(min=1e-6), torch.ones_like(num))


_LEG_TOKENS = ("CROTCH", "KNEE", "ANKLE")


def _joint_pos_track(env, command_name: str, std: float, upper: bool) -> "torch.Tensor":
  """Imitate the reference joint angles. Body-pose terms leave them free.

  Two groups because a single mean over 30 joints is dominated by the legs
  (rms 0.144 rad against 0.032 for the upper body, measured 2026-09-18), so the
  head and arms would barely pull -- the head drifted 5.5 deg with no term on it.
  """
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene["robot"]
  key = "_joint_track_upper" if upper else "_joint_track_legs"
  ids = getattr(env, key, None)
  if ids is None:
    names = list(robot.joint_names)
    is_leg = [any(t in n for t in _LEG_TOKENS) for n in names]
    ids = torch.tensor(
      [i for i, leg in enumerate(is_leg) if leg != upper], device=robot.data.joint_pos.device
    )
    setattr(env, key, ids)
  err = robot.data.joint_pos[:, ids] - cmd.joint_pos[:, ids]
  if std <= 0.0:
    # Quadratic cost, not a kernel: an exp over 18 upper-body joints diluted a
    # 5 deg head to 5 % of the term, and it was already at 0.69 -- nothing left
    # to win. Summed squared error keeps a gradient per joint at any amplitude.
    return (err * err).sum(dim=-1)
  return torch.exp(-(err * err).mean(dim=-1) / (std * std))


def _heading_hold(env, command_name: str) -> "torch.Tensor":
  """Low-passed yaw rate kernel, only when the direction command has no rotation.

  The policy cannot observe the reference heading, so anchor_ori cannot correct
  drift. tau 3 s filters the gait's yaw sway (BWC +-23 deg/s at ~0.56 Hz ->
  +-2 deg/s) while keeping the mean bias (4.4 deg/s measured, BWC 1.4).
  """
  wz = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  if not hasattr(env, "_heading_ema"):
    env._heading_ema = torch.zeros_like(wz)
  a = env.step_dt / HEADING_TAU
  fresh = env.episode_length_buf <= 1
  env._heading_ema = torch.where(fresh, wz, (1 - a) * env._heading_ema + a * wz)
  no_rot = _reference_velocity_command(env, command_name)[:, 2] == 0
  k = torch.exp(-((env._heading_ema / HEADING_STD) ** 2))
  return torch.where(no_rot, k, torch.zeros_like(k))


def _reference_velocity_command(env, command_name: str) -> "torch.Tensor":
  """La DIRECTION du mouvement, pas sa vitesse : signe(vx), signe(vy), signe(wz).

  La version precedente renvoyait la vitesse instantanee de la reference. Elle
  DECRIVAIT donc ce que le robot faisait deja, au lieu de le lui commander : la
  politique n'avait aucune raison d'apprendre le lien de causalite, et mesure du
  2026-09-15, elle ne l'a pas appris -- depart debout, consigne 0.15 m/s pendant
  15 s, 0.099 m parcourus au lieu de 2.25. Voir la memoire
  rhps1-tracking-policy-ignores-command.

  En signe, la commande cesse d'etre redondante avec l'etat : le robot ne peut
  pas deviner dans sa proprioception s'il doit partir en +y ou en -x. Et le zero
  -- les 14 s debout au debut du log BWC -- donne l'immobilite a consigne nulle.

  Calculee sur le DEPLACEMENT NET sur 4 s, pas sur la vitesse lissee. Lisser une
  vitesse ne donne pas une direction de marche : le bassin oscille lateralement a
  chaque pas et la marche de cote du BWC alterne gauche-droite, donc la moyenne
  s'annule. Mesure du 2026-09-15 sur une fenetre de 2 s : le segment avant
  ressortait [1,1,0] et le lateral [0,0,0]. A 4 s les quatre modes sortent
  correctement ; 6 s redonne [1,1,0] sur l'avant.

  Un axe est retenu s'il porte au moins la MOITIE du mouvement (dominance), avec
  un plancher de 3 cm pour ne pas qualifier du bruit.
  """
  cmd = env.command_manager.get_term(command_name)
  if not hasattr(cmd, "_dir_cmd"):
    from mjlab.utils.lab_api.math import matrix_from_quat

    ai = cmd.motion_anchor_body_index
    pos = cmd.motion.body_pos_w[:, ai, :2]
    rot = matrix_from_quat(cmd.motion.body_quat_w[:, ai])
    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
    T = pos.shape[0]
    W = int(4.0 / env.step_dt)  # 4 s : voir le docstring
    i0 = torch.clamp(torch.arange(T, device=pos.device) - W // 2, 0, T - 1)
    i1 = torch.clamp(torch.arange(T, device=pos.device) + W // 2, 0, T - 1)
    d = pos[i1] - pos[i0]
    c, sn = torch.cos(yaw), torch.sin(yaw)
    dx = d[:, 0] * c + d[:, 1] * sn
    dy = -d[:, 0] * sn + d[:, 1] * c
    dw = yaw[i1] - yaw[i0]
    dw = torch.atan2(torch.sin(dw), torch.cos(dw))
    # DOMINANCE, pas seuil absolu. Avec un seuil absolu de 3 cm, marcher droit a
    # 0.17 m/s pendant 4 s (68 cm) suffit a activer l'axe lateral des 4 % de
    # derive : les segments de marche avant ressortaient en [1,0,-1] ou [1,1,-1]
    # et la marche avant ne pesait que 6.6 % des trames. Un axe n'est retenu que
    # s'il porte au moins la moitie du mouvement -- mesure du 2026-09-15 : modes
    # purs 68.5 % -> 95.0 %, marche avant 6.6 % -> 28.1 %.
    R = 0.3  # bras de levier pour comparer une rotation a une translation
    mag = torch.stack([dx.abs(), dy.abs(), (dw * R).abs()], dim=1)
    actif = (mag > 0.5 * mag.max(dim=1, keepdim=True).values) & (mag > 0.03)
    cmd._dir_cmd = (
      torch.stack([torch.sign(dx), torch.sign(dy), torch.sign(dw)], dim=1) * actif
    )
    # Transitions: the 4 s window reads the BWC's weight shift as a mode of its
    # own ([0,-1,0] for 0.76 s before the first forward walk), so 0 -> forward,
    # the joystick push, never occurred at rest. Segments shorter than 1 s take
    # the label of the long segment they lead into.
    if os.environ.get("RHPS1_CMD_PREAMBLE", "0") == "1":
      D = cmd._dir_cmd
      change = torch.ones(D.shape[0], dtype=torch.bool, device=D.device)
      change[1:] = (D[1:] != D[:-1]).any(dim=1)
      starts = change.nonzero().flatten().tolist() + [D.shape[0]]
      short = int(1.0 / env.step_dt)
      nxt = None
      for a, b in reversed(list(zip(starts[:-1], starts[1:]))):
        if b - a >= short:
          nxt = D[a].clone()
        elif nxt is not None:
          D[a:b] = nxt
  return cmd._dir_cmd[cmd.time_steps]


def rhps1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  randomize: bool = True,
  student_actor: bool = False,
  deploy_v3: bool = False,
  cadence_term: float = 0.0,
  standing_start: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  """randomize=False : la PHASE 1, copier la trajectoire et rien d'autre.

  Poussees, friction variable, biais d'encodeur et decalage de masse ajoutent du
  bruit au signal d'imitation avant qu'il ne soit etabli. On copie d'abord, on
  robustifie ensuite -- la randomisation revient en phase 3, quand on cherche la
  robustesse et non plus la fidelite.

  La perturbation d'etat initial (pose_range / velocity_range du MotionCommand)
  est reduite mais PAS supprimee : elle sert a l'echantillonnage adaptatif des
  instants de depart, qui est ce qui fera travailler les transitions.
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_rhps1_robot_cfg()}

  # 200 Hz, comme la tache velocity et comme le deploiement. Le defaut de la
  # tache tracking est 0.005 x 4 = 50 Hz.
  cfg.sim.mujoco.timestep = 0.0025
  cfg.decimation = 2

  # Meme budget de contacts que la tache velocity : les sept capteurs de
  # proximite ajoutent leurs paires, et le defaut deborde des la mise en donnees
  # ("nconmax overflow, must be >= 53").
  cfg.sim.nconmax = 96

  prox = tuple(_proximity_sensor(n, a, b) for n, a, b, _ in _PROXIMITY)
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="BODY", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="BODY", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  ) + prox

  # Capteur pied-sol : absent de la tache tracking, repris tel quel de la tache
  # velocity. Il ne sert qu'au terme de cadence ci-dessous.
  if cadence_term:
    cfg.scene.sensors = cfg.scene.sensors + (
      ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
          mode="body", pattern=r"^(L_ANKLE_P_LINK|R_ANKLE_P_LINK)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      ),
    )
    cfg.sim.nconmax = 128  # le capteur pied-sol ajoute ses paires
    cfg.rewards["contact_schedule"] = RewardTermCfg(
      func=_contact_schedule_match,
      weight=float(cadence_term),
      params={
        "command_name": "motion",
        "sensor_name": "feet_ground_contact",
        "body_names": ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK"),
      },
    )

  # Kernels sized on the error the policy traverses (model_9000, standing start):
  # legs rms 0.144 rad, upper body 0.032. std 0.15 / 0.05 render 0.40 / 0.66 there.
  for _grp, _std, _w in (
    ("legs", float(os.environ.get("RHPS1_JOINT_STD_LEGS", "0.15")), 1.0),
    ("upper", float(os.environ.get("RHPS1_JOINT_STD_UPPER", "0")), -20.0),
  ):
    _jw = float(os.environ.get("RHPS1_JOINT_W", "1")) * _w
    if _jw:
      cfg.rewards[f"motion_joint_pos_{_grp}"] = RewardTermCfg(
        func=_joint_pos_track,
        weight=_jw,
        params={"command_name": "motion", "std": _std, "upper": _grp == "upper"},
      )

  _ds_w = float(os.environ.get("RHPS1_DOUBLE_SUPPORT_W", "0"))
  if _ds_w and cadence_term:
    cfg.rewards["double_support"] = RewardTermCfg(
      func=_double_support_match,
      weight=_ds_w,
      params={
        "command_name": "motion",
        "sensor_name": "feet_ground_contact",
        "body_names": ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK"),
      },
    )

  _land_w = float(os.environ.get("RHPS1_SOFT_LANDING_W", "0"))
  if _land_w and cadence_term:
    cfg.rewards["soft_landing"] = RewardTermCfg(
      func=_soft_landing, weight=_land_w, params={"sensor_name": "feet_ground_contact"}
    )

  _slip_w = float(os.environ.get("RHPS1_NOSLIP_W", "0"))
  if _slip_w and cadence_term:  # the foot-ground sensor comes with cadence_term
    cfg.rewards["no_slip"] = RewardTermCfg(
      func=_no_slip,
      weight=_slip_w,
      params={
        "sensor_name": "feet_ground_contact",
        "body_names": ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK"),
        "std": float(os.environ.get("RHPS1_NOSLIP_STD", "0.02")),
      },
    )

  _heading_w = float(os.environ.get("RHPS1_HEADING_W", "0"))
  if _heading_w:
    cfg.rewards["heading_hold"] = RewardTermCfg(
      func=_heading_hold, weight=_heading_w, params={"command_name": "motion"}
    )

  for name, _, _, min_dist in _PROXIMITY:
    cfg.rewards[name] = RewardTermCfg(
      func=velocity_mdp.leg_proximity_cost,
      weight=-2.0,
      params={"sensor_name": name, "min_dist": min_dist},
    )

  from dataclasses import replace as _arep

  from mjlab.tasks.tracking.config.rhps1.hold_action import (
    HoldableJointPositionActionCfg,
  )

  _ja = cfg.actions["joint_pos"]
  assert isinstance(_ja, JointPositionActionCfg)
  # Actions held at the start of a standing-start episode, like mc_mujoco's arming.
  cfg.actions["joint_pos"] = HoldableJointPositionActionCfg(
    **{f.name: getattr(_ja, f.name) for f in _ja.__dataclass_fields__.values()}
  )
  joint_pos_action = cfg.actions["joint_pos"]
  # x10 by default: at effort/stiffness (knee 0.0081 rad/unit) following BWC
  # needs 77-92 action units while exploration std sat at 0.28 unit (0.13 deg).
  _amult = float(os.environ.get("RHPS1_ACTION_SCALE_MULT", "10"))
  joint_pos_action.scale = {k: v * _amult for k, v in RHPS1_ACTION_SCALE.items()}
  # RHPS1_HEAD_SCALE=0 freezes the head. Off by default: the head is imitated by
  # motion_joint_pos_upper instead, like every other joint (Leo, 2026-09-18).
  _head = os.environ.get("RHPS1_HEAD_SCALE")
  if _head is not None:
    joint_pos_action.scale = {
      k: (float(_head) if k.startswith("HEAD_") else v)
      for k, v in joint_pos_action.scale.items()
    }
  # Rate cost is on raw units. -1.0 made exploration noise alone cost ~-35/step
  # and the policy learned to terminate early (run 2026-09-17_23-01-05). The
  # base -0.1 keeps ideal BWC tracking at ~-0.005/step with the x10 scale.
  cfg.rewards["action_rate_l2"].weight = float(os.environ.get("RHPS1_ACTION_RATE_W", "-0.1"))

  # Actuator latency, uniform in [0, max] ms, drawn per episode and per actuator
  # group, held 20 s. The real robot lags the target by 30-35 ms more than the
  # posture filter models; without it slot 4 oscillated on its first inference.
  _delay_ms = float(os.environ.get("RHPS1_ACTION_DELAY_MS", "0"))
  if _delay_ms > 0:
    _lag = int(round(_delay_ms / 1000.0 / cfg.sim.mujoco.timestep))
    for _act in cfg.scene.entities["robot"].articulation.actuators:
      _act.delay_min_lag = 0
      _act.delay_max_lag = _lag
      _act.delay_update_period = 8000
      _act.delay_per_env_phase = False

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # Le flottant comme ancre : c'est la pose que le CSV donne directement, donc
  # aucune reconstruction intermediaire entre le log et la reference.
  motion_cmd.anchor_body_name = "BODY"
  motion_cmd.body_names = _TRACKED_BODIES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(left|right)_foot[1-4]_collision$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("CHEST_P_LINK",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "L_ANKLE_P_LINK",
    "R_ANKLE_P_LINK",
    "L_WRIST_Y_LINK",
    "R_WRIST_Y_LINK",
  )

  # TERME DEDIE AUX PIEDS. motion_body_pos moyenne les quatorze corps a poids
  # egal, et le tronc et les bras -- faciles a suivre -- noient les pieds.
  # Mesure sur model_5000 : error_body_pos 0.0205 (~2 cm), alors que l'ecart sur
  # le seul pied vaut 2.5 cm (p90 1.05 cm contre 3.61 pour la reference). Les
  # pieds SONT l'essentiel de l'erreur residuelle, et rien ne les distingue.
  #
  # std DIMENSIONNE SUR L'ERREUR MESUREE, pas au jugé. A 0.10 le terme rendait
  # 1.8506 sur 2.0 -- sature a 93 %, donc sans gradient : un pied faux de 2.5 cm
  # le satisfaisait a 94 %, et la politique n'avait plus rien a gagner. Le lever
  # de pied a plafonne (p90 1.48 -> 1.66 cm en 1500 iterations) pendant que le
  # reward global restait plat a 62.4.
  #
  # A 0.03, une erreur de 2.5 cm ne le satisfait plus qu'a 50 %. Le POIDS reste
  # a 2.0 : dans un compromis a deux termes on resserre celui qui specifie, on
  # ne surenchérit pas sur celui qui incite -- augmenter le poids ferait monter
  # le maximum et enfoncerait encore la mediane, qui recule deja.
  #
  # NE PAS resserrer sous 0.03. Essaye a 0.015 le 2026-09-14, dimensionne sur
  # l'erreur d'une politique DEJA bonne (0.43 cm en vol) : exp(-(e/0.015)^2)
  # vaut zero des 3 cm, or la reference monte a 8.5 cm en vol. Cela cree une
  # zone de gradient NUL sur toute la phase de vol -- un pied en retard n'a plus
  # aucun signal pour remonter, tandis que l'appui rend toujours ~1. La
  # politique a appris l'optimum correspondant : rester au sol. Mesure a
  # l'iteration 2000 : pic par cycle 0.84 cm contre 7.96 a std 0.03, pour une
  # reference a 8.52. Dimensionner sur l'erreur TRAVERSEE pendant
  # l'apprentissage, pas sur l'erreur residuelle finale.
  #
  # Pilotable par RHPS1_FEET_STD.
  cfg.rewards["motion_feet_pos"] = RewardTermCfg(
    func=tracking_mdp.motion_relative_body_position_error_exp,
    weight=2.0,
    params={
      "command_name": "motion",
      "std": float(os.environ.get("RHPS1_FEET_STD", "0.03")),
      "body_names": ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK"),
    },
  )

  # NOYAUX RECALES SUR LES ERREURS MESUREES. Les valeurs heritees du G1 (std 0.3
  # a 3.14) etaient trois a trente fois plus larges que nos erreurs residuelles :
  # a l'iteration 19997, TOUS les termes de suivi rendaient 89 a 95 % de leur
  # maximum pendant que le robot levait le pied trois fois moins que la
  # reference. Le bareme etait satisfait par le comportement courant, donc sans
  # gradient -- et c'est pour ca qu'aucun terme ajoute sur les pieds n'a marche :
  # le troisieme a meme fait reculer p90 de 1.75 a 1.10 cm.
  #
  # Chaque std est choisi pour qu'une erreur EGALE A L'ERREUR MESUREE coute
  # environ la moitie du terme :
  #
  #     terme                   std G1   erreur   valeur    std ici   valeur
  #     motion_body_pos          0.30    0.0140    1.00       0.02     0.61
  #     motion_body_ori          0.40    0.0689    0.97       0.09     0.56
  #     motion_body_lin_vel      1.00    0.1015    0.99       0.15     0.63
  #     motion_body_ang_vel      3.14    0.50      0.97       0.70     0.60
  #     motion_global_root_ori   0.40    0.10      0.94       0.12     0.50
  #     motion_global_root_pos   0.30    0.197     0.65       0.25     0.54
  #
  # motion_feet_height est RETIRE : ajoute une heure plus tot, il a fait reculer
  # la regularite du lever au lieu de l'ameliorer. Le probleme n'etait pas un
  # terme manquant.
  for _name, _std in (
    ("motion_body_pos", 0.02),
    ("motion_body_ori", 0.09),
    ("motion_body_lin_vel", 0.15),
    ("motion_body_ang_vel", 0.70),
    ("motion_global_root_ori", 0.12),
    ("motion_global_root_pos", 0.25),
  ):
    cfg.rewards[_name].params["std"] = float(
      os.environ.get("RHPS1_STD_" + _name.removeprefix("motion_").upper(), _std)
    )

  # GROUPE "student" : ce que l'eleve verra, et uniquement ce qui est disponible
  # au deploiement. On garde les cinq blocs proprioceptifs de l'experte et on
  # remplace les trois blocs qui n'existent pas sur le robot -- command (les
  # angles cibles de la reference), motion_anchor_pos_b et motion_anchor_ori_b --
  # par la commande de vitesse a 3 dims.
  _student = {
    k: v
    for k, v in cfg.observations["actor"].terms.items()
    if k not in ("command", "motion_anchor_pos_b", "motion_anchor_ori_b")
  }
  _student["velocity_command"] = ObservationTermCfg(
    func=_reference_velocity_command, params={"command_name": "motion"}
  )
  cfg.observations["student"] = ObservationGroupCfg(
    terms=_student, concatenate_terms=True, enable_corruption=not play
  )

  cfg.viewer.body_name = "BODY"

  # Le RHPS1 n'a ni velocimetre ni gyro "imu_*" dans son XML -- ce sont des
  # capteurs du G1 (g1.xml:303). On lit donc la vitesse du corps racine, ce qui
  # est aussi plus proche du deploiement : mc_rtc reconstruit cette vitesse par
  # l'observateur MCWaiko, il ne lit pas un velocimetre.
  for grp in ("actor", "critic"):
    terms = cfg.observations[grp].terms
    if "base_lin_vel" in terms:
      terms["base_lin_vel"].func = velocity_mdp.base_lin_vel
      terms["base_lin_vel"].params = {}
    if "base_ang_vel" in terms:
      terms["base_ang_vel"].func = velocity_mdp.base_ang_vel
      terms["base_ang_vel"].params = {}

  # Depart arrete en posture de deploiement : voir standing_start.py.
  if standing_start > 0.0:
    from dataclasses import fields as _fields

    from .standing_start import StandingStartMotionCommandCfg

    _m = cfg.commands["motion"]
    _kw = {f.name: getattr(_m, f.name) for f in _fields(_m)}
    cfg.commands["motion"] = StandingStartMotionCommandCfg(
      **_kw,
      # 0.5 (was 0.3): only standing-start episodes tell a walking policy from
      # a stopping one -- RSI scored 5.01 for both 4500 and 6000 (2026-09-18).
      standing_start_prob=float(os.environ.get("RHPS1_STANDING_START", standing_start))
    )

  # FILTRE DE LA TACHE DE POSTURE DU QP, modelise a l'entrainement.
  #
  # mc_rtc n'envoie pas la cible de la politique directement aux articulations :
  # elle traverse la PostureTask du QP, un second ordre de raideur K.
  #
  # K = 1600, PAS le repli 0.2/(policy_step*timeStep) = 8000. Ce repli ne sert
  # que si la cle est absente de la configuration GLOBALE, or
  # mc_rtc_superbuild.yaml (robot reel) et mc_rtc_superbuild_mujoco.yaml portent
  # tous deux `posture_stiffness: 1600`. Le retard vaut 1/sqrt(K) : 25 ms a 1600,
  # 11.2 ms a 8000 -- j'ai modelise 8000 pendant toute la nuit du 2026-09-16,
  # soit un facteur 2.2 sur la grandeur meme qui pose probleme.
  #
  # La tache de tracking laissait ce parametre a None, donc elle entrainait sur
  # une reponse immediate.
  #
  # Mesure du 2026-09-16, MEME checkpoint, meme banc, seul le filtre change :
  #     sans filtre   +1.69 m   20.7 deg   6.0 cm
  #     K = 8000      -1.70 m   20.2 deg   5.9 cm
  # Le filtre ne casse pas la demarche, il en INVERSE le sens : meme amplitude,
  # meme lever, meme vitesse, propulsion de signe oppose. Sous mc_mujoco le QP
  # refuse ensuite de reculer pour tenir l'equilibre, et le robot se fige -- ce
  # qui explique l'ecart entrainement/deploiement signale le 2026-09-15.
  #
  # La policy 0, qui transfere, est entrainee avec ce filtre modelise.
  from dataclasses import replace as _replace

  # Foot-ground friction 1.0 = mc_mujoco (MuJoCo default, no priority there).
  # Was 0.5; the plant mismatch shows in base roll/yaw under replayed actions.
  _fric = float(os.environ.get("RHPS1_FOOT_FRICTION", "1.0"))
  _ent = cfg.scene.entities["robot"]
  _ent.collisions = tuple(
    _replace(c, friction={k: (_fric,) for k in c.friction})
    if isinstance(c.friction, dict)
    else c
    for c in _ent.collisions
  )

  _art = cfg.scene.entities["robot"].articulation
  _art.actuators = tuple(
    _replace(a, posture_task_stiffness=float(os.environ.get("RHPS1_POSTURE_K", "1600")))
    for a in _art.actuators
  )

  # RANDOMISATION du retard, plutot qu'une valeur fixe. Voir le docstring de
  # _randomize_posture_stiffness : a K fixe la politique s'effondre vers
  # l'immobilite, quel que soit le palier.
  _klo = float(os.environ.get("RHPS1_POSTURE_K_LO", "8000"))
  _khi = float(os.environ.get("RHPS1_POSTURE_K_HI", "200000"))
  if _khi > _klo:
    from mjlab.managers.event_manager import EventTermCfg as _Ev

    cfg.events["posture_stiffness"] = _Ev(
      func=_randomize_posture_stiffness, mode="reset",
      params={"k_range": (_klo, _khi)},
    )

  # TERMINATIONS DE SUIVI DESSERREES pendant l'adaptation au filtre de posture.
  #
  # Mesure du 2026-09-16 : avec le filtre, les episodes meurent en 0.5 s par
  # ee_body_pos (29.7) et anchor_pos (10.7), jamais par time_out (0.0) -- contre
  # 0.11 / 0.11 / 2.47 sans filtre. Le filtre fait partir le robot en ARRIERE,
  # donc il s'ecarte immediatement de la reference qui avance, et l'episode est
  # coupe avant qu'il ait pu apprendre a retourner sa propulsion. Les quatre
  # tentatives d'adaptation ont echoue pour cette raison, pas par difficulte.
  #
  # Desserrer laisse le temps de reconstruire la marche. A resserrer une fois la
  # politique adaptee, sinon elle n'a plus de garde-fou sur la derive.
  _term = float(os.environ.get("RHPS1_TERM_SCALE", "1.0"))
  if _term != 1.0:
    for _k in ("anchor_pos", "ee_body_pos"):
      if _k in cfg.terminations:
        cfg.terminations[_k].params["threshold"] *= _term
    if "anchor_ori" in cfg.terminations:
      cfg.terminations["anchor_ori"].params["threshold"] *= min(_term, 2.0)

  # Format V3 : la disposition EXACTE de utils.cpp case 0, celle qui a du temps
  # robot derriere elle. 126 dims = base_lin_vel[15], base_ang_vel[3],
  # projected_gravity[3], joint_pos[30], joint_vel[30], actions[30],
  # command[15] -- historique 5 sur la vitesse de base et la commande
  # seulement. L'ordre des termes EST l'ordre d'ecriture du C++ : ne pas le
  # reordonner sans reecrire le case.
  #
  # Le groupe "student" a 99 dims n'avait ni projected_gravity ni historique :
  # herite de la tache tracking du G1, qui tire son orientation de
  # motion_anchor_ori_b. L'eleve n'avait donc aucune perception de son
  # inclinaison -- aucun format deploye n'etait dans ce cas.
  if deploy_v3:
    _base = cfg.observations["student"].terms
    _v3: dict = {}
    for _k in ("base_lin_vel", "base_ang_vel"):
      _v3[_k] = _base[_k]
    _v3["projected_gravity"] = ObservationTermCfg(func=velocity_mdp.projected_gravity)
    for _k in ("joint_pos", "joint_vel", "actions", "velocity_command"):
      _v3[_k] = _base[_k]
    _v3["base_lin_vel"].history_length = 5
    _v3["velocity_command"].history_length = 5
    # Noise sized on mc_mujoco's estimator error vs sim truth (p99: lin vel
    # 0.036 m/s, ang vel 0.030 rad/s, joint vel < 0.01). The inherited +-0.5 m/s
    # on base_lin_vel was the kick out of standstill: noise-free the policy froze.
    from dataclasses import replace as _nrep

    from mjlab.utils.noise import UniformNoiseCfg as _Unoise

    for _k, _a in (("base_lin_vel", 0.05), ("base_ang_vel", 0.05), ("joint_vel", 0.1)):
      _v3[_k] = _nrep(_v3[_k], noise=_Unoise(n_min=-_a, n_max=_a))
    cfg.observations["student"] = ObservationGroupCfg(
      terms=_v3, concatenate_terms=True, enable_corruption=not play
    )

  # L'acteur n'observe plus que ce qui existe au deploiement. La critique garde
  # ses 291 dims privilegiees : c'est l'actor-critic asymetrique habituel, et ca
  # permet de repartir de la critique de l'experte telle quelle.
  if student_actor:
    cfg.observations["actor"] = cfg.observations["student"]

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if not randomize:
    for name in ("push_robot", "base_com", "encoder_bias", "foot_friction"):
      cfg.events.pop(name, None)
    for grp in ("actor", "critic"):
      cfg.observations[grp].enable_corruption = False
    m = cfg.commands["motion"]
    m.pose_range = {k: (v[0] * 0.25, v[1] * 0.25) for k, v in m.pose_range.items()}
    m.velocity_range = {
      k: (v[0] * 0.25, v[1] * 0.25) for k, v in m.velocity_range.items()
    }
    m.joint_position_range = (-0.025, 0.025)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # standing_start replaces cfg.commands["motion"]; motion_cmd is stale here.
    play_cmd = cfg.commands["motion"]
    play_cmd.pose_range = {}
    play_cmd.velocity_range = {}
    play_cmd.sampling_mode = "start"

  return cfg
