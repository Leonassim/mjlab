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


def _cap_gate(env, settle: float, ramp_steps: int) -> "torch.Tensor":
  """Constraint gate: 0 for `settle` s after each reset (RSI lands mid-step), times a
  linear weight ramp over the first `ramp_steps` env steps of the launch."""
  g = (env.episode_length_buf * env.step_dt >= settle).float()
  if ramp_steps > 0:
    g = g * min(1.0, env.common_step_counter / ramp_steps)
  return g


def _touchdown_speed_cap(env, sensor_name: str, body_names: tuple[str, ...], v_max: float,
                         settle: float = 0.0, ramp_steps: int = 0) -> "torch.Tensor":
  """Squared relative excess of the foot's landing speed over v_max, at each touchdown.

  Landing speed = fastest downward foot speed in the 50 ms before the load rises.
  v_max = the BWC's own fastest landing (0.21 m/s on bwc_motion_v3): zero for any
  landing the BWC makes, quadratic beyond (Leo, 2026-10-20: the impact limit is the
  BWC's). Event term divided by the step, so its weight reads per landing.
  """
  robot = env.scene["robot"]
  if not hasattr(env, "_tdc_ids"):
    env._tdc_ids = [list(robot.body_names).index(n) for n in body_names]
  vz = robot.data.body_link_lin_vel_w[:, env._tdc_ids, 2]
  force = env.scene[sensor_name].data.force.norm(dim=-1).reshape(vz.shape)
  n = int(round(0.05 / env.step_dt)) + 1
  buf = getattr(env, "_tdc_buf", None)
  if buf is None or buf.shape[0] != vz.shape[0]:
    buf = vz[:, None].repeat(1, n, 1)
  buf = torch.cat([buf[:, 1:], vz[:, None]], 1)
  fresh = env.episode_length_buf <= 1
  buf = torch.where(fresh[:, None, None], vz[:, None].expand_as(buf), buf)
  env._tdc_buf = buf
  prev = getattr(env, "_tdc_prev", None)
  if prev is None or prev.shape != force.shape:
    prev = force.clone()
  landed = (force > LAND_N) & (prev <= LAND_N) & ~fresh[:, None]
  env._tdc_prev = force.clone()
  v_down = (-buf.min(dim=1).values).clamp(min=0.0)
  # capped at 2: a fall lands at metres per second and would swamp the return
  excess = (torch.relu(v_down - v_max) / v_max).clamp(max=2.0)
  return (landed.float() * excess**2).sum(-1) / env.step_dt * _cap_gate(env, settle, ramp_steps)


def _impact_force_cap(env, sensor_name: str, f_max: float, window: float = 0.06,
                      settle: float = 0.0, ramp_steps: int = 0) -> "torch.Tensor":
  """Squared relative excess of the peak foot force over f_max in the `window` after each
  touchdown, emitted once when the window closes. f_max = slot 13's p99 (729 N in sim,
  dyn_envelope.py): impacts no harder than the BWC copy the robot already walks with.
  Event term divided by the step, so its weight reads per landing; excess capped at 2.
  """
  force = env.scene[sensor_name].data.force.norm(dim=-1).reshape(env.num_envs, -1)
  prev = getattr(env, "_ifc_prev", None)
  if prev is None or prev.shape != force.shape:
    prev = force.clone()
    env._ifc_since = torch.full_like(force, 1e4)
    env._ifc_peak = torch.zeros_like(force)
  landed = (force > LAND_N) & (prev <= LAND_N)
  env._ifc_prev = force.clone()
  since = torch.where(landed, torch.zeros_like(force), env._ifc_since + env.step_dt)
  peak = torch.where(landed, force, torch.maximum(env._ifc_peak, force))
  closing = (env._ifc_since < window) & (since >= window)
  env._ifc_since, env._ifc_peak = since, peak
  excess = (torch.relu(peak - f_max) / f_max).clamp(max=2.0)
  return (closing.float() * excess**2).sum(-1) / env.step_dt * _cap_gate(env, settle, ramp_steps)


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


def _joint_pos_track(
  env, command_name: str, std: float, upper: bool, skip: tuple[str, ...] = ()
) -> "torch.Tensor":
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
      [i for i, leg in enumerate(is_leg)
       if leg != upper and not any(t in names[i] for t in skip)],
      device=robot.data.joint_pos.device,
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


def _still_when_reference_still(
  env, command_name: str, std: float, gate: float, joint_std: float = 0.0
) -> "torch.Tensor":
  """Body angular-velocity tracking at a tight kernel, only while the reference is still.

  The BWC stays still for 1 s after the command flips. At motion_body_ang_vel's
  std 0.70 the jolt there (mean |w|^2 0.057, pelvis 18 deg/s) kept 89 % of the
  term; at std 0.2 it keeps 24 % while standing keeps 99 %. Gated on the
  reference's own rms body rate below `gate` (rad/s): at 0.02 that is the four
  standing segments of the clip (40 %) and 1.6 s after the flip, and under 1 %
  of walking frames, where errors (~0.4 rad/s) would leave the kernel flat.

  joint_std > 0 adds joint-velocity tracking over all joints to the exponent.
  The head is not a tracked body and each arm is diluted in the body mean: with
  the body part alone (model_57500) the pelvis fell to 11 deg/s while HEAD_Y
  still peaked at 33 deg/s, CHEST_Y at 22 and the arms at 15-21.
  """
  cmd = env.command_manager.get_term(command_name)
  ref = cmd.body_ang_vel_w
  still = (ref**2).sum(-1).mean(-1) < gate**2
  expo = ((ref - cmd.robot_body_ang_vel_w) ** 2).sum(-1).mean(-1) / std**2
  if joint_std > 0.0:
    expo = expo + ((cmd.joint_vel - cmd.robot_joint_vel) ** 2).mean(-1) / joint_std**2
  return torch.where(still, torch.exp(-expo), torch.zeros_like(expo))


def _biased_obs(func, attr: str):
  """Observation plus a per-env offset stored on the env (see _sample_obs_bias)."""

  def f(env, **kwargs):
    v = func(env, **kwargs)
    b = getattr(env, attr, None)
    return v if b is None else v + b

  return f


def _sample_obs_bias(env, env_ids, lin: float, grav: float) -> None:
  """Draw the estimator offsets held for the episode."""
  for attr, r in (("_rhps1_lin_bias", lin), ("_rhps1_grav_bias", grav)):
    if not hasattr(env, attr):
      setattr(env, attr, torch.zeros(env.num_envs, 3, device=env.device))
    getattr(env, attr)[env_ids] = (torch.rand(len(env_ids), 3, device=env.device) * 2 - 1) * r


def _ou_obs(func):
  """Observation plus a slowly varying error (Ornstein-Uhlenbeck, see _sample_ou)."""

  def f(env, **kwargs):
    v = func(env, **kwargs)
    n = getattr(env, "_rhps1_ou", None)
    if n is None:
      return v
    step = int(env.common_step_counter)
    if getattr(env, "_rhps1_ou_step", None) != step:
      a = torch.exp(-env.step_dt / env._rhps1_ou_tau)
      xi = torch.randn_like(n)
      env._rhps1_ou = n * a + env._rhps1_ou_sigma * torch.sqrt(1.0 - a * a) * xi
      env._rhps1_ou_step = step
    return v + env._rhps1_ou

  return f


def _sample_ou(env, env_ids, sigma: float, tau_min: float, tau_max: float) -> None:
  """Per episode: error amplitude in [0, sigma] (x, y; z a third) and time constant."""
  if not hasattr(env, "_rhps1_ou"):
    for attr in ("_rhps1_ou", "_rhps1_ou_sigma"):
      setattr(env, attr, torch.zeros(env.num_envs, 3, device=env.device))
    env._rhps1_ou_tau = torch.ones(env.num_envs, 1, device=env.device)
  k = len(env_ids)
  sig = torch.rand(k, 1, device=env.device) * sigma * torch.tensor([[1.0, 1.0, 1.0 / 3.0]], device=env.device)
  env._rhps1_ou_sigma[env_ids] = sig
  env._rhps1_ou_tau[env_ids] = tau_min + (tau_max - tau_min) * torch.rand(k, 1, device=env.device)
  env._rhps1_ou[env_ids] = sig * torch.randn(k, 3, device=env.device)


def _with_deflection(func, attr: str):
  """Joint observation plus the hidden transmission deflection (series compliance)."""

  def f(env, **kwargs):
    v = func(env, **kwargs)
    asset_cfg = kwargs.get("asset_cfg")
    ent = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    d = getattr(ent, attr, None)
    if d is None:
      return v
    ids = asset_cfg.joint_ids if asset_cfg is not None else slice(None)
    return v + d[:, ids]

  return f


def _sway_torque(env, env_ids, amp: tuple[float, float, float], tau: float) -> None:
  """Slowly varying random torque on the pelvis (OU, time constant tau), amplitude
  per axis drawn in [0, amp] at each episode start. Every env step."""
  robot = env.scene["robot"]
  n_env = env.num_envs
  if not hasattr(env, "_rhps1_sway"):
    env._rhps1_sway = torch.zeros(n_env, 3, device=env.device)
    env._rhps1_sway_sig = torch.zeros(n_env, 3, device=env.device)
    env._rhps1_sway_body = [list(robot.body_names).index("BODY")]
  fresh = env.episode_length_buf <= 1
  if torch.any(fresh):
    k = int(fresh.sum())
    a = torch.tensor(amp, device=env.device)
    env._rhps1_sway_sig[fresh] = torch.rand(k, 3, device=env.device) * a
    env._rhps1_sway[fresh] = 0.0
  c = math.exp(-env.step_dt / tau)
  env._rhps1_sway = env._rhps1_sway * c + env._rhps1_sway_sig * math.sqrt(1.0 - c * c) * torch.randn_like(env._rhps1_sway)
  tq = env._rhps1_sway.unsqueeze(1)
  robot.write_external_wrench_to_sim(torch.zeros_like(tq), tq, body_ids=env._rhps1_sway_body)


def _group_limits(robot, limits: dict[str, float], device) -> "torch.Tensor":
  lim = torch.full((len(robot.joint_names),), float("inf"), device=device)
  for i, n in enumerate(robot.joint_names):
    for g, v in limits.items():
      if g in n:
        lim[i] = v
  return lim


def _torque_envelope(env, limits: dict[str, float], settle: float = 0.0, ramp_steps: int = 0) -> "torch.Tensor":
  """Squared relative excess of |actuator torque| over a per-group ceiling.

  Ceilings = what slot 3 (BWC copy) already does: its p99 under training conditions
  (scripts/tools/train_cond_caps.py). Below them the term is exactly zero.
  """
  robot = env.scene["robot"]
  if not hasattr(env, "_tenv_lim"):
    env._tenv_lim = _group_limits(robot, limits, env.device)
  tau = robot.data.qfrc_actuator.abs()
  return ((torch.relu(tau - env._tenv_lim) / env._tenv_lim) ** 2).sum(-1) * _cap_gate(env, settle, ramp_steps)


def _motor_joint_vel(robot) -> "torch.Tensor":
  """Joint speed on the encoder side: the link's plus the hidden spring's deflection rate."""
  d = getattr(robot, "_rhps1_deflection_rate", None)
  return robot.data.joint_vel if d is None else robot.data.joint_vel + d


def _joint_speed_guard(env, vmax: float) -> "torch.Tensor":
  """rl_controller's runaway guard: any joint faster than vmax disarms the policy and
  the robot falls. Not while the hold ignores the policy's actions."""
  out = (_motor_joint_vel(env.scene["robot"]).abs() > vmax).any(-1)
  hold = getattr(env, "_action_hold_steps", None)
  return out if hold is None else out & ~(env.episode_length_buf <= hold)


def _joint_speed_cap(env, limits: dict[str, float], settle: float = 0.0, ramp_steps: int = 0) -> "torch.Tensor":
  """Squared relative excess of |joint speed| over a per-group ceiling (rl_controller
  disarms a policy above runaway_disarm_vel, 4 rad/s)."""
  robot = env.scene["robot"]
  if not hasattr(env, "_jsc_lim"):
    env._jsc_lim = _group_limits(robot, limits, env.device)
  qd = _motor_joint_vel(robot).abs()
  return ((torch.relu(qd - env._jsc_lim) / env._jsc_lim) ** 2).sum(-1) * _cap_gate(env, settle, ramp_steps)


def _arm_swing(env, k: float, std: float) -> "torch.Tensor":
  """Each arm swings against the leg on its own side, as in human walking.

  Shoulder and hip pitch share their sign (positive sends the limb back), so
  the target is shoulder - rest = -k (hip - rest), rest being the default pose.
  """
  robot = env.scene["robot"]
  ids = getattr(env, "_arm_swing_ids", None)
  if ids is None:
    n = list(robot.joint_names)
    ids = [(n.index(f"{s}_SHOULDER_P"), n.index(f"{s}_CROTCH_P")) for s in "LR"]
    env._arm_swing_ids = ids
  q, q0 = robot.data.joint_pos, robot.data.default_joint_pos
  err = torch.stack(
    [(q[:, sh] - q0[:, sh]) + k * (q[:, hp] - q0[:, hp]) for sh, hp in ids], dim=-1
  )
  return torch.exp(-(err * err).mean(-1) / (std * std))


def _head_id(env) -> int:
  robot = env.scene["robot"]
  i = getattr(env, "_head_body_id", None)
  if i is None:
    i = list(robot.body_names).index("HEAD_P_LINK")
    env._head_body_id = i
  return i


def _head_stable(env, std: float, tilt_only: bool = False) -> "torch.Tensor":
  """Low head angular velocity in the world frame: a steady gaze. tilt_only
  keeps the horizontal components (roll and pitch rates), so turning is free."""
  w = env.scene["robot"].data.body_link_ang_vel_w[:, _head_id(env)]
  if tilt_only:
    w = w[:, :2]
  return torch.exp(-(w * w).sum(-1) / (std * std))


def _head_yaw_still(env, std: float) -> "torch.Tensor":
  """Low HEAD_Y joint velocity: no yaw jitter, a slow turn toward the step stays free."""
  robot = env.scene["robot"]
  i = getattr(env, "_head_yaw_joint", None)
  if i is None:
    i = list(robot.joint_names).index("HEAD_Y")
    env._head_yaw_joint = i
  w = robot.data.joint_vel[:, i]
  return torch.exp(-(w * w) / (std * std))


def _centroidal_ang_mom(env) -> "torch.Tensor":
  """Angular momentum about the whole-body CoM, world frame (= mj_subtreeVel)."""
  from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

  robot = env.scene["robot"]
  d = robot.data
  ids = robot.indexing.body_ids
  m = env.sim.model.body_mass
  inertia = env.sim.model.body_inertia
  m = (m if m.dim() == 2 else m.unsqueeze(0))[:, ids]
  inertia = (inertia if inertia.dim() == 3 else inertia.unsqueeze(0))[:, ids]
  x, v = d.body_com_pos_w, d.body_com_lin_vel_w
  q, w = d.body_com_quat_w, d.body_com_ang_vel_w
  mt = m.sum(-1, keepdim=True)
  c = (m.unsqueeze(-1) * x).sum(1) / mt
  vc = (m.unsqueeze(-1) * v).sum(1) / mt
  orb = (m.unsqueeze(-1) * torch.cross(x - c.unsqueeze(1), v - vc.unsqueeze(1), dim=-1)).sum(1)
  spin = quat_apply(q, inertia.expand_as(w) * quat_apply_inverse(q, w)).sum(1)
  return orb + spin


def _yaw_momentum_still(env, std: float, tau: float) -> "torch.Tensor":
  """Low oscillation of the centroidal angular momentum about the vertical.

  L_z minus its low-pass at tau, so a steady turn costs nothing: only the
  per-step swing is penalised. The arms are free, so they swing only if that
  cancels the legs' yaw.
  """
  lz = _centroidal_ang_mom(env)[:, 2]
  step = int(env.common_step_counter)
  if not hasattr(env, "_lz_lp") or env._lz_lp.shape != lz.shape:
    env._lz_lp = lz.clone()
    env._lz_step = step
  if env._lz_step != step:
    env._lz_lp = env._lz_lp + (env.step_dt / tau) * (lz - env._lz_lp)
    env._lz_step = step
  env._lz_lp = torch.where(env.episode_length_buf == 0, lz, env._lz_lp)
  e = lz - env._lz_lp
  return torch.exp(-(e * e) / (std * std))


def _head_level(env, std: float) -> "torch.Tensor":
  """Head kept level: gravity along the head's own vertical axis."""
  from mjlab.utils.lab_api.math import quat_apply_inverse

  robot = env.scene["robot"]
  q = robot.data.body_link_quat_w[:, _head_id(env)]
  g = quat_apply_inverse(q, torch.tensor([0.0, 0.0, -1.0], device=q.device).expand(q.shape[0], 3))
  return torch.exp(-(g[:, :2] ** 2).sum(-1) / (std * std))


def _filtered_velocity_command(env, command_name: str, tau: float) -> "torch.Tensor":
  """The direction command through a first-order filter, as the controller applies it.

  A 0 -> 1 step drives the policy's reflex: every checkpoint jolts in the 0.3 s
  after the flip (pelvis 11-24 deg/s, upper body 30-50) while the BWC stays
  still, and rewards alone did not remove it. y += (dt / tau) * (u - y) at the
  policy rate, reset to the raw command at episode start so mid-walk starts get
  no artificial ramp. NewRLQPController's command_filter_tau must match.
  """
  u = _reference_velocity_command(env, command_name)
  step = int(env.common_step_counter)
  if not hasattr(env, "_cmd_filt") or env._cmd_filt.shape != u.shape:
    env._cmd_filt = u.clone()
    env._cmd_filt_step = step
  if env._cmd_filt_step != step:
    env._cmd_filt = env._cmd_filt + (env.step_dt / tau) * (u - env._cmd_filt)
    env._cmd_filt_step = step
  fresh = env.episode_length_buf == 0
  env._cmd_filt = torch.where(fresh[:, None], u, env._cmd_filt)
  return env._cmd_filt


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
    # Window kept inside each frame's own clip (several clips per file).
    t = torch.arange(T, device=pos.device)
    cs, ce = cmd.motion.clip_start.to(pos.device), cmd.motion.clip_end.to(pos.device)
    i0 = torch.maximum(t - W // 2, cs)
    i1 = torch.minimum(t + W // 2, ce - 1)
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
    # RHPS1_CLEAN_LABELS=1: a label lasting < 1 s with no foot lift within 0.3 s
    # is a weight shift read as a walk (-y before every start), not a command: 0.
    # While turning, the lateral sign flickers step to step: the BWC turns in
    # place, so yaw-active frames keep yaw only.
    if os.environ.get("RHPS1_CLEAN_LABELS", "0") == "1":
      D = cmd._dir_cmd
      D[D[:, 2] != 0, 1] = 0
      idx = [cmd.cfg.body_names.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
      lifted = cmd.motion.body_pos_w[:, idx, 2].max(dim=1).values > 0.11
      m = int(0.3 / env.step_dt)
      change = torch.ones(D.shape[0], dtype=torch.bool, device=D.device)
      change[1:] = (D[1:] != D[:-1]).any(dim=1)
      starts = change.nonzero().flatten().tolist() + [D.shape[0]]
      for a, b in zip(starts[:-1], starts[1:]):
        if b - a < int(1.0 / env.step_dt) and not bool(lifted[max(a - m, 0):b + m].any()):
          D[a:b] = 0
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

  _still_w = float(os.environ.get("RHPS1_STILL_W", "0"))
  if _still_w:
    cfg.rewards["still_when_reference_still"] = RewardTermCfg(
      func=_still_when_reference_still,
      weight=_still_w,
      params={
        "command_name": "motion",
        "std": float(os.environ.get("RHPS1_STILL_STD", "0.2")),
        "gate": float(os.environ.get("RHPS1_STILL_GATE", "0.02")),
        "joint_std": float(os.environ.get("RHPS1_STILL_JOINT_STD", "0")),
      },
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

  # RHPS1_DRIVE="ANKLE_P,ANKLE_R,CROTCH_P,CROTCH_R" (1 for these four): the real Elmo
  # cascade in place of the kinematic motor (FiniteDifferencePdActuatorCfg
  # drive_damping). Joint-side c at neutral [N.m.s/rad] = Kt*KP[2]*counts*gear^2,
  # KP[3] [/s], KI[2] [Hz]: RHPS1_drive_data (2026-08-03 export) and the Kaleido axis
  # table (hip cylinders Drive07/08/12/13, ankles 09/10/14/15). RHPS1_DRIVE_SCALE
  # scales c: the 0.97 Hz disarmed rocking measures 0.53x nominal on the ankles.
  _DRIVE = {"ANKLE_P": (252.0, 60.0, 2.0), "ANKLE_R": (84.0, 60.0, 2.0),
            "CROTCH_P": (280.7, 30.0, 45.0), "CROTCH_R": (137.4, 30.0, 45.0),
            "CROTCH_Y": (33.7, 39.0, 45.0), "KNEE": (838.6, 50.0, 0.02)}
  _dv = os.environ.get("RHPS1_DRIVE", "")
  _drive_groups = (("ANKLE_P", "ANKLE_R", "CROTCH_P", "CROTCH_R") if _dv == "1"
                   else tuple(g for g in _dv.split(",") if g))
  _drive_scale = tuple(float(x) for x in os.environ.get("RHPS1_DRIVE_SCALE", "0.53,0.53").split(","))
  for _act in cfg.scene.entities["robot"].articulation.actuators:
    _expr = " ".join(_act.target_names_expr)
    _g = next((g for g in _drive_groups if g in _expr), None)
    if _g:
      _act.drive_damping, _act.drive_kp_pos, _act.drive_ki_hz = _DRIVE[_g]
      _act.drive_gain_scale_range = _drive_scale
      # Stiff structure unless RHPS1_SERIES_K_<GROUP> says otherwise.
      if _act.series_stiffness_range is None:
        _act.series_stiffness_range = (5e4, 5e4)

  # RHPS1_SERIES_K="lo,hi" [N.m/rad]: hidden transmission compliance on the leg
  # joints, log-uniform per env and joint, the deflection added to the observed
  # joint position and velocity (see FiniteDifferencePdActuatorCfg).
  # RHPS1_SERIES_K_<GROUP> sets one group (ANKLE_P, ANKLE_R, KNEE, CROTCH_P...) and
  # wins over the legs-wide value. Measured on the robot (2026-10-16, hidden stance
  # tilt vs foot moment, r = -0.95/-0.91): ankle pitch ~1500-1700, roll ~800-1000.
  _sk = os.environ.get("RHPS1_SERIES_K")
  _groups = ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE", "ANKLE_P", "ANKLE_R")
  _per = {gname: os.environ.get(f"RHPS1_SERIES_K_{gname}") for gname in _groups}
  if _sk or any(_per.values()) or _drive_groups:
    for _act in cfg.scene.entities["robot"].articulation.actuators:
      _expr = " ".join(_act.target_names_expr)
      _gname = next((gn for gn in _groups if gn in _expr), None)
      _val = (_per.get(_gname) if _gname else None) or (_sk if _gname else None)
      if _val:
        assert hasattr(_act, "series_stiffness_range")
        _act.series_stiffness_range = tuple(float(x) for x in _val.split(","))
        # RHPS1_SERIES_DAMP="lo,hi": spring damping as a fraction of the PD damping;
        # RHPS1_SERIES_DAMP_<GROUP> sets one group and wins. The robot rocks heel-to-toe
        # at ~1 Hz with ~2% damping (static push test, 2026-10-22): ANKLE_P ~1400 with
        # a ratio ~0.01 reproduces it; the default 0.3-1.0 overdamps it.
        _sd = os.environ.get(f"RHPS1_SERIES_DAMP_{_gname}") or os.environ.get("RHPS1_SERIES_DAMP")
        if _sd:
          _act.series_damping_ratio = tuple(float(x) for x in _sd.split(","))
    from dataclasses import replace as _srep

    _at = cfg.observations["actor"].terms
    for _k, _attr in (("joint_pos", "_rhps1_deflection"), ("joint_vel", "_rhps1_deflection_rate")):
      if _k in _at:
        _at[_k] = _srep(_at[_k], func=_with_deflection(_at[_k].func, _attr))

  # RHPS1_LOWLEVEL_LAG=1: the real joints' low-pass (2026-10-15 logs, qOut -> qIn,
  # each joint ~ first order + 8-16 ms). The delay above was sized as a pure delay
  # standing in for both; with the low-pass on, its pure part is ~20 ms at most.
  if os.environ.get("RHPS1_LOWLEVEL_LAG", "0") == "1":
    _hz = {"CROTCH_Y": 5.75, "CROTCH_R": 4.9, "CROTCH_P": 4.75, "KNEE": 7.25,
           "ANKLE_P": 7.6, "ANKLE_R": 6.0, "CHEST": 6.4, "SHOULDER": 6.25,
           "ELBOW": 9.0, "WRIST": 9.0, "HAND": 9.0, "HEAD": 7.0}
    _rng = tuple(float(x) for x in os.environ.get("RHPS1_LOWLEVEL_LAG_RANGE", "0.75,1.25").split(","))
    for _act in cfg.scene.entities["robot"].articulation.actuators:
      _expr = " ".join(_act.target_names_expr)
      _f = next((v for k, v in _hz.items() if k in _expr), None)
      assert _f is not None and hasattr(_act, "lowlevel_lag_hz"), _expr
      # The drive's KP[3] is that pole already.
      _act.lowlevel_lag_hz = 0.0 if any(g in _expr for g in _drive_groups) else _f
      _act.lowlevel_lag_scale_range = _rng

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

  # RHPS1_FREE_ARMS=1: the arms leave the BWC (which holds them still) and swing
  # against the legs; trunk and head stay tracked, the head is also steadied.
  # Body terms keep their tracked-body list, so the critic input and the
  # checkpoints are unchanged.
  if os.environ.get("RHPS1_FREE_ARMS", "0") == "1":
    _arm = ("SHOULDER", "ELBOW", "WRIST")
    _kept = tuple(b for b in _TRACKED_BODIES if not any(t in b for t in _arm))
    for _k in ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
      if _k in cfg.rewards:
        cfg.rewards[_k].params["body_names"] = _kept
    cfg.terminations["ee_body_pos"].params["body_names"] = ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")
    # RHPS1_FREE_HEAD=1: the head leaves the BWC too; only its tilt rates and
    # its level are rewarded, its yaw is free (turning toward the step).
    _free_head = os.environ.get("RHPS1_FREE_HEAD", "0") == "1"
    if "motion_joint_pos_upper" in cfg.rewards:
      cfg.rewards["motion_joint_pos_upper"].params["skip"] = _arm + (("HEAD",) if _free_head else ())
    # Arm proximity weight: the swing brought the shoulder into the QP's 2 cm
    # shoulder-chest damper zone ~20 % of the time at -2.
    _apw = float(os.environ.get("RHPS1_ARM_PROX_W", "-2"))
    for _k in ("arm_torso_proximity", "shoulder_chest_proximity", "shoulder_body_proximity",
               "wrist_thigh_proximity"):
      if _k in cfg.rewards:
        cfg.rewards[_k].weight = _apw
    # RHPS1_ARM_PROX_MARGIN (m): the penalty starts that far before the QP zone.
    # S5 sat at 5.5 cm elbow-body in mjlab and 4.5 in mc_mujoco (QP braking
    # 61-68 %). Capped by the BWC's own posture: shoulder-chest 2.17 cm (1 mm
    # at most), wrist-thigh 0 (MuJoCo's hulls are already 1.5-2.7 cm wider).
    # Detection reaches 2 * RHPS1_PROX_GAP, raise it to cover the margin.
    _am = float(os.environ.get("RHPS1_ARM_PROX_MARGIN", "0"))
    if _am > 0:
      for _k, _cap in (("arm_torso_proximity", _am), ("shoulder_body_proximity", _am),
                       ("shoulder_chest_proximity", min(_am, 0.001))):
        _p = cfg.rewards[_k].params
        _p["min_dist"] = _p["min_dist"] + _cap
        assert _p["min_dist"] < 2 * float(os.environ.get("RHPS1_PROX_GAP", "0.025")), \
          f"{_k}: min_dist {_p['min_dist']} beyond detection, raise RHPS1_PROX_GAP"
    cfg.rewards["arm_swing"] = RewardTermCfg(
      func=_arm_swing,
      weight=float(os.environ.get("RHPS1_ARM_SWING_W", "1.0")),
      params={"k": float(os.environ.get("RHPS1_ARM_SWING_K", "0.6")),
              "std": float(os.environ.get("RHPS1_ARM_SWING_STD", "0.1"))},
    )
    cfg.rewards["head_stable"] = RewardTermCfg(
      func=_head_stable,
      weight=float(os.environ.get("RHPS1_HEAD_STAB_W", "1.0")),
      params={"std": float(os.environ.get("RHPS1_HEAD_STAB_STD", "0.2")), "tilt_only": _free_head},
    )
    if _free_head:
      cfg.rewards["head_level"] = RewardTermCfg(
        func=_head_level,
        weight=float(os.environ.get("RHPS1_HEAD_LEVEL_W", "1.0")),
        params={"std": float(os.environ.get("RHPS1_HEAD_LEVEL_STD", "0.1"))},
      )
      # RHPS1_HEAD_YAW_W: pushes doubled the HEAD_Y velocity (6 -> 10 deg/s rms)
      # while the tilt stayed at 3: the free yaw had no cost at all.
      _hy = float(os.environ.get("RHPS1_HEAD_YAW_W", "0"))
      if _hy > 0:
        cfg.rewards["head_yaw_still"] = RewardTermCfg(
          func=_head_yaw_still, weight=_hy,
          params={"std": float(os.environ.get("RHPS1_HEAD_YAW_STD", "0.15"))},
        )

  # Constraint gate for the caps below: RHPS1_CAP_SETTLE (s) ignored after each reset,
  # RHPS1_CAP_RAMP (env steps) linear weight ramp from the launch. Defaults off.
  _gate = {"settle": float(os.environ.get("RHPS1_CAP_SETTLE", "0")),
           "ramp_steps": int(os.environ.get("RHPS1_CAP_RAMP", "0"))}

  # RHPS1_TORQUE_ENV="KNEE:78,CROTCH_P:48,ANKLE_P:62": stay inside the slot 3
  # torque envelope while the reference changes.
  _te = os.environ.get("RHPS1_TORQUE_ENV")
  if _te:
    _tl = {k: float(v) for k, v in (kv.split(":") for kv in _te.split(","))}
    cfg.rewards["torque_envelope"] = RewardTermCfg(
      func=_torque_envelope, weight=float(os.environ.get("RHPS1_TORQUE_ENV_W", "-3")),
      params={"limits": _tl, **_gate},
    )

  # RHPS1_GUARD_VMAX=4: the controller's runaway guard as a termination. Hardened human
  # checkpoints armed at 3.4-3.7 rad/s in mjlab and were disarmed at 6 in mc_mujoco.
  _gv = os.environ.get("RHPS1_GUARD_VMAX")
  if _gv:
    from mjlab.managers.termination_manager import TerminationTermCfg

    cfg.terminations["joint_speed_guard"] = TerminationTermCfg(
      func=_joint_speed_guard, params={"vmax": float(_gv)})
  # RHPS1_JOINT_VMAX="KNEE:4,ANKLE_P:4": joint speed ceilings per group.
  _jv = os.environ.get("RHPS1_JOINT_VMAX")
  if _jv:
    _jl = {k: float(v) for k, v in (kv.split(":") for kv in _jv.split(","))}
    cfg.rewards["joint_speed_cap"] = RewardTermCfg(
      func=_joint_speed_cap, weight=float(os.environ.get("RHPS1_JOINT_VMAX_W", "-1")),
      params={"limits": _jl, **_gate},
    )

  # RHPS1_TOUCHDOWN_VMAX (m/s): feet land no faster than the BWC (0.21 m/s), weight
  # RHPS1_TOUCHDOWN_W per landing. Needs the foot-ground sensor (cadence_term).
  _tdv = os.environ.get("RHPS1_TOUCHDOWN_VMAX")
  if _tdv and cadence_term:
    cfg.rewards["touchdown_speed_cap"] = RewardTermCfg(
      func=_touchdown_speed_cap, weight=float(os.environ.get("RHPS1_TOUCHDOWN_W", "-5")),
      params={"sensor_name": "feet_ground_contact", "body_names": ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK"),
              "v_max": float(_tdv), **_gate},
    )

  # RHPS1_IMPACT_FMAX (N): touchdown force peaks no harder than slot 13's p99 (729 N),
  # weight RHPS1_IMPACT_W per landing.
  _ifm = os.environ.get("RHPS1_IMPACT_FMAX")
  if _ifm and cadence_term:
    cfg.rewards["impact_force_cap"] = RewardTermCfg(
      func=_impact_force_cap, weight=float(os.environ.get("RHPS1_IMPACT_W", "-5")),
      params={"sensor_name": "feet_ground_contact", "f_max": float(_ifm), **_gate},
    )

  # RHPS1_YAW_MOM_W: replaces the imposed arm swing (k 0.9 raised L_z rms from
  # 0.26 to 0.36-0.48: style, not dynamics). Pair with RHPS1_ARM_SWING_W=0.
  _ym = float(os.environ.get("RHPS1_YAW_MOM_W", "0"))
  if _ym > 0:
    cfg.rewards["yaw_momentum_still"] = RewardTermCfg(
      func=_yaw_momentum_still, weight=_ym,
      params={"std": float(os.environ.get("RHPS1_YAW_MOM_STD", "0.3")),
              "tau": float(os.environ.get("RHPS1_YAW_MOM_TAU", "1.0"))},
    )

  # RHPS1_FEET_HEIGHT_W: ankle HEIGHT tracking on its own. Slot 9 lifts 60-70 %
  # of the BWC in every direction (yaw 3 cm against 5): motion_feet_pos is mostly
  # fore-aft and lateral placement, see _feet_height_error_exp.
  _fh = float(os.environ.get("RHPS1_FEET_HEIGHT_W", "0"))
  if _fh > 0:
    cfg.rewards["motion_feet_height"] = RewardTermCfg(
      func=_feet_height_error_exp,
      weight=_fh,
      params={
        "command_name": "motion",
        "std": float(os.environ.get("RHPS1_FEET_HEIGHT_STD", "0.02")),
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
  # RHPS1_FOOT_SOLREF="timeconst,dampratio": softer foot-ground contact. On the
  # robot the pelvis tilts 0.5-2 deg during single support beyond what the
  # encoders explain (IMU vs flat-foot kinematics, 2026-10-16), against 0.03-0.25
  # in mc_mujoco: soles, foot rocking or hidden joint compliance. MuJoCo's
  # default is 0.02 s.
  _sr = os.environ.get("RHPS1_FOOT_SOLREF")
  if _sr:
    _srv = tuple(float(x) for x in _sr.split(","))
    _ent.collisions = tuple(
      _replace(c, solref={k: _srv for k in c.friction})
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
  # With RHPS1_HOLD_POSTURE_NOISE the held stance is off the reference: no
  # tracking termination while the policy's actions are ignored.
  if os.environ.get("RHPS1_HOLD_POSTURE_NOISE"):
    from mjlab.tasks.tracking.config.rhps1 import hold_action as _ha

    for _k, _f in (("anchor_pos", _ha.held_bad_anchor_pos_z_only), ("anchor_ori", _ha.held_bad_anchor_ori),
                   ("ee_body_pos", _ha.held_bad_motion_body_pos_z_only)):
      if _k in cfg.terminations:
        cfg.terminations[_k].func = _f
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
    # RHPS1_FULL_HISTORY=10: every term at that depth, oldest first per term --
    # rl_controller obs_format 2 (V3_DEEP_HISTORY_SIZE = 10, 1020 dims).
    _fh = int(os.environ.get("RHPS1_FULL_HISTORY", "0"))
    if _fh > 0:
      from dataclasses import replace as _hrep

      for _k in list(_v3):
        _v3[_k] = _hrep(_v3[_k], history_length=_fh)
    _tau = float(os.environ.get("RHPS1_CMD_TAU", "0"))
    if _tau > 0:
      from dataclasses import replace as _crep

      _v3["velocity_command"] = _crep(
        _v3["velocity_command"],
        func=_filtered_velocity_command,
        params={"command_name": "motion", "tau": _tau},
      )
    # RHPS1_HOLD_LAST_ACTION=1: the policy sees the held (applied) action, as
    # the controller seeds it at arming, not its own ignored output.
    if os.environ.get("RHPS1_HOLD_LAST_ACTION", "0") == "1":
      from dataclasses import replace as _arep

      from mjlab.tasks.tracking.config.rhps1.hold_action import held_last_action

      _v3["actions"] = _arep(_v3["actions"], func=held_last_action)
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

  _rand_events = {
    k: cfg.events.get(k) for k in ("push_robot", "base_com", "encoder_bias", "foot_friction")
  }
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

  # RHPS1_ROBUST: perturbations put back on the copy task one at a time,
  # comma-separated: obs (actor sensor noise + encoder bias), friction
  # (RHPS1_FRICTION_RANGE, default 0.5,1.2), com, push (velocity range scaled by
  # RHPS1_PUSH_SCALE, default 0.4).
  _rob = {s for s in os.environ.get("RHPS1_ROBUST", "").split(",") if s}
  if "obs" in _rob:
    cfg.observations["actor"].enable_corruption = not play
    cfg.events["encoder_bias"] = _rand_events["encoder_bias"]
    # IMU tilt error, ~3 deg: the V3 group left gravity noise-free.
    _at = cfg.observations["actor"].terms
    if "projected_gravity" in _at:
      from dataclasses import replace as _grep

      from mjlab.utils.noise import UniformNoiseCfg as _GUnoise

      _at["projected_gravity"] = _grep(
        _at["projected_gravity"], noise=_GUnoise(n_min=-0.05, n_max=0.05)
      )
  if "friction" in _rob:
    _f = [float(x) for x in os.environ.get("RHPS1_FRICTION_RANGE", "0.5,1.2").split(",")]
    cfg.events["foot_friction"] = _rand_events["foot_friction"]
    cfg.events["foot_friction"].params["ranges"] = (_f[0], _f[1])
  if "com" in _rob:
    cfg.events["base_com"] = _rand_events["base_com"]
  if "bias" in _rob:
    # Estimator offsets held for a whole episode: MCWaiko's velocity drifts
    # instead of being white noise, and the IMU sits slightly off level.
    _bl = float(os.environ.get("RHPS1_LINVEL_BIAS", "0.1"))
    _bg = float(os.environ.get("RHPS1_GRAVITY_BIAS", "0.03"))
    from dataclasses import replace as _brep

    _at = cfg.observations["actor"].terms
    for _k, _attr in (("base_lin_vel", "_rhps1_lin_bias"), ("projected_gravity", "_rhps1_grav_bias")):
      if _k in _at:
        _at[_k] = _brep(_at[_k], func=_biased_obs(_at[_k].func, _attr))
    from mjlab.managers.event_manager import EventTermCfg as _BEv

    cfg.events["obs_bias"] = _BEv(
      func=_sample_obs_bias, mode="reset", params={"lin": _bl, "grav": _bg}
    )
  if "sway" in _rob:
    # Continuous random pelvis torque. On the robot the body sways 2-3x more than
    # in sim for the same policy (gravity-y std 0.029 vs 0.013, gyro 2-3x,
    # 2026-10-16). Tested on slot 11 (2026-10-19): it raises sim sway but does not
    # shorten the steps, so it is not what shortens them on the robot.
    from mjlab.managers.event_manager import EventTermCfg as _SwEv

    _amp = tuple(float(x) for x in os.environ.get("RHPS1_SWAY_AMP", "20,20,10").split(","))
    cfg.events["sway_torque"] = _SwEv(func=_sway_torque, mode="step", params={
      "amp": _amp, "tau": float(os.environ.get("RHPS1_SWAY_TAU", "0.1"))})
  if "vel_ou" in _rob:
    # Slowly varying velocity-estimate error. On the robot, turning in place,
    # MCWaiko's forward velocity fluctuates 8x more than in mc_mujoco at the step
    # rate (1-2 Hz: 0.092 vs 0.011 m/s, 2026-10-16 slot 11 log), most likely the
    # stance foot pivoting under the leg odometry. A constant bias ("bias") is not
    # that; slot 12, trained with it, still shrank its steps on the robot.
    from dataclasses import replace as _orep

    from mjlab.managers.event_manager import EventTermCfg as _OEv

    _at = cfg.observations["actor"].terms
    if "base_lin_vel" in _at:
      _at["base_lin_vel"] = _orep(_at["base_lin_vel"], func=_ou_obs(_at["base_lin_vel"].func))
    cfg.events["vel_ou"] = _OEv(func=_sample_ou, mode="reset", params={
      "sigma": float(os.environ.get("RHPS1_VEL_OU_SIGMA", "0.12")),
      "tau_min": float(os.environ.get("RHPS1_VEL_OU_TAU_MIN", "0.1")),
      "tau_max": float(os.environ.get("RHPS1_VEL_OU_TAU_MAX", "0.5")),
    })
  if "gains" in _rob:
    from mjlab.envs.mdp import dr as _dr
    from mjlab.managers.event_manager import EventTermCfg as _GEv
    from mjlab.managers.scene_entity_config import SceneEntityCfg as _GSc

    _g = float(os.environ.get("RHPS1_GAIN_RANGE", "0.2"))
    cfg.events["pd_gains"] = _GEv(
      func=_dr.pd_gains, mode="reset",
      params={"asset_cfg": _GSc("robot"), "kp_range": (1 - _g, 1 + _g),
              "kd_range": (1 - _g, 1 + _g)},
    )
  if "push" in _rob:
    _ps = float(os.environ.get("RHPS1_PUSH_SCALE", "0.4"))
    cfg.events["push_robot"] = _rand_events["push_robot"]
    _vr = cfg.events["push_robot"].params["velocity_range"]
    cfg.events["push_robot"].params["velocity_range"] = {
      k: (v[0] * _ps, v[1] * _ps) for k, v in _vr.items()
    }

  # RHPS1_AMP=1: human style from a discriminator on retargeted LAFAN1 walking
  # (amp.py). The BWC stays the command schedule, reset states and z-only
  # terminations; its imitation terms give way to velocity tracking of the
  # direction command at RHPS1_AMP_SPEED (vx, vy, wz).
  if os.environ.get("RHPS1_AMP", "0") == "1":
    from . import amp as _amp

    for _k in [k for k in cfg.rewards if k.startswith("motion_")] + [
      "contact_schedule", "double_support"
    ]:
      cfg.rewards.pop(_k, None)
    cfg.rewards["track_lin_vel"] = RewardTermCfg(
      func=_amp.track_lin_vel, weight=float(os.environ.get("RHPS1_AMP_VEL_W", "2.0")),
      params={"command_name": "motion", "std": float(os.environ.get("RHPS1_AMP_VEL_STD", "0.15"))},
    )
    cfg.rewards["track_yaw_rate"] = RewardTermCfg(
      func=_amp.track_yaw_rate, weight=float(os.environ.get("RHPS1_AMP_YAW_W", "1.0")),
      params={"command_name": "motion", "std": float(os.environ.get("RHPS1_AMP_YAW_STD", "0.25"))},
    )
    cfg.observations["amp"] = ObservationGroupCfg(
      terms={"amp": ObservationTermCfg(func=_amp.env_features)},
      concatenate_terms=True, enable_corruption=False,
    )

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
