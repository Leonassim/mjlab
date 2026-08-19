# Ablation depuis policy 0

Chaque barreau part de zero et s'arrete a l'iteration 700 (~1 h). Deux mesures, 400 et 700, moyennees sur +/-25 iterations : un seul jalon s'est deja trompe dans les deux sens.

Policy 0 = run 2026-07-10_20-59-17, son ONNX est celui deploye. Ses reperes a l'iteration 600 : track_linear_velocity 2.60, stance_contacts_mean 2.81 ; regime etabli 3.0 / 2.89. Les runs qui echouent plafonnent a 1.95 et 3.6-3.8.

`verdict` = walks si tlv >= 2.4 et stance <= 3.0, broken si tlv <= 2.1 ou stance >= 3.4, sinon between. Cumulatif : un barreau casse designe sa propre deviation.

Cellules `400 / 700`. progress_ratio et sole_height_p90 n'existaient pas en juillet : ces colonnes ne comparent que les barreaux entre eux.

| ablation | verdict | status | track_linear_velocity | stance_contacts_mean | progress_ratio | sole_height_p90 | foot_vel_max | air_time_mean | fell_down | torque_limit_ratio_mean | mean_episode_length | mean_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p0 | broken | ok | 2.4603 / 1.3649 | 2.7128 / 2.5739 | 0.0149 / 0.0174 | 0.0063 / 0.0089 | 1.1433 / 1.267 | 3.1357 / 2.7532 | 0.0021 / 0.0252 | 0.3804 / 0.4612 | 3997.2947 / 3981.1007 | 0.4696 / 0.6099 |

## Runs

- `p0` : 2026-08-19_18-49-41, 62.0 min, `logs/ablation_p0.log`
