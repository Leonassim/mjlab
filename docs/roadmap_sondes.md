# Sondes, puis run propre

Méthode proposée par Léo, et elle corrige le défaut principal de la campagne :
quinze runs longs, quinze checkpoints de départ, des configurations qui se
recouvraient — rien n'était comparable à rien.

## Phase A — sondes

Chaque sonde : **400 itérations depuis le même checkpoint**, **un seul rung**,
puis un balayage avec randomisation. Elles ne cherchent pas la convergence,
seulement le **signe de la dérivée** : est-ce que ce terme déplace sa cible, et
qu'est-ce qu'il casse en le faisant.

| sonde | rung | question |
|---|---|---|
| P1 | `standfirm` | tenir debout à l'arrêt sous perturbation |
| P2 | période | allonger le cycle, qui régresse au lieu de croître |
| P3 | `wide` | élargir la randomisation, jamais commencé |
| P4 | contacts plats | un terme qui **paie** l'amélioration, pas une pénalité |
| P5 | `impactladder` | le curriculum d'impact, déjà en cours |

Sortie : une ligne par sonde — la cible visée, son delta, et les dégâts
collatéraux sur les quatre autres critères.

## Phase B — le run propre

L'ordre du curriculum se déduit des sondes : les rungs qui déplacent leur cible
sans casser le reste, enchaînés par dépendance, chacun franchi sur une réussite
mesurée et non sur un compteur.

Et la randomisation se décide à ce moment-là, sur ce que P3 aura montré — pas
avant, parce qu'élargir la randomisation change le plant et rendrait les autres
sondes incomparables si on le faisait en même temps.

## Ce qu'on sait déjà

`p0+rand` **restaure** la randomisation de policy 0, il ne l'élargit pas. Elle
n'a jamais été touchée de toute la campagne, dans aucun sens.
