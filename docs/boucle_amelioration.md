# La boucle

Protocole à suivre à chaque itération, jusqu'à ce que le balayage annonce que
tous les critères passent. Il existe parce que la campagne du 24-25 août ne
l'avait pas : sept changements choisis par le signal le plus bruyant du moment,
sur sept points de départ, jugés sur des métriques d'entraînement fausses.

## Un tour

1. **Entraîner** jusqu'au jalon, sans rien toucher. 2000 itérations au minimum —
   un changement de récompense met des milliers d'itérations à devenir un
   comportement, et conclure à 300 est ce qui a produit quatre verdicts faux.

2. **Balayer** le meilleur checkpoint :
   `uv run python scripts/tools/sweep_eval.py <run> <model.pt>`
   Le balayage se termine par un verdict qui **nomme le critère le plus en
   défaut**. C'est lui la cible, pas l'intuition du moment.

3. **Un seul changement**, celui qui vise ce critère. Un deuxième rend le
   résultat inattribuable, et la bande de bruit de cette échelle est large.

4. **Reprendre depuis le checkpoint que le balayage a validé**, pas depuis le
   dernier. Ils ne sont presque jamais le même : les 150 dernières itérations
   d'un run qui dérive rendent de la qualité contre du couple.

5. Retour en 1.

## Règles qui ne se négocient pas

- **L'acceptation se fait sur le balayage.** Les métriques d'entraînement ont
  annoncé 4.9 % de chutes et 0.31 d'écrêtage là où le déterministe lit 0.8 % et
  0.07. Elles servent à surveiller une dérive en cours de run, jamais à juger.
- **Les garde-fous ne bougent pas pendant un run.** Les déplacer, ce qui est
  arrivé quatre fois, rend tout verdict ininterprétable. S'ils sont mal réglés,
  on le corrige entre deux runs et on le note.
- **Les plafonds portent sur ce qui casse le robot**, pas sur ce qui est commode
  à mesurer, et ils se placent **au-dessus** de la cible que la récompense vise.
- **Mesurer avant de diagnostiquer.** Sur cette campagne, trois hypothèses
  posées avant mesure étaient fausses : la période sous-incitée alors qu'elle
  était contrainte par le budget, la saturation de la distance qui visait la
  mauvaise source, le balancement des bras qui ne décharge rien.

## Critères, dans l'ordre de priorité de Léo

| | seuil | pire cas sur la grille |
|---|---|---|
| ne jamais tomber | < 1 % | |
| couples faisables | écrêtage jambes < 0.25 | |
| impact faible | < 0.16 | |
| lever de pied | >= 0.030 | minimum sur les commandes en mouvement |
| pieds à plat | >= 3 coins sur 4 | idem |
| immobile à l'arrêt | pas de dérive à commande nulle | |

## État au 2026-08-25

`it15600` passe quatre critères sur six. Échecs : impact 0.272 contre 0.16, et
2.0 coins au sol sur 4. Itération en cours : `softland2`, qui tarife les deux.
