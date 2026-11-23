"""SynaptiQ — récompense et plasticité (lot 2, 23/11/2026). Fonctions PURES.

## Récompense (circuit de la dopamine)

Un souvenir que l'agent déclare UTILE gagne du poids au rappel ; un souvenir déclaré
inutile en perd. Le signal est EXPLICITE (outil `reinforce`), jamais déduit du nombre
d'accès : `access_count` mesure ce qui a été sélectionné, pas ce qui a servi, et s'en servir
comme récompense entretiendrait la boucle déjà relevée sur la récence (CLAUDE.md §7bis) —
plus un souvenir sort, plus il sortirait.

`memories.reward` est borné à [-RECOMPENSE_MAX, +RECOMPENSE_MAX] et le facteur de score qui
en dérive est borné à [1 - gain·max, 1 + gain·max] (défaut [0,5 ; 1,5]) : une récompense
nuance le classement, elle ne remplace jamais la pertinence.

## Plasticité (synapses hebbiennes)

« Ce qui s'active ensemble se lie ensemble. » Des souvenirs déclarés utiles ENSEMBLE
renforcent le lien qui les unit (créé s'il n'existe pas) ; pendant le sommeil, les liens
renforcés puis délaissés s'affaiblissent jusqu'au poids d'un lien neuf, jamais en dessous —
un lien n'est jamais supprimé (décision du 23/11 : aucune suppression de données).
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

RECOMPENSE_MAX = 5.0
POIDS_LIEN_NOUVEAU = 0.5
POIDS_LIEN_MAX = 1.0
POIDS_LIEN_PLANCHER = 0.1
# Au-delà, l'outil refuse : 10 souvenirs = 45 paires, assez pour un paquet de contexte.
SOUVENIRS_PAR_RENFORCEMENT_MAX = 10


def borner_recompense(valeur: float) -> float:
    return max(-RECOMPENSE_MAX, min(RECOMPENSE_MAX, float(valeur)))


def facteur_recompense(recompense: float | None, gain: float) -> float:
    """Multiplicateur de score : 1 + gain × récompense bornée."""
    if not recompense or gain <= 0:
        return 1.0
    return 1.0 + gain * borner_recompense(recompense)


def renforcer_poids(poids: float | None, pas: float) -> float:
    """Poids d'un lien après co-activation utile (plafonné)."""
    base = POIDS_LIEN_NOUVEAU if poids is None else float(poids)
    return min(POIDS_LIEN_MAX, base + pas)


def affaiblir_poids(poids: float, facteur: float, plancher: float = POIDS_LIEN_NOUVEAU) -> float:
    """Poids d'un lien RENFORCÉ puis délaissé, après une nuit.

    Ne concerne que les liens déjà renforcés (`last_reinforced_at` non nul) : affaiblir aussi
    les liens de similarité jamais renforcés effacerait en deux mois le graphe qui porte le
    gain du multi-saut. Le plancher par défaut est le poids d'un lien neuf : un lien délaissé
    revient à l'état d'un lien jamais renforcé, il ne devient pas plus faible que lui.
    """
    if float(poids) <= plancher:
        return float(poids)
    return max(plancher, float(poids) * (1.0 - facteur))


def paires_hebbiennes(ids: Sequence[str]) -> list[tuple[str, str]]:
    """Paires à renforcer, dans un ordre canonique (le plus petit id en source)."""
    uniques = sorted({str(i) for i in ids})
    return list(combinations(uniques, 2))
