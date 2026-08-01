"""Troncature au budget de tokens — règle PARTAGÉE par les bras du harness LOCOMO.

Un bras qui dispose de plus de contexte qu'un autre gagne pour une raison qui n'a rien à
voir avec la qualité de sa mémoire. La comparaison n'a donc de sens que si tous les bras
remplissent le MÊME budget avec le MÊME estimateur — `estimate_tokens`, celui qu'utilise
le collapse Q-EM côté serveur.

Ce module existe pour qu'il n'y ait qu'UNE implémentation de cette règle. Dupliquée dans
chaque bras, elle finirait par diverger, et la divergence serait invisible dans le rapport :
le lecteur verrait un écart d'exactitude là où il n'y aurait qu'un écart de budget.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synaptiq_core.qem import estimate_tokens


def fit_to_budget(contents: Iterable[str], max_tokens: int, *,
                  prefix: str = "- ") -> tuple[str, int]:
    """Remplit `max_tokens` avec les contenus reçus, dans l'ordre. Retourne (texte, tokens).

    Un contenu trop volumineux est SAUTÉ, jamais terminal : c'est exactement la règle de
    `collapse_by_utility` côté serveur, qui continue d'examiner les candidats suivants
    après un rejet. S'arrêter au premier dépassement laisserait du budget inutilisé qu'un
    souvenir plus court pourrait encore occuper — et le bras ainsi bridé perdrait des
    points pour une raison d'implémentation, pas de mémoire.
    """
    lignes: list[str] = []
    total = 0
    for content in contents:
        if not content:
            continue
        cout = estimate_tokens(content)
        if total + cout > max_tokens:
            continue
        lignes.append(f"{prefix}{content}")
        total += cout
    return "\n".join(lignes), total
