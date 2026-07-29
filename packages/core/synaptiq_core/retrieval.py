"""SynaptiQ — fusion de classements pour la recherche hybride.

Fonctions PURES, sans accès base ni dépendance framework : elles combinent des listes
d'identifiants déjà classées, quelle que soit leur provenance.

Pourquoi une fusion plutôt qu'un score composite pondéré : les scores d'une recherche
vectorielle (cosinus, borné [0,1]) et d'une recherche plein texte (`ts_rank`, non borné et
dépendant de la longueur du document) ne vivent PAS sur la même échelle. Les additionner
suppose une normalisation arbitraire qu'il faudrait recalibrer à chaque changement de
modèle d'embedding. La fusion par rang (RRF) ignore les valeurs et ne regarde que l'ordre :
elle est donc robuste par construction, et c'est l'approche retenue par les moteurs
hybrides de référence.
"""
from __future__ import annotations

from collections.abc import Iterable

# Constante d'amortissement de la RRF. 60 est la valeur de la publication d'origine
# (Cormack et al., 2009) et le défaut de la plupart des implémentations : elle empêche les
# tout premiers rangs d'écraser le reste, tout en gardant un ordre nettement décroissant.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Iterable[list[str]],
    k: int = DEFAULT_RRF_K,
    weights: Iterable[float] | None = None,
) -> dict[str, float]:
    """Fusionne plusieurs classements en un score unique par identifiant.

    Chaque classement contribue `poids / (k + rang)`, le rang commençant à 1. Un document
    bien placé dans PLUSIEURS classements dépasse un document premier dans un seul : c'est
    exactement l'effet recherché, un souvenir à la fois sémantiquement proche et contenant
    les bons termes littéraux doit primer.

    - `rankings` : listes d'identifiants, du plus au moins pertinent.
    - `weights`  : importance relative de chaque classement (défaut : 1.0 chacun).

    Retourne `{id: score}`, non trié — l'appelant décide de l'ordre et des coupes.
    Les identifiants dupliqués au sein d'un même classement ne comptent qu'une fois (au
    meilleur rang), sans quoi un doublon gonflerait artificiellement le score.
    """
    rankings = list(rankings)
    if weights is None:
        poids = [1.0] * len(rankings)
    else:
        poids = list(weights)
        if len(poids) != len(rankings):
            raise ValueError("weights doit avoir autant d'entrées que rankings")

    scores: dict[str, float] = {}
    for ranking, poids_i in zip(rankings, poids, strict=False):
        vus = set()
        for rang, doc_id in enumerate(ranking, start=1):
            if doc_id in vus:
                continue
            vus.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + poids_i / (k + rang)
    return scores


def fuse_and_rank(
    rankings: Iterable[list[str]],
    k: int = DEFAULT_RRF_K,
    weights: Iterable[float] | None = None,
    limit: int | None = None,
) -> list[str]:
    """`reciprocal_rank_fusion` suivie d'un tri décroissant, éventuellement tronqué.

    À score égal, l'ordre du premier classement fourni départage : le résultat reste
    déterministe d'une exécution à l'autre, ce qui est indispensable pour qu'un benchmark
    soit reproductible.
    """
    rankings = list(rankings)
    scores = reciprocal_rank_fusion(rankings, k=k, weights=weights)

    ordre_reference: dict[str, int] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            ordre_reference.setdefault(doc_id, position)

    classes = sorted(scores, key=lambda d: (-scores[d], ordre_reference.get(d, 1 << 30)))
    return classes[:limit] if limit is not None else classes
