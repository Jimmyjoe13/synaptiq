"""SynaptiQ — taxonomie des mémoires (type / sous-type) et routage vers les collections.

## Pourquoi ce module existe

La taxonomie vivait dans `apps/worker/worker.py`, donc appliquée UNIQUEMENT sur le chemin
d'extraction LLM. L'écriture directe (`POST /v1/memories`) n'en savait rien : n'importe
quel sous-type y passait. Constaté en production le 29/07 — des mémoires écrites par un
agent portaient `seo_audit_july_2026`, `nana_intelligence_lead_webhook`… et l'API ne
sourcillait pas, alors que le worker aurait corrigé ces valeurs.

Deux chemins d'écriture, deux règles : c'est une divergence garantie. La taxonomie est donc
partagée ici, et les deux chemins l'appliquent.

## Ce qui est refusé, et ce qui ne l'est pas

Un sous-type **inconnu est ACCEPTÉ** (champ libre). Ce n'est pas du laxisme : le routage
retombe proprement sur la collection du type (`semantic/*` -> `facts`), et un agent a de
bonnes raisons de vouloir un libellé métier (`nana_intelligence_lead_webhook`) plus précis
que `fact`. Interdire ces valeurs casserait des intégrations existantes sans rien protéger.

Un sous-type **appartenant à un AUTRE type est refusé**. `type=semantic` avec
`subtype=coding_best_practices` est une erreur de l'appelant, pas un libellé personnalisé :
le souvenir irait dans `facts` alors que son auteur visait `best_practices`. C'est la seule
classe d'erreur que la validation peut réellement attraper, donc la seule qu'elle attrape.
"""
from __future__ import annotations

from synaptiq_core.collections import (
    FAMILY_FALLBACK_KEY,
    SYSTEM_COLLECTIONS,
    CollectionRegistry,
)

# Types de mémoire reconnus, et leurs sous-types CANONIQUES (ceux qui pilotent le routage
# fin vers les 7 collections du context_packet — cf. `qem.route_memory`).
#
# DÉRIVÉ de `collections.SYSTEM_COLLECTIONS`, et non plus écrit à la main : c'était la même
# information tenue à deux endroits, donc une divergence en attente. Un sous-type canonique
# est exactement une collection livrée avec le moteur — le nom d'un rayon que SynaptiQ
# connaît d'avance, par opposition à ceux que l'agent se crée.
VALID_SUBTYPES: dict[str, set[str]] = {}
for _col in SYSTEM_COLLECTIONS:
    VALID_SUBTYPES.setdefault(_col.family, set()).add(_col.name)

# Sous-type retenu quand l'extraction LLM n'en propose pas d'exploitable : la collection
# système dont la clé de paquet est celle de repli de la famille (semantic -> facts, donc
# `fact` ; procedural -> rules, donc `rule`…). Ainsi le défaut suit automatiquement le
# routage au lieu d'être une troisième liste à maintenir.
DEFAULT_SUBTYPE: dict[str, str] = {
    _col.family: _col.name
    for _col in SYSTEM_COLLECTIONS
    if _col.packet_key == FAMILY_FALLBACK_KEY.get(_col.family)
}


class SubtypeMismatch(ValueError):
    """Le sous-type est canonique, mais rattaché à un autre type de mémoire."""


def owner_type_of(subtype: str) -> str | None:
    """Type auquel un sous-type canonique appartient, ou None s'il est libre."""
    for mtype, subtypes in VALID_SUBTYPES.items():
        if subtype in subtypes:
            return mtype
    return None


def is_canonical(memory_type: str, subtype: str | None) -> bool:
    """Le couple (type, sous-type) fait-il partie de la taxonomie canonique ?"""
    return bool(subtype) and subtype in VALID_SUBTYPES.get(memory_type, set())


def validate_subtype(memory_type: str, subtype: str | None) -> str | None:
    """Valide un couple (type, sous-type) fourni par un appelant.

    Retourne le sous-type inchangé s'il est acceptable. Lève `SubtypeMismatch` si le
    sous-type est canonique mais appartient à un autre type — le seul cas où l'intention
    de l'appelant est démontrablement trahie par le routage.
    """
    if not subtype:
        return subtype
    proprietaire = owner_type_of(subtype)
    if proprietaire is not None and proprietaire != memory_type:
        raise SubtypeMismatch(
            f"Le sous-type '{subtype}' appartient au type '{proprietaire}', pas à "
            f"'{memory_type}'. Utiliser type='{proprietaire}', ou un sous-type libre "
            f"si l'intention est différente."
        )
    return subtype


def normalize_extraction(memory_type: str | None, subtype: str | None,
                         registry: CollectionRegistry | None = None) -> tuple[str, str]:
    """Normalise une sortie de LLM en un couple (famille, collection) sûr.

    Contrairement à `validate_subtype`, on ne lève rien ici : un modèle qui hallucine un
    type doit produire une mémoire dégradée, jamais faire perdre l'événement.

    ## Ce que le registre change

    Sans registre, tout sous-type non canonique était ÉCRASÉ par le défaut de sa famille.
    Sur le chemin d'extraction, un `clients_paca` proposé par le LLM devenait donc `fact` —
    la collection que l'agent s'est donnée était détruite à l'écriture, alors même que
    l'écriture directe (`POST /v1/memories`) l'aurait acceptée. Encore deux chemins, deux
    règles.

    Avec un registre, une collection DÉCLARÉE par l'agent est conservée telle quelle. Un
    sous-type ni canonique ni déclaré reste ramené au défaut : sur ce chemin, la valeur
    vient d'un modèle et non d'une intention, et laisser un LLM peupler la taxonomie à
    chaque hallucination la rendrait illisible en quelques jours. La création reste un acte
    délibéré (lot 3).
    """
    mtype = memory_type if memory_type in VALID_SUBTYPES else "semantic"
    if subtype is not None and subtype in VALID_SUBTYPES[mtype]:
        return mtype, subtype
    if registry is not None and registry.get(mtype, subtype) is not None:
        # `get` ne renvoie jamais autre chose que None pour un `subtype` vide : le passage
        # ici garantit donc que `subtype` est une chaîne non vide.
        return mtype, str(subtype)
    return mtype, DEFAULT_SUBTYPE[mtype]
