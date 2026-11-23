"""SynaptiQ — le sommeil : consolidation nocturne de la mémoire (lot 2, 23/11/2026).

Fonctions PURES (aucune I/O) : l'orchestration SQL vit dans `apps/worker/sommeil.py`.

Le cerveau humain consolide pendant le sommeil : l'hippocampe rejoue les épisodes de la
journée, le cortex en retient l'essentiel, le reste s'efface. SynaptiQ fait de même, chaque
nuit, avec une contrainte posée par Jimmy le 23/11 : **rien n'est supprimé**. Ce qui
s'« efface » est rangé dans le compartiment `archives` (mémoire dormante), consultable et
restaurable.

Phases (dans cet ordre, chacune désactivable) :
  1. doublons      — quasi-identiques du même compartiment : le moins bon part en archives ;
  2. consolidation — épisodes rejoués par un LLM et transformés en savoirs durables ; les
                     épisodes source sont marqués « consolidés » (pas archivés tout de suite) ;
  3. oubli         — souvenirs jamais relus dont la récence est tombée sous un seuil :
                     archivés (jamais les réflexes ni les croyances) ;
  4. graphe        — souvenirs isolés des compartiments qui intriquent : reliés ;
  5. plasticité    — liens non renforcés : affaiblis (jamais supprimés).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from synaptiq_core.compartments import (
    COMPARTIMENT_ARCHIVES,
    COMPARTIMENT_CROYANCES,
    COMPARTIMENT_REFLEXES,
    PAR_NOM,
    cosinus,
    demi_vie,
)
from synaptiq_core.qem import compute_recency_factor

# Compartiments que l'oubli ne touche jamais : les réflexes (règles permanentes), les
# croyances (révisées ou contestées, jamais usées) et les archives elles-mêmes.
JAMAIS_OUBLIES = frozenset({COMPARTIMENT_REFLEXES, COMPARTIMENT_CROYANCES, COMPARTIMENT_ARCHIVES})

# Compartiments que la consolidation peut produire. Ni réflexes (ils exigent un choix
# explicite, cf. `compartments`), ni croyances, ni épisodes/observations (on consolide
# précisément pour en sortir).
CIBLES_CONSOLIDATION = ("decisions", "savoir_faire", "alertes", "environnement", "savoirs")


@dataclass(frozen=True)
class ReglagesSommeil:
    doublon_seuil: float = 0.95
    oubli_seuil: float = 0.05
    oubli_age_min_jours: float = 30.0
    consolidation_age_min_jours: float = 3.0
    consolidation_groupe_seuil: float = 0.75
    consolidation_groupes_max: int = 20
    consolidation_groupe_taille_max: int = 8
    graphe_max: int = 500
    plasticite_facteur: float = 0.05
    plasticite_repos_jours: float = 7.0


def score_survie(m: Mapping[str, Any]) -> tuple:
    """Clé de tri : le souvenir qui SURVIT à un doublon a le meilleur score.

    Ordre : récompense, nombre d'accès, importance, puis le plus récent. Le plus récent en
    dernier seulement : « le plus récent gagne » a déjà coûté 10 souvenirs sur 14 au banc
    du 26/09 (CLAUDE.md §7bis).
    """
    return (float(m.get("reward") or 0.0), int(m.get("access_count") or 0),
            float(m.get("importance") or 0.5), str(m.get("created_at") or ""))


def paire_doublon(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[str, str]:
    """(survivant, archivé) pour deux souvenirs jugés doublons."""
    if score_survie(a) >= score_survie(b):
        return str(a["id"]), str(b["id"])
    return str(b["id"]), str(a["id"])


def doit_oublier(compartiment: str, age_acces_s: float | None, access_count: int,
                 demi_vie_instance: float, reglages: ReglagesSommeil) -> bool:
    """Un souvenir doit-il passer en archives par oubli ?

    Seulement s'il n'a JAMAIS été relu, qu'il est assez vieux, et que sa récence (calculée
    avec la demi-vie de SON compartiment) est tombée sous le seuil. Un compartiment sans
    oubli (facteur 0) n'est jamais concerné.
    """
    if compartiment in JAMAIS_OUBLIES or compartiment not in PAR_NOM:
        return False
    if access_count > 0 or age_acces_s is None:
        return False
    if age_acces_s < reglages.oubli_age_min_jours * 86400:
        return False
    vie = demi_vie(compartiment, demi_vie_instance)
    if vie <= 0:
        return False
    return compute_recency_factor(age_acces_s, vie) < reglages.oubli_seuil


def grouper(souvenirs: Sequence[Mapping[str, Any]], seuil: float,
            taille_max: int) -> list[list[Mapping[str, Any]]]:
    """Regroupement glouton par similarité (épisodes d'un même sujet).

    Chaque souvenir rejoint le premier groupe dont le PREMIER membre lui est assez proche ;
    sinon il ouvre un groupe. Simple, déterministe, suffisant pour quelques centaines
    d'épisodes par nuit.
    """
    groupes: list[list[Mapping[str, Any]]] = []
    for s in souvenirs:
        for g in groupes:
            if len(g) < taille_max and cosinus(g[0]["embedding"], s["embedding"]) >= seuil:
                g.append(s)
                break
        else:
            groupes.append([s])
    return groupes


def prompt_consolidation(episodes: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Messages pour le LLM : rejouer des épisodes et en extraire le durable."""
    journal = "\n".join(f"- [{str(e.get('created_at') or '')[:10]}] {e['content']}"
                        for e in episodes)
    cibles = "\n".join(f"   - '{c}': {PAR_NOM[c].description}" for c in CIBLES_CONSOLIDATION)
    consigne = (
        "Below are episodes from an AI agent's memory (session logs, interactions).\n"
        "Extract only what is DURABLE and still useful later: decisions with their reason, "
        "proven procedures, resolved errors, locations/access, stable facts.\n"
        "Rules:\n"
        "1. 0 to 3 memories. Return an EMPTY list if nothing durable is stated.\n"
        "2. Each memory is a self-contained sentence, understandable without the episodes, "
        "with absolute dates when known.\n"
        "3. Write IN THE SAME LANGUAGE as the episodes. Never translate.\n"
        f"4. Put each memory in ONE compartment:\n{cibles}\n\n"
        f"Episodes:\n{journal}\n\n"
        'Reply with JSON only: {"memories": [{"compartment": "...", "content": "..."}]}'
    )
    return [{"role": "system", "content": "Memory consolidation. Reply with JSON only."},
            {"role": "user", "content": consigne}]


def valider_consolidation(donnees: Any) -> list[tuple[str, str]]:
    """Souvenirs consolidés exploitables : [(compartiment, contenu)], au plus 3.

    Ne lève jamais : un compartiment hors cible retombe sur `savoirs`, un contenu vide est
    écarté. Un modèle qui hallucine produit au pire un savoir mal rangé, jamais une perte.
    """
    items = donnees.get("memories") if isinstance(donnees, dict) else donnees
    if not isinstance(items, list):
        return []
    resultats: list[tuple[str, str]] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        contenu = str(item.get("content") or "").strip()
        if len(contenu) < 10:
            continue
        compartiment = str(item.get("compartment") or "")
        resultats.append((compartiment if compartiment in CIBLES_CONSOLIDATION else "savoirs",
                          contenu))
    return resultats
