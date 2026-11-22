"""SynaptiQ — orchestration du moteur Q-EM (assemblage d'un paquet de contexte).

## Pourquoi ce module existe

`build_context` vivait dans le handler HTTP (`apps/api/main.py`), sur 199 lignes mêlant
requêtes SQL, fusion de rangs, appel des phases Q-EM et sérialisation de la réponse.
Conséquence relevée à l'audit du 28/07 : les phases Q-EM étaient testables (elles sont
pures, dans `qem.py`) mais **l'orchestration ne l'était pas** — or c'est précisément là que
vivaient la fuite d'isolation F1 et le calcul du score de départ. Tout test devait passer
par HTTP et une base réelle.

L'accès aux données est donc isolé derrière `MemoryStore`. Ce module n'importe ni FastAPI,
ni psycopg2 : il s'exécute avec un magasin en mémoire, sans infrastructure.

## L'invariant d'isolation est porté par le type

Un `MemoryStore` est construit pour UN couple (tenant, agent) et toutes ses méthodes y sont
bornées. Aucune méthode ne prend de `tenant_id` ni d'`agent_id` en argument : il devient
donc structurellement impossible de rejouer F1, où la requête de complétion du graphe avait
« oublié » ces deux filtres. Le périmètre n'est plus une discipline d'écriture, c'est une
propriété du magasin.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from synaptiq_core.compartments import (
    COMPARTIMENT_CROYANCES,
    COMPARTIMENT_REFLEXES,
    PAR_NOM,
    demi_vie,
    packet_vide,
    parts_max,
)
from synaptiq_core.qem import (
    apply_contradictions,
    collapse_by_utility,
    compute_recency_factor,
    filter_redundancy,
    initial_score,
    propagate_entanglement,
)
from synaptiq_core.retrieval import DEFAULT_RRF_K, reciprocal_rank_fusion

logger = logging.getLogger("synaptiq-core.context")

# Paquet vide : les 9 compartiments (forme stable). Conservé comme constante publique,
# DÉRIVÉ de `compartments.COMPARTIMENTS` — jamais une seconde liste écrite à la main.
PACKET_VIDE: dict[str, list[str]] = packet_vide()


@dataclass(frozen=True)
class RetrievalConfig:
    """Réglages du moteur, tous externalisés en variables d'environnement côté API.

    Regroupés en un objet immuable plutôt que lus par `os.getenv` au fil du code : une
    exécution de benchmark peut ainsi faire varier une seule phase (`entangle_max_hops=0`,
    `hybrid=False`…) sans redéploiement ni variable globale, ce qui est la condition d'une
    table d'ablation honnête.
    """
    hybrid: bool = True
    candidates: int = 50
    rrf_k: int = DEFAULT_RRF_K
    weight_vector: float = 1.0
    weight_fts: float = 1.0
    entangle_damping: float = 0.5
    entangle_max_hops: int = 2
    # Agrégation des apports de plusieurs parents : "max" (défaut) ou "sum" (historique).
    entangle_aggregation: str = "max"
    redundancy_threshold: float = 0.90
    # Collapse (cf. `qem.collapse_by_utility`) : meilleurs admis d'office, exposant β.
    collapse_top_k: int = 3
    density_exponent: float = 0.5
    # Part max du budget par compartiment plafonné (rôle de l'hypothalamus). Les défauts
    # viennent des compartiments ; ces deux champs permettent de les surcharger par
    # instance ou par banc (QEM_CROYANCES_MAX_SHARE, QEM_REFLEXES_MAX_SHARE).
    croyances_max_share: float = 0.15
    reflexes_max_share: float = 0.20
    # Nombre max de réflexes injectés à CHAQUE contexte, quelle que soit la question
    # (tronc cérébral). 0 = désactive l'injection systématique.
    reflexes_injectes: int = 8
    # Demi-vie de la décroissance temporelle, en jours. Recalibrée de 90 → 14 le 11/08 :
    # à 90 jours sur un corpus de 13 jours, la décroissance était inerte (amplitude
    # 0,89–1,00 mesurée par l'audit) — elle ne triait rien. 14 jours la rend perceptible
    # sur l'horizon réel d'une mémoire d'agent.
    recency_halflife_days: float = 14.0

    def parts(self) -> dict[str, float]:
        """Parts max effectives par compartiment."""
        return parts_max({COMPARTIMENT_CROYANCES: self.croyances_max_share,
                          COMPARTIMENT_REFLEXES: self.reflexes_max_share})


class MemoryStore(Protocol):
    """Accès aux mémoires, borné à un (tenant, agent) fixés à la construction.

    Les lignes retournées sont des mappings portant au moins : `id`, `type`, `compartment`,
    `content`, `confidence`, `importance`, `last_accessed_at`, `created_at`, `occurred_at`,
    `embedding` (séquence de flottants, déjà désérialisée).
    """

    def fetch_candidates(self, query_vector: Sequence[float], query_text: str,
                         compartments: list[str]) -> list[dict]:
        """Candidats de la phase de superposition, avec `similarity`, `age_seconds`,
        `rank_vec` et `rank_fts` (ces deux derniers pouvant être None)."""

    def fetch_ancres(self, compartment: str, limit: int) -> list[dict]:
        """Souvenirs actifs d'un compartiment injecté d'office (réflexes), les plus
        importants d'abord, avec `age_seconds`. Même périmètre que la recherche (projet)."""

    def fetch_relationships(self, memory_ids: list[str]) -> list[dict]:
        """Arêtes touchant l'un des ids (comme source OU comme cible)."""

    def fetch_by_ids(self, memory_ids: list[str]) -> list[dict]:
        """Mémoires actives par id, dans le périmètre du magasin."""

    def mark_accessed(self, memory_ids: list[str]) -> None:
        """Incrémente le compteur d'accès et rafraîchit `last_accessed_at`."""


def _candidat_depuis_ligne(ligne: dict, score: float, recency_factor: float) -> dict:
    return {
        "id": str(ligne["id"]),
        "type": ligne["type"],
        "compartment": ligne["compartment"],
        "content": ligne["content"],
        "confidence": float(ligne["confidence"] or 1.0),
        "importance": float(ligne["importance"] or 0.5),
        "last_accessed_at": ligne["last_accessed_at"],
        "created_at": ligne["created_at"],
        # Date du FAIT (≠ created_at) : préfixée au contenu par le collapse, sans quoi le
        # LLM ne peut répondre à aucune question « quand… ».
        "occurred_at": ligne["occurred_at"],
        "embedding": ligne["embedding"],
        "similarity": max(0.0, float(ligne.get("similarity") or 0.0)),
        "recency_factor": recency_factor,
        "score": score,
    }


def _scores_de_fusion(lignes: list[dict], config: RetrievalConfig) -> dict[str, float]:
    """Rangs par chemin -> score RRF (indépendant des échelles de score de chaque chemin)."""
    if not config.hybrid:
        return {}
    rang_vectoriel = [str(r["id"]) for r in sorted(
        (r for r in lignes if r.get("rank_vec") is not None), key=lambda r: r["rank_vec"])]
    rang_plein_texte = [str(r["id"]) for r in sorted(
        (r for r in lignes if r.get("rank_fts") is not None), key=lambda r: r["rank_fts"])]
    return reciprocal_rank_fusion(
        [rang_vectoriel, rang_plein_texte],
        k=config.rrf_k,
        weights=[config.weight_vector, config.weight_fts],
    )


def build_context_packet(
    store: MemoryStore,
    query_vector: Sequence[float],
    query_text: str,
    compartments: list[str],
    max_tokens: int,
    config: RetrievalConfig,
    trace_id: str,
    explain: bool = False,
    explain_level: str = "selected",
    record_access: bool = True,
) -> dict[str, Any]:
    """Assemble un paquet de contexte compact selon les 4 phases de Q-EM.

    1. Superposition  — candidats (vectoriel + plein texte fusionnés par RRF).
    2. Intrication    — propagation d'activation amortie le long de `entangled_with`.
    3. Interférence   — annulation des contradictions puis des redondances sémantiques.
    4. Mesure         — collapse glouton par densité d'utilité sous budget de tokens.

    `explain_level="all"` trace TOUS les candidats avec la décision qui les concerne
    (retenu, annulé par redondance/contradiction et par qui, hors budget) : sans cela, un
    souvenir attendu disparaissait sans laisser de trace — 10 cas sur 14 au banc du 26/09.
    `record_access=False` n'écrit rien : un diagnostic ne doit pas modifier la récence
    qu'il mesure (le premier banc a réécrit `last_accessed_at` en production).
    """
    lignes = store.fetch_candidates(query_vector, query_text, compartments)

    scores_rrf = _scores_de_fusion(lignes, config)
    meilleur_rrf = max(scores_rrf.values()) if scores_rrf else 0.0

    # La réponse à la question ne doit pas pouvoir sortir du paquet sous la pression du
    # budget (audit 11/08 : le premier du rang fusionné disparaissait, noyé entre treize
    # sections qui saturent `max_tokens`). Le meilleur candidat est déclaré prioritaire au
    # collapse : il ne reste exclu que si l'interférence l'a annulé (score 0) ou si le
    # budget est inférieur à son coût seul. En mode vectoriel pur (pas de fusion), la
    # priorité est le plus proche voisin.
    if scores_rrf:
        # `__getitem__` plutôt que `.get` : la méthode surchargée (avec/sans défaut) ne se
        # laisse pas passer comme fonction de clé par mypy, et ici les clés existent.
        priorites = [max(scores_rrf, key=scores_rrf.__getitem__)]
    elif lignes:
        priorites = [str(max(lignes, key=lambda r: float(r.get("similarity") or 0.0))["id"])]
    else:
        priorites = []

    candidates: dict[str, dict] = {}
    for ligne in lignes:
        mem_id = str(ligne["id"])
        similarite = max(0.0, float(ligne.get("similarity") or 0.0))
        compartiment = ligne.get("compartment")
        croyance = compartiment == COMPARTIMENT_CROYANCES
        # Oubli PROPRE au compartiment : rapide pour les observations, nul pour les
        # réflexes et les croyances (une croyance se révise ou se conteste, elle ne s'use
        # pas : la décroissance la ferait disparaître sans que rien ne l'ait infirmée).
        recency_factor = compute_recency_factor(
            ligne.get("age_seconds"), demi_vie(str(compartiment), config.recency_halflife_days))
        # Pertinence de départ. En hybride, elle vient du rang FUSIONNÉ (normalisé sur le
        # meilleur candidat) et non du seul cosinus : sans cela, un souvenir trouvé
        # uniquement par le plein texte entrerait avec un score faible et serait éliminé par
        # le collapse — le rappel gagné serait aussitôt reperdu.
        if config.hybrid and mem_id in scores_rrf and meilleur_rrf:
            pertinence = scores_rrf[mem_id] / meilleur_rrf
        else:
            pertinence = similarite
        score = initial_score(pertinence, recency_factor)
        # Poids du compartiment (l'amygdale renforce les souvenirs marquants : alertes).
        if compartiment in PAR_NOM:
            score *= PAR_NOM[compartiment].poids
        if croyance:
            # Une hypothèse peu assurée ne doit pas passer devant un fait aussi pertinent.
            score *= float(ligne.get("confidence") or 0.0)
        candidates[mem_id] = _candidat_depuis_ligne(ligne, score, recency_factor)

    if not candidates and not _ancres_actives(compartments, config):
        return {
            "context_packet": packet_vide(),
            "token_estimate": 0,
            "selected_memory_ids": [],
            "trace_id": trace_id,
            "retrieval_trace": [] if explain else None,
        }

    relationships = store.fetch_relationships(list(candidates.keys()))

    # Compléter le graphe : les voisins intriqués que la recherche n'a pas ramenés.
    manquants: list[str] = []
    for rel in relationships:
        src = str(rel["source_memory_id"])
        tgt = str(rel["target_memory_id"])
        if src in candidates and tgt not in candidates and tgt not in manquants:
            manquants.append(tgt)
        elif tgt in candidates and src not in candidates and src not in manquants:
            manquants.append(src)

    if manquants:
        # `store` est borné au (tenant, agent) : la traversée du graphe ne peut pas
        # franchir la frontière d'isolation, quelle que soit l'arête suivie (audit F1).
        for ligne in store.fetch_by_ids(manquants):
            mem_id = str(ligne["id"])
            # similarity = 0 : ces mémoires n'ont pas matché la requête, elles n'entrent que
            # par activation. Un score initial nul les rend dépendantes de la propagation.
            candidates[mem_id] = _candidat_depuis_ligne(ligne, 0.0, 0.0)
            candidates[mem_id]["similarity"] = 0.0

    # ── Phases 2 à 4, déléguées au cœur pur (qem.py) ──
    propagate_entanglement(candidates, relationships,
                           config.entangle_damping, config.entangle_max_hops,
                           aggregation=config.entangle_aggregation)
    # Réflexes (tronc cérébral) : injectés à CHAQUE contexte, quelle que soit la question.
    # Ajoutés APRÈS la propagation, délibérément : présents dans tous les paquets, ils
    # deviendraient sinon des « hubs » qui diffusent leur activation vers les mêmes voisins
    # à chaque requête (le travers corrigé par l'agrégation `max` au banc du 26/09).
    ancres = _injecter_ancres(store, candidates, compartments, config)

    annulations: dict[str, dict] = {}
    apply_contradictions(candidates, relationships, journal=annulations)
    # Les ancres ne sont PAS protégées de la redondance : deux réflexes qui disent la même
    # chose n'ont pas à occuper deux fois le paquet (le mieux classé gagne, règle du 26/09).
    filter_redundancy(candidates, config.redundancy_threshold,
                      proteges=priorites, journal=annulations)
    context_packet, selected_ids, token_count = collapse_by_utility(
        candidates, max_tokens, priorites,
        top_k=config.collapse_top_k, density_exponent=config.density_exponent,
        parts_max=config.parts(), ancres=ancres)

    if selected_ids and record_access:
        store.mark_accessed(selected_ids)

    logger.info("Q-EM: mesure achevée (%d mémoires retenues, %d/%d tokens, trace=%s).",
                len(selected_ids), token_count, max_tokens, trace_id)

    return {
        "context_packet": context_packet,
        "token_estimate": token_count,
        "selected_memory_ids": selected_ids,
        "trace_id": trace_id,
        "retrieval_trace": _trace(candidates, selected_ids, annulations, priorites,
                                  explain_level, ancres) if explain else None,
    }


def _ancres_actives(compartments: list[str], config: RetrievalConfig) -> bool:
    return COMPARTIMENT_REFLEXES in compartments and config.reflexes_injectes > 0


def _injecter_ancres(store: MemoryStore, candidates: dict[str, dict],
                     compartments: list[str], config: RetrievalConfig) -> list[str]:
    """Ajoute les réflexes à injecter d'office ; renvoie leurs ids, du plus important au
    moins important. Un réflexe déjà ramené par la recherche garde son score (il est
    seulement promu en tête) ; les autres entrent avec leur importance comme score."""
    if not _ancres_actives(compartments, config):
        return []
    ancres: list[str] = []
    for ligne in store.fetch_ancres(COMPARTIMENT_REFLEXES, config.reflexes_injectes):
        mem_id = str(ligne["id"])
        if mem_id not in candidates:
            importance = max(0.01, float(ligne.get("importance") or 0.5))
            candidates[mem_id] = _candidat_depuis_ligne(ligne, importance, 1.0)
            candidates[mem_id]["similarity"] = 0.0
        ancres.append(mem_id)
    return ancres


def _trace(candidates: dict[str, dict], selected_ids: list[str],
           annulations: dict[str, dict], priorites: list[str], niveau: str,
           ancres: Sequence[str] = ()) -> list[dict]:
    """Trace d'explication. `selected` : les retenus (contrat historique) ; `all` : tous
    les candidats, retenus d'abord, avec le motif d'exclusion des autres."""
    retenus = set(selected_ids)
    ids = list(selected_ids)
    if niveau == "all":
        ids += sorted((i for i in candidates if i not in retenus),
                      key=lambda i: -candidates[i]["score"])
    trace = []
    for memory_id in ids:
        c = candidates[memory_id]
        entree: dict[str, Any] = {
            "memory_id": memory_id,
            "similarity": c["similarity"],
            "recency_factor": c.get("recency_factor", 0.0),
            "score": c["score"],
            "compartment": c.get("compartment"),
        }
        if memory_id in retenus:
            # Libellé historique conservé (contrat) ; la priorité est un champ à part.
            entree["selection_reason"] = "selected_by_utility_under_token_budget"
            entree["priority"] = memory_id in priorites
            entree["anchor"] = memory_id in ancres
        elif memory_id in annulations:
            entree["selection_reason"] = f"cancelled_{annulations[memory_id]['motif']}"
            entree["cancelled_by"] = annulations[memory_id]["par"]
            if "cosinus" in annulations[memory_id]:
                entree["cosine"] = annulations[memory_id]["cosinus"]
        elif c["score"] <= 0.0:
            entree["selection_reason"] = "zero_score"
        else:
            entree["selection_reason"] = "over_token_budget"
        trace.append(entree)
    return trace


class InMemoryStore:
    """Magasin en mémoire : sert les tests d'orchestration sans PostgreSQL.

    Fourni dans le paquet (et non dans `tests/`) parce qu'il fait partie du contrat
    `MemoryStore` : une implémentation tierce peut s'y comparer.
    """

    def __init__(self, memoires: list[dict] | None = None,
                 relations: list[dict] | None = None) -> None:
        self.memoires = {str(m["id"]): dict(m) for m in (memoires or [])}
        self.relations = [dict(r) for r in (relations or [])]
        self.acces_marques: list[str] = []

    def fetch_candidates(self, query_vector, query_text, compartments):
        return [dict(m) for m in self.memoires.values() if m["compartment"] in compartments]

    def fetch_ancres(self, compartment, limit):
        lignes = [dict(m) for m in self.memoires.values()
                  if m["compartment"] == compartment and m.get("status", "active") == "active"]
        lignes.sort(key=lambda m: -float(m.get("importance") or 0.5))
        return lignes[:limit]

    def fetch_relationships(self, memory_ids):
        connus = set(memory_ids)
        return [dict(r) for r in self.relations
                if str(r["source_memory_id"]) in connus or str(r["target_memory_id"]) in connus]

    def fetch_by_ids(self, memory_ids):
        return [dict(self.memoires[i]) for i in memory_ids if i in self.memoires]

    def mark_accessed(self, memory_ids):
        self.acces_marques.extend(memory_ids)
