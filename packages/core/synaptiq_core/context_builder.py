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

# Les 7 clés du context_packet, répétées ici pour la réponse vide (contrat stable).
PACKET_VIDE: dict[str, list[str]] = {
    "facts": [], "preferences": [], "episodes": [],
    "rules": [], "best_practices": [], "errors": [], "examples": []}


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
    redundancy_threshold: float = 0.75
    recency_halflife_days: float = 90.0


class MemoryStore(Protocol):
    """Accès aux mémoires, borné à un (tenant, agent) fixés à la construction.

    Les lignes retournées sont des mappings portant au moins : `id`, `type`, `subtype`,
    `content`, `confidence`, `importance`, `last_accessed_at`, `created_at`, `occurred_at`,
    `embedding` (séquence de flottants, déjà désérialisée).
    """

    def fetch_candidates(self, query_vector: Sequence[float], query_text: str,
                         memory_types: list[str]) -> list[dict]:
        """Candidats de la phase de superposition, avec `similarity`, `age_seconds`,
        `rank_vec` et `rank_fts` (ces deux derniers pouvant être None)."""

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
        "subtype": ligne["subtype"],
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
    memory_types: list[str],
    max_tokens: int,
    config: RetrievalConfig,
    trace_id: str,
    explain: bool = False,
) -> dict[str, Any]:
    """Assemble un paquet de contexte compact selon les 4 phases de Q-EM.

    1. Superposition  — candidats (vectoriel + plein texte fusionnés par RRF).
    2. Intrication    — propagation d'activation amortie le long de `entangled_with`.
    3. Interférence   — annulation des contradictions puis des redondances sémantiques.
    4. Mesure         — collapse glouton par densité d'utilité sous budget de tokens.
    """
    lignes = store.fetch_candidates(query_vector, query_text, memory_types)

    scores_rrf = _scores_de_fusion(lignes, config)
    meilleur_rrf = max(scores_rrf.values()) if scores_rrf else 0.0

    candidates: dict[str, dict] = {}
    for ligne in lignes:
        mem_id = str(ligne["id"])
        similarite = max(0.0, float(ligne.get("similarity") or 0.0))
        recency_factor = compute_recency_factor(ligne.get("age_seconds"),
                                                config.recency_halflife_days)
        # Pertinence de départ. En hybride, elle vient du rang FUSIONNÉ (normalisé sur le
        # meilleur candidat) et non du seul cosinus : sans cela, un souvenir trouvé
        # uniquement par le plein texte entrerait avec un score faible et serait éliminé par
        # le collapse — le rappel gagné serait aussitôt reperdu.
        if config.hybrid and mem_id in scores_rrf and meilleur_rrf:
            pertinence = scores_rrf[mem_id] / meilleur_rrf
        else:
            pertinence = similarite
        candidates[mem_id] = _candidat_depuis_ligne(
            ligne, initial_score(pertinence, recency_factor), recency_factor)

    if not candidates:
        return {
            "context_packet": dict(PACKET_VIDE),
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
                           config.entangle_damping, config.entangle_max_hops)
    apply_contradictions(candidates, relationships)
    filter_redundancy(candidates, config.redundancy_threshold)
    context_packet, selected_ids, token_count = collapse_by_utility(candidates, max_tokens)

    if selected_ids:
        store.mark_accessed(selected_ids)

    logger.info("Q-EM: mesure achevée (%d mémoires retenues, %d/%d tokens, trace=%s).",
                len(selected_ids), token_count, max_tokens, trace_id)

    return {
        "context_packet": context_packet,
        "token_estimate": token_count,
        "selected_memory_ids": selected_ids,
        "trace_id": trace_id,
        "retrieval_trace": [
            {
                "memory_id": memory_id,
                "similarity": candidates[memory_id]["similarity"],
                "recency_factor": candidates[memory_id].get("recency_factor", 0.0),
                "score": candidates[memory_id]["score"],
                "selection_reason": "selected_by_utility_under_token_budget",
            }
            for memory_id in selected_ids
        ] if explain else None,
    }


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

    def fetch_candidates(self, query_vector, query_text, memory_types):
        return [dict(m) for m in self.memoires.values() if m["type"] in memory_types]

    def fetch_relationships(self, memory_ids):
        connus = set(memory_ids)
        return [dict(r) for r in self.relations
                if str(r["source_memory_id"]) in connus or str(r["target_memory_id"]) in connus]

    def fetch_by_ids(self, memory_ids):
        return [dict(self.memoires[i]) for i in memory_ids if i in self.memoires]

    def mark_accessed(self, memory_ids):
        self.acces_marques.extend(memory_ids)
