"""synaptiq_core — logique partagée entre l'API et le worker (embeddings, gouvernance)."""

from synaptiq_core.compartments import (
    ALIAS,
    COMPARTIMENTS,
    NOMS,
    Compartiment,
    Rangement,
    catalogue,
    classer,
    depuis_ancien,
    normaliser_extraction,
    resoudre,
)
from synaptiq_core.context_builder import (
    InMemoryStore,
    MemoryStore,
    RetrievalConfig,
    build_context_packet,
)
from synaptiq_core.contradiction import (
    ContradictionJudge,
    LLMContradictionJudge,
    get_contradiction_judge,
    no_judge,
)
from synaptiq_core.embeddings import (
    Embedder,
    EmbeddingError,
    LMStudioEmbedder,
    MockEmbedder,
    OpenAICompatEmbedder,
    generate_mock_embedding,
    get_embedder,
    to_pgvector,
)
from synaptiq_core.entanglement import entangle, seuil_intrication
from synaptiq_core.governance import handle_contradictions, link_supersedes
from synaptiq_core.hashing import content_hash, normalize_for_hash
from synaptiq_core.qem import (
    apply_contradictions,
    collapse_by_utility,
    compute_recency_factor,
    estimate_tokens,
    filter_redundancy,
    format_entry,
    initial_score,
    propagate_entanglement,
    route_memory,
)
from synaptiq_core.retrieval import (
    DEFAULT_RRF_K,
    fuse_and_rank,
    reciprocal_rank_fusion,
)

__all__ = [
    "Embedder",
    "EmbeddingError",
    "LMStudioEmbedder",
    "MockEmbedder",
    "OpenAICompatEmbedder",
    "get_embedder",
    "generate_mock_embedding",
    "to_pgvector",
    "handle_contradictions",
    "link_supersedes",
    # Empreinte de contenu : même règle de déduplication sur les DEUX chemins d'écriture
    "content_hash",
    "normalize_for_hash",
    # Graphe d'intrication : même construction sur les DEUX chemins d'écriture
    "entangle",
    "seuil_intrication",
    # Verdict de contradiction (archivage sur décision explicite, jamais sur similarité)
    "ContradictionJudge",
    "LLMContradictionJudge",
    "get_contradiction_judge",
    "no_judge",
    # Orchestration des 4 phases (magasin injecté : testable sans SQL ni HTTP)
    "build_context_packet",
    "MemoryStore",
    "InMemoryStore",
    "RetrievalConfig",
    # Cœur algorithmique Q-EM (pur, testable sans infra)
    "compute_recency_factor",
    "initial_score",
    "propagate_entanglement",
    "apply_contradictions",
    "filter_redundancy",
    "collapse_by_utility",
    "route_memory",
    "estimate_tokens",
    "format_entry",
    # Compartiments (v0.4) : 9 régions fixes, propriété du moteur. L'agent juge seulement
    # où ranger ; le moteur classe par similarité quand il ne sait pas.
    "Compartiment",
    "COMPARTIMENTS",
    "NOMS",
    "ALIAS",
    "Rangement",
    "resoudre",
    "classer",
    "normaliser_extraction",
    "depuis_ancien",
    "catalogue",
    # Recherche hybride : fusion de classements (vectoriel + plein texte)
    "reciprocal_rank_fusion",
    "fuse_and_rank",
    "DEFAULT_RRF_K",
]
