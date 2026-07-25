"""synaptiq_core — logique partagée entre l'API et le worker (embeddings, gouvernance)."""

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
from synaptiq_core.governance import handle_contradictions
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
    # Recherche hybride : fusion de classements (vectoriel + plein texte)
    "reciprocal_rank_fusion",
    "fuse_and_rank",
    "DEFAULT_RRF_K",
]
