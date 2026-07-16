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
]
