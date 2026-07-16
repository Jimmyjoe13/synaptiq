"""SynaptiQ — couche d'embeddings pluggable.

Interface `Embedder` unique + implémentations sélectionnées par variable d'env :
- `LMStudioEmbedder` / `OpenAICompatEmbedder` : endpoint OpenAI-compatible
  (LM Studio en local par défaut, OpenAI / OpenRouter / NVIDIA NIM plus tard).
- `MockEmbedder` : vecteur déterministe par hash SHA256 — AUCUNE sémantique,
  réservé aux tests unitaires.

La factory `get_embedder()` lit la configuration dans l'environnement.
"""
from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

import requests

logger = logging.getLogger("synaptiq-core.embeddings")


class EmbeddingError(RuntimeError):
    """Erreur d'embedding (endpoint injoignable, dimension incohérente, etc.)."""


def _l2_normalize(vec: List[float]) -> List[float]:
    """Normalise L2 pour que le produit scalaire == similarité cosinus."""
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec


def to_pgvector(vec: List[float]) -> str:
    """Sérialise un vecteur au format littéral pgvector : '[0.1,0.2,...]'."""
    return "[" + ",".join(map(str, vec)) + "]"


class Embedder(ABC):
    """Contrat commun à tous les fournisseurs d'embeddings."""

    dim: int

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Encode un lot de textes → liste de vecteurs (même ordre que l'entrée)."""

    def embed_one(self, text: str) -> List[float]:
        """Encode un seul texte → un vecteur."""
        return self.embed([text])[0]


class MockEmbedder(Embedder):
    """Embedding déterministe basé sur SHA256. Sans propriété sémantique : tests uniquement."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            sha = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(sha[i % len(sha)] / 127.5) - 1.0 for i in range(self.dim)]
            out.append(_l2_normalize(vec))
        return out


class OpenAICompatEmbedder(Embedder):
    """Client pour tout endpoint `/embeddings` compatible OpenAI.

    Couvre LM Studio (local), OpenAI, OpenRouter, NVIDIA NIM… tant que l'API
    respecte le schéma `{"model", "input"}` → `{"data": [{"embedding": [...]}]}`.
    """

    def __init__(self, base_url: str, model: str, dim: int, api_key: str = "", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise EmbeddingError(
                f"Échec de l'appel embeddings sur {self.base_url} (modèle '{self.model}') : {e}. "
                "Vérifier que LM Studio est lancé, le modèle chargé et le serveur local actif."
            ) from e

        data = resp.json().get("data", [])
        if not data:
            raise EmbeddingError(f"Réponse embeddings vide depuis {self.base_url}.")
        # Respecter l'ordre d'entrée via le champ 'index' quand il est présent
        data_sorted = sorted(data, key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in data_sorted]

        got = len(vectors[0])
        if got != self.dim:
            raise EmbeddingError(
                f"Dimension d'embedding inattendue : reçu {got}, attendu {self.dim}. "
                "Aligner EMBEDDING_DIM et la colonne VECTOR(n) de la base."
            )
        return vectors


class LMStudioEmbedder(OpenAICompatEmbedder):
    """Alias explicite pour le fournisseur par défaut (LM Studio)."""


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Instancie l'embedder configuré (mis en cache pour tout le process).

    Variables d'environnement :
      EMBEDDING_PROVIDER : lmstudio (défaut) | openai | openrouter | mock
      EMBEDDING_BASE_URL : URL de l'endpoint OpenAI-compatible
      EMBEDDING_MODEL    : identifiant du modèle
      EMBEDDING_DIM      : dimension attendue (défaut 384)
      EMBEDDING_API_KEY  : clé API (facultative pour LM Studio)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "lmstudio").lower()
    dim = int(os.getenv("EMBEDDING_DIM", "384"))

    if provider == "mock":
        logger.warning("EMBEDDING_PROVIDER=mock : embeddings NON sémantiques (tests uniquement).")
        return MockEmbedder(dim=dim)

    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("EMBEDDING_MODEL", "all-minilm-l6-v2")
    api_key = os.getenv("EMBEDDING_API_KEY", "")

    if provider in ("lmstudio", "openai", "openrouter", "openai-compat"):
        logger.info("Embedder=%s base_url=%s model=%s dim=%d", provider, base_url, model, dim)
        return OpenAICompatEmbedder(base_url=base_url, model=model, dim=dim, api_key=api_key)

    raise EmbeddingError(f"EMBEDDING_PROVIDER inconnu : '{provider}'")


def generate_mock_embedding(text: str, dim: int = 384) -> List[float]:
    """Compat rétro : conservé pour l'ancien code et les tests. Utiliser get_embedder() ailleurs."""
    return MockEmbedder(dim=dim).embed_one(text)
