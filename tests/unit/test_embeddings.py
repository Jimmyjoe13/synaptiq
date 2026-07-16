"""Tests unitaires de la couche d'embeddings (aucune infra requise)."""
import pytest

from synaptiq_core.embeddings import (
    EmbeddingError,
    MockEmbedder,
    OpenAICompatEmbedder,
    get_embedder,
    to_pgvector,
)


def test_mock_embedder_dim_et_norme():
    e = MockEmbedder(dim=384)
    v = e.embed_one("bonjour le monde")
    assert len(v) == 384
    # vecteur normalisé L2 -> norme ~ 1
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6


def test_mock_embedder_deterministe():
    e = MockEmbedder()
    assert e.embed_one("texte identique") == e.embed_one("texte identique")


def test_mock_embedder_batch():
    e = MockEmbedder()
    out = e.embed(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 384 for v in out)


def test_to_pgvector_format():
    assert to_pgvector([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"


def test_factory_mode_mock(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    get_embedder.cache_clear()
    e = get_embedder()
    get_embedder.cache_clear()
    assert isinstance(e, MockEmbedder)
    assert e.dim == 384


def test_openai_compat_dimension_incoherente(monkeypatch):
    """Une dimension renvoyée != EMBEDDING_DIM doit lever EmbeddingError."""
    import synaptiq_core.embeddings as emb

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}  # 2 dims au lieu de 384

    monkeypatch.setattr(emb.requests, "post", lambda *a, **k: FakeResp())
    e = OpenAICompatEmbedder(base_url="http://fake/v1", model="m", dim=384)
    with pytest.raises(EmbeddingError):
        e.embed(["hello"])


def test_openai_compat_respecte_ordre(monkeypatch):
    """Les vecteurs doivent être réordonnés selon le champ 'index'."""
    import synaptiq_core.embeddings as emb

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [
                {"index": 1, "embedding": [2.0]},
                {"index": 0, "embedding": [1.0]},
            ]}

    monkeypatch.setattr(emb.requests, "post", lambda *a, **k: FakeResp())
    e = OpenAICompatEmbedder(base_url="http://fake/v1", model="m", dim=1)
    out = e.embed(["premier", "second"])
    assert out == [[1.0], [2.0]]
