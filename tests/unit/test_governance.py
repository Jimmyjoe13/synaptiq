"""Tests unitaires de la gouvernance (contradictions scopées sémantiquement)."""
from synaptiq_core.governance import handle_contradictions


class FakeCursor:
    """Curseur factice qui enregistre les requêtes SQL exécutées."""

    def __init__(self, rowcount=1):
        self.calls = []
        self.rowcount = rowcount

    def execute(self, query, params=None):
        self.calls.append((query, params))


def test_ignore_les_non_preferences():
    cur = FakeCursor()
    n = handle_contradictions(cur, "t", "a",
                              {"type": "semantic", "subtype": "fact", "content": "x"},
                              [0.1] * 384)
    assert n == 0
    assert cur.calls == []  # aucune requête d'archivage


def test_sans_embedding_aucun_archivage():
    """Sécurité : sans embedding, on n'archive rien (pas d'archivage en masse)."""
    cur = FakeCursor()
    n = handle_contradictions(cur, "t", "a",
                              {"type": "semantic", "subtype": "preference", "content": "x"},
                              None)
    assert n == 0
    assert cur.calls == []


def test_preference_declenche_requete_scopee():
    cur = FakeCursor(rowcount=2)
    n = handle_contradictions(cur, "tenant1", "agent1",
                              {"type": "semantic", "subtype": "preference", "content": "x"},
                              [0.1] * 384, threshold=0.8)
    assert n == 2
    assert len(cur.calls) == 1
    query, params = cur.calls[0]
    # Le scoping sémantique passe par l'opérateur de distance pgvector
    assert "embedding <=>" in query
    assert params[0] == "tenant1"
    assert params[1] == "agent1"
    assert params[3] == 0.8  # seuil de similarité
