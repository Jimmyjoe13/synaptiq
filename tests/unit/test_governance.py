"""Tests unitaires de la gouvernance : archivage sur VERDICT, jamais sur similarité.

Ces tests verrouillent le correctif F5 du 29/07. Le comportement précédent archivait toute
préférence dont le cosinus dépassait le seuil : deux préférences compatibles mais proches
(« mails courts » / « mails en français », ~0,85) se supprimaient l'une l'autre en silence.
Le test `test_proche_mais_pas_contradictoire_est_conservee` est exactement ce cas — il
échouait avant le correctif.
"""
from synaptiq_core.governance import handle_contradictions, link_supersedes


class FakeCursor:
    """Curseur factice : enregistre les requêtes et rejoue des lignes de pré-filtre."""

    def __init__(self, proches=None, rowcount=1):
        self.calls = []
        self.rowcount = rowcount
        # Lignes que le SELECT de pré-filtre sémantique doit retourner.
        self._proches = proches if proches is not None else []

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchall(self):
        return self._proches

    @property
    def queries(self):
        return [q for q, _ in self.calls]


def _oui(existant, nouveau):
    """Juge qui constate toujours une contradiction."""
    return True


def _non(existant, nouveau):
    """Juge qui ne constate jamais de contradiction (comportement sans LLM)."""
    return False


# ─── Cas où la gouvernance ne s'applique pas ─────────────────────────────────

def test_ignore_les_non_preferences():
    cur = FakeCursor()
    assert handle_contradictions(cur, "t", "a",
                                 {"type": "semantic", "subtype": "fact", "content": "x"},
                                 [0.1] * 384, judge=_oui) == []
    assert cur.calls == []  # aucune requête, même de lecture


def test_sans_embedding_aucun_archivage():
    """Sans embedding, le pré-filtre est impossible : on ne touche à rien."""
    cur = FakeCursor()
    assert handle_contradictions(cur, "t", "a",
                                 {"type": "semantic", "subtype": "preference", "content": "x"},
                                 None, judge=_oui) == []
    assert cur.calls == []


def test_aucune_preference_proche_aucun_appel_au_juge():
    """Pré-filtre vide -> pas de verdict à demander, pas d'écriture."""
    appels = []

    def juge_espion(a, b):
        appels.append((a, b))
        return True

    cur = FakeCursor(proches=[])
    assert handle_contradictions(cur, "t", "a",
                                 {"type": "semantic", "subtype": "preference", "content": "x"},
                                 [0.1] * 384, judge=juge_espion) == []
    assert appels == []
    assert len(cur.calls) == 1                      # le SELECT de pré-filtre, rien de plus
    assert "UPDATE" not in cur.queries[0]


# ─── Le cœur du correctif F5 ─────────────────────────────────────────────────

def test_proche_mais_pas_contradictoire_est_conservee():
    """RÉGRESSION F5 : la proximité sémantique seule ne doit RIEN archiver.

    Échouait avant le 29/07 : « mails en français » archivait « mails courts ».
    """
    cur = FakeCursor(proches=[("id-1", "Jimmy préfère les mails courts")])
    archives = handle_contradictions(
        cur, "t", "a",
        {"type": "semantic", "subtype": "preference",
         "content": "Jimmy préfère les mails en français"},
        [0.1] * 384, judge=_non,
    )
    assert archives == []
    # Une seule requête, en LECTURE : aucun UPDATE n'a été émis.
    assert len(cur.calls) == 1
    assert "SELECT" in cur.queries[0]
    assert not any("UPDATE" in q for q in cur.queries)


def test_contradiction_confirmee_archive():
    cur = FakeCursor(proches=[("id-1", "Jimmy préfère MySQL")])
    archives = handle_contradictions(
        cur, "t", "a",
        {"type": "semantic", "subtype": "preference", "content": "Jimmy préfère PostgreSQL"},
        [0.1] * 384, judge=_oui,
    )
    assert archives == ["id-1"]
    assert any("UPDATE memories" in q and "archived" in q for q in cur.queries)


def test_verdict_par_preference_seules_les_contredites_partent():
    """Le juge est interrogé préférence par préférence : un verdict ne contamine pas l'autre."""
    cur = FakeCursor(proches=[("garde-moi", "préférence compatible"),
                              ("archive-moi", "préférence contredite")])
    archives = handle_contradictions(
        cur, "t", "a",
        {"type": "semantic", "subtype": "preference", "content": "nouvelle"},
        [0.1] * 384,
        judge=lambda existant, nouveau: existant == "préférence contredite",
    )
    assert archives == ["archive-moi"]


def test_prefiltre_est_scope_tenant_agent_et_seuil():
    cur = FakeCursor(proches=[])
    handle_contradictions(cur, "tenant1", "agent1",
                          {"type": "semantic", "subtype": "preference", "content": "x"},
                          [0.1] * 384, threshold=0.9, judge=_non)
    query, params = cur.calls[0]
    assert "embedding <=>" in query       # scoping sémantique par pgvector
    assert params[0] == "tenant1"
    assert params[1] == "agent1"
    assert params[3] == 0.9               # seuil de pré-filtrage


# ─── Traçabilité de la supersession ──────────────────────────────────────────

def test_link_supersedes_cree_une_arete_par_ancien():
    cur = FakeCursor()
    link_supersedes(cur, "nouveau", ["vieux-1", "vieux-2"])
    assert len(cur.calls) == 2
    for query, params in cur.calls:
        assert "supersedes_by" in query
        assert params[0] == "nouveau"
    assert [p[1] for _, p in cur.calls] == ["vieux-1", "vieux-2"]


def test_link_supersedes_sans_ancien_ne_fait_rien():
    cur = FakeCursor()
    link_supersedes(cur, "nouveau", [])
    assert cur.calls == []
