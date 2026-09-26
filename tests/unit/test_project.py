"""Dimension projet (lot B, 26/09) : normalisation du nom et fragment de filtrage."""
import pytest

from synaptiq_core.project import activer_scan_iteratif, filtre_projet, normaliser_projet


@pytest.mark.parametrize("brut,attendu", [
    ("Emile", "emile"),
    ("  elu-scraper ", "elu-scraper"),
    ("dash.joe", "dash.joe"),
    ("jobxpress_v2", "jobxpress_v2"),
    (None, None),
    ("", None),
    ("   ", None),
])
def test_normalisation(brut, attendu):
    """Une seule orthographe par projet : sinon `Emile` et `emile` seraient deux partitions."""
    assert normaliser_projet(brut) == attendu


@pytest.mark.parametrize("invalide", ["-emile", "mon projet", "émile", "a" * 65, "x;drop"])
def test_nom_invalide_refuse(invalide):
    """Refus explicite plutôt que correction silencieuse (partition fantôme)."""
    with pytest.raises(ValueError):
        normaliser_projet(invalide)


def test_sans_projet_aucun_filtre():
    assert filtre_projet(None) == ("", [])


def test_projet_avec_globaux_par_defaut():
    frag, params = filtre_projet("emile")
    assert frag == "AND (project = %s OR project IS NULL)"
    assert params == ["emile"]


def test_projet_strict():
    frag, params = filtre_projet("emile", include_global=False)
    assert frag == "AND project = %s"
    assert params == ["emile"]


def test_alias_prefixe_les_colonnes():
    frag, _ = filtre_projet("emile", alias="m.")
    assert frag == "AND (m.project = %s OR m.project IS NULL)"


class _Curseur:
    def __init__(self, echouer=False):
        self.executees, self._echouer = [], echouer

    def execute(self, sql, params=None):
        self.executees.append(sql)
        if self._echouer and sql.startswith("SET LOCAL"):
            raise RuntimeError("unrecognized configuration parameter")


def test_scan_iteratif_active_sous_savepoint():
    cur = _Curseur()
    activer_scan_iteratif(cur, "strict_order")
    assert cur.executees == ["SAVEPOINT synaptiq_hnsw",
                             "SET LOCAL hnsw.iterative_scan = strict_order",
                             "RELEASE SAVEPOINT synaptiq_hnsw"]


def test_scan_iteratif_pgvector_ancien_n_avorte_pas():
    """pgvector < 0.8 : le paramètre n'existe pas, la transaction doit rester saine."""
    cur = _Curseur(echouer=True)
    activer_scan_iteratif(cur, "strict_order")
    assert "ROLLBACK TO SAVEPOINT synaptiq_hnsw" in cur.executees


def test_scan_iteratif_desactivable():
    cur = _Curseur()
    activer_scan_iteratif(cur, "off")
    assert cur.executees == []
