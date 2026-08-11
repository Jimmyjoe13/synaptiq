"""La construction du graphe, testée sans base : le curseur est un double.

`entangle()` n'est qu'une suite d'appels au curseur, donc un faux curseur suffit à verrouiller
ce qui est réellement subtil : le seuil STRICT, le fait qu'AUCUNE arête destructrice ne sorte
d'ici, et le fait que le seuil soit relu à chaque appel.

⚠️ Ces tests n'assertaient que la DIRECTION des arêtes, jamais l'issue de la phase
d'interférence qui les consomme. C'est ce trou-là qui a laissé passer la perte silencieuse
de bonnes pratiques ; le pendant se trouve dans `test_qem.py`
(`test_une_bonne_pratique_survit_a_l_erreur_ecrite_apres_elle`).

Le comportement de bout en bout (l'API tisse bien, le flag de collection est honoré) est
couvert par `tests/test_entanglement_direct_write.py`, qui exige Postgres.
"""
import pytest

from synaptiq_core import entangle, seuil_intrication


class _CurseurDouble:
    """Rejoue une liste de voisins et enregistre les arêtes insérées."""

    def __init__(self, voisins):
        self._voisins = voisins
        self.aretes = []          # (source, cible, relation, poids)
        self.derniers_params = None

    def execute(self, sql, params):
        if "INSERT INTO relationships" in sql:
            self.aretes.append(params)
        else:
            self.derniers_params = params

    def fetchall(self):
        return self._voisins


# (id, type, subtype, similarity)
def _voisin(identifiant, similarity, subtype="fact"):
    return (identifiant, "semantic", subtype, similarity)


def test_un_voisin_au_dessus_du_seuil_produit_une_arete():
    cur = _CurseurDouble([_voisin("cible", 0.9)])
    cree = entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8, threshold=0.7)
    assert cree == 1
    source, cible, relation, poids = cur.aretes[0]
    assert (source, cible, relation) == ("nouveau", "cible", "entangled_with")
    assert poids == 0.9


def test_le_seuil_est_strict():
    """`> seuil` et non `>=` : une similarité pile au seuil ne relie pas.

    Verrou volontaire — c'est le comportement historique du worker, et le modifier changerait
    la densité du graphe de toutes les instances existantes.
    """
    cur = _CurseurDouble([_voisin("cible", 0.7)])
    assert entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8, threshold=0.7) == 0
    assert cur.aretes == []


def test_les_voisins_sous_le_seuil_sont_ignores():
    cur = _CurseurDouble([_voisin("proche", 0.95), _voisin("loin", 0.3)])
    assert entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8, threshold=0.7) == 1
    assert cur.aretes[0][1] == "proche"


@pytest.mark.parametrize("sujet, voisin_subtype", [
    ("coding_best_practices", "code_error_resolution"),
    ("code_error_resolution", "coding_best_practices"),
])
def test_la_paire_pratique_erreur_ne_produit_plus_de_supersession(sujet, voisin_subtype):
    """RÉGRESSION — le cosinus ne prononce plus aucune supersession, dans AUCUN sens.

    Cette paire produisait un `supersedes_by`, donc une destruction à la lecture, décidée
    par la seule similarité : le motif « similaire ⇒ contradictoire » banni par F5, qui
    exige le verdict explicite d'un juge fail-closed. Toute supersession relève désormais
    de `governance`.
    """
    cur = _CurseurDouble([_voisin("voisin", 0.9, subtype=voisin_subtype)])
    assert entangle(cur, "t", "a", "nouveau", sujet, [0.1] * 8, threshold=0.7) == 1
    source, cible, relation, _ = cur.aretes[0]
    assert relation == "entangled_with"
    # Sens unique nouveau -> voisin : la lecture du graphe est bidirectionnelle.
    assert (source, cible) == ("nouveau", "voisin")


def test_aucune_arete_emise_n_est_destructrice():
    """Verrou de forme : quelle que soit la paire de sous-types, un seul type d'arête sort.

    Un test par paire particulière ne protège de rien — c'est une nouvelle règle de typage
    qui serait le retour du bug, pas celle-ci en particulier.
    """
    from synaptiq_core.entanglement import RELATION_INTRICATION

    sous_types = ["fact", "preference", "interaction", "rule",
                  "coding_best_practices", "code_error_resolution", "scratch", "libre"]
    for sujet in sous_types:
        cur = _CurseurDouble([_voisin(f"v_{cible}", 0.9, subtype=cible) for cible in sous_types])
        entangle(cur, "t", "a", "nouveau", sujet, [0.1] * 8, threshold=0.7)
        assert {a[2] for a in cur.aretes} == {RELATION_INTRICATION}
        assert all(a[0] == "nouveau" for a in cur.aretes)


def test_aucun_voisin_aucune_arete():
    cur = _CurseurDouble([])
    assert entangle(cur, "t", "a", "seul", "fact", [0.1] * 8, threshold=0.7) == 0


def test_une_similarite_nulle_ne_casse_pas():
    """`similarity` peut remonter NULL de PostgreSQL : ne pas exploser dessus."""
    cur = _CurseurDouble([(("cible"), "semantic", "fact", None)])
    assert entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8, threshold=0.7) == 0


def test_le_seuil_par_defaut_est_relu_a_chaque_appel(monkeypatch):
    """Convention du dépôt : les réglages ne sont pas figés à l'import.

    C'est ce qui permet une étude d'ablation ou un test sans redéploiement — et c'est ce que
    la constante figée du worker empêchait.
    """
    monkeypatch.setenv("QEM_ENTANGLE_THRESHOLD", "0.42")
    assert seuil_intrication() == pytest.approx(0.42)

    cur = _CurseurDouble([_voisin("cible", 0.5)])
    assert entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8) == 1

    monkeypatch.setenv("QEM_ENTANGLE_THRESHOLD", "0.9")
    cur = _CurseurDouble([_voisin("cible", 0.5)])
    assert entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8) == 0


def test_le_nombre_de_voisins_examines_est_borne():
    """La borne passe en paramètre lié, et vaut la constante du module."""
    from synaptiq_core.entanglement import VOISINS_EXAMINES
    cur = _CurseurDouble([])
    entangle(cur, "t", "a", "nouveau", "fact", [0.1] * 8, threshold=0.7)
    assert cur.derniers_params[-1] == VOISINS_EXAMINES


def test_le_worker_et_l_api_partagent_la_meme_fonction():
    """Régression : la fonction était définie dans le worker, donc l'API n'en avait aucune."""
    from apps.api.main import entangle as depuis_api
    from apps.worker.worker import entangle as depuis_worker

    assert depuis_api is entangle
    assert depuis_worker is entangle
