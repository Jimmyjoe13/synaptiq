"""La construction du graphe, testée sans base : le curseur est un double.

`entangle()` n'est qu'une suite d'appels au curseur, donc un faux curseur suffit à verrouiller
ce qui est réellement subtil : le seuil STRICT, la règle `supersedes_by` et son sens
d'inversion, et le fait que le seuil soit relu à chaque appel.

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


def test_une_bonne_pratique_supersede_l_erreur_associee():
    """La seule règle de typage : une bonne pratique remplace l'erreur qu'elle résout."""
    cur = _CurseurDouble([_voisin("erreur", 0.9, subtype="code_error_resolution")])
    entangle(cur, "t", "a", "nouvelle_pratique", "coding_best_practices", [0.1] * 8,
             threshold=0.7)
    source, cible, relation, _ = cur.aretes[0]
    assert relation == "supersedes_by"
    assert (source, cible) == ("nouvelle_pratique", "erreur")


def test_une_erreur_resolue_est_supersedee_par_la_pratique_existante():
    """Sens INVERSÉ : c'est la bonne pratique déjà en base qui remplace la nouvelle erreur.

    Si l'arête partait du nouveau souvenir, la phase d'interférence annulerait la bonne
    pratique — l'inverse de l'intention.
    """
    cur = _CurseurDouble([_voisin("pratique", 0.9, subtype="coding_best_practices")])
    entangle(cur, "t", "a", "nouvelle_erreur", "code_error_resolution", [0.1] * 8,
             threshold=0.7)
    source, cible, relation, _ = cur.aretes[0]
    assert relation == "supersedes_by"
    assert (source, cible) == ("pratique", "nouvelle_erreur")


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
