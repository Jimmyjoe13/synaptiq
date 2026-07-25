"""Tests unitaires de la fusion de classements (recherche hybride).

Purs : aucune infra. On vérifie la propriété qui justifie la RRF — un document présent
dans PLUSIEURS classements doit primer sur un document premier dans un seul — ainsi que
le déterminisme, indispensable à la reproductibilité d'un benchmark.
"""
import pytest

from synaptiq_core.retrieval import (
    DEFAULT_RRF_K,
    fuse_and_rank,
    reciprocal_rank_fusion,
)


def test_document_present_dans_les_deux_classements_prime():
    """Le cœur de l'hybride : sémantiquement proche ET littéralement présent > l'un ou l'autre.

    'b' n'est premier nulle part (2e et 2e) mais apparaît des deux côtés ; 'a' et 'c' sont
    premiers dans un seul classement. 'b' doit passer devant.
    """
    vectoriel = ["a", "b", "x"]
    plein_texte = ["c", "b", "y"]

    ordre = fuse_and_rank([vectoriel, plein_texte])

    assert ordre[0] == "b"


def test_un_seul_classement_preserve_l_ordre():
    """Chemin dégradé (plein texte sans résultat) : la fusion ne doit rien réordonner."""
    assert fuse_and_rank([["a", "b", "c"]]) == ["a", "b", "c"]


def test_classement_vide_ignore():
    """Une requête sans correspondance plein texte ne doit pas casser la fusion."""
    assert fuse_and_rank([["a", "b"], []]) == ["a", "b"]


def test_tous_classements_vides():
    assert fuse_and_rank([[], []]) == []


def test_scores_conformes_a_la_formule():
    scores = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_poids_favorisent_un_chemin():
    """Pondérer permet d'ajuster la confiance accordée à chaque chemin."""
    vectoriel, plein_texte = ["a"], ["b"]

    assert fuse_and_rank([vectoriel, plein_texte], weights=[10.0, 1.0])[0] == "a"
    assert fuse_and_rank([vectoriel, plein_texte], weights=[1.0, 10.0])[0] == "b"


def test_poids_incoherents_rejetes():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_doublon_dans_un_classement_ne_gonfle_pas_le_score():
    """Sans dédup, un identifiant répété cumulerait plusieurs contributions."""
    avec_doublon = reciprocal_rank_fusion([["a", "a", "b"]])
    sans_doublon = reciprocal_rank_fusion([["a", "b"]])
    assert avec_doublon["a"] == pytest.approx(sans_doublon["a"])


def test_egalite_departagee_de_facon_deterministe():
    """À score égal, l'ordre du premier classement tranche : résultat rejouable."""
    resultat = [fuse_and_rank([["a", "b"], ["b", "a"]]) for _ in range(5)]
    assert all(r == resultat[0] for r in resultat)
    assert resultat[0] == ["a", "b"]


def test_limite_tronque_apres_fusion():
    """La troncature doit intervenir APRÈS fusion, sinon on perd le gain de rappel."""
    ordre = fuse_and_rank([["a", "b", "c"], ["d", "b", "e"]], limit=2)
    assert len(ordre) == 2
    assert "b" in ordre          # b est le meilleur candidat fusionné


def test_k_faible_accentue_les_premiers_rangs():
    """k règle l'écart entre le 1er et les suivants : petit k = 1er très dominant."""
    ecart_k_petit = (lambda s: s["a"] - s["b"])(reciprocal_rank_fusion([["a", "b"]], k=1))
    ecart_k_grand = (lambda s: s["a"] - s["b"])(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert ecart_k_petit > ecart_k_grand


def test_valeur_par_defaut_de_k():
    assert DEFAULT_RRF_K == 60
    assert reciprocal_rank_fusion([["a"]])["a"] == pytest.approx(1 / 61)


def test_rappel_elargi_par_le_second_chemin():
    """Propriété visée : le plein texte fait ENTRER des documents absents du vectoriel.

    C'est le point du benchmark — 47 % des questions échouaient parce que la bonne
    mémoire n'était dans aucun résultat.
    """
    vectoriel = ["a", "b"]
    plein_texte = ["reponse_exacte"]

    ordre = fuse_and_rank([vectoriel, plein_texte])

    assert "reponse_exacte" in ordre
