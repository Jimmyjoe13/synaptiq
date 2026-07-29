"""Tests unitaires des statistiques de benchmark (audit §7.1).

L'enjeu : rendre impossible la publication d'un écart sans son incertitude. Le test
`test_le_gain_annonce_du_readme_n_est_pas_significatif` reproduit exactement le chiffre
publié (51,32 % contre 48,03 % sur 152 questions) et vérifie que l'outil le qualifie
correctement de non significatif.
"""
import pytest

from synaptiq_core.stats import (
    Difference,
    Proportion,
    required_sample_size,
    wilson_interval,
)

# ─── Intervalle de Wilson ────────────────────────────────────────────────────

def test_wilson_encadre_la_proportion():
    bas, haut = wilson_interval(50, 100)
    assert bas < 0.5 < haut
    assert 0.39 < bas < 0.41
    assert 0.59 < haut < 0.61


def test_wilson_reste_dans_zero_un():
    """L'approximation normale sortirait de [0, 1] sur ces cas ; Wilson non."""
    for succes, total in [(0, 10), (10, 10), (1, 3), (0, 1)]:
        bas, haut = wilson_interval(succes, total)
        assert 0.0 <= bas <= haut <= 1.0


def test_wilson_total_nul():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_l_intervalle_se_resserre_en_racine_de_l_echantillon():
    """La marge décroît en 1/√n : quadrupler l'échantillon la divise par deux.

    C'est la raison pour laquelle passer de 152 à ~1 990 questions (les 10 conversations
    LOCOMO) fait tomber la marge de ~±8 à ~±2 points, et pas à ±0,6.
    """
    petit = Proportion(50, 100).margin
    quadruple = Proportion(200, 400).margin
    decuple = Proportion(500, 1000).margin
    assert decuple < quadruple < petit
    # x4 d'échantillon -> marge / 2
    assert 0.45 < quadruple / petit < 0.55
    # x10 d'échantillon -> marge / sqrt(10) ~ 3,16
    assert 0.28 < decuple / petit < 0.36


# ─── Proportion ──────────────────────────────────────────────────────────────

def test_proportion_valeur_et_serialisation():
    p = Proportion(78, 152)
    assert p.value == pytest.approx(0.5132, abs=1e-4)
    charge = p.as_dict()
    assert charge["total"] == 152
    assert charge["ci95_low"] < charge["value"] < charge["ci95_high"]
    # C'est LE chiffre qui manquait au README : ~±8 points sur 152 questions.
    assert 7.0 < charge["ci95_margin_points"] < 9.0


def test_proportion_vide():
    p = Proportion(0, 0)
    assert p.value == 0.0
    assert p.margin == 0.0


def test_proportion_lisible():
    assert "n=152" in str(Proportion(78, 152))
    assert "%" in str(Proportion(78, 152))


# ─── Différence entre deux bras ──────────────────────────────────────────────

def test_le_gain_annonce_du_readme_n_est_pas_significatif():
    """51,32 % vs 48,03 % sur 152 questions : l'intervalle contient zéro.

    Ce n'est pas un échec du moteur, c'est une insuffisance d'échantillon — et c'est
    exactement ce que l'outil doit dire au lieu de laisser lire « +3,29 pts ».
    """
    qem = Proportion(78, 152)      # 51,32 %
    vecteur = Proportion(73, 152)  # 48,03 %
    difference = Difference(a=qem, b=vecteur)
    assert difference.delta == pytest.approx(0.0329, abs=1e-3)
    bas, haut = difference.interval
    assert bas < 0 < haut
    assert difference.significant is False
    assert "NON significatif" in difference.as_dict()["verdict"]


def test_un_ecart_franc_sur_grand_echantillon_est_significatif():
    difference = Difference(a=Proportion(1200, 2000), b=Proportion(1000, 2000))
    assert difference.significant is True
    assert difference.as_dict()["ci95_low_points"] > 0
    assert "significatif à 95" in difference.as_dict()["verdict"]


def test_difference_symetrique():
    a, b = Proportion(1200, 2000), Proportion(1000, 2000)
    assert Difference(a, b).delta == pytest.approx(-Difference(b, a).delta)


def test_difference_sur_echantillon_vide():
    difference = Difference(a=Proportion(0, 0), b=Proportion(0, 0))
    assert difference.interval == (0.0, 0.0)
    assert difference.significant is False


# ─── Dimensionnement d'un run ────────────────────────────────────────────────

def test_taille_requise_pour_deux_points_de_marge():
    """~2 400 questions pour ±2 points : justifie le passage aux 10 conversations LOCOMO."""
    n = required_sample_size(2.0)
    assert 2300 < n < 2500


def test_taille_requise_decroit_avec_la_marge_toleree():
    assert required_sample_size(8.0) < required_sample_size(2.0)


def test_marge_invalide_refusee():
    with pytest.raises(ValueError):
        required_sample_size(0)
