"""Garde-fou des croyances (famille `reflective`, lot C du 26/09)."""
import pytest

from synaptiq_core.belief_guard import (
    CONFIANCE_MAX,
    CroyanceRefusee,
    categorie_sensible,
    valider_croyance,
)
from synaptiq_core.taxonomy import normalize_extraction


@pytest.mark.parametrize("texte,categorie", [
    ("Jimmy semble souffrir de depression", "sante"),
    ("Il est probablement de gauche", "politique"),
    ("Regis est sans doute catholique", "religion"),
    ("Je pense qu'il est homosexuel", "orientation"),
    ("Son origine ethnique explique son accent", "origine"),
    ("Herve est syndique", "syndicat"),
    ("He is clearly depressed lately", "sante"),
])
def test_categories_sensibles_detectees(texte, categorie):
    assert categorie_sensible(texte) == categorie


@pytest.mark.parametrize("texte", [
    "Jimmy prefere des reponses courtes et directes",
    "Regis fournit les CSV en fin de semaine",
    "Les humains relisent rarement une note plus longue qu'un ecran",
    "Il assainit toujours ses donnees avant import",   # « assainit » ne doit pas matcher « santé »
])
def test_comportements_observables_acceptes(texte):
    assert categorie_sensible(texte) is None


def test_une_croyance_sensible_est_refusee_avec_une_consigne():
    with pytest.raises(CroyanceRefusee, match="Reformuler"):
        valider_croyance("Jimmy est sans doute anxieux, anxiete chronique", 0.4)


def test_confiance_forte_sans_indice_refusee():
    with pytest.raises(CroyanceRefusee, match="evidence_ids"):
        valider_croyance("Jimmy prefere decider seul", 0.8)


def test_confiance_forte_avec_indice_acceptee_et_plafonnee():
    v = valider_croyance("Jimmy prefere decider seul", 1.0, ["mem-1"])
    assert v.confidence == CONFIANCE_MAX
    assert v.evidence == ("mem-1",)


def test_confiance_faible_sans_indice_acceptee():
    assert valider_croyance("Jimmy aime peut-etre les tableaux", 0.4).confidence == 0.4


def test_les_tiers_sont_autorises():
    """Décision de Jimmy du 26/09 : nommer un client ou un collègue est permis."""
    assert valider_croyance("Regis repond plus vite le matin", 0.4)


def test_un_extracteur_ne_produit_jamais_de_croyance():
    """Une croyance est formulée délibérément, jamais extraite d'un tour de dialogue."""
    assert normalize_extraction("reflective", "user_model") == ("semantic", "fact")
