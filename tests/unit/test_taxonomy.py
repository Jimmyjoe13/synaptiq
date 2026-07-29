"""Tests unitaires de la taxonomie partagée (incident de production du 29/07).

Contexte : la taxonomie vivait dans le worker, donc n'était appliquée qu'au chemin
d'extraction LLM. `POST /v1/memories` acceptait n'importe quel sous-type — constaté en
prod, des mémoires portaient `seo_audit_july_2026`, `nana_intelligence_lead_webhook`.

Ce qui est verrouillé ici : un sous-type LIBRE reste accepté (des intégrations réelles en
dépendent), un sous-type canonique rattaché au MAUVAIS type est refusé.
"""
import pytest

from synaptiq_core.qem import route_memory
from synaptiq_core.taxonomy import (
    DEFAULT_SUBTYPE,
    VALID_SUBTYPES,
    SubtypeMismatch,
    is_canonical,
    normalize_extraction,
    owner_type_of,
    validate_subtype,
)

# ─── Appartenance ────────────────────────────────────────────────────────────

def test_owner_type_of_sous_types_canoniques():
    assert owner_type_of("preference") == "semantic"
    assert owner_type_of("coding_best_practices") == "procedural"
    assert owner_type_of("interaction") == "episodic"
    assert owner_type_of("scratch") == "working"


def test_owner_type_of_sous_type_libre():
    assert owner_type_of("nana_intelligence_lead_webhook") is None


def test_is_canonical():
    assert is_canonical("semantic", "preference") is True
    assert is_canonical("semantic", "seo_audit_july_2026") is False
    assert is_canonical("semantic", None) is False


# ─── Validation d'une écriture directe ───────────────────────────────────────

def test_sous_type_canonique_accepte():
    assert validate_subtype("semantic", "preference") == "preference"


def test_sous_type_libre_accepte():
    """Un libellé métier est légitime : le routage retombe sur la collection du type.

    Refuser ces valeurs casserait des intégrations existantes sans rien protéger.
    """
    for libre in ("nana_intelligence_lead_webhook", "seo_audit_july_2026", "project_info"):
        assert validate_subtype("semantic", libre) == libre


def test_sous_type_absent_accepte():
    assert validate_subtype("semantic", None) is None
    assert validate_subtype("semantic", "") == ""


def test_sous_type_du_mauvais_type_refuse():
    """LE cas que la validation doit attraper : l'intention de l'appelant est trahie.

    `semantic` + `coding_best_practices` partirait dans `facts`, alors que son auteur
    visait manifestement `best_practices`.
    """
    with pytest.raises(SubtypeMismatch) as exc:
        validate_subtype("semantic", "coding_best_practices")
    assert "procedural" in str(exc.value)      # le message dit quel type utiliser
    assert "coding_best_practices" in str(exc.value)


@pytest.mark.parametrize(("mauvais_type", "sous_type"), [
    ("semantic", "code_error_resolution"),
    ("semantic", "interaction"),
    ("procedural", "preference"),
    ("episodic", "rule"),
    ("working", "fact"),
])
def test_toutes_les_permutations_croisees_sont_refusees(mauvais_type, sous_type):
    with pytest.raises(SubtypeMismatch):
        validate_subtype(mauvais_type, sous_type)


def test_chaque_sous_type_canonique_passe_avec_son_propre_type():
    for mtype, sous_types in VALID_SUBTYPES.items():
        for st in sous_types:
            assert validate_subtype(mtype, st) == st


# ─── Normalisation de la sortie LLM (ne lève jamais) ─────────────────────────

def test_normalisation_conserve_un_couple_valide():
    assert normalize_extraction("procedural", "rule") == ("procedural", "rule")


def test_normalisation_corrige_un_sous_type_hors_type():
    """Côté LLM on corrige au lieu de refuser : un événement ne doit jamais être perdu."""
    assert normalize_extraction("semantic", "coding_best_practices") == ("semantic", "fact")


def test_normalisation_type_hallucine_retombe_sur_semantic():
    assert normalize_extraction("type_invente", "peu_importe") == ("semantic", "fact")
    assert normalize_extraction(None, None) == ("semantic", "fact")


def test_les_defauts_couvrent_tous_les_types():
    assert set(DEFAULT_SUBTYPE) == set(VALID_SUBTYPES)
    for mtype, defaut in DEFAULT_SUBTYPE.items():
        assert defaut in VALID_SUBTYPES[mtype]


# ─── Cohérence avec le routage effectif ──────────────────────────────────────

def test_tout_sous_type_canonique_est_route_quelque_part():
    for mtype, sous_types in VALID_SUBTYPES.items():
        for st in sous_types:
            assert route_memory(mtype, st) is not None


def test_un_sous_type_libre_retombe_sur_la_collection_du_type():
    """Le comportement observé en prod, désormais explicite et testé."""
    assert route_memory("semantic", "nana_intelligence_lead_webhook") == "facts"
    assert route_memory("episodic", "seo_audit_july_2026") == "episodes"
    assert route_memory("procedural", "nextjs_static_export_rules") == "rules"
