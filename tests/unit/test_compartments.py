"""Compartiments (v0.4) : 9 régions fixes, propriété du moteur.

Ce qui est verrouillé ici :
  - la forme fixe (9 compartiments, ordre du paquet, une famille déduite par compartiment) ;
  - la résolution à l'écriture : nom valide > ancien nom (alias) > classement par similarité,
    et JAMAIS un rangement vide ;
  - les croyances ne sont ni classables ni écrites par la voie ordinaire ;
  - la copie FIGÉE de la table d'alias dans la migration correspond au cœur au jour de sa
    création (une divergence future est voulue, une divergence initiale est un bug).
"""
import importlib.util
from pathlib import Path

import pytest

from synaptiq_core import compartments as cp

ROOT = Path(__file__).resolve().parents[2]


# ─── Forme fixe ───────────────────────────────────────────────────────────────

def test_neuf_compartiments_dans_l_ordre_du_paquet():
    assert cp.NOMS == ("reflexes", "decisions", "savoir_faire", "alertes", "environnement",
                       "savoirs", "episodes", "observations", "croyances")
    assert list(cp.packet_vide()) == list(cp.NOMS)


def test_chaque_compartiment_a_une_region_et_une_famille_valide():
    for c in cp.COMPARTIMENTS:
        assert c.region and c.description
        assert c.famille in cp.FAMILIES
        assert cp.famille_de(c.nom) == c.famille


def test_famille_inconnue_refusee_a_la_definition():
    with pytest.raises(ValueError):
        cp.Compartiment("x", "r", "t", "working", "d")


def test_part_max_hors_bornes_refusee():
    with pytest.raises(ValueError):
        cp.Compartiment("x", "r", "t", "semantic", "d", part_max=0.0)


def test_compartiment_inconnu_leve_un_message_explicite():
    with pytest.raises(KeyError, match="Valides"):
        cp.compartiment("clients_paca")


def test_famille_de_un_inconnu_retombe_sur_semantic():
    assert cp.famille_de("inconnu") == "semantic"


def test_seules_les_croyances_sont_fermees_a_l_ecriture_directe():
    fermes = [c.nom for c in cp.COMPARTIMENTS if not c.ecriture_directe]
    assert fermes == ["croyances"]
    assert "croyances" not in cp.CLASSABLES


def test_seuls_les_reflexes_sont_injectes_d_office():
    assert [c.nom for c in cp.COMPARTIMENTS if c.toujours_injecte] == ["reflexes"]


# ─── Anciens noms (migration et transition) ───────────────────────────────────

@pytest.mark.parametrize("famille,nom,attendu", [
    ("semantic", "preference", "reflexes"),
    ("procedural", "rule", "reflexes"),
    ("procedural", "jimmy_workflow", "reflexes"),
    ("semantic", "decisions", "decisions"),
    ("procedural", "decisions", "decisions"),
    ("procedural", "deploy_playbooks", "savoir_faire"),
    ("procedural", "coding_best_practices", "savoir_faire"),
    ("procedural", "code_error_resolution", "alertes"),
    ("procedural", "bugfix", "alertes"),
    ("semantic", "infra_access", "environnement"),
    ("semantic", "fact", "savoirs"),
    ("semantic", "reflexions_humaines", "savoirs"),
    ("episodic", "session_log", "episodes"),
    ("episodic", "interaction", "episodes"),
    ("working", "scratch", "observations"),
    ("reflective", "user_model", "croyances"),
    # Le cas d'école de l'audit : le même nom, deux sens selon la famille d'origine.
    ("episodic", "project_state", "episodes"),
    ("semantic", "project_state", "savoirs"),
    ("procedural", "project_state", "savoirs"),
])
def test_conversion_exacte_des_anciens_noms(famille, nom, attendu):
    assert cp.depuis_ancien(famille, nom) == (attendu, True)


@pytest.mark.parametrize("famille,attendu", [
    ("semantic", "savoirs"), ("episodic", "episodes"), ("procedural", "savoir_faire"),
    ("working", "observations"), ("reflective", "croyances"), (None, "savoirs"),
])
def test_repli_par_famille_pour_un_nom_inconnu(famille, attendu):
    assert cp.depuis_ancien(famille, "crm_ultimate_deployment") == (attendu, False)
    assert cp.depuis_ancien(famille, None) == (attendu, False)


def test_un_nom_de_compartiment_se_convertit_en_lui_meme():
    assert cp.depuis_alias("alertes") == "alertes"


def test_la_migration_part_de_la_meme_table_que_le_coeur():
    """Copie figée au 22/11 : identique au cœur le jour de sa création."""
    chemin = ROOT / "migrations" / "versions" / "20261122_compartments.py"
    spec = importlib.util.spec_from_file_location("migration_compartments", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ALIAS == cp.ALIAS
    assert module.ALIAS_PAR_FAMILLE == cp.ALIAS_PAR_FAMILLE
    assert module.REPLI_FAMILLE == cp.REPLI_FAMILLE
    assert module.COMPARTIMENTS == cp.NOMS
    assert module.FAMILLE == {c.nom: c.famille for c in cp.COMPARTIMENTS}


def test_le_case_sql_de_la_migration_couvre_tous_les_cas():
    chemin = ROOT / "migrations" / "versions" / "20261122_compartments.py"
    spec = importlib.util.spec_from_file_location("migration_compartments2", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = module._case_compartiment()
    # Le cas (famille, nom) doit précéder le nom seul : sinon `project_state` épisodique
    # tomberait dans le repli par famille… qui donnerait le même résultat, mais
    # `project_state` sémantique non. L'ordre est une partie du contrat.
    assert case.index("subtype = 'project_state'") < case.index("subtype = 'preference'")
    assert case.endswith("ELSE 'savoirs' END")
    assert module._lit("l'apostrophe") == "'l''apostrophe'"


# ─── Classement par similarité ─────────────────────────────────────────────────

def _protos():
    """Prototypes orthogonaux : un axe par compartiment classable."""
    noms = cp.CLASSABLES
    return {nom: [1.0 if i == j else 0.0 for j in range(len(noms))]
            for i, nom in enumerate(noms)}


def test_classer_prend_le_plus_proche():
    protos = _protos()
    vecteur = [0.0] * len(cp.CLASSABLES)
    vecteur[cp.CLASSABLES.index("alertes")] = 1.0
    nom, sim = cp.classer(vecteur, protos)
    assert (nom, round(sim, 3)) == ("alertes", 1.0)


def test_classer_ignore_un_prototype_de_croyances():
    """Même si un prototype `croyances` traînait dans le dictionnaire, il est ignoré."""
    protos = {"croyances": [1.0, 0.0], "savoirs": [0.0, 1.0]}
    assert cp.classer([1.0, 0.1], protos)[0] == "savoirs"


def test_classer_sans_vecteur_retombe_sur_le_defaut():
    assert cp.classer(None, _protos()) == ("savoirs", 0.0)
    assert cp.classer([1.0, 0.0], {}) == ("savoirs", 0.0)


def test_cosinus_robuste():
    assert cp.cosinus([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cp.cosinus([1.0], [1.0, 0.0]) == 0.0


def test_centroide_normalise():
    c = cp.centroide([[2.0, 0.0], [0.0, 2.0]])
    assert round(c[0], 4) == round(c[1], 4) == round(2 ** -0.5, 4)
    assert cp.centroide([]) == []


# ─── Résolution à l'écriture ───────────────────────────────────────────────────

def test_resoudre_nom_valide():
    r = cp.resoudre("decisions", None, {})
    assert (r.compartiment, r.origine) == ("decisions", "agent")


def test_resoudre_ancien_nom():
    r = cp.resoudre("preference", None, {})
    assert (r.compartiment, r.origine, r.demande) == ("reflexes", "alias", "preference")


def test_resoudre_inconnu_classe_par_le_moteur():
    protos = _protos()
    vecteur = [0.0] * len(cp.CLASSABLES)
    vecteur[cp.CLASSABLES.index("environnement")] = 1.0
    r = cp.resoudre("clients_paca", vecteur, protos)
    assert (r.compartiment, r.origine, r.similarite) == ("environnement", "moteur", 1.0)


def test_resoudre_absent_classe_par_le_moteur():
    r = cp.resoudre(None, None, {})
    assert (r.compartiment, r.origine, r.demande) == ("savoirs", "moteur", None)


def test_resoudre_rend_croyances_si_demande_explicitement():
    """La décision de refuser appartient à l'appelant (store_memory) ; le cœur ne ment pas
    sur ce qui a été demandé."""
    assert cp.resoudre("croyances", None, {}).compartiment == "croyances"


def test_un_extracteur_ne_produit_jamais_de_croyance():
    for demande in ("croyances", "user_model", "human_insights"):
        r = cp.normaliser_extraction(demande, None, {})
        assert r.compartiment != "croyances" and r.origine == "moteur"


def test_extraction_valide_conservee():
    r = cp.normaliser_extraction("alertes", None, {})
    assert (r.compartiment, r.origine) == ("alertes", "agent")


# ─── Prototypes (cache par modèle) ─────────────────────────────────────────────

class _EmbedderCompteur:
    def __init__(self, model):
        self.model = model
        self.appels = 0

    def embed(self, textes):
        self.appels += 1
        return [[float(len(t)), 1.0] for t in textes]


def test_prototypes_calcules_une_fois_par_modele():
    cp._PROTOTYPES.clear()
    e = _EmbedderCompteur("modele-a")
    p1 = cp.prototypes(e)
    p2 = cp.prototypes(e)
    assert p1 is p2 and e.appels == 1
    assert set(p1) == set(cp.CLASSABLES)
    autre = _EmbedderCompteur("modele-b")
    cp.prototypes(autre)
    assert autre.appels == 1          # un autre modèle ne réutilise pas le cache
    cp._PROTOTYPES.clear()


def test_textes_prototypes_excluent_les_croyances():
    assert "croyances" not in cp.textes_prototypes()


# ─── Comportements ─────────────────────────────────────────────────────────────

def test_demi_vie_par_compartiment():
    assert cp.demi_vie("observations", 14.0) == 7.0
    assert cp.demi_vie("savoir_faire", 14.0) == 56.0
    assert cp.demi_vie("reflexes", 14.0) == 0.0
    assert cp.demi_vie("savoirs", 0.0) == 0.0          # oubli désactivé pour l'instance
    assert cp.demi_vie("inconnu", 14.0) == 14.0


def test_parts_max_et_surcharges():
    assert cp.parts_max() == {"reflexes": 0.20, "croyances": 0.15}
    assert cp.parts_max({"reflexes": 0.3, "inconnu": 0.5, "savoirs": 2.0}) == {
        "reflexes": 0.3, "croyances": 0.15}


def test_catalogue_public():
    fiches = cp.catalogue()
    assert [f["name"] for f in fiches] == list(cp.NOMS)
    reflexes = fiches[0]
    assert reflexes["always_injected"] is True and reflexes["forgetting"] == "aucun"
    assert fiches[-1]["direct_write"] is False
