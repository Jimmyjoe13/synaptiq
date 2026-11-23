"""Sommeil, récompense, plasticité (lot 2) : les décisions PURES du cœur.

Ce qui est verrouillé : rien n'est jamais supprimé (l'oubli et les doublons vont en
archives) ; les réflexes et les croyances ne s'oublient pas ; la consolidation ne produit
jamais de réflexe ni de croyance ; un lien délaissé ne descend pas sous le poids d'un lien
neuf ; la récompense nuance la pertinence sans la remplacer.
"""
from datetime import datetime

import pytest

from synaptiq_core.context_builder import InMemoryStore, RetrievalConfig, build_context_packet
from synaptiq_core.plasticite import (
    POIDS_LIEN_MAX,
    POIDS_LIEN_NOUVEAU,
    affaiblir_poids,
    borner_recompense,
    facteur_recompense,
    paires_hebbiennes,
    renforcer_poids,
)
from synaptiq_core.sommeil import (
    CIBLES_CONSOLIDATION,
    ReglagesSommeil,
    doit_oublier,
    grouper,
    paire_doublon,
    prompt_consolidation,
    valider_consolidation,
)

JOUR = 86400.0
R = ReglagesSommeil()


# ─── Récompense ───────────────────────────────────────────────────────────────

def test_facteur_de_recompense_borne():
    assert facteur_recompense(0, 0.1) == 1.0
    assert facteur_recompense(None, 0.1) == 1.0
    assert facteur_recompense(3, 0.1) == pytest.approx(1.3)
    assert facteur_recompense(50, 0.1) == pytest.approx(1.5)
    assert facteur_recompense(-50, 0.1) == pytest.approx(0.5)
    assert facteur_recompense(3, 0.0) == 1.0
    assert borner_recompense(7) == 5.0


def test_la_recompense_pondere_le_score_au_rappel():
    def _mem(i, reward):
        return {"id": i, "type": "semantic", "compartment": "savoirs", "content": i,
                "confidence": 1.0, "importance": 0.5, "reward": reward,
                "last_accessed_at": datetime(2026, 1, 1), "created_at": datetime(2026, 1, 1),
                "occurred_at": None,
                "embedding": [1.0, 0.0] if i == "U" else [0.0, 1.0], "similarity": 1.0,
                "age_seconds": 0.0, "rank_vec": 1, "rank_fts": None}
    store = InMemoryStore([_mem("U", 2.0), _mem("N", 0.0)])
    trace = build_context_packet(store, [1.0, 0.0], "q", ["savoirs"], 1000,
                                 RetrievalConfig(hybrid=False, recency_halflife_days=0),
                                 "t", explain=True, explain_level="all")["retrieval_trace"]
    scores = {e["memory_id"]: e["score"] for e in trace}
    assert scores["U"] == pytest.approx(1.2) and scores["N"] == pytest.approx(1.0)


# ─── Plasticité ───────────────────────────────────────────────────────────────

def test_renforcement_plafonne_et_lien_neuf():
    assert renforcer_poids(None, 0.0) == POIDS_LIEN_NOUVEAU
    assert renforcer_poids(0.95, 0.1) == POIDS_LIEN_MAX


def test_affaiblissement_jamais_sous_le_poids_d_un_lien_neuf():
    assert affaiblir_poids(1.0, 0.05) == pytest.approx(0.95)
    assert affaiblir_poids(0.52, 0.5) == POIDS_LIEN_NOUVEAU
    assert affaiblir_poids(0.3, 0.5) == 0.3          # déjà sous le plancher : inchangé


def test_paires_hebbiennes_canoniques_et_dedoublonnees():
    assert paires_hebbiennes(["c", "a", "b", "a"]) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert paires_hebbiennes(["seul"]) == []


# ─── Doublons ─────────────────────────────────────────────────────────────────

def test_doublon_le_mieux_note_survit_pas_le_plus_recent():
    ancien_utile = {"id": "A", "reward": 2, "access_count": 0, "importance": 0.5,
                    "created_at": "2026-01-01"}
    recent = {"id": "B", "reward": 0, "access_count": 9, "importance": 0.9,
              "created_at": "2026-11-01"}
    assert paire_doublon(ancien_utile, recent) == ("A", "B")
    assert paire_doublon(recent, {**ancien_utile, "reward": 0}) == ("B", "A")


# ─── Oubli ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("compartiment,age_jours,acces,attendu", [
    ("observations", 40, 0, True),      # demi-vie 7 j : 40 j -> récence 0,019
    ("observations", 40, 1, False),     # déjà relu : jamais oublié
    ("observations", 20, 0, False),     # trop jeune (âge minimal 30 j)
    ("episodes", 40, 0, False),         # demi-vie 14 j : récence 0,138
    ("episodes", 70, 0, True),
    ("savoir_faire", 200, 0, False),    # demi-vie 56 j : récence 0,084
    ("reflexes", 5000, 0, False),       # jamais oublié
    ("croyances", 5000, 0, False),
    ("archives", 5000, 0, False),
    ("inconnu", 5000, 0, False),
])
def test_oubli(compartiment, age_jours, acces, attendu):
    assert doit_oublier(compartiment, age_jours * JOUR, acces, 14.0, R) is attendu


def test_oubli_desactive_si_l_instance_n_oublie_pas():
    assert doit_oublier("observations", 400 * JOUR, 0, 0.0, R) is False


# ─── Consolidation ────────────────────────────────────────────────────────────

def test_grouper_par_similarite_et_taille():
    souvenirs = [{"id": i, "embedding": v} for i, v in
                 [("a", [1, 0]), ("b", [0.99, 0.1]), ("c", [0, 1]), ("d", [1, 0.01])]]
    groupes = grouper(souvenirs, 0.9, taille_max=2)
    assert [[s["id"] for s in g] for g in groupes] == [["a", "b"], ["c"], ["d"]]


def test_prompt_liste_les_cibles_et_les_episodes():
    messages = prompt_consolidation([{"content": "Session du jour : migration OK",
                                      "created_at": "2026-11-20T10:00:00"}])
    texte = messages[-1]["content"]
    assert "[2026-11-20] Session du jour" in texte
    for cible in CIBLES_CONSOLIDATION:
        assert f"'{cible}'" in texte
    assert "'reflexes'" not in texte and "'croyances'" not in texte


def test_valider_consolidation_ne_produit_ni_reflexe_ni_croyance():
    sortie = valider_consolidation({"memories": [
        {"compartment": "reflexes", "content": "Toujours tutoyer Jimmy."},
        {"compartment": "croyances", "content": "Jimmy semble pressé ce matin."},
        {"compartment": "alertes", "content": "Le port 6399 est réservé par Windows."},
        {"compartment": "decisions", "content": "court"},
        "pas un objet",
    ]})
    assert sortie == [("savoirs", "Toujours tutoyer Jimmy."),
                      ("savoirs", "Jimmy semble pressé ce matin."),
                      ("alertes", "Le port 6399 est réservé par Windows.")]


@pytest.mark.parametrize("donnees", [None, "texte", {"memories": "x"}, {}])
def test_valider_consolidation_tolere_le_bruit(donnees):
    assert valider_consolidation(donnees) == []


def test_valider_consolidation_au_plus_trois():
    items = [{"compartment": "savoirs", "content": f"Souvenir durable numéro {i}"}
             for i in range(6)]
    assert len(valider_consolidation(items)) == 3
