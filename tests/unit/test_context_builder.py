"""Tests unitaires de l'orchestration Q-EM — sans HTTP, sans PostgreSQL (audit F16).

Ce qui était intestable avant l'extraction : le scoring de départ (RRF normalisé vs
cosinus), la complétion du graphe, le marquage des accès et la forme de la réponse. Ces
quatre points ne pouvaient être exercés qu'à travers un `TestClient` et une base réelle.
"""
from datetime import datetime

from synaptiq_core.context_builder import (
    InMemoryStore,
    RetrievalConfig,
    build_context_packet,
)

TOUS_TYPES = ["semantic", "episodic", "procedural", "working"]
CONFIG = RetrievalConfig(hybrid=False, recency_halflife_days=0.0)


def _mem(mem_id, contenu="contenu", type_="semantic", subtype="fact",
         similarity=1.0, embedding=None, importance=0.5, **extra):
    """Ligne telle que la renvoie un MemoryStore (vecteur déjà désérialisé)."""
    ligne = {
        "id": mem_id,
        "type": type_,
        "subtype": subtype,
        "content": contenu,
        "confidence": 1.0,
        "importance": importance,
        "last_accessed_at": datetime(2026, 7, 1),
        "created_at": datetime(2026, 7, 1),
        "occurred_at": None,
        "embedding": embedding if embedding is not None else [1.0, 0.0, 0.0],
        "similarity": similarity,
        "age_seconds": 0.0,
        "rank_vec": 1,
        "rank_fts": None,
    }
    ligne.update(extra)
    return ligne


def _construire(store, **kw):
    params = dict(query_vector=[1.0, 0.0, 0.0], query_text="requête",
                  memory_types=TOUS_TYPES, max_tokens=1000, config=CONFIG,
                  trace_id="trace_test")
    params.update(kw)
    return build_context_packet(store, **params)


# ─── Contrat de réponse ──────────────────────────────────────────────────────

def test_magasin_vide_renvoie_les_7_cles():
    """Contrat stable côté consommateur : le paquet porte toujours ses 7 collections."""
    resultat = _construire(InMemoryStore())
    assert set(resultat["context_packet"]) == {
        "facts", "preferences", "episodes", "rules", "best_practices", "errors", "examples"}
    assert all(v == [] for v in resultat["context_packet"].values())
    assert resultat["token_estimate"] == 0
    assert resultat["selected_memory_ids"] == []
    assert resultat["trace_id"] == "trace_test"


def test_trace_id_est_celui_fourni_par_l_appelant():
    """Le trace_id vient du handler pour être corrélable avec ses logs."""
    store = InMemoryStore([_mem("A")])
    assert _construire(store, trace_id="trace_abc")["trace_id"] == "trace_abc"


def test_explain_absent_par_defaut():
    store = InMemoryStore([_mem("A")])
    assert _construire(store)["retrieval_trace"] is None
    trace = _construire(store, explain=True)["retrieval_trace"]
    assert len(trace) == 1
    assert trace[0]["memory_id"] == "A"
    assert trace[0]["selection_reason"] == "selected_by_utility_under_token_budget"


def test_routage_par_type_et_soustype():
    store = InMemoryStore([
        _mem("F", contenu="un fait", subtype="fact"),
        _mem("P", contenu="une préférence", subtype="preference",
             embedding=[0.0, 1.0, 0.0]),
        _mem("R", contenu="une règle", type_="procedural", subtype="rule",
             embedding=[0.0, 0.0, 1.0]),
    ])
    packet = _construire(store)["context_packet"]
    assert packet["facts"] == ["un fait"]
    assert packet["preferences"] == ["une préférence"]
    assert packet["rules"] == ["une règle"]


# ─── Complétion du graphe d'intrication ──────────────────────────────────────

def test_intrication_ramene_un_voisin_non_trouve_par_la_recherche():
    """Une mémoire absente des candidats entre par ACTIVATION, pas par similarité."""
    trouvee = _mem("SEED", contenu="point d'entrée")
    voisine = _mem("VOISINE", contenu="ramenée par le graphe",
                   embedding=[0.0, 1.0, 0.0], similarity=0.0)
    store = InMemoryStore(
        memoires=[trouvee],
        relations=[{"source_memory_id": "SEED", "target_memory_id": "VOISINE",
                    "relation_type": "entangled_with", "weight": 1.0}],
    )
    # La voisine n'est pas candidate à la recherche, mais existe pour `fetch_by_ids`.
    store.memoires["VOISINE"] = voisine

    config = RetrievalConfig(hybrid=False, recency_halflife_days=0.0,
                             entangle_damping=0.5, entangle_max_hops=1)
    packet = _construire(store, config=config)["context_packet"]
    assert "point d'entrée" in packet["facts"]
    assert "ramenée par le graphe" in packet["facts"]


def test_intrication_desactivee_ne_ramene_rien():
    """max_hops=0 : le voisin reste dehors (levier d'ablation de la phase 2)."""
    store = InMemoryStore(
        memoires=[_mem("SEED", contenu="point d'entrée")],
        relations=[{"source_memory_id": "SEED", "target_memory_id": "VOISINE",
                    "relation_type": "entangled_with", "weight": 1.0}],
    )
    store.memoires["VOISINE"] = _mem("VOISINE", contenu="ramenée par le graphe",
                                     embedding=[0.0, 1.0, 0.0], similarity=0.0)
    config = RetrievalConfig(hybrid=False, recency_halflife_days=0.0, entangle_max_hops=0)
    packet = _construire(store, config=config)["context_packet"]
    assert "ramenée par le graphe" not in packet["facts"]


def test_l_isolation_repose_sur_le_magasin_pas_sur_l_orchestration():
    """RÉGRESSION F1, verrouillée par le TYPE.

    L'orchestration demande des ids au magasin sans jamais fournir de tenant ni d'agent :
    elle n'a donc pas les moyens de franchir la frontière d'isolation. Ici le magasin ne
    connaît pas la voisine (comme le ferait un magasin borné à un autre agent) et rien ne
    remonte — l'ancienne requête, elle, l'aurait ramenée.
    """
    store = InMemoryStore(
        memoires=[_mem("SEED", contenu="dans le périmètre")],
        relations=[{"source_memory_id": "SEED", "target_memory_id": "HORS_PERIMETRE",
                    "relation_type": "entangled_with", "weight": 1.0}],
    )
    resultat = _construire(store)
    tout = " ".join(c for entrees in resultat["context_packet"].values() for c in entrees)
    assert "dans le périmètre" in tout
    assert "HORS_PERIMETRE" not in tout


# ─── Effets de bord ──────────────────────────────────────────────────────────

def test_les_memoires_retenues_sont_marquees_accedees():
    """La récence se réactive à la lecture : sans ce marquage, la décroissance est fausse."""
    store = InMemoryStore([_mem("A"), _mem("B", embedding=[0.0, 1.0, 0.0])])
    resultat = _construire(store)
    assert sorted(store.acces_marques) == sorted(resultat["selected_memory_ids"])


def test_aucun_marquage_si_rien_n_est_retenu():
    store = InMemoryStore()
    _construire(store)
    assert store.acces_marques == []


# ─── Scoring de départ ───────────────────────────────────────────────────────

def test_hybride_le_score_vient_du_rang_fusionne():
    """En hybride, un souvenir trouvé UNIQUEMENT par le plein texte doit survivre.

    C'est la raison d'être de la normalisation RRF : avec le seul cosinus, ce candidat
    entrerait à un score quasi nul et le collapse l'éliminerait — le rappel gagné par la
    recherche plein texte serait aussitôt reperdu.
    """
    store = InMemoryStore([
        _mem("VECTORIEL", contenu="trouvé par le vecteur", similarity=0.9,
             rank_vec=1, rank_fts=None),
        # Cosinus quasi nul, mais premier en plein texte.
        _mem("LITTERAL", contenu="trouvé par le plein texte", similarity=0.01,
             embedding=[0.0, 1.0, 0.0], rank_vec=None, rank_fts=1),
    ])
    config = RetrievalConfig(hybrid=True, recency_halflife_days=0.0)
    resultat = _construire(store, config=config)
    assert set(resultat["selected_memory_ids"]) == {"VECTORIEL", "LITTERAL"}
    trace = _construire(store, config=config, explain=True)["retrieval_trace"]
    scores = {t["memory_id"]: t["score"] for t in trace}
    # Le score du candidat littéral n'est PAS son cosinus (0.01) : il vient du rang fusionné.
    assert scores["LITTERAL"] > 0.5


def test_non_hybride_le_score_est_le_cosinus():
    store = InMemoryStore([_mem("A", similarity=0.42)])
    trace = _construire(store, config=RetrievalConfig(hybrid=False, recency_halflife_days=0.0),
                        explain=True)["retrieval_trace"]
    assert trace[0]["score"] == 0.42


def test_similarite_negative_ne_produit_pas_de_score_negatif():
    """Un cosinus négatif est ramené à 0, donc le candidat n'entre pas dans le paquet.

    Sans cet écrêtage, un score négatif s'inviterait dans le tri par densité d'utilité et
    dans la propagation d'activation, où il RETIRERAIT de l'activation à ses voisins.
    """
    store = InMemoryStore([_mem("A", similarity=-0.3)])
    resultat = _construire(store, explain=True)
    assert resultat["selected_memory_ids"] == []
    assert resultat["retrieval_trace"] == []
    assert resultat["token_estimate"] == 0


def test_budget_de_tokens_respecte():
    store = InMemoryStore([
        _mem("COURT", contenu="un"),
        _mem("LONG", contenu="a b c d e f g h i j", embedding=[0.0, 1.0, 0.0]),
    ])
    resultat = _construire(store, max_tokens=2)
    assert resultat["selected_memory_ids"] == ["COURT"]
    assert resultat["token_estimate"] <= 2
