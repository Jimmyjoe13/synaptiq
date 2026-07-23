"""Tests unitaires du cœur algorithmique Q-EM (packages/core/synaptiq_core/qem.py).

Purs : aucune infra (ni Postgres, ni Redis), embeddings déterministes à la main.
Répliquent en isolation la logique des tests d'intégration `tests/test_q_em.py`.
"""
from datetime import datetime

from synaptiq_core.qem import (
    apply_contradictions,
    collapse_by_utility,
    compute_recency_factor,
    filter_redundancy,
    initial_score,
    propagate_entanglement,
    route_memory,
)

HALFLIFE = 90.0  # jours


def _cand(mem_id, **kw):
    """Fabrique un candidat avec des valeurs par défaut sûres."""
    base = {
        "id": mem_id,
        "type": "semantic",
        "subtype": "fact",
        "content": "contenu",
        "confidence": 1.0,
        "importance": 0.5,
        "created_at": datetime(2026, 1, 1),
        "last_accessed_at": datetime(2026, 1, 1),
        "embedding": [1.0, 0.0, 0.0],
        "similarity": 0.0,
        "recency_factor": 1.0,
        "score": 0.0,
    }
    base.update(kw)
    return base


# ─── Phase 1 : récence + score initial ───────────────────────────────────────

def test_recency_factor_demi_vie():
    """age=0 -> 1.0 ; age=halflife -> 0.5 ; age=2*halflife -> 0.25."""
    assert compute_recency_factor(0, HALFLIFE) == 1.0
    assert compute_recency_factor(HALFLIFE * 86400, HALFLIFE) == 0.5
    assert compute_recency_factor(2 * HALFLIFE * 86400, HALFLIFE) == 0.25


def test_recency_factor_desactive():
    """halflife <= 0 => décroissance neutralisée (1.0)."""
    assert compute_recency_factor(999 * 86400, 0) == 1.0
    assert compute_recency_factor(999 * 86400, -5) == 1.0


def test_initial_score():
    """Le score de départ = similarité x facteur de récence."""
    assert initial_score(0.8, 0.5) == 0.4
    assert initial_score(1.0, 1.0) == 1.0


# ─── Phase 2 : intrication (propagation d'activation) ────────────────────────

def test_propagation_intrication():
    """M2 sans similarité directe mais intriquée à M1 (similaire) -> score M2 > 0.

    Réplique en pur la logique de `test_q_em_entanglement_propagation`.
    """
    candidates = {
        "M1": _cand("M1", type="semantic", subtype="fact", similarity=1.0, score=1.0),
        "M2": _cand("M2", type="procedural", subtype="rule", similarity=0.0, score=0.0),
    }
    relationships = [
        {"source_memory_id": "M1", "target_memory_id": "M2",
         "relation_type": "entangled_with", "weight": 1.0},
    ]
    propagate_entanglement(candidates, relationships, damping=0.5)

    # M2 reçoit M1.similarity(1.0) * weight(1.0) * damping(0.5) = 0.5
    assert candidates["M2"]["score"] == 0.5
    # M1 reçoit M2.similarity(0.0) * ... = +0, inchangé
    assert candidates["M1"]["score"] == 1.0


def test_propagation_ignore_extremites_absentes():
    """Un lien vers une mémoire hors des candidats ne propage rien (pas de KeyError)."""
    candidates = {"M1": _cand("M1", similarity=1.0, score=1.0)}
    relationships = [
        {"source_memory_id": "M1", "target_memory_id": "ABSENT",
         "relation_type": "entangled_with", "weight": 1.0},
    ]
    propagate_entanglement(candidates, relationships, damping=0.5)
    assert candidates["M1"]["score"] == 1.0


# ─── Phase 3 : interférences destructives ────────────────────────────────────

def test_contradiction_annule_la_plus_ancienne():
    """Sur un couple contradictoire, la mémoire au created_at le plus ancien est annulée."""
    candidates = {
        "OLD": _cand("OLD", created_at=datetime(2026, 7, 9, 8), score=1.0),
        "NEW": _cand("NEW", created_at=datetime(2026, 7, 9, 9), score=1.0),
    }
    relationships = [
        {"source_memory_id": "OLD", "target_memory_id": "NEW",
         "relation_type": "contradicts", "weight": 1.0},
    ]
    apply_contradictions(candidates, relationships)
    assert candidates["OLD"]["score"] == 0.0
    assert candidates["NEW"]["score"] == 1.0


def test_supersedes_by_traite_comme_contradiction():
    """'supersedes_by' déclenche le même filtre que 'contradicts'."""
    candidates = {
        "OLD": _cand("OLD", created_at=datetime(2026, 7, 9, 8), score=1.0),
        "NEW": _cand("NEW", created_at=datetime(2026, 7, 9, 9), score=1.0),
    }
    relationships = [
        {"source_memory_id": "NEW", "target_memory_id": "OLD",
         "relation_type": "supersedes_by", "weight": 1.0},
    ]
    apply_contradictions(candidates, relationships)
    assert candidates["OLD"]["score"] == 0.0
    assert candidates["NEW"]["score"] == 1.0


def test_redondance_annule_le_moins_important():
    """Deux embeddings identiques (cosinus 1.0 > seuil) -> seul le plus important survit."""
    candidates = {
        "HI": _cand("HI", importance=0.8, embedding=[1.0, 0.0, 0.0], score=1.0),
        "LO": _cand("LO", importance=0.5, embedding=[1.0, 0.0, 0.0], score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["HI"]["score"] == 1.0
    assert candidates["LO"]["score"] == 0.0


def test_redondance_embeddings_distincts_conserves():
    """Embeddings orthogonaux (cosinus 0 < seuil) -> aucune annulation."""
    candidates = {
        "A": _cand("A", importance=0.8, embedding=[1.0, 0.0, 0.0], score=1.0),
        "B": _cand("B", importance=0.5, embedding=[0.0, 1.0, 0.0], score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["A"]["score"] == 1.0
    assert candidates["B"]["score"] == 1.0


# ─── Phase 4 : mesure (collapse + routage) ───────────────────────────────────

def test_collapse_respecte_budget():
    """Une mémoire dont les tokens dépassent le budget restant est exclue."""
    candidates = {
        "A": _cand("A", content="un", score=1.0),           # 1 token, densité 1.0
        "B": _cand("B", content="a b c d e f", score=1.0),  # ~7 tokens
    }
    packet, selected_ids, token_count = collapse_by_utility(candidates, max_tokens=1)
    # A tient dans le budget (1 <= 1), B est hors budget
    assert selected_ids == ["A"]
    assert token_count == 1
    assert "un" in packet["facts"]


def test_collapse_routage_par_type():
    """Routage effectif : semantic/fact->facts, procedural/rule->rules,
    episodic->episodes, working->examples."""
    candidates = {
        "F": _cand("F", type="semantic", subtype="fact", content="fait", score=1.0),
        "R": _cand("R", type="procedural", subtype="rule", content="regle", score=1.0),
        "E": _cand("E", type="episodic", subtype="interaction", content="episode", score=1.0),
        "W": _cand("W", type="working", subtype=None, content="exemple", score=1.0),
    }
    packet, selected_ids, _ = collapse_by_utility(candidates, max_tokens=1000)
    assert set(selected_ids) == {"F", "R", "E", "W"}
    assert packet["facts"] == ["fait"]
    assert packet["rules"] == ["regle"]
    assert packet["episodes"] == ["episode"]
    assert packet["examples"] == ["exemple"]


def test_collapse_packet_toujours_7_cles():
    """Le context_packet expose toujours ses 7 clés, même à vide."""
    packet, _, _ = collapse_by_utility({}, max_tokens=1000)
    assert set(packet.keys()) == {
        "facts", "preferences", "episodes", "rules", "best_practices", "errors", "examples"
    }
    assert all(v == [] for v in packet.values())


def test_collapse_ignore_score_nul():
    """Un candidat au score nul (filtré en amont) n'est pas mesuré."""
    candidates = {
        "OK": _cand("OK", content="garde", score=1.0),
        "KO": _cand("KO", content="jette", score=0.0),
    }
    packet, selected_ids, _ = collapse_by_utility(candidates, max_tokens=1000)
    assert selected_ids == ["OK"]
    assert "jette" not in packet["facts"]


# ─── Routage correct (helper pur) : type/subtype -> clé du packet ────────────

def test_route_memory_routage_correct():
    """route_memory implémente le routage INTENDÉ par type/subtype (7 clés)."""
    assert route_memory("semantic", "preference") == "preferences"
    assert route_memory("semantic", "fact") == "facts"
    assert route_memory("episodic", "interaction") == "episodes"
    assert route_memory("procedural", "coding_best_practices") == "best_practices"
    assert route_memory("procedural", "code_error_resolution") == "errors"
    assert route_memory("procedural", "rule") == "rules"
    assert route_memory("working", None) == "examples"
    assert route_memory("inconnu", None) is None


def test_collapse_routage_par_soustype():
    """Le collapse propage le sous-type : les collections logiques dédiées sont remplies.

    semantic/preference -> preferences, procedural/coding_best_practices -> best_practices,
    procedural/code_error_resolution -> errors.
    """
    candidates = {
        "P": _cand("P", type="semantic", subtype="preference", content="pref", score=1.0),
        "BP": _cand("BP", type="procedural", subtype="coding_best_practices",
                    content="bp", score=1.0),
        "ER": _cand("ER", type="procedural", subtype="code_error_resolution",
                    content="err", score=1.0),
    }
    packet, _, _ = collapse_by_utility(candidates, max_tokens=1000)
    assert packet["preferences"] == ["pref"]
    assert packet["best_practices"] == ["bp"]
    assert packet["errors"] == ["err"]
    # Ces contenus ne doivent plus « fuiter » dans facts/rules.
    assert packet["facts"] == []
    assert packet["rules"] == []
