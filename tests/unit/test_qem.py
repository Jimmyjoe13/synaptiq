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


def test_propagation_multihop():
    """Chaîne M1(seed)→M2→M3 : l'activation se propage à 2 sauts, atténuée par damping.

    Verrouille le comportement multi-hop (vrai spreading activation) : M3, à 2 liens
    du seed, remonte alors qu'il ne matche pas la requête.
    """
    candidates = {
        "M1": _cand("M1", similarity=1.0, score=1.0),
        "M2": _cand("M2", similarity=0.0, score=0.0),
        "M3": _cand("M3", similarity=0.0, score=0.0),
    }
    relationships = [
        {"source_memory_id": "M1", "target_memory_id": "M2",
         "relation_type": "entangled_with", "weight": 1.0},
        {"source_memory_id": "M2", "target_memory_id": "M3",
         "relation_type": "entangled_with", "weight": 1.0},
    ]
    propagate_entanglement(candidates, relationships, damping=0.5, max_hops=2)

    # hop1 : M2 += M1.sim(1.0)*w*damping(0.5) = 0.5
    assert candidates["M2"]["score"] == 0.5
    # hop2 : M3 += activation(M2)=0.5 * w * damping(0.5) = 0.25
    assert candidates["M3"]["score"] == 0.25
    # M1 (seed, déjà visité) n'est jamais re-boosté : pas de retour d'onde
    assert candidates["M1"]["score"] == 1.0


def test_propagation_mono_hop_borne():
    """max_hops=1 : seul le voisin direct est activé, M3 (2 sauts) reste à 0."""
    candidates = {
        "M1": _cand("M1", similarity=1.0, score=1.0),
        "M2": _cand("M2", similarity=0.0, score=0.0),
        "M3": _cand("M3", similarity=0.0, score=0.0),
    }
    relationships = [
        {"source_memory_id": "M1", "target_memory_id": "M2",
         "relation_type": "entangled_with", "weight": 1.0},
        {"source_memory_id": "M2", "target_memory_id": "M3",
         "relation_type": "entangled_with", "weight": 1.0},
    ]
    propagate_entanglement(candidates, relationships, damping=0.5, max_hops=1)
    assert candidates["M2"]["score"] == 0.5
    assert candidates["M3"]["score"] == 0.0


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


# ─── Invariants de la version vectorisée du filtre de redondance (audit F9) ──

def test_redondance_ne_propage_pas_en_chaine():
    """Une mémoire ANNULÉE n'annule personne : la chaîne est rompue.

    A et C sont orthogonaux (cosinus 0), mais B est proche des deux. B est annulé par A ;
    il ne doit donc pas à son tour annuler C. Sans cette rupture, un maillon intermédiaire
    propagerait des suppressions en cascade et viderait le contexte.
    """
    proche_de_a = [0.98, 0.199, 0.0]     # cos(A) ~0.98 > seuil
    candidates = {
        "A": _cand("A", importance=0.9, embedding=[1.0, 0.0, 0.0], score=1.0),
        "B": _cand("B", importance=0.8, embedding=proche_de_a, score=1.0),
        "C": _cand("C", importance=0.7, embedding=[0.0, 1.0, 0.0], score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["A"]["score"] == 1.0
    assert candidates["B"]["score"] == 0.0   # annulé par A
    assert candidates["C"]["score"] == 1.0   # PAS annulé par B, qui est mort


def test_redondance_compare_a_tous_les_conserves():
    """Un candidat est annulé s'il est redondant avec N'IMPORTE lequel des conservés."""
    candidates = {
        "A": _cand("A", importance=0.9, embedding=[1.0, 0.0, 0.0], score=1.0),
        "B": _cand("B", importance=0.8, embedding=[0.0, 1.0, 0.0], score=1.0),
        # Orthogonal à A, mais quasi identique à B -> doit tomber.
        "C": _cand("C", importance=0.7, embedding=[0.0, 0.99, 0.141], score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["A"]["score"] == 1.0
    assert candidates["B"]["score"] == 1.0
    assert candidates["C"]["score"] == 0.0


def test_redondance_ignore_les_candidats_sans_embedding():
    """Sans vecteur, aucune redondance n'est démontrable : ni annulé, ni annulateur."""
    candidates = {
        "AVEC": _cand("AVEC", importance=0.9, embedding=[1.0, 0.0, 0.0], score=1.0),
        "SANS": _cand("SANS", importance=0.8, embedding=[], score=1.0),
        "COPIE": _cand("COPIE", importance=0.7, embedding=[1.0, 0.0, 0.0], score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["AVEC"]["score"] == 1.0
    assert candidates["SANS"]["score"] == 1.0    # conservé faute de vecteur
    assert candidates["COPIE"]["score"] == 0.0   # annulé par AVEC malgré le trou


def test_redondance_ecarte_les_dimensions_incoherentes():
    """Un vecteur d'une autre dimension est écarté, jamais tronqué en silence."""
    candidates = {
        "REF": _cand("REF", importance=0.9, embedding=[1.0, 0.0, 0.0], score=1.0),
        "AUTRE_DIM": _cand("AUTRE_DIM", importance=0.8, embedding=[1.0, 0.0], score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["REF"]["score"] == 1.0
    assert candidates["AUTRE_DIM"]["score"] == 1.0


def test_redondance_accepte_les_tableaux_numpy():
    """`parse_embedding` renvoie désormais un ndarray : le filtre doit l'accepter."""
    import numpy as np
    candidates = {
        "HI": _cand("HI", importance=0.8, embedding=np.array([1.0, 0.0, 0.0]), score=1.0),
        "LO": _cand("LO", importance=0.5, embedding=np.array([1.0, 0.0, 0.0]), score=1.0),
    }
    filter_redundancy(candidates, threshold=0.75)
    assert candidates["HI"]["score"] == 1.0
    assert candidates["LO"]["score"] == 0.0


def test_redondance_equivalente_a_la_reference_naive():
    """Équivalence avec l'implémentation de référence sur des données pseudo-aléatoires.

    Garde-fou contre une divergence subtile introduite par la vectorisation : on rejoue
    l'algorithme naïf (double boucle, rupture de chaîne) et on compare les scores.
    """
    import random
    rng = random.Random(20260729)

    def _reference(cands, seuil):
        actifs = [c for c, v in cands.items() if v["score"] > 0.0]
        actifs.sort(key=lambda c: (cands[c]["importance"], cands[c]["created_at"]), reverse=True)
        for i in range(len(actifs)):
            if cands[actifs[i]]["score"] == 0.0:
                continue
            for j in range(i + 1, len(actifs)):
                if cands[actifs[j]]["score"] == 0.0:
                    continue
                cos = sum(x * y for x, y in zip(cands[actifs[i]]["embedding"],
                                                cands[actifs[j]]["embedding"], strict=False))
                if cos > seuil:
                    cands[actifs[j]]["score"] = 0.0

    for essai in range(30):
        base = {}
        for k in range(8):
            vec = [rng.gauss(0, 1) for _ in range(6)]
            norme = sum(x * x for x in vec) ** 0.5
            base[f"m{k}"] = {"importance": round(rng.uniform(0, 1), 3),
                             "embedding": [x / norme for x in vec]}
        vectorise = {k: _cand(k, score=1.0, **v) for k, v in base.items()}
        naif = {k: _cand(k, score=1.0, **v) for k, v in base.items()}

        filter_redundancy(vectorise, threshold=0.3)
        _reference(naif, 0.3)

        assert {k: v["score"] for k, v in vectorise.items()} == \
               {k: v["score"] for k, v in naif.items()}, f"divergence à l'essai {essai}"


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
    # Une famille inconnue renvoyait `None`, et `collapse_by_utility` retirait alors la
    # mémoire du paquet — après l'avoir comptée dans `selected_ids` et facturée en tokens.
    # Elle retombe désormais sur une section de repli (cf. tests/unit/test_collections.py).
    assert route_memory("inconnu", None) == "facts"


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


# ─── Selection sous max_tokens ───

def test_collapse_conserve_candidats_sous_max_tokens():
    """Tant que max_tokens le permet, tous les candidats actifs sont conservés."""
    candidates = {
        "fort": _cand("fort", score=1.0),
        "traine": _cand("traine", score=0.01),
    }
    _, ids, _ = collapse_by_utility(candidates, max_tokens=10_000)
    assert set(ids) == {"fort", "traine"}


# ─── Datation du contexte ───

def test_la_date_prefixe_le_souvenir():
    """Sans la date dans le texte, le LLM ne peut répondre à aucune question « quand »."""
    candidates = {"m1": _cand("m1", content="Caroline est allee a un groupe de soutien",
                              score=1.0, occurred_at=datetime(2023, 5, 7))}
    packet, _, _ = collapse_by_utility(candidates, max_tokens=10_000)
    assert packet["facts"] == ["[2023-05-07] Caroline est allee a un groupe de soutien"]


def test_souvenir_sans_date_inchange():
    candidates = {"m1": _cand("m1", content="Caroline aime les beagles", score=1.0)}
    packet, _, _ = collapse_by_utility(candidates, max_tokens=10_000)
    assert packet["facts"] == ["Caroline aime les beagles"]


def test_le_prefixe_est_compte_dans_le_budget():
    """La date consomme des tokens : l'omettre du décompte ferait déborder le budget."""
    from synaptiq_core.qem import estimate_tokens, format_entry
    contenu = "Caroline est allee a un groupe de soutien"
    date = datetime(2023, 5, 7)
    candidates = {"m1": _cand("m1", content=contenu, score=1.0, occurred_at=date)}
    _, _, tokens = collapse_by_utility(candidates, max_tokens=10_000)
    assert tokens == estimate_tokens(format_entry(contenu, date))
