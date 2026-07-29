"""Tests unitaires du harness de benchmark LOCOMO (benchmarks/locomo_runner.py).

Un benchmark qui se trompe en silence est pire que pas de benchmark : il produit des
chiffres publiables et faux. Ces tests verrouillent les deux modes de défaillance déjà
rencontrés — un juge mal parsé et une baseline pas comparable — sans appeler de LLM.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "packages" / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

locomo = pytest.importorskip("benchmarks.locomo_runner", reason="harness LOCOMO absent")
from synaptiq_core.qem import estimate_tokens


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeClient:
    """Client HTTP minimal : renvoie une réponse figée et mémorise la requête."""

    def __init__(self, payload, status_code=200):
        self._response = FakeResponse(payload, status_code)
        self.last_path = None
        self.last_json = None

    def post(self, path, json=None):
        self.last_path = path
        self.last_json = json
        return self._response


# ─── Parsing du verdict du juge ───

@pytest.mark.parametrize("verdict,attendu", [
    ("oui", True),
    ("Oui.", True),
    ("yes", True),
    ("non", False),
    ("Non, la réponse diffère.", False),
    # Les modèles de raisonnement préfixent leur verdict d'une justification.
    ("Après analyse des deux formulations, oui", True),
    ("Le sens n'est pas le même : non", False),
    # Pièges de sous-chaîne : sans limite de mot, « Louis » contient « oui »
    # et « nous » contient « no » — ces cas ont produit de vrais faux positifs.
    ("Louis est mentionné, mais la réponse est fausse : non", False),
    ("Nous confirmons que la réponse est correcte : oui", True),
])
def test_verdict_du_juge(monkeypatch, verdict, attendu):
    monkeypatch.setattr(locomo, "_llm_chat", lambda *a, **k: verdict)
    assert locomo.judge("q", "gold", "hyp", pace=0) is attendu


def test_verdict_illisible_compte_comme_incorrect():
    """Sans marqueur exploitable, on ne crédite JAMAIS un point au hasard."""
    import unittest.mock as mock
    with mock.patch.object(locomo, "_llm_chat", return_value="je ne peux pas trancher"):
        assert locomo.judge("q", "gold", "hyp", pace=0) is False


# ─── Comparabilité des deux bras ───

def test_baseline_vectorielle_respecte_le_budget_de_tokens():
    """La baseline doit être tronquée au MÊME budget, sinon la comparaison est biaisée."""
    longue = " ".join(["mot"] * 400)   # bien au-delà du budget
    courte = "un souvenir court"
    client = FakeClient({"memories": [{"content": longue}, {"content": courte}]})

    ctx, tokens = locomo.context_vector(client, "agent", "question", max_tokens=50, top_k=20)

    assert client.last_path == "/retrieve"
    assert tokens <= 50
    assert courte in ctx and longue not in ctx


def test_baseline_utilise_le_meme_estimateur_que_qem():
    """Le décompte doit venir de `estimate_tokens`, celui du collapse Q-EM."""
    contenus = ["premier souvenir de test", "second souvenir de test"]
    client = FakeClient({"memories": [{"content": c} for c in contenus]})

    _, tokens = locomo.context_vector(client, "agent", "q", max_tokens=10_000, top_k=20)

    assert tokens == sum(estimate_tokens(c) for c in contenus)


def test_bras_qem_rapporte_le_decompte_de_l_api():
    client = FakeClient({
        "context_packet": {"facts": ["fait A"], "preferences": ["pref B"], "episodes": []},
        "token_estimate": 42,
    })

    ctx, tokens = locomo.context_qem(client, "agent", "question", max_tokens=1500)

    assert client.last_path == "/context/build"
    assert tokens == 42
    assert "fait A" in ctx and "pref B" in ctx


def test_contexte_vide_si_l_api_echoue():
    """Une API en erreur doit donner un contexte vide explicite, jamais planter le run."""
    assert locomo.context_qem(FakeClient({}, status_code=503), "a", "q", 1500) == ("", 0)
    assert locomo.context_vector(FakeClient({}, status_code=503), "a", "q", 1500, 20) == ("", 0)


# ─── Agrégation des scores ───

def test_rapport_par_bras_calcule_exactitude_et_cout():
    rows = [
        {"category": 1, "correct": True, "context_tokens": 100},
        {"category": 1, "correct": False, "context_tokens": 200},
        {"category": 2, "correct": True, "context_tokens": 300},
    ]
    rapport = locomo._arm_report(rows)

    assert rapport["accuracy_overall"] == pytest.approx(2 / 3, abs=1e-4)
    assert rapport["accuracy_by_category"]["multi-hop"] == 0.5
    assert rapport["accuracy_by_category"]["temporal"] == 1.0
    assert rapport["avg_context_tokens"] == 200.0


def test_rapport_sur_bras_vide_ne_divise_pas_par_zero():
    assert locomo._arm_report([])["accuracy_overall"] == 0.0


# ─── Garde-fou anti-corpus-dégradé ───

def test_compteur_de_repli_intercepte_les_extractions_regex():
    """Le repli du worker doit être compté, sinon un run dégradé passe inaperçu."""
    worker = locomo.worker
    original = worker._heuristic_extract

    with locomo._FallbackCounter() as compteur:
        assert worker._heuristic_extract is not original  # instrumenté
        worker._heuristic_extract("Erreur : traceback à l'import")
        worker._heuristic_extract("Bonne pratique : borner les requêtes")
        assert compteur.count == 2

    # La fonction d'origine est restaurée à la sortie du contexte.
    assert worker._heuristic_extract is original


def test_compteur_de_repli_preserve_le_resultat():
    """L'instrumentation ne doit rien changer à la classification renvoyée."""
    worker = locomo.worker
    attendu = worker._heuristic_extract("Je préfère les réponses courtes")
    with locomo._FallbackCounter():
        obtenu = worker._heuristic_extract("Je préfère les réponses courtes")
    assert obtenu == attendu


def test_compteur_restaure_meme_en_cas_d_exception():
    worker = locomo.worker
    original = worker._heuristic_extract
    with pytest.raises(ValueError):
        with locomo._FallbackCounter():
            raise ValueError("panne pendant l'ingestion")
    assert worker._heuristic_extract is original
