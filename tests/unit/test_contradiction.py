"""Tests unitaires du juge de contradiction (sélection et fail-closed).

Invariant à ne jamais casser : **un juge en panne n'archive rien**. Une erreur réseau, un
JSON illisible ou une réponse ambiguë doivent tous produire « pas de contradiction », faute
de quoi une indisponibilité du LLM se traduirait par une destruction de préférences.
"""
import pytest

from synaptiq_core.contradiction import (
    LLMContradictionJudge,
    get_contradiction_judge,
    no_judge,
)


@pytest.fixture(autouse=True)
def _vide_le_cache():
    """`get_contradiction_judge` est mis en cache pour le process : purge entre les tests."""
    get_contradiction_judge.cache_clear()
    yield
    get_contradiction_judge.cache_clear()


# ─── Sélection du juge ───────────────────────────────────────────────────────

def test_sans_llm_le_juge_est_neutre(monkeypatch):
    """LLM_PROVIDER=mock -> aucun archivage automatique possible."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("CONTRADICTION_JUDGE", raising=False)
    assert get_contradiction_judge() is no_judge


def test_mode_off_desactive_meme_avec_un_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "sk-vraie-cle")
    monkeypatch.setenv("CONTRADICTION_JUDGE", "off")
    assert get_contradiction_judge() is no_judge


def test_endpoint_local_sans_cle_active_le_juge(monkeypatch):
    """LM Studio / Ollama en local n'exigent aucune clé : le juge doit s'activer."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("CONTRADICTION_JUDGE", raising=False)
    assert isinstance(get_contradiction_judge(), LLMContradictionJudge)


def test_provider_distant_sans_cle_valide_reste_neutre(monkeypatch):
    """Placeholder de clé non remplacé -> pas de juge, donc pas d'archivage."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "your_api_key_here")
    monkeypatch.delenv("CONTRADICTION_JUDGE", raising=False)
    assert get_contradiction_judge() is no_judge


# ─── Verdict du juge LLM ─────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, content, status=200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _juge_repondant(monkeypatch, content=None, exception=None, status=200):
    def _post(url, **kwargs):
        if exception is not None:
            raise exception
        return _FakeResponse(content, status)

    monkeypatch.setattr("synaptiq_core.contradiction.requests.post", _post)
    return LLMContradictionJudge(base_url="http://x/v1", model="m")


def test_verdict_oui(monkeypatch):
    juge = _juge_repondant(monkeypatch, content="YES")
    assert juge("Jimmy préfère MySQL", "Jimmy préfère PostgreSQL") is True


def test_verdict_non(monkeypatch):
    juge = _juge_repondant(monkeypatch, content="NO")
    assert juge("mails courts", "mails en français") is False


def test_reponse_bavarde_toleree(monkeypatch):
    """Un modèle qui ne respecte pas « un seul mot » ne doit pas fausser le verdict."""
    juge = _juge_repondant(monkeypatch, content="yes, they contradict each other")
    assert juge("a", "b") is True


def test_reponse_ambigue_ne_archive_pas(monkeypatch):
    """Tout ce qui ne commence pas par YES est traité comme « pas de contradiction »."""
    juge = _juge_repondant(monkeypatch, content="peut-être, cela dépend du contexte")
    assert juge("a", "b") is False


def test_erreur_reseau_fail_closed(monkeypatch):
    juge = _juge_repondant(monkeypatch, exception=ConnectionError("endpoint injoignable"))
    assert juge("a", "b") is False


def test_erreur_http_fail_closed(monkeypatch):
    juge = _juge_repondant(monkeypatch, content="YES", status=500)
    assert juge("a", "b") is False


def test_reponse_malformee_fail_closed(monkeypatch):
    class _Cassee(_FakeResponse):
        def json(self):
            return {"inattendu": True}

    monkeypatch.setattr("synaptiq_core.contradiction.requests.post",
                        lambda url, **kw: _Cassee("", 200))
    juge = LLMContradictionJudge(base_url="http://x/v1", model="m")
    assert juge("a", "b") is False
