"""Tests unitaires du SDK Python (audit F18 : le SDK n'avait aucun test).

Le SDK est le contrat que voient les intégrateurs. Ce qui est verrouillé ici : les URLs
versionnées, la propagation de l'en-tête d'authentification, la transmission de
`idempotency_key`/`explain`, et le fait qu'une panne réseau lève une erreur explicite
plutôt qu'un `None` silencieux.
"""
import pytest

from synaptiq_sdk.client import SynaptiqClient


class _Reponse:
    def __init__(self, charge, status=200):
        self._charge = charge
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._charge


@pytest.fixture
def appels(monkeypatch):
    enregistres = []

    def _requete(url, json=None, headers=None, timeout=None, params=None, **kw):
        enregistres.append({"url": url, "payload": json, "headers": headers,
                            "timeout": timeout, "params": params})
        return _Reponse({"status": "ok", "memories": [], "memory_id": "mem-1",
                         "compartments": []})

    import synaptiq_sdk.client as module
    monkeypatch.setattr(module.requests, "post", _requete)
    monkeypatch.setattr(module.requests, "get", _requete)
    return enregistres


def test_url_de_base_normalisee():
    """Une barre oblique finale ne doit pas produire d'URL à double barre."""
    client = SynaptiqClient(base_url="http://exemple:8000/")
    assert client.base_url == "http://exemple:8000"


def test_sans_cle_aucun_en_tete_d_auth():
    assert SynaptiqClient().headers == {}


def test_la_cle_est_propagee_en_bearer(appels):
    client = SynaptiqClient(api_key="sk-test")
    client.capture(agent_id="a", session_id="s", content="c")
    assert appels[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_capture_appelle_events_versionne(appels):
    client = SynaptiqClient()
    client.capture(agent_id="agentA", session_id="s1", content="une interaction",
                   idempotency_key="evt-1")
    assert appels[0]["url"].endswith("/v1/events")
    charge = appels[0]["payload"]
    assert charge["agent_id"] == "agentA"
    assert charge["content"] == "une interaction"
    assert charge["idempotency_key"] == "evt-1"


def test_build_context_transmet_budget_et_explain(appels):
    client = SynaptiqClient()
    client.build_context(agent_id="agentA", session_id="s1", task="t", query="q",
                         max_tokens=500, explain=True)
    charge = appels[0]["payload"]
    assert appels[0]["url"].endswith("/v1/context/build")
    assert charge["constraints"]["max_tokens"] == 500
    assert charge["explain"] is True


def test_build_context_tous_les_compartiments_par_defaut(appels):
    """Rien n'est imposé : le serveur prend les 9 compartiments."""
    SynaptiqClient().build_context(agent_id="a", session_id="s", task="t", query="q")
    assert appels[0]["payload"]["constraints"] == {"max_tokens": 1200}


def test_panne_reseau_leve_une_erreur_explicite(monkeypatch):
    """Une erreur silencieuse ferait écrire l'agent dans le vide sans qu'il le sache."""
    import synaptiq_sdk.client as module

    def _casse(*a, **kw):
        raise ConnectionError("API injoignable")

    monkeypatch.setattr(module.requests, "post", _casse)
    with pytest.raises(RuntimeError, match="injoignable"):
        SynaptiqClient().capture(agent_id="a", session_id="s", content="c")


def test_health_ne_leve_jamais(monkeypatch):
    """`health()` sert justement à diagnostiquer une panne : il doit la RETOURNER."""
    import synaptiq_sdk.client as module

    def _casse(*a, **kw):
        raise ConnectionError("injoignable")

    monkeypatch.setattr(module.requests, "get", _casse)
    resultat = SynaptiqClient().health()
    assert resultat["status"] == "unhealthy"
    assert "injoignable" in resultat["error"]


def test_les_appels_ont_un_timeout(appels):
    """Sans timeout, un agent se bloquerait indéfiniment sur une API muette."""
    SynaptiqClient().capture(agent_id="a", session_id="s", content="c")
    assert appels[0]["timeout"] is not None


# ─── Compartiments : 9 régions fixes, l'agent choisit où ranger ─────────────

def test_les_compartiments_sont_omis_quand_absents(appels):
    """Une liste vide serait refusée ; l'absence doit tout balayer.

    C'est la distinction que `None` porte et qu'une valeur par défaut `[]` détruirait.
    """
    client = SynaptiqClient()
    client.build_context(agent_id="a", session_id="s", task="t", query="q")
    client.retrieve(agent_id="a", query="q")
    assert "compartments" not in appels[0]["payload"]["constraints"]
    assert "compartments" not in appels[1]["payload"]


def test_les_compartiments_sont_transmis_quand_fournis(appels):
    client = SynaptiqClient()
    client.build_context(agent_id="a", session_id="s", task="t", query="q",
                         compartments=["alertes"])
    client.retrieve(agent_id="a", query="q", compartments=["alertes"])
    assert appels[0]["payload"]["constraints"]["compartments"] == ["alertes"]
    assert appels[1]["payload"]["compartments"] == ["alertes"]


def test_store_memory_compartiment_facultatif(appels):
    client = SynaptiqClient()
    client.store_memory(agent_id="a", content="x")
    client.store_memory(agent_id="a", content="y", compartment="savoir_faire")
    assert "compartment" not in appels[0]["payload"]
    assert appels[1]["payload"]["compartment"] == "savoir_faire"
    assert "type" not in appels[1]["payload"] and "subtype" not in appels[1]["payload"]


def test_note_belief_vise_le_compartiment_croyances(appels):
    SynaptiqClient().note_belief(agent_id="a", content="x", about="humans")
    assert appels[0]["payload"]["compartment"] == "croyances"
    assert appels[0]["payload"]["about"] == "humans"


def test_list_compartments_est_un_get_versionne(appels):
    SynaptiqClient().list_compartments(agent_id="agentA")
    assert appels[0]["url"].endswith("/v1/compartments")
    assert appels[0]["params"] == {"agent_id": "agentA"}


def test_une_panne_sur_les_compartiments_leve_une_erreur_explicite(monkeypatch):
    """Comme le reste du SDK : jamais de `None` silencieux sur une panne réseau."""
    import synaptiq_sdk.client as module

    def _casse(*a, **kw):
        raise ConnectionError("injoignable")

    monkeypatch.setattr(module.requests, "get", _casse)
    with pytest.raises(RuntimeError, match="compartiments"):
        SynaptiqClient().list_compartments(agent_id="a")
