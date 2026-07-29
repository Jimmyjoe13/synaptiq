"""Tests unitaires du serveur MCP — la surface exposée aux agents (audit F18).

136 lignes sans aucun test jusqu'ici, alors que c'est le point d'entrée par lequel un LLM
écrit et lit la mémoire. Le test le plus important est
`test_agent_id_n_est_pas_un_parametre_d_outil` : il verrouille le correctif F2, où
`agent_id` était un paramètre choisi par le modèle lui-même.
"""
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastmcp", reason="serveur MCP non installé (requirements-dev)")

import apps.mcp.server as mcp_server


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
    """Intercepte les appels HTTP sortants et rejoue une réponse contrôlée.

    Fixe aussi une identité : `SYNAPTIQ_AGENT_ID` n'a plus de défaut, les outils refusent
    donc de partir sans elle (c'est le correctif de l'incident du 29/07).
    """
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_de_test")
    enregistres = []
    reponse = {"charge": {"memory_id": "mem-1", "memories": [], "context_packet": {}}}

    def _post(url, json=None, headers=None, timeout=None):
        enregistres.append({"url": url, "payload": json, "headers": headers})
        return _Reponse(reponse["charge"])

    monkeypatch.setattr(mcp_server.requests, "post", _post)
    enregistres_et_reponse = (enregistres, reponse)
    return enregistres_et_reponse


# ─── Le correctif de sécurité F2 ─────────────────────────────────────────────

@pytest.mark.parametrize("outil", ["store_memory", "recall_memories", "build_context"])
def test_agent_id_n_est_pas_un_parametre_d_outil(outil):
    """RÉGRESSION F2 : aucun outil ne doit exposer `agent_id`.

    Tant qu'il était paramètre, l'identité mémoire était choisie par le LLM : il suffisait
    d'une autre chaîne dans un appel d'outil pour lire la mémoire d'un autre agent.
    """
    fonction = getattr(mcp_server, outil)
    # `@mcp.tool()` enveloppe la fonction : on inspecte la fonction sous-jacente si besoin.
    cible = getattr(fonction, "fn", getattr(fonction, "__wrapped__", fonction))
    parametres = inspect.signature(cible).parameters
    assert "agent_id" not in parametres, f"{outil} expose encore agent_id au modèle"


def test_l_identite_vient_de_l_environnement(appels, monkeypatch):
    enregistres, _ = appels
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_configure")
    mcp_server.store_memory.fn(content="un fait", memory_type="semantic")
    assert enregistres[0]["payload"]["agent_id"] == "agent_configure"


# ─── Identité obligatoire (incident de production du 29/07) ───────────────────

def test_aucun_defaut_d_identite():
    """RÉGRESSION : le défaut `qwen_code_agent` a causé une panne en production.

    Le serveur lisait une partition vide et répondait « aucun souvenir trouvé », sans
    erreur — symptôme indiscernable d'une mémoire réellement vide.
    """
    import inspect
    source = inspect.getsource(mcp_server)
    assert 'os.getenv("SYNAPTIQ_AGENT_ID", "qwen_code_agent")' not in source


def test_require_agent_id_echoue_avec_un_message_actionnable(monkeypatch):
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "")
    with pytest.raises(RuntimeError) as exc:
        mcp_server.require_agent_id()
    message = str(exc.value)
    assert "SYNAPTIQ_AGENT_ID" in message
    # Le message doit dire QUOI faire, pas seulement ce qui manque.
    assert "env" in message
    assert "agent_id FROM memories" in message


@pytest.mark.parametrize("outil", ["store_memory", "recall_memories", "build_context"])
def test_sans_identite_les_outils_refusent_plutot_que_de_lire_a_cote(outil, monkeypatch):
    """Aucune requête ne doit partir sans identité : mieux vaut une erreur qu'un vide."""
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "")
    partis = []
    monkeypatch.setattr(mcp_server.requests, "post",
                        lambda *a, **kw: partis.append(kw) or _Reponse({}))

    fonction = getattr(mcp_server, outil).fn
    arguments = {
        "store_memory": {"content": "x", "memory_type": "semantic"},
        "recall_memories": {"query": "x"},
        "build_context": {"task": "t", "query": "q"},
    }[outil]
    resultat = fonction(**arguments)
    assert resultat.startswith("[ERROR]")
    assert "SYNAPTIQ_AGENT_ID" in resultat
    assert partis == []          # rien n'a été envoyé à l'API


# ─── Aucun effet de bord à l'import ──────────────────────────────────────────

def test_l_import_ne_demarre_aucun_serveur():
    """RÉGRESSION : `ensure_api_running()` était appelée au niveau module.

    Importer ce module lançait donc un uvicorn — y compris depuis la suite de tests, où
    cela ajoutait plusieurs secondes d'attente et laissait un processus orphelin.
    """
    import inspect
    lignes = inspect.getsource(mcp_server).splitlines()
    # Un appel au niveau module (colonne 0), hors du bloc __main__.
    appels_module = [n for n, ligne in enumerate(lignes)
                     if ligne.startswith("ensure_api_running(")]
    assert appels_module == [], f"appel à l'import en ligne(s) {appels_module}"


def test_le_sous_processus_ne_pollue_pas_stdout():
    """En transport stdio, stdout porte le JSON-RPC : uvicorn ne doit pas y écrire."""
    import inspect
    source = inspect.getsource(mcp_server.ensure_api_running)
    assert "stdout=sortie" in source
    assert "stderr=sortie" in source


def test_ensure_api_running_ne_lance_rien_si_l_api_repond(monkeypatch):
    lances = []
    monkeypatch.setattr(mcp_server.requests, "get", lambda *a, **kw: _Reponse({}, 200))
    monkeypatch.setattr(mcp_server.subprocess, "Popen",
                        lambda *a, **kw: lances.append(a) or None)
    assert mcp_server.ensure_api_running() is True
    assert lances == []


def test_le_demarrage_de_l_api_ne_bloque_pas_le_handshake(monkeypatch):
    """RÉGRESSION : l'attente de l'API retardait le handshake MCP de 14 s.

    Le client MCP a son propre délai d'initialisation, bien plus court : il tuait le
    process, ce qui se lit `exit status 1` sur Windows et faisait échouer le rechargement
    de TOUS les serveurs du client. Un handshake de protocole ne doit jamais attendre une
    tâche annexe.
    """
    monkeypatch.delenv("SYNAPTIQ_AUTOSTART_WAIT_S", raising=False)
    attentes = []

    def _get_injoignable(*a, **kw):
        raise ConnectionError("refuse")

    monkeypatch.setattr(mcp_server.requests, "get", _get_injoignable)
    monkeypatch.setattr(mcp_server.subprocess, "Popen", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_server.time, "sleep", lambda s: attentes.append(s))

    assert mcp_server.ensure_api_running() is False
    assert attentes == [], f"a attendu {sum(attentes)} s au lieu de rendre la main"


def test_l_attente_reste_possible_explicitement(monkeypatch):
    """`SYNAPTIQ_AUTOSTART_WAIT_S` permet de la réactiver (scripts, mise au point)."""
    monkeypatch.setattr(mcp_server.requests, "get",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refuse")))
    monkeypatch.setattr(mcp_server.subprocess, "Popen", lambda *a, **kw: None)
    dormi = []
    monkeypatch.setattr(mcp_server.time, "sleep", lambda s: dormi.append(s))

    assert mcp_server.ensure_api_running(timeout_s=1.0) is False
    assert sum(dormi) > 0      # il a bien patienté


def test_un_premier_appel_trop_tot_reessaie_une_fois(monkeypatch):
    """L'API peut ne pas encore écouter : `_poster` réessaie au lieu d'échouer."""
    tentatives = []

    def _post(url, **kw):
        tentatives.append(url)
        if len(tentatives) == 1:
            raise mcp_server.requests.ConnectionError("pas encore d'ecoute")
        return _Reponse({"memories": []})

    monkeypatch.setattr(mcp_server.requests, "post", _post)
    monkeypatch.setattr(mcp_server.time, "sleep", lambda s: None)
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_de_test")

    resultat = mcp_server.recall_memories.fn(query="q")
    assert len(tentatives) == 2
    assert not resultat.startswith("[ERROR]")


def test_le_retry_ne_masque_pas_une_panne_durable(monkeypatch):
    """Deux échecs de connexion doivent remonter une erreur, pas boucler."""
    tentatives = []

    def _post(url, **kw):
        tentatives.append(url)
        raise mcp_server.requests.ConnectionError("API morte")

    monkeypatch.setattr(mcp_server.requests, "post", _post)
    monkeypatch.setattr(mcp_server.time, "sleep", lambda s: None)
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_de_test")

    resultat = mcp_server.recall_memories.fn(query="q")
    assert len(tentatives) == 2
    assert resultat.startswith("[ERROR]")


# ─── Contrats d'appel ────────────────────────────────────────────────────────

def test_store_memory_appelle_le_bon_endpoint(appels):
    enregistres, _ = appels
    resultat = mcp_server.store_memory.fn(content="un fait", memory_type="semantic",
                                          subtype="fact")
    assert enregistres[0]["url"].endswith("/v1/memories")
    assert enregistres[0]["payload"]["type"] == "semantic"
    assert enregistres[0]["payload"]["subtype"] == "fact"
    assert "mem-1" in resultat


def test_recall_memories_formate_les_resultats(appels):
    enregistres, reponse = appels
    reponse["charge"] = {"memories": [
        {"type": "semantic", "subtype": "preference", "content": "mails courts",
         "confidence": 0.9},
    ]}
    resultat = mcp_server.recall_memories.fn(query="style")
    assert enregistres[0]["url"].endswith("/v1/retrieve")
    assert "mails courts" in resultat
    # Le type est mis en capitales, le sous-type reste tel quel.
    assert "[SEMANTIC / preference]" in resultat


def test_recall_memories_sans_resultat(appels):
    _, reponse = appels
    reponse["charge"] = {"memories": []}
    assert "Aucun souvenir" in mcp_server.recall_memories.fn(query="inconnu")


def test_build_context_aplati_les_7_collections(appels):
    enregistres, reponse = appels
    reponse["charge"] = {
        "token_estimate": 42,
        "context_packet": {"facts": ["un fait"], "preferences": ["une pref"],
                           "errors": ["une erreur"]},
    }
    resultat = mcp_server.build_context.fn(task="tache", query="requete")
    assert enregistres[0]["url"].endswith("/v1/context/build")
    assert "42" in resultat
    assert "[FAITS] un fait" in resultat
    assert "[PREFERENCES] une pref" in resultat
    assert "[ERREURS] une erreur" in resultat


def test_erreur_reseau_rendue_lisible_pour_l_agent(monkeypatch):
    """Un outil MCP ne doit pas lever : l'agent doit recevoir un message exploitable."""
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_de_test")

    def _post_casse(*a, **kw):
        raise ConnectionError("API injoignable")

    monkeypatch.setattr(mcp_server.requests, "post", _post_casse)
    resultat = mcp_server.store_memory.fn(content="x", memory_type="semantic")
    assert resultat.startswith("[ERROR]")
    assert "injoignable" in resultat
