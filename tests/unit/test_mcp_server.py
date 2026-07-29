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
    """Intercepte les appels HTTP sortants et rejoue une réponse contrôlée."""
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
    def _post_casse(*a, **kw):
        raise ConnectionError("API injoignable")

    monkeypatch.setattr(mcp_server.requests, "post", _post_casse)
    resultat = mcp_server.store_memory.fn(content="x", memory_type="semantic")
    assert resultat.startswith("[ERROR]")
    assert "injoignable" in resultat
