"""Outils MCP des compartiments (v0.4) : l'agent juge où ranger, le moteur range et le dit.

Hérite du test central de la v0.3 (`store_memory` rend son verdict de rangement) : un
moteur de mémoire ne doit jamais laisser croire à un classement qui n'a pas eu lieu.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastmcp", reason="serveur MCP non installé (requirements-dev)")

import apps.mcp.server as mcp_server


class _Reponse:
    def __init__(self, charge, status=200):
        self._charge, self.status_code = charge, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._charge

    @property
    def text(self):
        return str(self._charge)


@pytest.fixture
def http(monkeypatch):
    """Intercepte GET et POST, rejoue une réponse contrôlée, enregistre les appels."""
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_de_test")
    appels: list[dict] = []
    etat = {"charge": {}, "status": 200}

    def _post(url, json=None, headers=None, timeout=None):
        appels.append({"methode": "POST", "url": url, "payload": json})
        return _Reponse(etat["charge"], etat["status"])

    def _get(url, params=None, headers=None, timeout=None):
        appels.append({"methode": "GET", "url": url, "params": params})
        return _Reponse(etat["charge"], etat["status"])

    monkeypatch.setattr(mcp_server.requests, "post", _post)
    monkeypatch.setattr(mcp_server.requests, "get", _get)
    return appels, etat


def _outil(nom):
    """Fonction Python sous-jacente à un outil MCP (FastMCP enveloppe l'original)."""
    fn = getattr(mcp_server, nom)
    return getattr(fn, "fn", fn)


# ─── 1. store_memory : l'agent choisit, le moteur range et le DIT ────────────

def test_store_memory_annonce_le_compartiment(http):
    appels, etat = http
    etat["charge"] = {"status": "created", "memory_id": "m1", "compartment": "alertes",
                      "region": "Amygdale", "compartment_origin": "agent"}
    message = _outil("store_memory")("erreur corrigee", "alertes")
    assert appels[0]["payload"]["compartment"] == "alertes"
    assert "'alertes' (Amygdale)" in message
    assert "[INFO]" not in message


def test_store_memory_sans_compartiment_laisse_classer_le_moteur(http):
    """Compartiment omis : rien n'est envoyé, le moteur classe et l'agent le sait."""
    appels, etat = http
    etat["charge"] = {"status": "created", "memory_id": "m1", "compartment": "savoir_faire",
                      "region": "Cervelet", "compartment_origin": "moteur",
                      "classification_similarity": 0.71, "requested": None}
    message = _outil("store_memory")("toujours borner les requetes")
    assert "compartment" not in appels[0]["payload"]
    assert "aucun compartiment indique" in message and "0.71" in message


def test_store_memory_signale_un_nom_inconnu(http):
    _, etat = http
    etat["charge"] = {"status": "created", "memory_id": "m1", "compartment": "savoirs",
                      "region": "Lobe temporal", "compartment_origin": "moteur",
                      "classification_similarity": 0.4, "requested": "clients_paca"}
    message = _outil("store_memory")("x", "clients_paca")
    assert "'clients_paca' n'est pas un compartiment" in message


def test_store_memory_signale_un_ancien_nom(http):
    _, etat = http
    etat["charge"] = {"status": "created", "memory_id": "m1", "compartment": "reflexes",
                      "region": "Tronc cérébral", "compartment_origin": "alias",
                      "requested": "preference"}
    message = _outil("store_memory")("x", "preference")
    assert "ancien nom" in message and "'reflexes'" in message


def test_store_memory_refuse_les_croyances(http):
    """Les croyances passent par le garde-fou de note_belief, jamais par store_memory."""
    appels, _ = http
    message = _outil("store_memory")("Jimmy aime X", "croyances")
    assert message.startswith("[REFUSE]") and "note_belief" in message
    assert appels == []


def _outils():
    import asyncio
    return asyncio.run(mcp_server.mcp.get_tools())


def test_la_description_de_store_memory_liste_les_compartiments():
    """La fiche montrée à l'agent DÉRIVE du cœur : jamais une liste à la main."""
    from synaptiq_core.compartments import COMPARTIMENTS
    description = _outils()["store_memory"].description
    for c in COMPARTIMENTS:
        if c.ecriture_directe:
            assert f"'{c.nom}'" in description
    assert "'croyances' (" not in description


# ─── 2. list_compartments ────────────────────────────────────────────────────

def test_list_compartments(http):
    appels, etat = http
    etat["charge"] = {"compartments": [
        {"name": "reflexes", "region": "Tronc cérébral", "description": "regles",
         "forgetting": "aucun", "memory_count": 3, "always_injected": True,
         "direct_write": True, "max_share": 0.2},
        {"name": "croyances", "region": "Cortex préfrontal", "description": "hypotheses",
         "forgetting": "aucun", "memory_count": 0, "always_injected": False,
         "direct_write": False, "max_share": 0.15},
    ]}
    sortie = _outil("list_compartments")()
    assert appels[0]["methode"] == "GET" and appels[0]["url"].endswith("/v1/compartments")
    assert "injecte a chaque contexte" in sortie and "max 20 %" in sortie
    assert "via note_belief seulement" in sortie


# ─── 3. Filtres de lecture ───────────────────────────────────────────────────

def test_recall_memories_transmet_le_filtre(http):
    appels, etat = http
    etat["charge"] = {"memories": []}
    _outil("recall_memories")("q", compartments=["alertes"])
    assert appels[0]["payload"]["compartments"] == ["alertes"]


def test_recall_memories_omet_le_filtre_quand_il_est_absent(http):
    appels, etat = http
    etat["charge"] = {"memories": []}
    _outil("recall_memories")("q")
    assert "compartments" not in appels[0]["payload"]


def test_build_context_transmet_le_filtre(http):
    appels, etat = http
    etat["charge"] = {"token_estimate": 0, "context_packet": {}}
    _outil("build_context")("t", "q", compartments=["savoir_faire"])
    assert appels[0]["payload"]["constraints"]["compartments"] == ["savoir_faire"]


# ─── 4. Plus aucun outil de création ou de fusion ────────────────────────────

def test_les_outils_de_collections_ont_disparu():
    outils = _outils()
    assert not {"create_collection", "merge_collections", "list_collections"} & set(outils)
    assert "list_compartments" in outils


def test_list_compartments_n_expose_pas_agent_id():
    assert "agent_id" not in _outils()["list_compartments"].parameters.get("properties", {})


# ─── 6. Projet (lot B) et refus lisibles ─────────────────────────────────────

def test_store_memory_transmet_le_projet(http):
    appels, etat = http
    etat["charge"] = {"status": "created", "memory_id": "m1", "compartment": "savoirs", "region": "Lobe temporal", "compartment_origin": "agent", "project": "emile"}
    message = _outil("store_memory")("x", "savoirs", project="emile")
    assert appels[0]["payload"]["project"] == "emile"
    assert "projet 'emile'" in message


def test_sans_projet_le_souvenir_est_global(http, monkeypatch):
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_PROJECT", None)
    appels, etat = http
    etat["charge"] = {"status": "created", "memory_id": "m1", "compartment": "savoirs", "region": "Lobe temporal", "compartment_origin": "agent", "project": None}
    message = _outil("store_memory")("x", "savoirs")
    assert "project" not in appels[0]["payload"]
    assert "global" in message


def test_le_projet_par_defaut_du_serveur_s_applique(http, monkeypatch):
    """`SYNAPTIQ_PROJECT` (déclaré par dépôt) évite à l'agent de répéter le projet."""
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_PROJECT", "synaptiq")
    appels, etat = http
    etat["charge"] = {"memories": []}
    _outil("recall_memories")("q")
    assert appels[0]["payload"]["project"] == "synaptiq"


def test_global_explicite_neutralise_le_projet_par_defaut(http, monkeypatch):
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_PROJECT", "synaptiq")
    appels, etat = http
    etat["charge"] = {"token_estimate": 0, "context_packet": {}}
    _outil("build_context")("t", "q", project="global")
    assert "project" not in appels[0]["payload"]["constraints"]


def test_un_422_dit_quel_champ_est_refuse(http):
    """Incident du 22/09 : un 422 opaque cachait `memory_type` envoyé au lieu de `type`."""
    _, etat = http
    etat["status"] = 422
    etat["charge"] = {"detail": [{"loc": ["body", "project"], "msg": "Nom de projet invalide"}]}
    message = _outil("store_memory")("x", project="mon projet")
    assert message == "[REFUSE] project: Nom de projet invalide"



# ─── 7. Croyances (compartiment `croyances`, lot C) ──────────────────────────────

def test_note_belief_ecrit_une_croyance(http):
    appels, etat = http
    etat["charge"] = {"status": "created", "memory_id": "b1", "compartment": "croyances", "region": "Cortex préfrontal"}
    message = _outil("note_belief")("Jimmy decide vite", confidence=0.8, evidence_ids=["m1"])
    payload = appels[0]["payload"]
    assert payload["compartment"] == "croyances" and payload["about"] == "user"
    assert payload["evidence"] == ["m1"]
    assert "hypothese" in message


def test_note_belief_sur_les_humains_porte_le_sujet(http):
    appels, etat = http
    etat["charge"] = {"status": "created", "memory_id": "b1", "compartment": "croyances", "region": "Cortex préfrontal"}
    _outil("note_belief")("Regis repond le matin", about="humans")
    assert appels[0]["payload"]["about"] == "humans"


def test_note_belief_transmet_le_refus_du_garde_fou(http):
    _, etat = http
    etat["status"] = 422
    etat["charge"] = {"detail": [{"loc": ["body"], "msg": "Croyance refusée : catégorie sensible"}]}
    assert _outil("note_belief")("x").startswith("[REFUSE]")


def test_list_beliefs_affiche_confiance_et_indices(http):
    appels, etat = http
    etat["charge"] = {"beliefs": [{"id": "b1", "about": "user", "content": "Jimmy decide vite",
                                   "confidence": 0.8, "evidence": ["m1", "m2"]}]}
    sortie = _outil("list_beliefs")()
    assert appels[0]["methode"] == "GET"
    assert "confiance 0.8, 2 indice(s)" in sortie and "b1" in sortie


def test_contest_belief_appelle_l_endpoint_dedie(http):
    appels, etat = http
    etat["charge"] = {"status": "contested"}
    _outil("contest_belief")("b1", reason="faux")
    assert appels[0]["url"].endswith("/v1/beliefs/b1/contest")
