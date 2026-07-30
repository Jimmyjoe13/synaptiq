"""Tests unitaires des permissions de clé API et du périmètre d'agents.

Verrouillent le correctif du 29/07 (audit F2) : avant celui-ci, une clé API valait TOUS les
droits sur TOUS les agents de son tenant, et `agent_id` — fourni par l'appelant, et côté
MCP choisi par le LLM lui-même — n'était vérifié par personne.
"""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "core"))

from apps.api.main import AuthContext, require_scope, resolve_agent

# ─── Permissions (scopes) ────────────────────────────────────────────────────

def test_scopes_par_defaut_lecture_ecriture_sans_admin():
    """Une clé sans scopes explicites (colonne à son DEFAULT) lit et écrit, mais ne purge pas."""
    auth = AuthContext(tenant_id="t")
    require_scope(auth, "read")
    require_scope(auth, "write")
    with pytest.raises(HTTPException) as exc:
        require_scope(auth, "admin")
    assert exc.value.status_code == 403


def test_cle_lecture_seule_ne_peut_pas_ecrire():
    auth = AuthContext(tenant_id="t", scopes=["read"])
    require_scope(auth, "read")
    with pytest.raises(HTTPException) as exc:
        require_scope(auth, "write")
    assert exc.value.status_code == 403
    # Le message nomme les scopes réellement portés : diagnostic immédiat côté appelant.
    assert "read" in exc.value.detail


def test_scope_admin_accorde_explicitement():
    auth = AuthContext(tenant_id="t", scopes=["read", "write", "admin"])
    require_scope(auth, "admin")  # ne lève pas


def test_sans_auth_lecture_et_ecriture_sont_permises():
    """Mode dev (SYNAPTIQ_AUTH_REQUIRED=false) : read et write passent sans clé."""
    for scope in ("read", "write"):
        require_scope(None, scope)


def test_sans_auth_admin_reste_refuse():
    """RÉGRESSION 30/07 : `admin` n'est JAMAIS implicite, même sans authentification.

    Ce test asseyait l'inverse jusqu'ici (« sans auth, tout est permis ») et c'est exactement
    ce qui rendait la purge de l'instance de production joignable sans clé : un `DELETE
    /v1/memories?confirm=default` anonyme répondait 400, pas 401. Le nom du tenant était le
    seul secret — et le message d'erreur 400 le donnait.

    Un mode de confort ne doit pas ouvrir un endpoint irréversible.
    """
    with pytest.raises(HTTPException) as exc:
        require_scope(None, "admin")
    assert exc.value.status_code == 403
    # Le message doit dire quoi faire, pas seulement refuser.
    assert "create_api_key" in exc.value.detail


# ─── Périmètre d'agents ──────────────────────────────────────────────────────

def test_agent_hors_perimetre_refuse():
    """Le cœur de F2 : une clé bornée à agentA ne peut pas agir comme agentB."""
    auth = AuthContext(tenant_id="t", agent_scope=["agentA"])
    assert resolve_agent(auth, "agentA") == "agentA"
    with pytest.raises(HTTPException) as exc:
        resolve_agent(auth, "agentB")
    assert exc.value.status_code == 403
    assert "agentB" in exc.value.detail


def test_agent_scope_absent_autorise_tous_les_agents():
    """Compatibilité ascendante : agent_scope NULL en base = tous les agents du tenant."""
    auth = AuthContext(tenant_id="t", agent_scope=None)
    assert resolve_agent(auth, "n_importe_quel_agent") == "n_importe_quel_agent"


def test_agent_scope_multiple():
    auth = AuthContext(tenant_id="t", agent_scope=["agentA", "agentB"])
    assert resolve_agent(auth, "agentB") == "agentB"
    with pytest.raises(HTTPException):
        resolve_agent(auth, "agentC")


def test_sans_auth_aucun_controle_d_agent():
    assert resolve_agent(None, "agentX") == "agentX"
