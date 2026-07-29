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


def test_sans_auth_tout_est_permis():
    """Mode dev (SYNAPTIQ_AUTH_REQUIRED=false) : aucun scope à vérifier."""
    for scope in ("read", "write", "admin"):
        require_scope(None, scope)


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
