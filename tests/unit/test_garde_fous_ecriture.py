"""Garde-fous ajoutés le 11/08 après audit — pannes silencieuses.

Dans le job CI BLOQUANT (tests/unit/, aucune infrastructure requise) : un garde-fou de
sécurité vérifié dans un job `continue-on-error` ne garde rien.

1. `DELETE /v1/memories` sans `agent_id` ne consultait PAS le périmètre de la clé.
4. `create_api_key.py` produisait par défaut une clé valable pour tous les agents.

(Les garde-fous 2, 3 et 5 portaient sur le registre de collections, supprimé en v0.4 au
profit des compartiments fixes : `tests/unit/test_compartments.py`.)
"""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))

from apps.api.main import (
    AuthContext,
    purge_memories,
)

# ─── 1. La purge ne doit pas déborder du périmètre de la clé ─────────────────

def test_purge_globale_refusee_a_une_cle_bornee_a_un_agent():
    """LE cas grave : une clé admin bornée à agentA purgeait TOUT le tenant.

    `resolve_agent` n'était appelé que `if agent_id`. Omettre le filtre élargissait donc
    l'effet de la requête au lieu de le restreindre — et la suppression est physique.
    """
    auth = AuthContext(tenant_id="t", scopes=["read", "write", "admin"],
                       agent_scope=["agentA"])
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id=None, confirm="t", auth=auth)
    assert exc.value.status_code == 403
    # Le message doit nommer le périmètre ET la sortie : sans quoi l'appelant relance à
    # l'identique.
    assert "agentA" in exc.value.detail
    assert "agent_id" in exc.value.detail


def test_purge_globale_refusee_avant_le_controle_de_confirmation():
    """Une autorisation manquante prime sur une confirmation manquante.

    Sinon le 400 « rappeler avec ?confirm=… » servirait de mode d'emploi pour une purge
    que la clé n'a de toute façon pas le droit de faire.
    """
    auth = AuthContext(tenant_id="t", scopes=["admin"], agent_scope=["agentA"])
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id=None, confirm=None, auth=auth)
    assert exc.value.status_code == 403


def test_purge_d_un_agent_hors_perimetre_toujours_refusee():
    """Non-régression du contrôle existant (F2 du 29/07)."""
    auth = AuthContext(tenant_id="t", scopes=["admin"], agent_scope=["agentA"])
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id="agentB", confirm="t", auth=auth)
    assert exc.value.status_code == 403


def test_purge_globale_reste_possible_pour_une_cle_sans_perimetre(monkeypatch):
    """Le cas légitime ne doit pas être fermé : clé admin non bornée, purge du tenant.

    Elle échoue plus loin (pas de pool en test unitaire) : ce qui compte est que le refus
    ne soit PAS un 403 de périmètre. Le pool est forcé à None : la suite d'intégration,
    qui tourne dans le même process, en crée un vrai — sans ce forçage l'assertion
    « pool absent » ne tiendrait pas en run complet.
    """
    import apps.api.main as main

    monkeypatch.setattr(main, "db_pool", None)
    auth = AuthContext(tenant_id="t", scopes=["admin"], agent_scope=None)
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id=None, confirm="t", auth=auth)
    assert exc.value.status_code == 503        # pool absent, donc le contrôle est passé


def test_purge_confirmation_toujours_exigee_hors_perimetre_borne():
    auth = AuthContext(tenant_id="t", scopes=["admin"])
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id=None, confirm="mauvais", auth=auth)
    assert exc.value.status_code == 400


# ─── 4. Le périmètre d'agents d'une clé API est un choix explicite ───────────

def _perimetre(argv):
    """Résout le périmètre d'agents à partir d'une ligne de commande (sans base)."""
    import create_api_key

    parser = create_api_key.construire_parseur()
    args = parser.parse_args(argv)
    return create_api_key.resoudre_perimetre(args, parser)


def test_cle_sans_perimetre_refusee(capsys):
    """L'écart le plus net entre la promesse produit et le comportement réel.

    `--agents` valait None par défaut -> `agent_scope IS NULL` -> `resolve_agent` ne
    vérifiait rien -> la clé valait pour TOUS les agents de l'instance.
    """
    with pytest.raises(SystemExit) as exc:
        _perimetre(["--name", "sans-perimetre"])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "--agents" in message and "--all-agents" in message
    assert "TOUS" in message                   # le risque est nommé, pas seulement refusé


def test_perimetre_explicite():
    assert _perimetre(["--name", "k", "--agents", "agentA", "agentB"]) == ["agentA", "agentB"]


def test_echappatoire_mono_agent():
    """Le cas légitime (instance mono-agent) reste possible, mais assumé."""
    assert _perimetre(["--name", "k", "--all-agents"]) is None


def test_les_deux_drapeaux_ensemble_sont_refuses(capsys):
    with pytest.raises(SystemExit):
        _perimetre(["--name", "k", "--agents", "agentA", "--all-agents"])
    assert "s'excluent" in capsys.readouterr().err
