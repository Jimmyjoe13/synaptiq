"""Garde-fous ajoutés le 11/08 après audit — quatre pannes silencieuses.

Les quatre sont dans le job CI BLOQUANT (tests/unit/, aucune infrastructure requise) :
un garde-fou de sécurité vérifié dans un job `continue-on-error` ne garde rien.

1. `DELETE /v1/memories` sans `agent_id` ne consultait PAS le périmètre de la clé.
2. `POST /v1/collections` acceptait une `packet_key` canonique (`rules`…).
3. `POST /v1/memories` répondait `201 created` pour une ligne jamais écrite.
4. `create_api_key.py` produisait par défaut une clé valable pour tous les agents.
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
    CLES_PAQUET_CANONIQUES,
    AuthContext,
    CollectionInput,
    _charger_registre_isole,
    create_collection,
    purge_memories,
)
from synaptiq_core.collections import SYSTEM_COLLECTIONS

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


def test_purge_globale_reste_possible_pour_une_cle_sans_perimetre():
    """Le cas légitime ne doit pas être fermé : clé admin non bornée, purge du tenant.

    Elle échoue plus loin (pas de pool en test unitaire) : ce qui compte est que le refus
    ne soit PAS un 403 de périmètre.
    """
    auth = AuthContext(tenant_id="t", scopes=["admin"], agent_scope=None)
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id=None, confirm="t", auth=auth)
    assert exc.value.status_code == 503        # pool absent, donc le contrôle est passé


def test_purge_confirmation_toujours_exigee_hors_perimetre_borne():
    auth = AuthContext(tenant_id="t", scopes=["admin"])
    with pytest.raises(HTTPException) as exc:
        purge_memories(agent_id=None, confirm="mauvais", auth=auth)
    assert exc.value.status_code == 400


# ─── 2. Aucune collection d'agent ne se greffe sur une section canonique ─────

def test_les_cles_canoniques_derivent_des_collections_systeme():
    """Interdiction du CLAUDE.md §3 : pas de seconde liste écrite à la main."""
    assert CLES_PAQUET_CANONIQUES == {c.packet_key for c in SYSTEM_COLLECTIONS}


@pytest.mark.parametrize("cle", sorted(c.packet_key for c in SYSTEM_COLLECTIONS))
def test_packet_key_canonique_refusee(cle):
    """`packet_key="rules"` servait les souvenirs de l'agent dans la rubrique Règles.

    Pas un défaut d'isolation (c'est son propre paquet), mais une injection STRUCTURELLE :
    un contenu mémorisé, éventuellement dicté par un tiers, se retrouvait rangé dans la
    section la plus impérative du prompt système.
    """
    payload = CollectionInput(agent_id="agentA", name="un_rayon", family="semantic",
                              description="Une description suffisamment longue.",
                              packet_key=cle)
    with pytest.raises(HTTPException) as exc:
        create_collection(payload, auth=None)
    assert exc.value.status_code == 422
    assert cle in exc.value.detail


def test_nom_valant_une_cle_canonique_refuse_aussi():
    """À défaut de `packet_key`, c'est le NOM qui la porte.

    `is_canonical` protège le nom d'une collection (`rule`), pas la clé de section
    (`rules`) : contrôler seulement le champ explicite laissait la porte grande ouverte.
    """
    payload = CollectionInput(agent_id="agentA", name="rules", family="semantic",
                              description="Une description suffisamment longue.")
    with pytest.raises(HTTPException) as exc:
        create_collection(payload, auth=None)
    assert exc.value.status_code == 422


def test_packet_key_libre_toujours_acceptee():
    """Le cas normal ne doit pas être fermé : plusieurs collections peuvent partager une
    section, tant qu'elle n'est pas canonique."""
    payload = CollectionInput(agent_id="agentA", name="clients_paca", family="semantic",
                              description="Les clients de la région PACA.",
                              packet_key="clients")
    with pytest.raises(HTTPException) as exc:
        create_collection(payload, auth=None)
    assert exc.value.status_code == 503        # pool absent : le contrôle est passé


# ─── 3. Un registre illisible n'avorte pas la transaction d'écriture ─────────

class CurseurTransactionnel:
    """Curseur qui reproduit la sémantique psycopg2/PostgreSQL des transactions avortées.

    Une fois une requête en échec, TOUTE requête suivante est refusée
    (`InFailedSqlTransaction`) jusqu'à un `ROLLBACK TO SAVEPOINT` — mais `commit()` est,
    lui, traduit en ROLLBACK silencieux par le serveur. C'est ce silence-là qui rendait la
    perte d'écriture invisible.
    """

    def __init__(self, echouer_sur: str) -> None:
        self.echouer_sur = echouer_sur
        self.avortee = False
        self.executees: list[str] = []

    def execute(self, sql, params=None):
        self.executees.append(sql.strip().split("\n")[0])
        if sql.strip().upper().startswith("ROLLBACK TO SAVEPOINT"):
            self.avortee = False
            return
        if self.avortee:
            raise RuntimeError("current transaction is aborted, commands ignored")
        if self.echouer_sur in sql:
            self.avortee = True
            erreur = RuntimeError('relation "memory_collections" does not exist')
            erreur.pgcode = "42P01"
            raise erreur

    def fetchall(self):
        return []

    def fetchone(self):
        return None


def test_registre_illisible_ne_laisse_pas_la_transaction_avortee():
    """LE correctif : après le repli, la transaction doit rester utilisable.

    Sans le SAVEPOINT, l'INSERT était déjà passé mais `conn.commit()` était exécuté comme
    un ROLLBACK par PostgreSQL, sans exception : l'API répondait `201 created` pour une
    ligne inexistante. Sur une collection sans intrication, aucune requête ne suivait —
    donc rien ne révélait la perte.
    """
    cur = CurseurTransactionnel(echouer_sur="memory_collections")

    registre = _charger_registre_isole(cur, "t", "agentA")

    # Repli sur les collections système, comme le veut `charger_registre`…
    assert len(registre.collections) == len(SYSTEM_COLLECTIONS)
    # …mais la transaction est de nouveau saine : c'est ce qui manquait.
    assert cur.avortee is False
    cur.execute("INSERT INTO memories DEFAULT VALUES")      # ne doit plus lever
    assert any(s.startswith("SAVEPOINT") for s in cur.executees)
    assert any(s.startswith("ROLLBACK TO SAVEPOINT") for s in cur.executees)


def test_savepoint_libere_sur_le_chemin_nominal():
    """Un registre lisible ne doit laisser aucun savepoint ouvert derrière lui."""
    cur = CurseurTransactionnel(echouer_sur="__jamais__")

    _charger_registre_isole(cur, "t", "agentA")

    assert cur.executees[0] == "SAVEPOINT synaptiq_registre"
    assert cur.executees[-1] == "RELEASE SAVEPOINT synaptiq_registre"
    assert cur.avortee is False


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
