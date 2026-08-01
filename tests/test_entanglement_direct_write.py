"""L'écriture directe tisse désormais le graphe (lot 1 du 01/08).

Avant ce lot, `_entangle` n'existait que dans `apps/worker/worker.py` : un agent qui écrivait
uniquement par `store_memory` ne construisait AUCUNE arête, et la phase 2 de Q-EM tournait sur
un graphe vide. Sans erreur, sans log. Mesuré sur une instance réelle : 28 souvenirs, 0 arête,
après des semaines d'usage.

Exige Postgres + Redis (marqué integration via conftest). L'embedder `mock` est déterministe :
deux contenus proches produisent des vecteurs proches, ce qui rend le seuil pilotable en test.
"""
import os

import psycopg2
import pytest
from conftest import purge_tenants
from fastapi.testclient import TestClient

import apps.api.main as main
from apps.api.main import app

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
TENANT = "entangle_test_tenant"
AGENT = "agent_entangle"


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True

    def _purge():
        purge_tenants(conn, TENANT)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_collections WHERE tenant_id = %s", (TENANT,))
            conn.commit()

    _purge()
    yield conn
    _purge()
    conn.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_TENANT", TENANT)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    # Seuil très bas : l'embedder mock ne garantit pas 0,7 de cosinus entre deux phrases
    # proches. Ce test vérifie que le CÂBLAGE existe, pas la valeur du défaut.
    monkeypatch.setenv("QEM_ENTANGLE_THRESHOLD", "0.0")
    main.invalidate_auth_cache()
    with TestClient(app) as c:
        yield c


def _ecrire(client, contenu, subtype="fact", type_="semantic"):
    return client.post("/memories", json={"agent_id": AGENT, "type": type_,
                                          "subtype": subtype, "content": contenu})


def _aretes(db, agent=AGENT):
    with db.cursor() as cur:
        cur.execute("""
            SELECT count(*) FROM relationships r
            JOIN memories m ON m.id = r.source_memory_id
            WHERE m.tenant_id = %s AND m.agent_id = %s AND r.relation_type = 'entangled_with'
        """, (TENANT, agent))
        return cur.fetchone()[0]


def test_une_ecriture_directe_tisse_des_aretes(client, db):
    """LE test du lot : `POST /v1/memories` construit du graphe."""
    _ecrire(client, "Le serveur de recette ecoute sur le port 8443.")
    assert _aretes(db) == 0, "le premier souvenir n'a aucun voisin a relier"

    _ecrire(client, "Le serveur de production ecoute sur le port 443.")
    assert _aretes(db) >= 1, "le second souvenir doit etre relie au premier"


def test_le_premier_souvenir_ne_s_intrique_pas_a_lui_meme(client, db):
    _ecrire(client, "Un fait isole, sans voisin possible.")
    with db.cursor() as cur:
        cur.execute("""
            SELECT count(*) FROM relationships r
            JOIN memories m ON m.id = r.source_memory_id
            WHERE m.tenant_id = %s AND r.source_memory_id = r.target_memory_id
        """, (TENANT,))
        assert cur.fetchone()[0] == 0


def test_une_collection_non_structurante_ne_tisse_rien(client, db):
    """Le flag `entangle` de la collection décide, et il est honoré ici.

    `scratch` (famille `working`) est livrée avec `entangle=False` : c'est de la mémoire de
    travail volatile, l'intriquer densifierait le graphe sans gain de pertinence.
    """
    _ecrire(client, "Un premier fait durable, pour avoir un voisin disponible.")
    aretes_avant = _aretes(db)

    _ecrire(client, "Une note de travail jetable.", subtype="scratch", type_="working")

    assert _aretes(db) == aretes_avant, "une collection non structurante ne doit rien tisser"


def test_le_seuil_est_respecte(client, db, monkeypatch):
    """Un seuil inatteignable ne doit produire aucune arête.

    Verrouille que le seuil est bien lu À L'APPEL et non figé à l'import : c'est la convention
    du dépôt, et c'est ce qui rend une étude d'ablation possible sans redéploiement.
    """
    _ecrire(client, "Premier fait, pour peupler la memoire de l'agent.")
    aretes_avant = _aretes(db)

    monkeypatch.setenv("QEM_ENTANGLE_THRESHOLD", "1.1")   # cosinus jamais atteignable
    _ecrire(client, "Second fait, qui ne devrait se relier a rien.")

    assert _aretes(db) == aretes_avant


def test_le_doublon_ne_tisse_pas_de_seconde_arete(client, db):
    """Croisement avec l'idempotence : une relance ne doit rien ajouter, graphe compris."""
    _ecrire(client, "Un fait de reference pour ce test.")
    _ecrire(client, "Un second fait, qui cree une arete vers le premier.")
    aretes_apres_deux = _aretes(db)

    relance = _ecrire(client, "Un second fait, qui cree une arete vers le premier.")

    assert relance.json()["status"] == "duplicate"
    assert _aretes(db) == aretes_apres_deux
