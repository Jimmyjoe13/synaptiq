"""Le registre de collections, de bout en bout : base -> registre -> API.

Ce que ces tests verrouillent au-delà des tests unitaires du registre :

- une collection déclarée par un agent est INVISIBLE depuis un autre agent (l'isolation
  vaut pour la taxonomie comme pour les souvenirs) ;
- le routage renvoyé par `POST /v1/memories` provient bien du registre de CET agent, et
  non plus d'une cascade codée en dur.

Exige Postgres + Redis (marqué integration via conftest). Auth désactivée par la fixture.
"""
import os

import psycopg2
import pytest
from conftest import purge_tenants
from fastapi.testclient import TestClient

import apps.api.main as main
from apps.api.main import app

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
TENANT = "col_test_tenant"


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True

    def _purge():
        purge_tenants(conn, TENANT)
        with conn.cursor() as cur:
            # Les collections d'agent ne sont pas des souvenirs : `purge_tenants` ne les
            # touche pas (et ne le doit pas — un effacement RGPD n'a pas à détruire la
            # structure que l'agent s'est donnée).
            cur.execute("DELETE FROM memory_collections WHERE tenant_id = %s", (TENANT,))
            conn.commit()

    _purge()
    yield conn
    _purge()
    conn.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_TENANT", TENANT)
    main.invalidate_auth_cache()
    with TestClient(app) as c:
        yield c


def _declarer_collection(db, agent_id, nom, famille, cle, entangle=True):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memory_collections "
            "(tenant_id, agent_id, name, family, packet_key, entangle, description, "
            " created_by) VALUES (%s, %s, %s, %s, %s, %s, %s, 'agent')",
            (TENANT, agent_id, nom, famille, cle, entangle, "Collection de test."),
        )
        db.commit()


def test_les_sept_collections_systeme_sont_servies(client, db):
    """Un agent tout neuf voit déjà les sept rayons livrés avec le moteur."""
    resp = client.get("/collections", params={"agent_id": "agent_neuf"})
    assert resp.status_code == 200, resp.text
    corps = resp.json()
    systeme = {c["name"] for c in corps["collections"] if c["created_by"] == "system"}
    assert systeme == {"fact", "preference", "interaction", "rule",
                       "coding_best_practices", "code_error_resolution", "scratch"}
    assert corps["packet_keys"][:7] == ["facts", "preferences", "episodes", "rules",
                                        "best_practices", "errors", "examples"]


def test_la_collection_d_un_agent_est_invisible_pour_un_autre(client, db):
    """L'isolation porte AUSSI sur la taxonomie, pas seulement sur les souvenirs.

    La structure qu'un agent se donne en dit long sur ce qu'il sait : la fuiter
    renseignerait un autre agent sur un périmètre qui ne le regarde pas.
    """
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")

    vue_a = client.get("/collections", params={"agent_id": "agentA"}).json()
    vue_b = client.get("/collections", params={"agent_id": "agentB"}).json()

    assert "clients_paca" in {c["name"] for c in vue_a["collections"]}
    assert "clients_paca" not in {c["name"] for c in vue_b["collections"]}
    # Et la clé de paquet supplémentaire ne fuite pas non plus.
    assert "clients_paca" in vue_a["packet_keys"]
    assert "clients_paca" not in vue_b["packet_keys"]


def test_le_routage_retourne_par_l_ecriture_vient_du_registre(client, db):
    """`POST /v1/memories` doit honorer une collection déclarée par CET agent.

    Avant ce lot, `collection` sortait d'une cascade de `if` : un sous-type libre était
    systématiquement annoncé dans `facts`, quelle que soit l'intention de l'agent.
    """
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")

    resp = client.post("/memories", json={
        "agent_id": "agentA", "type": "semantic", "subtype": "clients_paca",
        "content": "Nana Intelligence couvre Marseille, Aix, Toulon et Nice.",
    })
    assert resp.status_code == 201, resp.text
    assert resp.json()["collection"] == "clients_paca"
    # Le nom n'est pas canonique : l'agent doit pouvoir le savoir.
    assert resp.json()["canonical_subtype"] is False

    # Le MÊME sous-type, écrit par un agent qui ne l'a pas déclaré, retombe sur la section
    # de repli de sa famille — comportement historique, préservé.
    autre = client.post("/memories", json={
        "agent_id": "agentB", "type": "semantic", "subtype": "clients_paca",
        "content": "Contenu ecrit par un autre agent.",
    })
    assert autre.status_code == 201, autre.text
    assert autre.json()["collection"] == "facts"


def test_memory_count_reflete_les_souvenirs_reels(client, db):
    """Une collection déclarée mais vide est une information utile, pas un bug."""
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")
    _declarer_collection(db, "agentA", "jamais_utilisee", "semantic", "facts")

    client.post("/memories", json={
        "agent_id": "agentA", "type": "semantic", "subtype": "clients_paca",
        "content": "Un souvenir range dans la collection dediee.",
    })

    par_nom = {c["name"]: c for c in
               client.get("/collections", params={"agent_id": "agentA"}).json()["collections"]}
    assert par_nom["clients_paca"]["memory_count"] == 1
    assert par_nom["jamais_utilisee"]["memory_count"] == 0


def test_lecture_du_registre_exige_le_scope_read(client, db, monkeypatch):
    """L'endpoint suit le même régime de permissions que le reste de la surface."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    resp = client.get("/collections", params={"agent_id": "agentA"})
    assert resp.status_code == 401
