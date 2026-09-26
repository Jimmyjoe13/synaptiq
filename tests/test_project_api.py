"""Dimension PROJET de bout en bout (lot B, 26/09).

Un agent généraliste mêlait tous ses projets dans une même partition : un rappel sur
« état de SynaptiQ » ramenait surtout d'autres projets. Ces tests verrouillent que le filtre
projet vaut dans CHAQUE chemin — recherche, construction de contexte, complétion du graphe —
et que le graphe lui-même ne relie pas deux projets entre eux.

Exige Postgres + Redis (marqué integration via conftest).
"""
import json
import os

import psycopg2
import pytest
from conftest import purge_tenants
from fastapi.testclient import TestClient

import apps.api.main as main
from apps.api.main import app

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
TENANT = "project_test_tenant"
AGENT = "agent_projet"


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True

    def _purge():
        purge_tenants(conn, TENANT)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_collections WHERE tenant_id = %s", (TENANT,))

    _purge()
    yield conn
    _purge()
    conn.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_TENANT", TENANT)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    # Seuil -1 : TOUT voisin examiné est relié (l'embedder mock produit aussi des cosinus
    # négatifs, un seuil nul en écarterait). Si une arête inter-projets était possible, ce
    # réglage la produirait à coup sûr — c'est ce qui rend le test d'isolation probant.
    monkeypatch.setenv("QEM_ENTANGLE_THRESHOLD", "-1.0")
    main.invalidate_auth_cache()
    with TestClient(app) as c:
        yield c


def _ecrire(client, contenu, project=None):
    corps = {"agent_id": AGENT, "type": "semantic", "subtype": "fact", "content": contenu}
    if project is not None:
        corps["project"] = project
    r = client.post("/v1/memories", json=corps)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def trois_souvenirs(client, db):
    """Un souvenir Émile, un souvenir SynaptiQ, un souvenir global."""
    return {
        "emile": _ecrire(client, "Le domaine d'envoi d'Emile est plafonne a 400 mails par jour.",
                         "emile")["memory_id"],
        "synaptiq": _ecrire(client, "Le port de l'API SynaptiQ est 8000.", "synaptiq")["memory_id"],
        "global": _ecrire(client, "Jimmy veut des reponses courtes en francais.")["memory_id"],
    }


def _ids_retrieve(client, **filtre):
    r = client.post("/v1/retrieve", json={"agent_id": AGENT, "query": "port domaine reponses",
                                          "limit": 10, **filtre})
    assert r.status_code == 200, r.text
    return {str(m["id"]) for m in r.json()["memories"]}


def test_le_projet_est_normalise_et_rendu(client, db):
    corps = _ecrire(client, "Un fait de projet.", " Emile ")
    assert corps["project"] == "emile"


def test_nom_de_projet_invalide_refuse_en_422(client, db):
    r = client.post("/v1/memories", json={"agent_id": AGENT, "type": "semantic",
                                          "subtype": "fact", "content": "x",
                                          "project": "mon projet"})
    assert r.status_code == 422


def test_retrieve_sans_projet_voit_tout(client, trois_souvenirs):
    assert _ids_retrieve(client) == set(trois_souvenirs.values())


def test_retrieve_projet_inclut_les_globaux(client, trois_souvenirs):
    assert _ids_retrieve(client, project="emile") == {
        trois_souvenirs["emile"], trois_souvenirs["global"]}


def test_retrieve_projet_strict(client, trois_souvenirs):
    assert _ids_retrieve(client, project="emile", include_global=False) == {
        trois_souvenirs["emile"]}


def test_context_build_ne_sert_pas_un_autre_projet(client, trois_souvenirs):
    r = client.post("/v1/context/build", json={
        "agent_id": AGENT, "session_id": "s", "task": "t", "query": "port domaine reponses",
        "constraints": {"max_tokens": 2000, "project": "synaptiq"}})
    assert r.status_code == 200, r.text
    selection = set(map(str, r.json()["selected_memory_ids"]))
    assert trois_souvenirs["emile"] not in selection
    assert trois_souvenirs["synaptiq"] in selection


def test_le_graphe_ne_relie_jamais_deux_projets(client, db, trois_souvenirs):
    """Seuil d'intrication à -1 : sans le filtre, Émile et SynaptiQ seraient reliés."""
    with db.cursor() as cur:
        cur.execute("""
            SELECT count(*) FROM relationships r
            JOIN memories a ON a.id = r.source_memory_id
            JOIN memories b ON b.id = r.target_memory_id
            WHERE a.tenant_id = %s AND a.project IS NOT NULL AND b.project IS NOT NULL
              AND a.project <> b.project
        """, (TENANT,))
        assert cur.fetchone()[0] == 0
        # Le câblage existe bien : le global, lui, est relié aux projets.
        cur.execute("""
            SELECT count(*) FROM relationships r
            JOIN memories m ON m.id = r.source_memory_id WHERE m.tenant_id = %s
        """, (TENANT,))
        assert cur.fetchone()[0] >= 1


def test_le_projet_voyage_jusqu_au_worker_par_l_outbox(client, db):
    r = client.post("/v1/events", json={"agent_id": AGENT, "session_id": "s",
                                        "content": "Un evenement de projet.",
                                        "project": "Emile"})
    assert r.status_code in (200, 201), r.text
    with db.cursor() as cur:
        cur.execute("""
            SELECT e.project, o.payload FROM events e JOIN event_outbox o ON o.event_id = e.id
            WHERE e.tenant_id = %s
        """, (TENANT,))
        projet, payload = cur.fetchone()
    assert projet == "emile"
    charge = payload if isinstance(payload, dict) else json.loads(payload)
    assert charge["project"] == "emile"
