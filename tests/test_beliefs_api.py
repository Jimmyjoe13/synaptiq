"""Croyances de l'agent de bout en bout (famille `reflective`, lot C du 26/09).

Ce que l'agent pense de l'utilisateur doit être : validé à l'écriture (garde-fou), servi
comme une HYPOTHÈSE et jamais comme un fait, révisable par remplacement, consultable d'un
seul appel, et contestable — une croyance contestée ne revient pas à l'identique.

Exige Postgres + Redis (marqué integration via conftest).
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
TENANT = "beliefs_test_tenant"
AGENT = "agent_croyances"


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    purge_tenants(conn, TENANT)
    yield conn
    purge_tenants(conn, TENANT)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM audit_log WHERE tenant_id = %s", (TENANT,))
    conn.close()


@pytest.fixture
def client(monkeypatch, db):
    monkeypatch.setenv("SYNAPTIQ_TENANT", TENANT)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    main.invalidate_auth_cache()
    with TestClient(app) as c:
        yield c


def _noter(client, contenu, confiance=0.4, subtype="user_model", **extra):
    return client.post("/v1/memories", json={
        "agent_id": AGENT, "type": "reflective", "subtype": subtype,
        "content": contenu, "confidence": confiance, **extra})


def _fait(client, contenu):
    r = client.post("/v1/memories", json={"agent_id": AGENT, "type": "episodic",
                                          "subtype": "interaction", "content": contenu})
    assert r.status_code == 201, r.text
    return r.json()["memory_id"]


def test_une_croyance_est_enregistree_puis_listee(client):
    r = _noter(client, "Jimmy prefere qu'on lui propose un plan court avant d'agir")
    assert r.status_code == 201, r.text
    assert r.json()["collection"] == "user_model"
    croyances = client.get("/v1/beliefs", params={"agent_id": AGENT}).json()["beliefs"]
    assert len(croyances) == 1
    assert croyances[0]["about"] == "user"
    assert croyances[0]["confidence"] == 0.4


def test_categorie_sensible_refusee_en_422(client):
    r = _noter(client, "Jimmy traverse sans doute une depression")
    assert r.status_code == 422
    assert "sensible" in r.text


def test_confiance_forte_exige_des_indices(client):
    assert _noter(client, "Jimmy decide toujours vite", 0.8).status_code == 422
    indice = _fait(client, "Jimmy a valide le plan en deux minutes.")
    r = _noter(client, "Jimmy decide toujours vite", 1.0, evidence=[indice])
    assert r.status_code == 201, r.text
    croyance = client.get("/v1/beliefs", params={"agent_id": AGENT}).json()["beliefs"][0]
    assert croyance["confidence"] == 0.9            # plafond d'une croyance
    assert croyance["evidence"] == [indice]


def test_un_tiers_peut_etre_nomme(client):
    r = _noter(client, "Regis prefere recevoir les CSV le lundi", subtype="human_insights")
    assert r.status_code == 201, r.text
    croyances = client.get("/v1/beliefs", params={"agent_id": AGENT, "about": "humans"})
    assert croyances.json()["beliefs"][0]["about"] == "humans"


def test_evidence_hors_croyance_refusee(client):
    r = client.post("/v1/memories", json={"agent_id": AGENT, "type": "semantic",
                                          "content": "un fait", "evidence": ["x"]})
    assert r.status_code == 422


def test_une_croyance_revisee_remplace_l_ancienne(client, db):
    ancienne = _noter(client, "Jimmy aime les longs rapports").json()["memory_id"]
    r = _noter(client, "Jimmy prefere finalement les rapports courts", replaces=ancienne)
    assert r.status_code == 201, r.text
    actives = client.get("/v1/beliefs", params={"agent_id": AGENT}).json()["beliefs"]
    assert [c["content"] for c in actives] == ["Jimmy prefere finalement les rapports courts"]
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM relationships WHERE target_memory_id = %s "
                    "AND relation_type = 'supersedes_by'", (ancienne,))
        assert cur.fetchone()[0] == 1


def test_replaces_ne_peut_pas_archiver_un_fait(client):
    fait = _fait(client, "Un simple episode.")
    assert _noter(client, "Une hypothese", replaces=fait).status_code == 404


def test_une_croyance_contestee_sort_et_ne_revient_pas(client, db):
    contenu = "Jimmy n'aime pas qu'on lui pose des questions"
    croyance = _noter(client, contenu).json()["memory_id"]
    r = client.post(f"/v1/beliefs/{croyance}/contest",
                    json={"agent_id": AGENT, "reason": "faux"})
    assert r.status_code == 200, r.text
    assert client.get("/v1/beliefs", params={"agent_id": AGENT}).json()["beliefs"] == []
    # Réécriture à l'identique refusée ; la contestation est tracée sans contenu.
    assert _noter(client, contenu).status_code == 409
    with db.cursor() as cur:
        cur.execute("SELECT details FROM audit_log WHERE tenant_id = %s "
                    "AND action = 'belief_contested'", (TENANT,))
        details = cur.fetchone()[0]
    assert details["reason_length"] == 4 and "reason" not in details


def test_la_croyance_est_servie_comme_hypothese(client):
    _noter(client, "Jimmy valide plus vite un plan illustre par un exemple", 0.5)
    r = client.post("/v1/context/build", json={
        "agent_id": AGENT, "session_id": "s", "task": "t",
        "query": "comment presenter un plan a Jimmy", "constraints": {"max_tokens": 1200}})
    assert r.status_code == 200, r.text
    assert r.json()["context_packet"]["user_model"] == [
        "(hypothèse, confiance 0.5) Jimmy valide plus vite un plan illustre par un exemple"]
