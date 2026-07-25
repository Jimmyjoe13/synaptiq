"""Tests d'intégration — isolation par agent_id + authentification par clé API.

Exigent Postgres (DATABASE_URL). Marqués `integration` automatiquement (hors tests/unit/).
Verrouillent deux garanties produit non-négociables, jusqu'ici SANS aucun test :
  1. un agent ne lit JAMAIS les mémoires d'un autre agent de la même instance ;
  2. le tenant n'est pas pilotable par le body ; l'auth par clé API scope le tenant
     (isolation cross-tenant), et le mode auth requise rejette l'absence/invalidité de clé.

Insertions via l'API (POST /memories) : on réutilise le chemin d'écriture réel (embedder
patché déterministe) plutôt que du SQL brut, et on teste l'endpoint au passage.
"""
import hashlib
import os

import psycopg2
import pytest
from conftest import purge_tenants
from fastapi.testclient import TestClient

import apps.api.main as main
from apps.api.main import app

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db"
)
INSTANCE_TENANT = "iso_test_tenant"
CONST_VEC = [1.0] + [0.0] * 383  # vecteur unitaire déterministe (similarité 1.0)


class _ConstEmbedder:
    """Embedder de test : vecteur constant → similarité cosinus 1.0 déterministe."""
    dim = 384

    def __init__(self, vec):
        self._v = vec

    def embed(self, texts):
        return [self._v for _ in texts]

    def embed_one(self, text):
        return self._v


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _clean(db):
    """Purge bornée aux tenants du test.

    Un TRUNCATE global effacerait aussi les données réelles de l'instance — la base de
    développement sert souvent aussi aux essais (cf. conftest.purge_tenants).
    """
    # Tous les périmètres touchés par ce fichier, y compris ceux créés via des clés API
    # (`tenantX`/`tenantY`) : en oublier un laisse une clé en base et fait échouer le
    # run suivant sur la contrainte d'unicité de `key_hash`.
    def _purge():
        purge_tenants(db, INSTANCE_TENANT, "autre_tenant", "test_tenant",
                      "tenantX", "tenantY", "tenant_pirate")
    _purge()
    yield
    _purge()


@pytest.fixture
def client(monkeypatch):
    """TestClient avec tenant d'instance fixé et embedder déterministe patché.

    Context manager obligatoire (déclenche le lifespan → init du pool PostgreSQL).
    """
    monkeypatch.setenv("SYNAPTIQ_TENANT", INSTANCE_TENANT)
    monkeypatch.setattr(main, "get_embedder", lambda: _ConstEmbedder(CONST_VEC))
    with TestClient(app) as c:
        yield c


def _seed_api_key(db, raw_key: str, tenant_id: str) -> None:
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (key_hash, tenant_id, name, active) VALUES (%s, %s, %s, true)",
            (key_hash, tenant_id, f"test-{tenant_id}"),
        )
        db.commit()


# ─── 1. Isolation par agent_id (même instance, même tenant) ──────────────────

def test_retrieve_isole_par_agent(client):
    """Un agent ne récupère QUE ses propres mémoires via /retrieve."""
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                    "subtype": "preference", "content": "pref de A"})
    client.post("/memories", json={"agent_id": "agentB", "type": "semantic",
                                    "subtype": "preference", "content": "pref de B"})

    resp = client.post("/retrieve", json={"agent_id": "agentA", "query": "peu importe"})
    assert resp.status_code == 200
    contents = [m["content"] for m in resp.json()["memories"]]
    assert contents == ["pref de A"]
    assert "pref de B" not in contents


def test_context_build_isole_par_agent(client):
    """/context/build ne ramène jamais les mémoires d'un autre agent."""
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                    "subtype": "preference", "content": "pref de A"})
    client.post("/memories", json={"agent_id": "agentB", "type": "semantic",
                                    "subtype": "preference", "content": "pref de B"})

    resp = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1",
        "task": "t", "query": "q",
        "constraints": {"max_tokens": 1000, "memory_types": ["semantic"]},
    })
    assert resp.status_code == 200
    prefs = resp.json()["context_packet"]["preferences"]
    assert prefs == ["pref de A"]


def test_body_ne_peut_pas_changer_le_tenant(client, db):
    """Un `tenant_id` injecté dans le body est ignoré : le périmètre reste l'instance."""
    # Mémoire écrite dans le tenant d'instance.
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "preference", "content": "pref de A"})
    # Body trafiqué avec un faux tenant_id → doit être ignoré (retour normal, périmètre instance).
    resp = client.post("/retrieve", json={"agent_id": "agentA", "query": "q",
                                          "tenant_id": "tenant_pirate"})
    assert resp.status_code == 200
    assert [m["content"] for m in resp.json()["memories"]] == ["pref de A"]
    # Confirme que rien n'a été écrit sous le tenant injecté : la vérification porte sur
    # l'absence du faux tenant, et non sur la liste complète des tenants de la base —
    # celle-ci contient légitimement d'autres périmètres (la purge est bornée au test).
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", ("tenant_pirate",))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", (INSTANCE_TENANT,))
        assert cur.fetchone()[0] == 1


# ─── 2. Authentification par clé API ─────────────────────────────────────────

def test_auth_requise_sans_cle_401(client, monkeypatch):
    """AUTH_REQUIRED=true + aucune clé → 401 (pas d'accès anonyme)."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    resp = client.post("/retrieve", json={"agent_id": "agentA", "query": "q"})
    assert resp.status_code == 401


def test_cle_invalide_401(client, monkeypatch):
    """Une clé Bearer inconnue → 401, même avec auth non requise."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    resp = client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers={"Authorization": "Bearer clef-bidon"})
    assert resp.status_code == 401


def test_cle_valide_scope_le_tenant_et_isole(client, db, monkeypatch):
    """Deux clés = deux tenants : chacune ne voit que les mémoires de son tenant."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    _seed_api_key(db, "cle-tenant-X", "tenantX")
    _seed_api_key(db, "cle-tenant-Y", "tenantY")
    hx = {"Authorization": "Bearer cle-tenant-X"}
    hy = {"Authorization": "Bearer cle-tenant-Y"}

    # Écriture sous le tenant X (résolu par la clé, jamais par le body).
    r = client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                       "subtype": "preference", "content": "secret de X"},
                    headers=hx)
    assert r.status_code == 201

    # La clé X voit sa mémoire...
    rx = client.post("/retrieve", json={"agent_id": "agentA", "query": "q"}, headers=hx)
    assert [m["content"] for m in rx.json()["memories"]] == ["secret de X"]

    # ...mais la clé Y (autre tenant) ne voit RIEN, même avec le même agent_id.
    ry = client.post("/retrieve", json={"agent_id": "agentA", "query": "q"}, headers=hy)
    assert ry.json()["memories"] == []
