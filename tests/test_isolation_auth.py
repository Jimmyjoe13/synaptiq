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
        # Les clés API sont mises en cache par hash (AUTH_CACHE_TTL) : sans purge du cache,
        # une clé supprimée en base resterait acceptée et un même libellé réutilisé par un
        # autre test résoudrait vers l'ancien tenant.
        main.invalidate_auth_cache()
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


def _seed_api_key(db, raw_key: str, tenant_id: str, scopes=None, agents=None) -> None:
    """Insère une clé API. `scopes=None` -> DEFAULT de la colonne (read+write, sans admin).

    `agents=None` -> agent_scope NULL, soit l'accès à tous les agents du tenant.
    """
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    with db.cursor() as cur:
        if scopes is None and agents is None:
            cur.execute(
                "INSERT INTO api_keys (key_hash, tenant_id, name, active) VALUES (%s, %s, %s, true)",
                (key_hash, tenant_id, f"test-{tenant_id}"),
            )
        else:
            cur.execute(
                "INSERT INTO api_keys (key_hash, tenant_id, name, active, scopes, agent_scope) "
                "VALUES (%s, %s, %s, true, COALESCE(%s, ARRAY['read','write']::text[]), %s)",
                (key_hash, tenant_id, f"test-{tenant_id}", scopes, agents),
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


def test_revocation_effective_apres_invalidation_du_cache(client, db, monkeypatch):
    """Documente le compromis du cache d'auth (F10) : la révocation a une fenêtre bornée.

    Une clé désactivée en base reste acceptée jusqu'à l'expiration de son entrée de cache
    (`AUTH_CACHE_TTL`, 60 s par défaut). C'est le prix explicite de la suppression du
    SELECT+UPDATE par requête. `invalidate_auth_cache()` rend la révocation immédiate.
    """
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    _seed_api_key(db, "cle-a-revoquer", INSTANCE_TENANT)
    h = {"Authorization": "Bearer cle-a-revoquer"}
    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 200

    with db.cursor() as cur:
        cur.execute("UPDATE api_keys SET active = false WHERE tenant_id = %s", (INSTANCE_TENANT,))
        db.commit()

    # Toujours acceptée : l'entrée de cache n'a pas expiré. Comportement voulu, pas un bug.
    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 200

    main.invalidate_auth_cache()
    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 401


def test_cle_invalide_jamais_mise_en_cache(client, db, monkeypatch):
    """Une clé rejetée ne doit pas être mémorisée : sinon une clé réactivée resterait morte."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    h = {"Authorization": "Bearer cle-inexistante-puis-creee"}
    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 401

    _seed_api_key(db, "cle-inexistante-puis-creee", INSTANCE_TENANT)
    # Sans purge de cache : la clé fonctionne immédiatement.
    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 200


# ─── 2bis. Codes d'erreur honnêtes (audit F15) ───────────────────────────────

def test_events_signale_503_quand_la_base_est_absente(client, monkeypatch):
    """RÉGRESSION F15 : base indisponible -> 503, et non 500.

    `capture_event` enveloppait tout dans un `except Exception` qui ravalait le 503 de
    `get_conn()` : un client ne pouvait plus distinguer « réessaie » de « bug serveur »,
    donc ne pouvait pas implémenter de retry correct.
    """
    monkeypatch.setattr(main, "db_pool", None)
    resp = client.post("/events", json={"agent_id": "agentA", "session_id": "s1",
                                        "content": "un evenement"})
    assert resp.status_code == 503


# ─── Taxonomie appliquée à l'écriture directe (incident du 29/07) ─────────────

def test_sous_type_libre_accepte_et_routage_annonce(client):
    """Un libellé métier reste accepté, et la réponse dit OÙ le souvenir sera servi.

    Constaté en prod : des mémoires écrites via cet endpoint portaient des sous-types hors
    taxonomie (`nana_intelligence_lead_webhook`). C'est légitime, mais l'appelant ne pouvait
    pas savoir que son libellé ne produisait pas le routage fin qu'il imaginait.
    """
    resp = client.post("/memories", json={
        "agent_id": "agentA", "type": "semantic",
        "subtype": "nana_intelligence_lead_webhook", "content": "un webhook n8n",
    })
    assert resp.status_code == 201
    corps = resp.json()
    assert corps["collection"] == "facts"          # retombe sur la collection du type
    assert corps["canonical_subtype"] is False     # dit explicitement que c'est un libellé libre


def test_sous_type_canonique_annonce_sa_collection_fine(client):
    resp = client.post("/memories", json={
        "agent_id": "agentA", "type": "semantic",
        "subtype": "preference", "content": "Jimmy préfère les mails courts",
    })
    assert resp.status_code == 201
    assert resp.json()["collection"] == "preferences"
    assert resp.json()["canonical_subtype"] is True


def test_sous_type_du_mauvais_type_refuse_en_422(client):
    """Seule erreur démontrable : un sous-type canonique rattaché au mauvais type.

    `semantic` + `coding_best_practices` partirait dans `facts` alors que l'auteur visait
    `best_practices`. Avant le 29/07, l'API l'acceptait sans broncher.
    """
    resp = client.post("/memories", json={
        "agent_id": "agentA", "type": "semantic",
        "subtype": "coding_best_practices", "content": "toujours borner les requêtes",
    })
    assert resp.status_code == 422
    assert "procedural" in resp.text     # le message indique le bon type


def test_metrics_expose_les_jauges_du_pipeline(client):
    """Les deux métriques qui préviennent l'incident (audit F14) sont exposées et à jour."""
    corps = client.get("/metrics").text
    assert "synaptiq_outbox_pending" in corps
    assert "synaptiq_outbox_oldest_age_seconds" in corps
    assert "synaptiq_dlq_depth" in corps


def test_trace_id_unique_par_requete(client):
    """RÉGRESSION F14 : le trace_id était un horodatage à la seconde, donc partagé.

    Deux appels successifs doivent recevoir deux identifiants distincts, sinon corréler
    des logs à une requête est impossible.
    """
    corps = {"agent_id": "agentA", "session_id": "s1", "task": "t", "query": "q"}
    premier = client.post("/context/build", json=corps).json()["trace_id"]
    second = client.post("/context/build", json=corps).json()["trace_id"]
    assert premier != second
    assert premier.startswith("trace_")


# ─── 3. Le graphe d'intrication ne franchit pas la frontière d'agent (audit F1) ──

def _insert_memory_sql(db, tenant, agent, content, vec):
    """Insère une mémoire en SQL brut, avec un vecteur choisi.

    Nécessaire ici : l'embedder patché du test renvoie un vecteur CONSTANT, or ce test a
    besoin de deux vecteurs ORTHOGONAUX pour que le filtre de redondance de Q-EM
    (cosinus > 0,75) n'élimine pas l'un des deux et ne masque pas la fuite testée.
    """
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, "
            "confidence, importance, status) "
            "VALUES (%s, %s, 'semantic', 'fact', %s, %s, 1.0, 0.5, 'active') RETURNING id",
            (tenant, agent, content, "[" + ",".join(map(str, vec)) + "]"),
        )
        mem_id = cur.fetchone()[0]
        db.commit()
    return mem_id


def test_graphe_ne_franchit_pas_la_frontiere_d_agent(client, db):
    """RÉGRESSION F1 : une arête d'intrication vers un AUTRE agent ne doit rien ramener.

    `build_context` complète ses candidats par les mémoires « manquantes » du graphe. Cette
    requête ne filtrait ni le tenant ni l'agent : une seule arête traversante suffisait à
    injecter la mémoire d'un tiers dans le contexte envoyé au LLM. Ce test échoue sans le
    filtre ajouté le 29/07.
    """
    # agentA : vecteur aligné sur celui que l'embedder patché produira pour la requête.
    mem_a = _insert_memory_sql(db, INSTANCE_TENANT, "agentA", "donnee de A", CONST_VEC)
    # agentB : vecteur orthogonal -> jamais ramené par la similarité, seulement par le graphe.
    autre_vec = [0.0, 1.0] + [0.0] * 382
    mem_b = _insert_memory_sql(db, INSTANCE_TENANT, "agentB", "SECRET de B", autre_vec)

    # Arête traversant la frontière d'agent (cas d'un import ou d'un outil d'admin).
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight) "
            "VALUES (%s, %s, 'entangled_with', 1.0)",
            (mem_a, mem_b),
        )
        db.commit()

    resp = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1", "task": "t", "query": "q",
        "constraints": {"max_tokens": 2000, "memory_types": ["semantic"]},
    })
    assert resp.status_code == 200
    tout_le_contexte = " ".join(
        contenu for entrees in resp.json()["context_packet"].values() for contenu in entrees
    )
    assert "donnee de A" in tout_le_contexte
    assert "SECRET de B" not in tout_le_contexte


# ─── 4. Périmètre d'agents porté par la clé (audit F2) ───────────────────────

def test_cle_bornee_a_un_agent_refuse_les_autres(client, db, monkeypatch):
    """Une clé restreinte à agentA ne peut ni lire ni écrire au nom d'agentB."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    _seed_api_key(db, "cle-agentA", INSTANCE_TENANT, agents=["agentA"])
    h = {"Authorization": "Bearer cle-agentA"}

    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 200
    assert client.post("/retrieve", json={"agent_id": "agentB", "query": "q"},
                       headers=h).status_code == 403
    assert client.post("/memories", json={"agent_id": "agentB", "type": "semantic",
                                          "content": "intrusion"},
                       headers=h).status_code == 403
    assert client.post("/context/build", json={
        "agent_id": "agentB", "session_id": "s", "task": "t", "query": "q"},
        headers=h).status_code == 403


def test_cle_lecture_seule_ne_peut_pas_ecrire(client, db, monkeypatch):
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    _seed_api_key(db, "cle-lecture", INSTANCE_TENANT, scopes=["read"])
    h = {"Authorization": "Bearer cle-lecture"}

    assert client.post("/retrieve", json={"agent_id": "agentA", "query": "q"},
                       headers=h).status_code == 200
    assert client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                          "content": "ecriture interdite"},
                       headers=h).status_code == 403
    assert client.post("/events", json={"agent_id": "agentA", "session_id": "s",
                                        "content": "ecriture interdite"},
                       headers=h).status_code == 403


# ─── 5. Purge protégée (audit F3) ────────────────────────────────────────────

def test_purge_refusee_sans_scope_admin(client, db, monkeypatch):
    """RÉGRESSION F3 : une clé d'agent ordinaire ne doit pas pouvoir vider l'instance."""
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    _seed_api_key(db, "cle-normale", INSTANCE_TENANT)  # scopes par défaut : read + write
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "content": "a garder"},
                headers={"Authorization": "Bearer cle-normale"})

    resp = client.delete(f"/memories?confirm={INSTANCE_TENANT}",
                         headers={"Authorization": "Bearer cle-normale"})
    assert resp.status_code == 403
    # La donnée est toujours là.
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", (INSTANCE_TENANT,))
        assert cur.fetchone()[0] == 1


def test_purge_exige_la_confirmation(client, db):
    """Sans ?confirm=<tenant>, la purge échoue en 400 et ne supprime rien."""
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "content": "a garder"})
    assert client.delete("/memories").status_code == 400
    assert client.delete("/memories?confirm=mauvais_tenant").status_code == 400
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", (INSTANCE_TENANT,))
        assert cur.fetchone()[0] == 1


def test_purge_confirmee_supprime_et_laisse_une_trace(client, db, monkeypatch):
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    _seed_api_key(db, "cle-admin", INSTANCE_TENANT, scopes=["read", "write", "admin"])
    h = {"Authorization": "Bearer cle-admin"}
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "content": "a supprimer"}, headers=h)

    resp = client.delete(f"/memories?confirm={INSTANCE_TENANT}", headers=h)
    assert resp.status_code == 200
    assert resp.json()["deleted_memories"] == 1

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", (INSTANCE_TENANT,))
        assert cur.fetchone()[0] == 0
        # La trace d'audit survit à la purge : c'en est le seul témoignage.
        cur.execute("SELECT action, actor, details FROM audit_log WHERE tenant_id = %s",
                    (INSTANCE_TENANT,))
        lignes = cur.fetchall()
    assert len(lignes) == 1
    action, actor, details = lignes[0]
    assert action == "purge_memories"
    assert details["deleted_memories"] == 1
    assert details["scope"] == "tenant"
    # L'auteur est identifié par un préfixe de hash, jamais par la clé elle-même.
    assert actor and "cle-admin" not in actor
