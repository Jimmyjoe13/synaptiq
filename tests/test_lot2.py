"""Lot 2 de bout en bout : archives, récompense, plasticité, restauration, sommeil.

Règle verrouillée avant tout le reste (décision de Jimmy du 23/11/2026) : AUCUNE
suppression. Le sommeil range dans `archives`, il ne supprime jamais une ligne.

Exige Postgres + Redis (marqué integration via conftest). Embeddings `mock`, LLM simulé.
"""
import json
import os
from datetime import UTC, datetime, timedelta

import psycopg2
import pytest
from conftest import purge_tenants
from fastapi.testclient import TestClient

import apps.api.main as main
import apps.worker.sommeil as sommeil
from apps.api.main import app
from synaptiq_core.sommeil import ReglagesSommeil

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
TENANT = "lot2_test_tenant"
AGENT = "agent_lot2"


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    purge_tenants(conn, TENANT)
    yield conn
    purge_tenants(conn, TENANT)
    conn.close()


@pytest.fixture
def client(monkeypatch, db):
    monkeypatch.setenv("SYNAPTIQ_TENANT", TENANT)
    main.invalidate_auth_cache()
    with TestClient(app) as c:
        yield c


def _vec(*composantes):
    """Vecteur 384 dimensions, normalisé, à partir de quelques composantes."""
    v = list(composantes) + [0.0] * (384 - len(composantes))
    n = sum(x * x for x in v) ** 0.5
    return "[" + ",".join(str(x / n) for x in v) + "]"


def _inserer(db, contenu, compartiment, vec, age_jours=0, acces=0, reward=0.0,
             famille=None, agent=AGENT):
    quand = datetime.now(UTC) - timedelta(days=age_jours)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (tenant_id, agent_id, type, compartment, content, embedding, "
            "confidence, importance, status, created_at, last_accessed_at, access_count, "
            "reward, content_hash) VALUES (%s, %s, %s, %s, %s, %s, 1.0, 0.5, 'active', %s, %s, "
            "%s, %s, md5(%s)) RETURNING id::text",
            (TENANT, agent, famille or main.PAR_NOM[compartiment].famille, compartiment, contenu,
             vec, quand, quand, acces, reward, contenu))
        return cur.fetchone()[0]


def _ligne(db, mem_id):
    with db.cursor() as cur:
        cur.execute("SELECT compartment, type, reward, provenance FROM memories WHERE id = %s",
                    (mem_id,))
        return cur.fetchone()


# ─── API : archives hors des lectures par défaut, restauration ───────────────

def test_les_archives_ne_remontent_que_sur_demande(client, db):
    _inserer(db, "Souvenir actif sur le serveur", "savoirs", _vec(1, 0))
    _inserer(db, "Souvenir archivé sur le serveur", "archives", _vec(1, 0.01), famille="dormant")
    defaut = client.post("/v1/retrieve", json={"agent_id": AGENT, "query": "serveur"}).json()
    assert {m["compartment"] for m in defaut["memories"]} == {"savoirs"}
    archives = client.post("/v1/retrieve", json={"agent_id": AGENT, "query": "serveur",
                                                  "compartments": ["archives"]}).json()
    assert [m["compartment"] for m in archives["memories"]] == ["archives"]
    paquet = client.post("/v1/context/build", json={
        "agent_id": AGENT, "session_id": "s", "task": "t", "query": "serveur",
        "record_access": False}).json()["context_packet"]
    assert paquet["archives"] == [] and paquet["savoirs"]


def test_restauration_vers_le_compartiment_d_origine(client, db):
    mem = _inserer(db, "Ancienne procédure de déploiement", "archives", _vec(0, 1),
                   famille="dormant")
    with db.cursor() as cur:
        cur.execute("UPDATE memories SET provenance = %s WHERE id = %s",
                    (json.dumps({"archived_from": "savoir_faire", "archived_reason": "oubli"}),
                     mem))
    r = client.post(f"/v1/memories/{mem}/restore", json={"agent_id": AGENT})
    assert r.status_code == 200, r.text
    compartiment, famille, _, provenance = _ligne(db, mem)
    assert (compartiment, famille) == ("savoir_faire", "procedural")
    assert provenance["archived_reason"] == "oubli" and "restored_at" in provenance
    # Restaurer un souvenir qui n'est pas en archives : 404.
    assert client.post(f"/v1/memories/{mem}/restore",
                       json={"agent_id": AGENT}).status_code == 404


# ─── API : récompense et plasticité ──────────────────────────────────────────

def test_reinforce_recompense_et_tisse_un_lien(client, db):
    a = _inserer(db, "Procédure A", "savoir_faire", _vec(1, 0))
    b = _inserer(db, "Erreur B résolue", "alertes", _vec(0, 1))
    r = client.post("/v1/memories/reinforce", json={"agent_id": AGENT, "memory_ids": [a, b]})
    assert r.status_code == 200, r.text
    assert r.json()["links"] == 1
    assert _ligne(db, a)[2] == 1.0 and _ligne(db, b)[2] == 1.0
    with db.cursor() as cur:
        cur.execute("SELECT weight, last_reinforced_at IS NOT NULL FROM relationships "
                    "WHERE source_memory_id = %s AND target_memory_id = %s", tuple(sorted([a, b])))
        assert cur.fetchone() == (0.5, True)
    # Second renforcement : le même lien se renforce, il n'est pas dupliqué.
    client.post("/v1/memories/reinforce", json={"agent_id": AGENT, "memory_ids": [b, a]})
    with db.cursor() as cur:
        cur.execute("SELECT count(*), max(weight) FROM relationships WHERE source_memory_id "
                    "= ANY(%s::uuid[]) OR target_memory_id = ANY(%s::uuid[])", ([a, b], [a, b]))
        n, poids = cur.fetchone()
    assert n == 1 and poids == pytest.approx(0.6)


def test_reinforce_negatif_et_borne(client, db):
    a = _inserer(db, "Souvenir trompeur", "savoirs", _vec(1, 0), reward=-4.5)
    r = client.post("/v1/memories/reinforce", json={"agent_id": AGENT, "memory_ids": [a],
                                                     "useful": False})
    assert r.json()["links"] == 0
    assert _ligne(db, a)[2] == -5.0


def test_reinforce_ne_touche_pas_un_autre_agent(client, db):
    autre = _inserer(db, "Souvenir d'un autre agent", "savoirs", _vec(1, 0), agent="intrus")
    r = client.post("/v1/memories/reinforce", json={"agent_id": AGENT, "memory_ids": [autre]})
    assert r.json() == {"status": "reinforced", "useful": True, "memories": 0, "ignored": 1,
                        "links": 0}
    assert _ligne(db, autre)[2] == 0.0


# ─── Sommeil ─────────────────────────────────────────────────────────────────

class _FauxLLM:
    rotation = True

    def __init__(self):
        self.appels = 0

    def chat_json(self, messages, max_tokens=1500):
        self.appels += 1
        return ({"memories": [{"compartment": "decisions",
                               "content": "Décision : migrer la base vers PostgreSQL 17 en "
                                          "décembre, pour pgvector 0.9."}]}, "faux-free")


def _nombre(db):
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", (TENANT,))
        return cur.fetchone()[0]


def test_une_nuit_de_sommeil_complete(db, monkeypatch):
    llm = _FauxLLM()
    monkeypatch.setattr(sommeil, "get_client_llm", lambda: llm)
    monkeypatch.setenv("QEM_ENTANGLE_THRESHOLD", "0.7")
    # Doublons : même compartiment, même vecteur ; le récompensé survit.
    garde = _inserer(db, "Le vault Obsidian est dans C:/Obsidian/Jimmy", "environnement",
                     _vec(1, 0, 0), reward=1.0)
    double = _inserer(db, "Vault Obsidian : C:/Obsidian/Jimmy", "environnement", _vec(1, 0, 0))
    # Oubli : vieille observation jamais relue ; l'autre a été relue ; le réflexe est éternel.
    oubliee = _inserer(db, "Mesure : 1 Go de swap", "observations", _vec(0, 1, 0), age_jours=60)
    relue = _inserer(db, "Mesure : 2 Go de RAM", "observations", _vec(0, 0.9, 0.4),
                     age_jours=60, acces=3)
    reflexe = _inserer(db, "Toujours tutoyer Jimmy", "reflexes", _vec(0, 0, 1), age_jours=900)
    # Consolidation : deux épisodes proches (cosinus 0,78 : regroupés, pas doublons).
    e1 = _inserer(db, "Session : discussion migration PostgreSQL 17", "episodes",
                  _vec(0, 0, 0, 1, 0), age_jours=5)
    e2 = _inserer(db, "Session : migration PostgreSQL 17 validée pour décembre", "episodes",
                  _vec(0, 0, 0, 1, 0.8), age_jours=5)
    # Graphe : un savoir isolé, voisin d'un autre (cosinus 0,74 > seuil 0,7).
    isole = _inserer(db, "SynaptiQ stocke ses vecteurs en 384 dimensions", "savoirs",
                     _vec(0, 0, 0, 0, 0, 1, 0))
    _inserer(db, "Les vecteurs SynaptiQ font 384 dimensions (pgvector)", "savoirs",
             _vec(0, 0, 0, 0, 0, 1, 0.9))
    # Plasticité : un lien renforcé il y a 30 jours, puis délaissé.
    with db.cursor() as cur:
        cur.execute("INSERT INTO relationships (source_memory_id, target_memory_id, "
                    "relation_type, weight, last_reinforced_at) VALUES (%s, %s, "
                    "'entangled_with', 0.9, now() - interval '30 days')", (garde, reflexe))
    avant = _nombre(db)

    conn = psycopg2.connect(DATABASE_URL)
    try:
        rapport = sommeil.dormir(conn, TENANT, AGENT, sommeil.PHASES, ReglagesSommeil())
    finally:
        conn.close()

    # AUCUNE suppression : seulement +1 (le souvenir consolidé).
    assert _nombre(db) == avant + 1
    assert rapport["doublons"] == 1
    assert _ligne(db, garde)[0] == "environnement"
    archive = _ligne(db, double)
    assert archive[0:2] == ("archives", "dormant")
    assert archive[3]["archived_from"] == "environnement"
    assert archive[3]["merged_into"] == garde
    assert rapport["oubli"] == 1
    assert _ligne(db, oubliee)[0] == "archives" and _ligne(db, oubliee)[3]["archived_reason"] == "oubli"
    assert _ligne(db, relue)[0] == "observations" and _ligne(db, reflexe)[0] == "reflexes"
    assert rapport["consolidation"]["souvenirs_crees"] == 1 and llm.appels == 1
    with db.cursor() as cur:
        cur.execute("SELECT compartment, provenance FROM memories WHERE tenant_id = %s "
                    "AND provenance->>'source' = 'sommeil'", (TENANT,))
        compartiment, provenance = cur.fetchone()
    assert compartiment == "decisions" and sorted(provenance["consolidated_from"]) == sorted([e1, e2])
    assert _ligne(db, e1)[3]["consolidated"] is True and _ligne(db, e1)[0] == "episodes"
    assert rapport["graphe"] >= 1
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM relationships WHERE source_memory_id = %s "
                    "OR target_memory_id = %s", (isole, isole))
        assert cur.fetchone()[0] >= 1
        cur.execute("SELECT weight FROM relationships WHERE source_memory_id = %s "
                    "AND target_memory_id = %s", (garde, reflexe))
        assert cur.fetchone()[0] == pytest.approx(0.855)
        cur.execute("SELECT details FROM audit_log WHERE tenant_id = %s AND action = 'sommeil'",
                    (TENANT,))
        assert cur.fetchone()[0]["agent_id"] == AGENT

    # Une seconde nuit ne rejoue pas les épisodes déjà consolidés.
    conn = psycopg2.connect(DATABASE_URL)
    try:
        sommeil.dormir(conn, TENANT, AGENT, ("consolidation",), ReglagesSommeil())
    finally:
        conn.close()
    assert llm.appels == 1


def test_sommeil_a_blanc_n_ecrit_rien(db, monkeypatch):
    monkeypatch.setattr(sommeil, "get_client_llm", lambda: _FauxLLM())
    double_a = _inserer(db, "Fait A", "savoirs", _vec(1, 0))
    _inserer(db, "Fait A bis", "savoirs", _vec(1, 0))
    conn = psycopg2.connect(DATABASE_URL)
    try:
        rapport = sommeil.dormir(conn, TENANT, AGENT, sommeil.PHASES, ReglagesSommeil(),
                                 dry=True)
    finally:
        conn.close()
    assert rapport["doublons"] == 1
    assert _ligne(db, double_a)[0] == "savoirs"
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM audit_log WHERE tenant_id = %s", (TENANT,))
        assert cur.fetchone()[0] == 0


def test_sommeil_sans_llm_reporte_la_consolidation(db, monkeypatch):
    monkeypatch.setattr(sommeil, "get_client_llm", lambda: None)
    _inserer(db, "Session ancienne", "episodes", _vec(1, 0), age_jours=5)
    conn = psycopg2.connect(DATABASE_URL)
    try:
        rapport = sommeil.dormir(conn, TENANT, AGENT, ("consolidation",), ReglagesSommeil())
    finally:
        conn.close()
    assert rapport["consolidation"]["souvenirs_crees"] == 0
    assert "aucun LLM" in rapport["consolidation"]["note"]
