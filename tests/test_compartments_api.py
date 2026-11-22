"""Les compartiments (v0.4), de bout en bout : API -> base -> paquet de contexte.

Ce que ces tests verrouillent au-delà des tests unitaires du cœur :

- la base REFUSE un compartiment hors des 9 (contrainte CHECK de la migration) ;
- l'écriture range toujours quelque part, et dit où et par qui (agent, alias, moteur) ;
- la famille interne est DÉDUITE du compartiment, plus jamais fournie ;
- les réflexes sont injectés dans le contexte même sans rapport avec la question ;
- le comptage par compartiment est borné à l'agent (isolation).

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
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
TENANT = "comp_test_tenant"
AGENT = "agent_comp"


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


def _ecrire(client, contenu, **champs):
    r = client.post("/v1/memories", json={"agent_id": AGENT, "content": contenu, **champs})
    assert r.status_code == 201, r.text
    return r.json()


def _ligne(db, memory_id):
    with db.cursor() as cur:
        cur.execute("SELECT type, compartment, legacy_subtype, provenance FROM memories "
                    "WHERE id = %s", (memory_id,))
        return cur.fetchone()


# ─── Écriture ─────────────────────────────────────────────────────────────────

def test_compartiment_choisi_par_l_agent(client, db):
    corps = _ecrire(client, "Toujours lancer la suite complète avant un déploiement.",
                    compartment="savoir_faire")
    assert (corps["compartment"], corps["compartment_origin"]) == ("savoir_faire", "agent")
    assert corps["region"] == "Cervelet"
    famille, compartiment, ancien, _ = _ligne(db, corps["memory_id"])
    assert (famille, compartiment, ancien) == ("procedural", "savoir_faire", None)


def test_compartiment_absent_classe_par_le_moteur(client, db):
    corps = _ecrire(client, "Le serveur de prod écoute sur le port 8443.")
    assert corps["compartment_origin"] == "moteur"
    assert corps["compartment"] in main.NOMS and corps["compartment"] != "croyances"
    assert isinstance(corps["classification_similarity"], float)


def test_nom_inconnu_classe_et_trace_dans_la_provenance(client, db):
    corps = _ecrire(client, "Contact principal du client PACA.", compartment="clients_paca")
    assert corps["compartment_origin"] == "moteur"
    assert corps["requested"] == "clients_paca"
    _, _, _, provenance = _ligne(db, corps["memory_id"])
    assert provenance == {"requested_compartment": "clients_paca"}


def test_ancien_client_v03_converti_par_alias(client, db):
    corps = _ecrire(client, "Erreur 500 corrigée en épinglant fastapi.",
                    type="procedural", subtype="code_error_resolution")
    assert (corps["compartment"], corps["compartment_origin"]) == ("alertes", "alias")
    famille, compartiment, _, _ = _ligne(db, corps["memory_id"])
    assert (famille, compartiment) == ("procedural", "alertes")


def test_un_doublon_garde_son_rangement_d_origine(client, db):
    premier = _ecrire(client, "Le vault est dans C:/Obsidian.", compartment="environnement")
    second = _ecrire(client, "Le vault est dans C:/Obsidian.", compartment="savoirs")
    assert second["status"] == "duplicate"
    assert second["memory_id"] == premier["memory_id"]
    assert (second["compartment"], second["compartment_origin"]) == ("environnement", "existant")


def test_la_base_refuse_un_compartiment_hors_liste(db):
    """La contrainte CHECK est le dernier rempart : même un INSERT direct ne passe pas."""
    with db.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute(
            "INSERT INTO memories (tenant_id, agent_id, type, compartment, content, embedding, "
            "status) VALUES (%s, %s, 'semantic', 'clients_paca', 'x', %s, 'active')",
            (TENANT, AGENT, "[" + ",".join(["0.1"] * 384) + "]"))


# ─── Lecture ──────────────────────────────────────────────────────────────────

def test_le_paquet_a_toujours_les_9_compartiments(client, db):
    r = client.post("/v1/context/build", json={
        "agent_id": AGENT, "session_id": "s", "task": "t", "query": "rien"})
    assert r.status_code == 200, r.text
    assert list(r.json()["context_packet"]) == list(main.NOMS)


def test_les_reflexes_sont_injectes_sans_rapport_avec_la_question(client, db):
    _ecrire(client, "Toujours répondre en français.", compartment="reflexes", importance=0.9)
    r = client.post("/v1/context/build", json={
        "agent_id": AGENT, "session_id": "s", "task": "t",
        "query": "quelle est la taille du disque du serveur", "record_access": False})
    assert r.status_code == 200, r.text
    assert r.json()["context_packet"]["reflexes"] == ["Toujours répondre en français."]


def test_filtre_par_compartiment_strict(client, db):
    _ecrire(client, "Piège : docker cp écrase les fichiers.", compartment="alertes")
    _ecrire(client, "Décision : rester sur PostgreSQL.", compartment="decisions")
    r = client.post("/v1/retrieve", json={"agent_id": AGENT, "query": "docker",
                                          "compartments": ["alertes"]})
    assert r.status_code == 200, r.text
    assert {m["compartment"] for m in r.json()["memories"]} == {"alertes"}
    # Un nom inconnu dans un FILTRE est une erreur de l'appelant (et non un vide muet).
    r = client.post("/v1/retrieve", json={"agent_id": AGENT, "query": "x",
                                          "compartments": ["clients_paca"]})
    assert r.status_code == 422


def test_list_compartments_compte_par_agent(client, db):
    _ecrire(client, "Décision A.", compartment="decisions")
    client.post("/v1/memories", json={"agent_id": "autre_agent", "content": "Décision B.",
                                      "compartment": "decisions"})
    r = client.get("/v1/compartments", params={"agent_id": AGENT})
    assert r.status_code == 200, r.text
    comptes = {c["name"]: c["memory_count"] for c in r.json()["compartments"]}
    assert comptes["decisions"] == 1
    assert list(comptes) == list(main.NOMS)
    assert r.json()["legacy_aliases"]["preference"] == "reflexes"


def test_les_routes_de_collections_ont_disparu(client):
    assert client.get("/v1/collections", params={"agent_id": AGENT}).status_code == 404
    assert client.post("/v1/collections", json={}).status_code in (404, 405)
