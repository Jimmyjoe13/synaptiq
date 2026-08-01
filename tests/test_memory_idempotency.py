"""Idempotence de `POST /v1/memories`, de bout en bout (correctif du 01/08).

Le bug : le chemin direct était un `INSERT` nu. Un client qui relançait après un timeout
perçu — alors que le premier appel avait abouti côté serveur — créait une SECONDE ligne, sans
erreur ni trace. Reproduit deux fois en conditions réelles avant correctif.

Le coût n'était pas surtout dans le rappel : la phase 3 de Q-EM annule les redondances
au-dessus de 0,75, et deux copies ont un cosinus de 1,0, donc `build_context` n'en servait
qu'une. Il était dans le GRAPHE — un clone est fatalement le premier des 3 voisins retenus
par l'intrication, et rien n'annule une arête déjà écrite.

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
TENANT = "idem_test_tenant"
CONTENU = "Le serveur de recette ecoute sur 10.1.2.3 port 8443, certificat auto-signe."


@pytest.fixture
def db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    purge_tenants(conn, TENANT)
    yield conn
    purge_tenants(conn, TENANT)
    conn.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_TENANT", TENANT)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    main.invalidate_auth_cache()
    with TestClient(app) as c:
        yield c


def _ecrire(client, contenu=CONTENU, **extra):
    charge = {"agent_id": "agent_idem", "type": "semantic", "subtype": "fact",
              "content": contenu}
    charge.update(extra)
    return client.post("/memories", json=charge)


def _compter(db, contenu=CONTENU):
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s AND content = %s "
                    "AND status = 'active'", (TENANT, contenu))
        return cur.fetchone()[0]


def test_deux_ecritures_identiques_ne_font_qu_une_ligne(client, db):
    """LE test de régression : la relance après timeout perçu."""
    premiere = _ecrire(client)
    seconde = _ecrire(client)

    assert premiere.status_code == 201, premiere.text
    assert seconde.status_code == 201, seconde.text
    assert premiere.json()["status"] == "created"
    assert seconde.json()["status"] == "duplicate"
    # Même identifiant rendu : l'appelant obtient une référence utilisable, pas une erreur.
    assert seconde.json()["memory_id"] == premiere.json()["memory_id"]
    assert _compter(db) == 1


def test_la_reponse_dupliquee_a_la_meme_forme_que_la_creation(client, db):
    """Deux formes de réponse obligeraient chaque client à traiter le cas dégradé à part."""
    cles_creation = set(_ecrire(client).json())
    cles_doublon = set(_ecrire(client).json())
    assert cles_creation == cles_doublon


def test_la_casse_et_les_blancs_ne_creent_pas_un_nouveau_souvenir(client, db):
    """La déduplication porte sur le contenu NORMALISÉ (cf. `normalize_for_hash`)."""
    _ecrire(client)
    variante = _ecrire(client, contenu="  LE SERVEUR DE RECETTE  ECOUTE SUR 10.1.2.3 "
                                       "PORT 8443, CERTIFICAT AUTO-SIGNE.  ")
    assert variante.json()["status"] == "duplicate"
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s AND status = 'active'",
                    (TENANT,))
        assert cur.fetchone()[0] == 1


def test_un_contenu_different_cree_bien_une_ligne(client, db):
    """Garde-fou : la déduplication ne doit pas devenir un refus d'écrire."""
    _ecrire(client)
    autre = _ecrire(client, contenu="Le serveur de production ecoute sur 10.9.9.9 port 443.")
    assert autre.json()["status"] == "created"
    assert autre.json()["memory_id"] != _ecrire(client).json()["memory_id"]


def test_un_souvenir_archive_peut_etre_re_affirme(client, db):
    """Pourquoi les index sont bornés à `status = 'active'`.

    Archiver un fait puis le ré-affirmer plus tard est légitime : une décision revient, une
    préférence redevient vraie. Contraindre sur TOUTES les lignes rendrait tout archivage
    définitif — un moteur de mémoire ne peut pas s'interdire de changer d'avis deux fois.
    """
    premier = _ecrire(client).json()["memory_id"]
    with db.cursor() as cur:
        cur.execute("UPDATE memories SET status = 'archived' WHERE id = %s", (premier,))
        db.commit()

    renaissance = _ecrire(client)

    assert renaissance.json()["status"] == "created"
    assert renaissance.json()["memory_id"] != premier
    assert _compter(db) == 1  # une seule ACTIVE : l'ancienne est archivée


def test_la_cle_d_idempotence_deduplique_des_contenus_differents(client, db):
    """Le complément explicite, pour les appelants qui ont une vraie clé stable."""
    premiere = _ecrire(client, idempotency_key="import-4711")
    seconde = _ecrire(client, contenu="Formulation entierement differente du meme fait.",
                      idempotency_key="import-4711")

    assert premiere.json()["status"] == "created"
    assert seconde.json()["status"] == "duplicate"
    assert seconde.json()["memory_id"] == premiere.json()["memory_id"]


def test_le_content_hash_est_desormais_renseigne(client, db):
    """La colonne existait déjà et le chemin direct ne l'alimentait jamais.

    Sans elle, l'index unique ne peut rien contraindre : c'est la cause racine.
    """
    identifiant = _ecrire(client).json()["memory_id"]
    with db.cursor() as cur:
        cur.execute("SELECT content_hash FROM memories WHERE id = %s", (identifiant,))
        empreinte = cur.fetchone()[0]
    assert empreinte is not None and len(empreinte) == 64


def test_le_doublon_ne_pollue_pas_le_paquet_de_contexte(client, db):
    """Vérification du bout de la chaîne : un seul exemplaire servi à l'agent."""
    _ecrire(client)
    _ecrire(client)

    resp = client.post("/context/build", json={
        "agent_id": "agent_idem", "session_id": "s1",
        "task": "Se connecter au serveur de recette",
        "query": "serveur de recette port certificat",
        "constraints": {"max_tokens": 800}})

    assert resp.status_code == 200, resp.text
    assert len(resp.json()["selected_memory_ids"]) == 1
