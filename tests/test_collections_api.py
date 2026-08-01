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
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
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


# ─── Lot 2 : le paquet servi suit le registre ────────────────────────────────

# Les deux branches SQL de `_fetch_candidates` sont modifiées par le filtre de collection :
# elles doivent donc TOUTES DEUX être exercées. `RETRIEVAL_HYBRID` est lu à l'appel, un
# monkeypatch du module suffit.
#
# ⚠️ Les requêtes reprennent le contenu MOT POUR MOT. Avec `EMBEDDING_PROVIDER=mock`, les
# vecteurs ne sont pas sémantiques : deux textes différents sont quasi orthogonaux, donc
# `similarity ≈ 0`. En hybride le plein texte rattrape, mais en vectoriel pur le score de
# départ vaut le seul cosinus et le collapse écarte tout candidat à score nul. Un texte
# identique garantit un cosinus de 1 dans les deux modes — sinon le test serait
# intermittent, et sa couleur dépendrait du hasard du vecteur factice.
CONTENU_PACA = "Nana Intelligence couvre Marseille, Aix, Toulon et Nice."
CONTENU_AUTRE = "Un fait sans aucun rapport avec la prospection."


@pytest.fixture(params=[True, False], ids=["hybride", "vectoriel_pur"])
def mode_recherche(request, monkeypatch):
    monkeypatch.setattr(main, "RETRIEVAL_HYBRID", request.param)
    return request.param


def test_le_context_packet_porte_la_section_de_l_agent(client, db, mode_recherche):
    """Bout en bout : la collection déclarée devient une SECTION du contexte servi au LLM.

    C'est le livrable observable du lot 2 : jusqu'ici le libellé métier de l'agent était
    dilué dans `facts` et n'apparaissait nulle part dans le paquet.
    """
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")
    client.post("/memories", json={
        "agent_id": "agentA", "type": "semantic", "subtype": "clients_paca",
        "content": CONTENU_PACA,
    })

    resp = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1",
        "task": "Preparer une prospection", "query": CONTENU_PACA,
    })
    assert resp.status_code == 200, resp.text
    packet = resp.json()["context_packet"]
    assert "clients_paca" in packet
    assert any("Marseille" in e for e in packet["clients_paca"])
    # Les sept canoniques restent présentes : le contrat public ne se réduit pas.
    for canonique in ("facts", "preferences", "episodes", "rules", "best_practices",
                      "errors", "examples"):
        assert canonique in packet


def test_la_section_apparait_meme_quand_la_recherche_ne_ramene_rien(client, db):
    """La forme de la réponse ne doit pas dépendre de la présence de résultats."""
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")
    resp = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1", "task": "t", "query": "aucun souvenir",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["context_packet"]["clients_paca"] == []


def test_le_filtre_par_collection_borne_la_recherche(client, db, mode_recherche):
    """Viser un rayon précis au lieu de ratisser toute une famille.

    Exercé sur les DEUX branches SQL : le filtre y est répété, comme le filtre
    tenant/agent, et une seule des deux aurait pu être oubliée.
    """
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "clients_paca", "content": CONTENU_PACA})
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "fact", "content": CONTENU_AUTRE})

    # La requête vise le contenu HORS collection : sans filtre il remonterait en premier.
    resp = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1", "task": "t", "query": CONTENU_AUTRE,
        "constraints": {"max_tokens": 1200, "collections": ["clients_paca"]},
    })
    assert resp.status_code == 200, resp.text
    packet = resp.json()["context_packet"]
    # Le fait hors périmètre est exclu par le SQL, alors même qu'il matche parfaitement.
    assert packet["facts"] == []
    assert all("rapport avec la prospection" not in e
               for section in packet.values() for e in section)


def test_sans_filtre_le_hors_perimetre_remonte_bien(client, db, mode_recherche):
    """Contre-épreuve du test précédent : c'est bien le FILTRE qui excluait, pas le hasard."""
    _declarer_collection(db, "agentA", "clients_paca", "semantic", "clients_paca")
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "clients_paca", "content": CONTENU_PACA})
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "fact", "content": CONTENU_AUTRE})

    packet = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1", "task": "t", "query": CONTENU_AUTRE,
    }).json()["context_packet"]
    assert any("rapport avec la prospection" in e for e in packet["facts"])


# ─── Lot 3 : l'agent crée lui-même ses rayons ────────────────────────────────

def _creer(client, **kwargs):
    corps = {"agent_id": "agentA", "name": "clients_paca", "family": "semantic",
             "description": "Clients et prospects de la region PACA.", **kwargs}
    return client.post("/collections", json=corps)


def test_un_agent_cree_sa_collection_et_ecrit_dedans(client, db):
    """Le cycle complet d'autonomie : je déclare un rayon, j'y range, il est servi."""
    resp = _creer(client)
    assert resp.status_code == 201, resp.text
    corps = resp.json()
    # La réponse dit comment s'en servir : sans ça l'agent doit deviner que « collection »
    # se traduit par le couple (type, subtype).
    assert corps["usage"] == {"type": "semantic", "subtype": "clients_paca"}

    ecriture = client.post("/memories", json={
        "agent_id": "agentA", **corps["usage"], "content": CONTENU_PACA})
    assert ecriture.status_code == 201, ecriture.text
    assert ecriture.json()["collection"] == "clients_paca"

    packet = client.post("/context/build", json={
        "agent_id": "agentA", "session_id": "s1", "task": "t", "query": CONTENU_PACA,
    }).json()["context_packet"]
    assert any("Marseille" in e for e in packet["clients_paca"])


def test_la_creation_est_journalisee(client, db):
    """Structurer sa mémoire est une opération sensible : elle laisse une trace."""
    _creer(client)
    with db.cursor() as cur:
        cur.execute("SELECT action, details FROM audit_log WHERE tenant_id = %s "
                    "AND action = 'create_collection'", (TENANT,))
        lignes = cur.fetchall()
    assert len(lignes) == 1
    assert lignes[0][1]["name"] == "clients_paca"


def test_un_nom_canonique_est_refuse(client, db):
    """Laisser un agent redéfinir `fact` reroute en silence tout ce qui y est déjà rangé."""
    resp = _creer(client, name="fact")
    assert resp.status_code == 422
    assert "systeme" in resp.json()["detail"] or "système" in resp.json()["detail"]


def test_le_meme_nom_deux_fois_est_refuse(client, db):
    assert _creer(client).status_code == 201
    doublon = _creer(client, description="Une autre description, meme nom.")
    assert doublon.status_code == 409
    assert "existe deja" in doublon.json()["detail"] or "existe déjà" in doublon.json()["detail"]


def test_un_nom_mal_forme_est_refuse(client, db):
    """Le nom voyage jusque dans les clés du paquet : il doit rester lisible et stable."""
    assert _creer(client, name="Clients PACA !").status_code == 422
    assert _creer(client, name="a").status_code == 422


def test_une_famille_inventee_est_refusee(client, db):
    """La famille porte un comportement du moteur : elle n'est pas ouverte."""
    assert _creer(client, family="marketing").status_code == 422


def test_une_description_indigente_est_refusee(client, db):
    """La description sert à décider plus tard s'il faut réutiliser : elle est obligatoire."""
    assert _creer(client, description="x").status_code == 422


def test_le_plafond_protege_de_l_emballement(client, db, monkeypatch):
    """Un LLM crée une catégorie à chaque nouveauté si on le laisse faire.

    Quarante rayons, c'est un paquet éclaté en quarante sections dont l'essentiel est vide —
    plus dur à exploiter que les sept d'origine.
    """
    monkeypatch.setattr(main, "MAX_COLLECTIONS_PER_AGENT", 2)
    # Descriptions DISTINCTES : sinon c'est l'anti-doublon sémantique qui refuserait, et le
    # test passerait au vert pour la mauvaise raison.
    assert _creer(client, name="rayon_un",
                  description="Contrats signes et avenants clients.").status_code == 201
    assert _creer(client, name="rayon_deux",
                  description="Incidents de production horodates.").status_code == 201
    trop = _creer(client, name="rayon_trois",
                  description="Recettes de cuisine et menus hebdomadaires.")
    assert trop.status_code == 409
    assert "Plafond" in trop.json()["detail"]


def test_la_creation_exige_le_scope_write(client, db, monkeypatch):
    monkeypatch.setattr(main, "AUTH_REQUIRED", True)
    assert _creer(client).status_code == 401


def test_une_collection_hors_graphe_est_bien_enregistree(client, db):
    """`entangle=False` : l'agent retire volontairement du bruit du graphe."""
    resp = _creer(client, name="brouillons", entangle=False,
                  description="Notes de travail jetables, non structurantes.")
    assert resp.status_code == 201
    assert resp.json()["entangle"] is False

    par_nom = {c["name"]: c for c in
               client.get("/collections", params={"agent_id": "agentA"}).json()["collections"]}
    assert par_nom["brouillons"]["entangle"] is False


# ─── Lot 4 : les garde-fous anti-dérive ──────────────────────────────────────

DESC_PACA = "Clients et prospects de la region PACA, Marseille Aix Toulon Nice."


def test_un_doublon_semantique_est_refuse(client, db):
    """LE garde-fou du lot : deux NOMS distincts pour la même chose.

    `clients_paca` et `clients_region_paca` sont deux chaînes différentes — l'unicité du
    nom ne protège de rien. C'est la description qu'il faut comparer, avec l'outil que le
    produit sait déjà manier : la similarité vectorielle.
    """
    assert _creer(client, name="clients_paca", description=DESC_PACA).status_code == 201

    # Description IDENTIQUE : avec l'embedder mock (déterministe sur le texte), le cosinus
    # vaut 1.0 — le cas du doublon franc, sans dépendre d'une sémantique simulée.
    doublon = _creer(client, name="clients_region_paca", description=DESC_PACA)
    assert doublon.status_code == 409
    detail = doublon.json()["detail"]
    # Le refus doit NOMMER la collection proche : sinon l'agent ne sait pas où ranger.
    assert "clients_paca" in detail
    assert "similarit" in detail


def test_une_collection_reellement_distincte_passe(client, db):
    assert _creer(client, name="clients_paca", description=DESC_PACA).status_code == 201
    autre = _creer(client, name="incidents_prod",
                   description="Pannes de production et leur resolution, horodatees.")
    assert autre.status_code == 201, autre.text


def test_le_seuil_de_doublon_est_reglable(client, db, monkeypatch):
    """Un seuil à 1.01 rend le contrôle inopérant : preuve que c'est bien lui qui refuse."""
    monkeypatch.setattr(main, "COLLECTION_DUP_THRESHOLD", 1.01)
    assert _creer(client, name="clients_paca", description=DESC_PACA).status_code == 201
    assert _creer(client, name="clients_region_paca", description=DESC_PACA).status_code == 201


def test_le_doublon_est_detecte_contre_les_collections_systeme(client, db):
    """Le cas le plus probable pour un agent qui débute : redoubler un rayon livré.

    Les collections système n'ont pas de vecteur en base ; sans l'embarquement à la volée,
    la protection serait inopérante précisément là où elle sert le plus.
    """
    systeme = "Faits stables sur une personne, une entite ou le monde."
    resp = _creer(client, name="faits_generaux", description=systeme)
    assert resp.status_code == 409
    assert "fact" in resp.json()["detail"]


# ─── Fusion ──────────────────────────────────────────────────────────────────

def test_la_fusion_deplace_les_souvenirs_et_supprime_la_source(client, db):
    """Sans fusion, une taxonomie ne peut que grossir."""
    _creer(client, name="clients_paca", description=DESC_PACA)
    _creer(client, name="prospects_paca",
           description="Contacts commerciaux pas encore convertis en clients signes.")
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "prospects_paca", "content": CONTENU_PACA})

    resp = client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "prospects_paca", "target": "clients_paca"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["moved_memories"] == 1

    noms = {c["name"] for c in
            client.get("/collections", params={"agent_id": "agentA"}).json()["collections"]}
    assert "prospects_paca" not in noms
    assert "clients_paca" in noms

    # Le souvenir n'est pas détruit : il a changé d'étiquette.
    with db.cursor() as cur:
        cur.execute("SELECT subtype FROM memories WHERE tenant_id = %s AND agent_id = %s",
                    (TENANT, "agentA"))
        assert [li[0] for li in cur.fetchall()] == ["clients_paca"]


def test_la_fusion_est_journalisee(client, db):
    _creer(client, name="clients_paca", description=DESC_PACA)
    _creer(client, name="prospects_paca", description="Contacts pas encore convertis.")
    client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "prospects_paca", "target": "clients_paca"})
    with db.cursor() as cur:
        cur.execute("SELECT details FROM audit_log WHERE tenant_id = %s "
                    "AND action = 'merge_collections'", (TENANT,))
        lignes = cur.fetchall()
    assert len(lignes) == 1
    assert lignes[0][0]["source"] == "prospects_paca"


def test_une_collection_systeme_ne_peut_pas_etre_fusionnee(client, db):
    """Elle sert TOUS les agents : la retirer depuis une requête d'agent serait un dégât."""
    _creer(client, name="clients_paca", description=DESC_PACA)
    resp = client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "fact", "target": "clients_paca"})
    assert resp.status_code == 422
    assert "systeme" in resp.json()["detail"] or "système" in resp.json()["detail"]


def test_la_fusion_entre_familles_differentes_est_refusee(client, db):
    """La famille porte un COMPORTEMENT : la changer ne serait pas qu'un rangement."""
    _creer(client, name="clients_paca", description=DESC_PACA)
    _creer(client, name="incidents", family="episodic",
           description="Incidents de production horodates et leur deroule.")
    resp = client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "incidents", "target": "clients_paca"})
    assert resp.status_code == 422
    assert "amilles" in resp.json()["detail"]


def test_la_fusion_refuse_une_collection_inconnue(client, db):
    _creer(client, name="clients_paca", description=DESC_PACA)
    assert client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "inexistante", "target": "clients_paca"
    }).status_code == 404
    assert client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "clients_paca", "target": "inexistante"
    }).status_code == 404


def test_la_fusion_sur_soi_meme_est_refusee(client, db):
    _creer(client, name="clients_paca", description=DESC_PACA)
    assert client.post("/collections/merge", json={
        "agent_id": "agentA", "source": "clients_paca", "target": "clients_paca"
    }).status_code == 422


def test_la_fusion_ne_franchit_pas_la_frontiere_entre_agents(client, db):
    """L'isolation vaut aussi pour l'entretien de la taxonomie."""
    _creer(client, name="clients_paca", description=DESC_PACA)
    resp = client.post("/collections/merge", json={
        "agent_id": "agentB", "source": "clients_paca", "target": "fact"})
    assert resp.status_code == 404


# ─── Collections dormantes ───────────────────────────────────────────────────

def test_une_collection_vide_et_ancienne_est_signalee(client, db, monkeypatch):
    """Créée puis jamais remplie : premier symptôme d'une taxonomie qui se disperse."""
    monkeypatch.setattr(main, "COLLECTION_STALE_DAYS", 0)
    _creer(client, name="jamais_utilisee", description="Un rayon cree puis oublie aussitot.")
    par_nom = {c["name"]: c for c in
               client.get("/collections", params={"agent_id": "agentA"}).json()["collections"]}
    assert par_nom["jamais_utilisee"]["stale"] is True
    # Une collection SYSTÈME vide n'est jamais dormante : elle est livrée, pas créée.
    assert par_nom["scratch"]["stale"] is False


def test_une_collection_vide_mais_recente_n_est_pas_signalee(client, db):
    """Créée il y a dix minutes et encore vide : l'agent est en train de la remplir."""
    _creer(client, name="toute_neuve", description="Un rayon cree a l'instant meme.")
    par_nom = {c["name"]: c for c in
               client.get("/collections", params={"agent_id": "agentA"}).json()["collections"]}
    assert par_nom["toute_neuve"]["stale"] is False


def test_le_quota_est_expose(client, db):
    """L'agent doit pouvoir anticiper le plafond, pas le découvrir en s'y cognant."""
    _creer(client, name="clients_paca", description=DESC_PACA)
    limites = client.get("/collections",
                         params={"agent_id": "agentA"}).json()["limits"]
    assert limites["used"] == 1
    assert limites["max_collections"] == main.MAX_COLLECTIONS_PER_AGENT


# ─── Filtre par collection sur /v1/retrieve ──────────────────────────────────

def test_retrieve_filtre_par_collection(client, db, mode_recherche):
    _creer(client, name="clients_paca", description=DESC_PACA)
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "clients_paca", "content": CONTENU_PACA})
    client.post("/memories", json={"agent_id": "agentA", "type": "semantic",
                                   "subtype": "fact", "content": CONTENU_AUTRE})

    # La requête vise le contenu HORS collection : seul le filtre peut l'écarter.
    resp = client.post("/retrieve", json={
        "agent_id": "agentA", "query": CONTENU_AUTRE, "limit": 10,
        "collections": ["clients_paca"]})
    assert resp.status_code == 200, resp.text
    contenus = [m["content"] for m in resp.json()["memories"]]
    assert all("rapport avec la prospection" not in c for c in contenus)

    sans_filtre = client.post("/retrieve", json={
        "agent_id": "agentA", "query": CONTENU_AUTRE, "limit": 10}).json()["memories"]
    assert any("rapport avec la prospection" in m["content"] for m in sans_filtre)
