"""Lot 3 : les outils MCP par lesquels l'agent structure sa propre mémoire.

Le test central est `test_store_memory_previent_quand_le_rangement_n_a_pas_eu_lieu` : c'est
l'aveuglement qui rendait l'autonomie illusoire. L'API renvoyait déjà `collection` et
`canonical_subtype` ; l'outil MCP les jetait, et l'agent croyait ranger finement alors que
son libellé retombait dans la section par défaut de sa famille.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastmcp", reason="serveur MCP non installé (requirements-dev)")

import apps.mcp.server as mcp_server


class _Reponse:
    def __init__(self, charge, status=200):
        self._charge, self.status_code = charge, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._charge

    @property
    def text(self):
        return str(self._charge)


@pytest.fixture
def http(monkeypatch):
    """Intercepte GET et POST, rejoue une réponse contrôlée, enregistre les appels."""
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "agent_de_test")
    appels: list[dict] = []
    etat = {"charge": {}, "status": 200}

    def _post(url, json=None, headers=None, timeout=None):
        appels.append({"methode": "POST", "url": url, "payload": json})
        return _Reponse(etat["charge"], etat["status"])

    def _get(url, params=None, headers=None, timeout=None):
        appels.append({"methode": "GET", "url": url, "params": params})
        return _Reponse(etat["charge"], etat["status"])

    monkeypatch.setattr(mcp_server.requests, "post", _post)
    monkeypatch.setattr(mcp_server.requests, "get", _get)
    return appels, etat


def _outil(nom):
    """Fonction Python sous-jacente à un outil MCP (FastMCP enveloppe l'original)."""
    fn = getattr(mcp_server, nom)
    return getattr(fn, "fn", fn)


# ─── 1. store_memory rend son verdict de rangement ───────────────────────────

def test_store_memory_annonce_la_section_servie(http):
    _, etat = http
    etat["charge"] = {"memory_id": "mem-1", "collection": "clients_paca",
                      "canonical_subtype": False}
    resultat = _outil("store_memory")("Nana couvre Marseille", "semantic", "clients_paca")
    assert "[SUCCESS]" in resultat
    assert "clients_paca" in resultat


def test_store_memory_previent_quand_le_rangement_n_a_pas_eu_lieu(http):
    """LE correctif du lot : l'agent doit savoir que son libellé n'a pas produit de rayon.

    Avant, la réponse était `[SUCCESS] Memoire enregistree. ID: ...` — strictement la même
    que pour un rangement réussi. L'agent n'avait aucun moyen de distinguer les deux, donc
    aucun moyen d'apprendre.
    """
    _, etat = http
    # Libellé libre non déclaré : l'API le range dans `facts`, le repli de `semantic`.
    etat["charge"] = {"memory_id": "mem-1", "collection": "facts",
                      "canonical_subtype": False}
    resultat = _outil("store_memory")("un contenu", "semantic", "clients_paca")

    assert "[INFO]" in resultat
    assert "n'est pas une collection declaree" in resultat
    assert "facts" in resultat
    # Et le message doit dire QUOI FAIRE, pas seulement constater.
    assert "create_collection" in resultat


def test_store_memory_ne_previent_pas_sur_un_sous_type_canonique(http):
    """`preference` est canonique : aucun avertissement, ce serait du bruit."""
    _, etat = http
    etat["charge"] = {"memory_id": "mem-1", "collection": "preferences",
                      "canonical_subtype": True}
    resultat = _outil("store_memory")("Jimmy aime les mails courts", "semantic", "preference")
    assert "[INFO]" not in resultat


def test_store_memory_ne_previent_pas_sans_sous_type(http):
    _, etat = http
    etat["charge"] = {"memory_id": "mem-1", "collection": "facts", "canonical_subtype": False}
    assert "[INFO]" not in _outil("store_memory")("un fait", "semantic")


# ─── 2. list_collections ─────────────────────────────────────────────────────

def test_list_collections_affiche_origine_volume_et_graphe(http):
    appels, etat = http
    etat["charge"] = {"collections": [
        {"name": "fact", "family": "semantic", "packet_key": "facts", "description": "",
         "entangle": True, "created_by": "system", "memory_count": 12},
        {"name": "clients_paca", "family": "semantic", "packet_key": "clients_paca",
         "description": "Clients de la region PACA.", "entangle": False,
         "created_by": "agent", "memory_count": 0},
    ]}
    resultat = _outil("list_collections")()

    assert appels[0]["methode"] == "GET"
    assert appels[0]["params"] == {"agent_id": "agent_de_test"}
    assert "systeme" in resultat and "agent" in resultat
    assert "12 souvenir(s)" in resultat
    # Une collection déclarée mais vide doit apparaître : c'est une information.
    assert "0 souvenir(s)" in resultat
    assert "hors graphe" in resultat and "intriquee" in resultat
    assert "Clients de la region PACA." in resultat


def test_list_collections_sur_registre_vide(http):
    _, etat = http
    etat["charge"] = {"collections": []}
    assert "Aucune collection" in _outil("list_collections")()


# ─── 3. create_collection ────────────────────────────────────────────────────

def test_create_collection_rend_le_mode_d_emploi(http):
    appels, etat = http
    etat["charge"] = {"status": "created", "name": "clients_paca", "family": "semantic",
                      "packet_key": "clients_paca", "entangle": True,
                      "usage": {"type": "semantic", "subtype": "clients_paca"}}
    resultat = _outil("create_collection")(
        "clients_paca", "semantic", "Clients de la region PACA.")

    assert "[SUCCESS]" in resultat
    # L'agent doit repartir en sachant écrire dedans, sans avoir à déduire le couple.
    assert "memory_type='semantic'" in resultat
    assert "subtype='clients_paca'" in resultat
    assert appels[0]["payload"]["agent_id"] == "agent_de_test"
    assert appels[0]["payload"]["entangle"] is True


def test_create_collection_transmet_le_refus_metier_tel_quel(http):
    """Un `HTTP 409` ne dit rien à un modèle ; le détail de l'API, si."""
    _, etat = http
    etat["status"] = 409
    etat["charge"] = {"detail": "Plafond atteint (50 collections). Reutiliser ou fusionner."}
    resultat = _outil("create_collection")("un_rayon", "semantic", "Une description.")
    assert resultat.startswith("[REFUSE]")
    assert "Plafond atteint" in resultat


def test_create_collection_transmet_aussi_le_422(http):
    _, etat = http
    etat["status"] = 422
    etat["charge"] = {"detail": "'fact' est une collection systeme."}
    assert "[REFUSE]" in _outil("create_collection")("fact", "semantic", "Une description.")


# ─── 4. build_context sert AUSSI les sections de l'agent ─────────────────────

def test_build_context_affiche_les_sections_de_l_agent(http):
    """RÉGRESSION : l'outil itérait sur sept libellés codés en dur.

    Les sections créées par l'agent auraient été écartées en silence — son propre rangement
    invisible dans son propre contexte, sans le moindre signal. Tout le lot 2 aurait été
    inopérant vu depuis MCP.
    """
    _, etat = http
    etat["charge"] = {"token_estimate": 42, "context_packet": {
        "facts": ["un fait ordinaire"],
        "preferences": [], "episodes": [], "rules": [], "best_practices": [],
        "errors": [], "examples": [],
        "clients_paca": ["Nana couvre Marseille"],
    }}
    resultat = _outil("build_context")("une tache", "une requete")

    assert "[FAITS] un fait ordinaire" in resultat
    assert "[CLIENTS_PACA] Nana couvre Marseille" in resultat
    # Les canoniques restent en tête, dans leur ordre.
    assert resultat.index("[FAITS]") < resultat.index("[CLIENTS_PACA]")


def test_les_sections_vides_ne_produisent_aucune_ligne(http):
    """Pourquoi il n'y a PAS de plafond sur le nombre de sections servies.

    Le plan prévoyait de borner les sections du paquet, par crainte qu'un agent à quarante
    collections noie le modèle sous trente-cinq rubriques vides. Vérification faite, ce
    plafond serait inutile ET nuisible : il entrerait en conflit avec la garantie de forme
    stable du paquet (une collection déclarée apparaît même vide), alors que le rendu, lui,
    n'imprime QUE les sections qui ont du contenu. Le prompt ne voit donc jamais de rubrique
    vide, quel que soit le nombre de collections.

    Ce test verrouille cette propriété : c'est elle qui rend le plafond superflu.
    """
    _, etat = http
    etat["charge"] = {"token_estimate": 5, "context_packet": {
        "facts": ["le seul contenu"],
        "preferences": [], "episodes": [], "rules": [], "best_practices": [],
        "errors": [], "examples": [],
        **{f"rayon_{i}": [] for i in range(40)},
    }}
    resultat = _outil("build_context")("t", "q")

    lignes = [li for li in resultat.splitlines() if li.startswith("- ")]
    assert lignes == ["- [FAITS] le seul contenu"]
    assert "RAYON_0" not in resultat


def test_build_context_transmet_le_filtre_de_collections(http):
    appels, etat = http
    etat["charge"] = {"token_estimate": 0, "context_packet": {}}
    _outil("build_context")("t", "q", collections=["clients_paca"])
    assert appels[0]["payload"]["constraints"]["collections"] == ["clients_paca"]


def test_build_context_omet_le_filtre_quand_il_est_absent(http):
    """Une liste vide serait un filtre qui ne ramène rien : l'absence doit rester l'absence."""
    appels, etat = http
    etat["charge"] = {"token_estimate": 0, "context_packet": {}}
    _outil("build_context")("t", "q")
    assert "collections" not in appels[0]["payload"]["constraints"]


# ─── 5. L'identité reste hors de portée du modèle ────────────────────────────

def test_les_nouveaux_outils_n_exposent_pas_agent_id(monkeypatch):
    """RÉGRESSION F2, étendue aux outils du lot 3.

    `agent_id` était un paramètre d'outil : c'était donc le LLM qui choisissait sous quelle
    identité lire et écrire. Deux outils de plus, deux occasions de le réintroduire.
    """
    import inspect
    for nom in ("store_memory", "recall_memories", "build_context",
                "list_collections", "create_collection", "merge_collections"):
        params = inspect.signature(_outil(nom)).parameters
        assert "agent_id" not in params, f"{nom} expose agent_id"
        assert "tenant_id" not in params, f"{nom} expose tenant_id"


def test_les_nouveaux_outils_echouent_sans_identite(monkeypatch):
    """Sans `SYNAPTIQ_AGENT_ID`, chaque outil doit rendre un message actionnable."""
    monkeypatch.setattr(mcp_server, "SYNAPTIQ_AGENT_ID", "")
    for nom, args in (("list_collections", ()),
                      ("merge_collections", ("a_vider", "la_cible")),
                      ("create_collection", ("un_rayon", "semantic", "Une description."))):
        resultat = _outil(nom)(*args)
        assert "[ERROR]" in resultat
        assert "SYNAPTIQ_AGENT_ID" in resultat


# ─── 6. Lot 4 : les garde-fous vus depuis l'agent ────────────────────────────

def test_list_collections_signale_les_dormantes_et_le_quota(http):
    """Un défaut que l'agent ne voit pas est un défaut qu'il ne corrigera pas."""
    _, etat = http
    etat["charge"] = {
        "collections": [
            {"name": "clients_paca", "family": "semantic", "packet_key": "clients_paca",
             "description": "Clients PACA.", "entangle": True, "created_by": "agent",
             "memory_count": 7, "stale": False},
            {"name": "jamais_utilisee", "family": "semantic", "packet_key": "jamais_utilisee",
             "description": "Un rayon oublie.", "entangle": True, "created_by": "agent",
             "memory_count": 0, "stale": True},
        ],
        "limits": {"max_collections": 50, "used": 2},
    }
    resultat = _outil("list_collections")()

    assert "DORMANTE" in resultat
    assert "[ATTENTION]" in resultat
    assert "jamais_utilisee" in resultat
    # L'issue doit être nommée, pas seulement le problème.
    assert "merge_collections" in resultat
    assert "2 / 50" in resultat


def test_list_collections_sans_dormante_ne_crie_pas(http):
    _, etat = http
    etat["charge"] = {"collections": [
        {"name": "clients_paca", "family": "semantic", "packet_key": "clients_paca",
         "description": "", "entangle": True, "created_by": "agent",
         "memory_count": 7, "stale": False}], "limits": {"max_collections": 50, "used": 1}}
    assert "[ATTENTION]" not in _outil("list_collections")()


def test_merge_collections_rend_le_nombre_de_souvenirs_deplaces(http):
    appels, etat = http
    etat["charge"] = {"status": "merged", "source": "prospects_paca",
                      "target": "clients_paca", "moved_memories": 12}
    resultat = _outil("merge_collections")("prospects_paca", "clients_paca")

    assert "[SUCCESS]" in resultat
    assert "12 souvenir(s) deplace(s)" in resultat
    assert appels[0]["payload"] == {"agent_id": "agent_de_test",
                                    "source": "prospects_paca", "target": "clients_paca"}


def test_merge_collections_transmet_le_refus_metier(http):
    """Un `HTTP 422` ne dit rien à un modèle ; la raison, si."""
    _, etat = http
    etat["status"] = 422
    etat["charge"] = {"detail": "Familles differentes ('episodic' vers 'semantic')."}
    resultat = _outil("merge_collections")("incidents", "clients_paca")
    assert resultat.startswith("[REFUSE]")
    assert "Familles differentes" in resultat


def test_create_collection_transmet_le_doublon_semantique(http):
    """Le refus doit NOMMER le rayon proche, sinon l'agent ne sait pas où ranger."""
    _, etat = http
    etat["status"] = 409
    etat["charge"] = {"detail": "'clients_paca' decrit deja la meme chose (similarite 0.93)."}
    resultat = _outil("create_collection")(
        "clients_region_paca", "semantic", "Clients de la region PACA.")
    assert "[REFUSE]" in resultat
    assert "clients_paca" in resultat


def test_recall_memories_transmet_le_filtre_de_collections(http):
    appels, etat = http
    etat["charge"] = {"memories": []}
    _outil("recall_memories")("une requete", collections=["clients_paca"])
    assert appels[0]["payload"]["collections"] == ["clients_paca"]


def test_recall_memories_omet_le_filtre_quand_il_est_absent(http):
    appels, etat = http
    etat["charge"] = {"memories": []}
    _outil("recall_memories")("une requete")
    assert "collections" not in appels[0]["payload"]
