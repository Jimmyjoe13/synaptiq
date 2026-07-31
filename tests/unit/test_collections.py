"""Registre de collections : le rangement devient un objet que l'agent possède.

Deux exigences dominent ces tests :

1. **Aucune régression de routage.** Le registre système doit produire EXACTEMENT le même
   rangement que la cascade de `if` qu'il remplace, sur les sept collections canoniques
   comme sur les sous-types libres.
2. **Plus jamais de disparition silencieuse.** `route_memory` renvoyait `None` sur un type
   inconnu et `collapse_by_utility` retirait alors la mémoire du paquet — après l'avoir
   comptée et facturée en tokens. C'est cette classe de panne que le registre ferme.
"""
import pytest

from synaptiq_core.collections import (
    FAMILY_FALLBACK_KEY,
    SYSTEM_COLLECTIONS,
    SYSTEM_PACKET_KEYS,
    Collection,
    CollectionRegistry,
)
from synaptiq_core.qem import collapse_by_utility, route_memory
from synaptiq_core.taxonomy import DEFAULT_SUBTYPE, VALID_SUBTYPES

# ─── 1. Non-régression du routage ────────────────────────────────────────────

@pytest.mark.parametrize(("famille", "collection", "attendu"), [
    ("semantic", "preference", "preferences"),
    ("semantic", "fact", "facts"),
    ("semantic", None, "facts"),
    ("semantic", "nana_intelligence_lead_webhook", "facts"),   # libre -> repli famille
    ("episodic", "interaction", "episodes"),
    ("episodic", "n_importe_quoi", "episodes"),
    ("procedural", "coding_best_practices", "best_practices"),
    ("procedural", "code_error_resolution", "errors"),
    ("procedural", "rule", "rules"),
    ("procedural", "libre", "rules"),
    ("working", "scratch", "examples"),
])
def test_routage_identique_a_la_cascade_historique(famille, collection, attendu):
    """Le registre système reproduit le comportement d'origine, cas par cas."""
    assert route_memory(famille, collection) == attendu


def test_les_sept_collections_systeme_couvrent_les_sept_sections():
    """Une section de paquet sans collection qui l'alimente serait morte."""
    assert {c.packet_key for c in SYSTEM_COLLECTIONS} == set(SYSTEM_PACKET_KEYS)


# ─── 2. Plus de disparition silencieuse ──────────────────────────────────────

def test_famille_inconnue_ne_renvoie_plus_none():
    """RÉGRESSION : `None` faisait disparaître la mémoire du paquet, sans aucun signal.

    Elle était pourtant comptée dans `selected_ids` et son budget de tokens dépensé. Un
    rangement de repli est visible et corrigeable ; une disparition muette, non.
    """
    assert route_memory("famille_inventee", "peu importe") == "facts"


def test_une_memoire_de_famille_inconnue_atteint_le_paquet():
    """Le bout de la chaîne : elle doit sortir dans le paquet, pas s'évaporer."""
    candidats = {
        "m1": {"type": "famille_inventee", "subtype": "x", "content": "un souvenir orphelin",
               "score": 1.0},
    }
    packet, selectionnes, tokens = collapse_by_utility(candidats, max_tokens=100)
    assert selectionnes == ["m1"]
    assert tokens > 0
    # Avant : selected_ids=['m1'], tokens dépensés, et AUCUNE section ne contenait le texte.
    assert any("un souvenir orphelin" in entree
               for section in packet.values() for entree in section)


def test_une_cle_de_paquet_personnalisee_ne_leve_pas():
    """Une collection d'agent peut porter une clé hors des sept : pas de KeyError."""
    registre = CollectionRegistry.depuis([
        Collection("client_nana", "semantic", "clients", created_by="agent"),
    ])
    assert registre.packet_key("semantic", "client_nana") == "clients"


# ─── 3. Priorité agent > système, et complétion ──────────────────────────────

def test_la_collection_de_l_agent_gagne_sur_l_homonyme_systeme():
    """Un agent peut spécialiser un nom canonique pour lui, sans affecter les autres."""
    registre = CollectionRegistry.depuis([
        Collection("fact", "semantic", "clients", created_by="agent"),
    ])
    assert registre.packet_key("semantic", "fact") == "clients"
    # Le registre système, lui, n'a pas bougé.
    assert CollectionRegistry.systeme().packet_key("semantic", "fact") == "facts"


def test_les_collections_systeme_sont_toujours_completees():
    """Un magasin qui ne renvoie que les collections d'agent ne doit rien casser."""
    registre = CollectionRegistry.depuis([
        Collection("client_nana", "semantic", "facts", created_by="agent"),
    ])
    assert registre.packet_key("procedural", "code_error_resolution") == "errors"
    assert registre.get("semantic", "client_nana") is not None


def test_packet_keys_ajoute_les_cles_de_l_agent_apres_les_canoniques():
    registre = CollectionRegistry.depuis([
        Collection("client_nana", "semantic", "clients", created_by="agent"),
    ])
    cles = registre.packet_keys()
    assert cles[:len(SYSTEM_PACKET_KEYS)] == SYSTEM_PACKET_KEYS
    assert cles[-1] == "clients"


# ─── 4. Intrication portée par la collection ─────────────────────────────────

def test_entangle_reproduit_le_defaut_historique():
    """Défaut `QEM_ENTANGLE_TYPES=procedural,semantic` : inchangé tant que rien n'est déclaré."""
    registre = CollectionRegistry.systeme()
    assert registre.entangle_pour("semantic", "fact") is True
    assert registre.entangle_pour("procedural", "rule") is True
    assert registre.entangle_pour("episodic", "interaction") is False
    # Collection libre : hérite de sa famille.
    assert registre.entangle_pour("semantic", "libre") is True
    assert registre.entangle_pour("episodic", "libre") is False


def test_l_agent_peut_demander_l_intrication_d_une_collection_d_episodes():
    """Le gain réel du lot : les épisodes ne tissaient AUCUNE arête, globalement.

    Le multi-hop est la dimension où Q-EM creuse l'écart ; pouvoir décider collection par
    collection, c'est pouvoir densifier le graphe là où ça compte.
    """
    registre = CollectionRegistry.depuis([
        Collection("reunions_client", "episodic", "episodes", entangle=True,
                   created_by="agent"),
    ])
    assert registre.entangle_pour("episodic", "reunions_client") is True
    assert registre.entangle_pour("episodic", "interaction") is False


# ─── 5. Cohérence avec la taxonomie dérivée ──────────────────────────────────

def test_la_taxonomie_derive_bien_des_collections_systeme():
    """`VALID_SUBTYPES` était une seconde liste tenue à la main : elle dérive désormais."""
    assert VALID_SUBTYPES == {
        "procedural": {"code_error_resolution", "coding_best_practices", "rule"},
        "semantic": {"preference", "fact"},
        "episodic": {"interaction"},
        "working": {"scratch"},
    }


def test_les_defauts_suivent_le_repli_de_famille():
    """Le sous-type par défaut est celui qui sert la section de repli de sa famille."""
    assert DEFAULT_SUBTYPE == {"procedural": "rule", "semantic": "fact",
                               "episodic": "interaction", "working": "scratch"}
    for famille, nom in DEFAULT_SUBTYPE.items():
        assert route_memory(famille, nom) == FAMILY_FALLBACK_KEY[famille]


def test_une_famille_hors_des_quatre_est_refusee_a_la_construction():
    """La famille porte le comportement du moteur : elle ne peut pas être inventée."""
    with pytest.raises(ValueError, match="Famille inconnue"):
        Collection("x", "marketing", "facts")
