"""Lot 2 : le context_packet suit le registre, et l'intrication se décide par collection.

Trois exigences :

1. **Aucune régression sans registre.** Tout appel qui n'en fournit pas doit produire
   exactement les sept sections d'avant.
2. **Une collection déclarée est VISIBLE dans le paquet**, y compris quand elle est vide —
   sinon la forme de la réponse changerait selon qu'il y a des souvenirs ou non.
3. **L'intrication cesse d'être un réglage d'instance.** C'est le seul gain de ce lot qui
   touche la qualité du rappel : le multi-hop est la dimension où Q-EM creuse l'écart, et
   `episodic` n'y participait jamais, pour personne.
"""
from synaptiq_core.collections import (
    SYSTEM_PACKET_KEYS,
    Collection,
    CollectionRegistry,
    charger_registre,
)
from synaptiq_core.context_builder import (
    PACKET_VIDE,
    InMemoryStore,
    RetrievalConfig,
    build_context_packet,
    packet_vide,
)
from synaptiq_core.qem import collapse_by_utility
from synaptiq_core.taxonomy import normalize_extraction


def _candidat(mem_id, famille, collection, contenu, score=1.0):
    return {mem_id: {"type": famille, "subtype": collection, "content": contenu,
                     "score": score}}


# ─── 1. Non-régression sans registre ─────────────────────────────────────────

def test_sans_registre_le_paquet_garde_ses_sept_sections():
    packet, _, _ = collapse_by_utility(
        _candidat("m1", "semantic", "fact", "un fait"), max_tokens=100)
    assert tuple(packet.keys()) == SYSTEM_PACKET_KEYS
    assert packet["facts"] == ["un fait"]


def test_packet_vide_sans_registre_est_le_contrat_historique():
    assert packet_vide() == PACKET_VIDE
    assert packet_vide() is not PACKET_VIDE, "doit être une copie, pas la constante partagée"


# ─── 2. Le paquet suit le registre ───────────────────────────────────────────

def test_une_collection_declaree_ouvre_sa_propre_section():
    """Le libellé de l'agent cesse d'être dilué dans `facts`."""
    registre = CollectionRegistry.depuis([
        Collection("clients_paca", "semantic", "clients_paca", created_by="agent"),
    ])
    packet, _, _ = collapse_by_utility(
        _candidat("m1", "semantic", "clients_paca", "Nana couvre Marseille"),
        max_tokens=100, registry=registre)

    assert packet["clients_paca"] == ["Nana couvre Marseille"]
    assert packet["facts"] == []          # ne tombe plus dans le fourre-tout
    assert tuple(packet.keys())[:7] == SYSTEM_PACKET_KEYS


def test_une_collection_declaree_mais_vide_apparait_quand_meme():
    """« J'ai créé ce rayon et il ne contient rien » est une information, pas un vide."""
    registre = CollectionRegistry.depuis([
        Collection("clients_paca", "semantic", "clients_paca", created_by="agent"),
    ])
    packet, _, _ = collapse_by_utility(
        _candidat("m1", "semantic", "fact", "un fait ordinaire"),
        max_tokens=100, registry=registre)
    assert packet["clients_paca"] == []


def test_le_paquet_vide_a_la_meme_forme_que_le_paquet_plein():
    """RÉGRESSION : sans ça, la réponse changeait de forme selon qu'il y a des résultats.

    Une recherche infructueuse renvoyait les sept clés canoniques, une recherche fructueuse
    en renvoyait davantage. Le consommateur aurait dû tester l'existence de chaque clé.
    """
    registre = CollectionRegistry.depuis([
        Collection("clients_paca", "semantic", "clients_paca", created_by="agent"),
    ])
    vide = build_context_packet(
        store=InMemoryStore(), query_vector=[0.0], query_text="rien",
        memory_types=["semantic"], max_tokens=100, config=RetrievalConfig(),
        trace_id="t", registry=registre)["context_packet"]
    plein, _, _ = collapse_by_utility(
        _candidat("m1", "semantic", "clients_paca", "x"), max_tokens=100, registry=registre)

    assert set(vide.keys()) == set(plein.keys())
    assert "clients_paca" in vide


def test_le_registre_est_propage_jusqu_au_collapse():
    """Bout en bout : build_context_packet -> collapse, avec un magasin en mémoire."""
    registre = CollectionRegistry.depuis([
        Collection("clients_paca", "semantic", "clients_paca", created_by="agent"),
    ])
    store = InMemoryStore(memoires=[{
        "id": "m1", "type": "semantic", "subtype": "clients_paca",
        "content": "Nana couvre Marseille", "confidence": 1.0, "importance": 0.5,
        "last_accessed_at": None, "created_at": 1, "occurred_at": None,
        "embedding": [1.0], "similarity": 0.9,
    }])
    resultat = build_context_packet(
        store=store, query_vector=[1.0], query_text="paca", memory_types=["semantic"],
        max_tokens=200, config=RetrievalConfig(hybrid=False), trace_id="t",
        registry=registre)
    assert resultat["context_packet"]["clients_paca"] == ["Nana couvre Marseille"]


# ─── 3. L'intrication se décide par collection ───────────────────────────────

def test_une_collection_d_episodes_peut_etre_intriquee():
    """LE gain mesurable du lot : `episodic` ne tissait AUCUNE arête, pour personne.

    `QEM_ENTANGLE_TYPES` vaut pour l'instance entière. Des comptes rendus de réunion sont
    des épisodes, mais structurants : les priver de graphe prive le multi-hop de matière.
    """
    from apps.worker.worker import _is_entanglement_candidate

    reunion = {"type": "episodic", "subtype": "reunions_client"}
    brut = {"type": "episodic", "subtype": "interaction"}

    # Sans registre : comportement historique, aucun épisode n'est intriqué.
    assert _is_entanglement_candidate(reunion) is False
    assert _is_entanglement_candidate(brut) is False

    registre = CollectionRegistry.depuis([
        Collection("reunions_client", "episodic", "episodes", entangle=True,
                   created_by="agent"),
    ])
    assert _is_entanglement_candidate(reunion, registre) is True
    # L'épisode brut, lui, reste hors du graphe : la décision est bien PAR COLLECTION.
    assert _is_entanglement_candidate(brut, registre) is False


def test_une_collection_peut_aussi_etre_exclue_du_graphe():
    """La réciproque doit tenir : l'agent peut retirer du bruit du graphe."""
    from apps.worker.worker import _is_entanglement_candidate

    registre = CollectionRegistry.depuis([
        Collection("brouillons", "semantic", "facts", entangle=False, created_by="agent"),
    ])
    assert _is_entanglement_candidate({"type": "semantic", "subtype": "brouillons"},
                                      registre) is False
    assert _is_entanglement_candidate({"type": "semantic", "subtype": "fact"},
                                      registre) is True


# ─── 4. L'extraction ne détruit plus la collection de l'agent ────────────────

def test_l_extraction_preserve_une_collection_declaree():
    """RÉGRESSION : deux chemins, deux règles.

    `normalize_extraction` écrasait tout sous-type non canonique par le défaut de sa
    famille. Un `clients_paca` proposé par le LLM devenait `fact` — alors que le même
    libellé passé à `POST /v1/memories` était conservé.
    """
    registre = CollectionRegistry.depuis([
        Collection("clients_paca", "semantic", "clients_paca", created_by="agent"),
    ])
    assert normalize_extraction("semantic", "clients_paca") == ("semantic", "fact")
    assert normalize_extraction("semantic", "clients_paca", registre) == (
        "semantic", "clients_paca")


def test_l_extraction_ecrase_toujours_un_libelle_non_declare():
    """Un LLM ne doit pas peupler la taxonomie à chaque hallucination.

    La création reste un acte délibéré (lot 3) : sur ce chemin, la valeur vient d'un modèle
    et non d'une intention.
    """
    registre = CollectionRegistry.systeme()
    assert normalize_extraction("semantic", "invente_par_le_llm", registre) == (
        "semantic", "fact")


def test_une_famille_hallucinee_retombe_toujours_sur_semantic():
    assert normalize_extraction("marketing", "x") == ("semantic", "fact")


# ─── 5. Le chargeur partagé par les DEUX chemins d'écriture ──────────────────
# `charger_registre` vit dans le cœur précisément pour que l'API et le worker voient le
# même registre. Ses replis doivent donc être testés ici, une seule fois.

class _CurseurFactice:
    def __init__(self, lignes=None, lever=None):
        self._lignes, self._lever = lignes or [], lever
        self.requete = None

    def execute(self, sql, params=None):
        if self._lever is not None:
            raise self._lever
        self.requete = (sql, params)

    def fetchall(self):
        return self._lignes


def test_charger_registre_construit_depuis_des_tuples():
    """Le curseur peut être à tuples (`create_memory`) : lecture par POSITION."""
    cur = _CurseurFactice([
        ("clients_paca", "semantic", "clients_paca", "Clients PACA", True, "agent"),
    ])
    registre = charger_registre(cur, "t1", "agentA")
    assert registre.packet_key("semantic", "clients_paca") == "clients_paca"
    # Les collections système complètent toujours.
    assert registre.packet_key("procedural", "code_error_resolution") == "errors"
    # Le périmètre est bien passé en paramètres liés.
    assert cur.requete[1] == ("t1", "agentA")


def test_charger_registre_accepte_aussi_un_curseur_a_dictionnaires():
    cur = _CurseurFactice([{
        "name": "clients_paca", "family": "semantic", "packet_key": "clients_paca",
        "description": "", "entangle": True, "created_by": "agent",
    }])
    assert charger_registre(cur, "t1", "agentA").packet_key(
        "semantic", "clients_paca") == "clients_paca"


def test_charger_registre_replie_sur_le_systeme_si_la_table_manque():
    """Migration pas encore tirée : état d'exploitation légitime, pas une panne."""
    manquante = Exception("relation does not exist")
    manquante.pgcode = "42P01"          # UndefinedTable
    registre = charger_registre(_CurseurFactice(lever=manquante), "t1", "agentA")
    assert registre.packet_keys() == SYSTEM_PACKET_KEYS


def test_charger_registre_replie_aussi_sur_une_erreur_inattendue():
    """La taxonomie ne doit jamais être une dépendance dure du rappel.

    Refuser une mémoire parce que son ÉTIQUETTE est illisible serait disproportionné — mais
    ce repli-là est journalisé en ERROR, contrairement au précédent.
    """
    registre = charger_registre(_CurseurFactice(lever=RuntimeError("boum")), "t1", "agentA")
    assert registre.packet_keys() == SYSTEM_PACKET_KEYS


def test_charger_registre_ecarte_une_ligne_a_famille_invalide():
    """Une seule ligne corrompue ne doit pas priver l'agent de TOUT son registre."""
    cur = _CurseurFactice([
        ("bidon", "marketing", "facts", "", True, "agent"),
        ("clients_paca", "semantic", "clients_paca", "", True, "agent"),
    ])
    registre = charger_registre(cur, "t1", "agentA")
    assert registre.get("marketing", "bidon") is None
    assert registre.get("semantic", "clients_paca") is not None
