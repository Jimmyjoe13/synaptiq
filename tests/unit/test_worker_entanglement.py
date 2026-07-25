"""Tests unitaires : éligibilité à l'intrication automatique + validation d'extraction.

Sans infra (ni Postgres, ni Redis) : on ne teste que les fonctions pures du worker qui
décident du TYPE d'un souvenir et de son éligibilité au graphe `entangled_with`.

Enjeu : c'est ce prédicat qui détermine si le graphe se remplit. S'il ne matche jamais,
`propagate_entanglement` n'a aucune arête à parcourir et Q-EM dégénère en top-k vectoriel.
"""
import pytest

from apps.worker.worker import (
    _heuristic_extract,
    _is_entanglement_candidate,
    _validate_extraction,
)


# ─── Éligibilité à l'intrication ───

@pytest.mark.parametrize("mtype,subtype", [
    ("procedural", "code_error_resolution"),
    ("procedural", "coding_best_practices"),
    ("procedural", "rule"),
    ("semantic", "fact"),
    ("semantic", "preference"),
])
def test_souvenirs_durables_sont_intricables(mtype, subtype):
    assert _is_entanglement_candidate({"type": mtype, "subtype": subtype})


def test_episodic_exclu_par_defaut():
    """Les épisodes bruts ne tissent pas le graphe (densité sans pertinence)."""
    assert not _is_entanglement_candidate({"type": "episodic", "subtype": "interaction"})


def test_preference_intricable_meme_si_type_inattendu():
    """Garde-fou historique : une préférence reste intricable quel que soit son type."""
    assert _is_entanglement_candidate({"type": "working", "subtype": "preference"})


def test_types_intricables_configurables(monkeypatch):
    """QEM_ENTANGLE_TYPES pilote le prédicat sans redéploiement de code."""
    import apps.worker.worker as worker

    monkeypatch.setattr(worker, "QEM_ENTANGLE_TYPES", {"episodic"})
    assert worker._is_entanglement_candidate({"type": "episodic", "subtype": "interaction"})
    assert not worker._is_entanglement_candidate({"type": "semantic", "subtype": "fact"})


# ─── Validation de l'extraction : c'est elle qui produit le type ───

def test_extraction_llm_valide_est_intricable():
    """Une classification LLM correcte doit déboucher sur un souvenir intricable."""
    data = _validate_extraction(
        {"type": "semantic", "subtype": "preference", "content": "Jimmy préfère les mails courts",
         "summary": "Préférence", "confidence": 0.9, "importance": 0.8},
        "peu importe",
    )
    assert data["type"] == "semantic"
    assert data["subtype"] == "preference"
    assert _is_entanglement_candidate(data)


def test_extraction_type_inconnu_retombe_sur_semantic():
    """Un type hors taxonomie est normalisé en semantic/fact — donc intricable."""
    data = _validate_extraction({"type": "inventé", "subtype": "n'importe quoi"}, "contenu brut")
    assert data["type"] == "semantic"
    assert data["subtype"] == "fact"
    assert data["content"] == "contenu brut"
    assert _is_entanglement_candidate(data)


def test_extraction_borne_les_scores():
    data = _validate_extraction(
        {"type": "semantic", "subtype": "fact", "confidence": 42, "importance": -3}, "x"
    )
    assert data["confidence"] == 1.0
    assert data["importance"] == 0.0


def test_extraction_scores_non_numeriques_prennent_le_defaut():
    data = _validate_extraction(
        {"type": "semantic", "subtype": "fact", "confidence": "élevée", "importance": None}, "x"
    )
    assert data["confidence"] == 0.9
    assert data["importance"] == 0.5


# ─── Repli heuristique : documente ce qu'on perd sans LLM ───

@pytest.mark.parametrize("texte,attendu_type,attendu_subtype", [
    ("Erreur critique : traceback à l'import du module socket", "procedural", "code_error_resolution"),
    ("Bonne pratique : toujours borner les requêtes SQL", "procedural", "coding_best_practices"),
    ("Je préfère les réponses courtes et directes", "semantic", "preference"),
])
def test_heuristique_reconnait_les_tournures_cibles(texte, attendu_type, attendu_subtype):
    data = _heuristic_extract(texte)
    assert (data["type"], data["subtype"]) == (attendu_type, attendu_subtype)
    assert _is_entanglement_candidate(data)


def test_heuristique_sans_motif_produit_un_episode_non_intricable():
    """Le cas dominant sans LLM : aucune tournure reconnue -> episodic -> hors graphe.

    C'est précisément pourquoi `.env.example` livre un LLM d'extraction par défaut.
    """
    data = _heuristic_extract("Le rapport trimestriel a été transmis au client hier.")
    assert data["type"] == "episodic"
    assert not _is_entanglement_candidate(data)


# ─── Négociation du format de sortie structurée ───

class _FakeResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {"choices": [{"message": {"content": '{"type":"semantic",'
                                                             '"subtype":"fact","content":"c",'
                                                             '"summary":"s","confidence":0.9,'
                                                             '"importance":0.5}'}}]}
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _reset_negociation(worker):
    worker._format_negotiated = False
    worker._negotiated_format_mode = None


def test_negocie_json_object_quand_l_endpoint_l_accepte(monkeypatch):
    """OpenAI / Groq / OpenRouter acceptent json_object : on doit s'y tenir."""
    import apps.worker.worker as worker
    _reset_negociation(worker)
    vus = []

    def fake_post(url, headers=None, json=None, timeout=None):
        vus.append(json.get("response_format", {}).get("type"))
        return _FakeResp(200)

    monkeypatch.setattr(worker.requests, "post", fake_post)
    monkeypatch.setattr(worker, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(worker, "LLM_API_KEY", "clé-valide")
    worker.call_llm_extractor("Caroline a adopté un beagle")

    assert vus[0] == "json_object"
    assert worker._negotiated_format_mode == "json_object"


def test_bascule_sur_json_schema_si_json_object_refuse(monkeypatch):
    """Cas LM Studio : json_object -> 400. Sans bascule, TOUTE extraction échouait
    et retombait silencieusement sur les regex."""
    import apps.worker.worker as worker
    _reset_negociation(worker)
    vus = []

    def fake_post(url, headers=None, json=None, timeout=None):
        mode = json.get("response_format", {}).get("type")
        vus.append(mode)
        return _FakeResp(400) if mode == "json_object" else _FakeResp(200)

    monkeypatch.setattr(worker.requests, "post", fake_post)
    monkeypatch.setattr(worker, "LLM_PROVIDER", "lmstudio")
    monkeypatch.setattr(worker, "LLM_API_KEY", "")
    monkeypatch.setattr(worker, "LLM_BASE_URL", "http://localhost:1234/v1")
    res = worker.call_llm_extractor("Caroline a adopté un beagle")

    assert vus[:2] == ["json_object", "json_schema"]
    assert worker._negotiated_format_mode == "json_schema"
    assert res[0]["type"] == "semantic"  # extraction LLM réussie, pas un repli regex


def test_retombe_en_texte_libre_si_aucun_format_accepte(monkeypatch):
    import apps.worker.worker as worker
    _reset_negociation(worker)

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(400) if "response_format" in json else _FakeResp(200)

    monkeypatch.setattr(worker.requests, "post", fake_post)
    monkeypatch.setattr(worker, "LLM_PROVIDER", "lmstudio")
    monkeypatch.setattr(worker, "LLM_BASE_URL", "http://localhost:1234/v1")
    worker.call_llm_extractor("un contenu")

    assert worker._negotiated_format_mode is None


def test_negociation_faite_une_seule_fois(monkeypatch):
    """La négociation ne doit pas re-sonder l'endpoint à chaque événement."""
    import apps.worker.worker as worker
    _reset_negociation(worker)
    appels = []

    def fake_post(url, headers=None, json=None, timeout=None):
        appels.append(1)
        return _FakeResp(200)

    monkeypatch.setattr(worker.requests, "post", fake_post)
    monkeypatch.setattr(worker, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(worker, "LLM_API_KEY", "clé")
    worker.call_llm_extractor("un")
    worker.call_llm_extractor("deux")
    worker.call_llm_extractor("trois")

    # 1 sonde de négociation + 1 appel réel par événement.
    assert len(appels) == 4


# ─── Extraction multi-faits ───

def _payload(memories):
    """Réponse LLM au format attendu : {"memories": [...]}."""
    import json as _json
    return _FakeResp(200, {"choices": [{"message": {"content": _json.dumps({"memories": memories})}}]})


def test_un_evenement_produit_plusieurs_faits(monkeypatch):
    """Un tour énonce souvent 2-3 faits ; n'en garder qu'un perdait l'information."""
    import apps.worker.worker as worker
    _reset_negociation(worker)
    memories = [
        {"type": "semantic", "subtype": "fact", "content": "Caroline a adopté un beagle",
         "summary": "Adoption", "occurred_at": "2023-05-08", "confidence": 1, "importance": 0.8},
        {"type": "semantic", "subtype": "preference", "content": "Caroline aime courir le matin",
         "summary": "Préférence", "occurred_at": "", "confidence": 0.9, "importance": 0.6},
    ]
    monkeypatch.setattr(worker.requests, "post",
                        lambda *a, **k: _payload(memories))
    monkeypatch.setattr(worker, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(worker, "LLM_API_KEY", "clé")

    faits = worker.call_llm_extractor("peu importe", occurred_at="2023-05-08")

    assert len(faits) == 2
    assert faits[0]["subtype"] == "fact" and faits[1]["subtype"] == "preference"


def test_date_relative_resolue_en_datetime(monkeypatch):
    """`occurred_at` ISO devient un datetime exploitable en base."""
    import apps.worker.worker as worker
    from datetime import datetime
    _reset_negociation(worker)
    monkeypatch.setattr(worker.requests, "post", lambda *a, **k: _payload([
        {"type": "semantic", "subtype": "fact", "content": "Caroline y est allée",
         "summary": "s", "occurred_at": "2023-05-07", "confidence": 1, "importance": 0.5}]))
    monkeypatch.setattr(worker, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(worker, "LLM_API_KEY", "clé")

    fait = worker.call_llm_extractor("hier je suis allée...", occurred_at="2023-05-08")[0]

    assert fait["occurred_at"] == datetime(2023, 5, 7)


def test_horodatage_transmis_au_prompt(monkeypatch):
    """Sans la référence temporelle dans le prompt, le modèle ne peut rien résoudre."""
    import apps.worker.worker as worker
    _reset_negociation(worker)
    vus = []

    def capture(url, headers=None, json=None, timeout=None):
        vus.append(json["messages"][-1]["content"])
        return _payload([{"type": "semantic", "subtype": "fact", "content": "c",
                          "summary": "s", "occurred_at": "", "confidence": 1, "importance": 0.5}])

    monkeypatch.setattr(worker.requests, "post", capture)
    monkeypatch.setattr(worker, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(worker, "LLM_API_KEY", "clé")
    worker.call_llm_extractor("un contenu", occurred_at="2023-08-14")

    assert "2023-08-14" in vus[-1]


@pytest.mark.parametrize("valeur,attendu_none", [
    ("", True), (None, True), ("pas une date", True), ("last week", True),
    ("2023-05-07", False), ("2023-05-07T14:30:00", False), ("2023-05-07T14:30:00Z", False),
])
def test_dates_invalides_ignorees(valeur, attendu_none):
    """Une date fausse serait pire que pas de date : on préfère None."""
    from apps.worker.worker import _parse_occurred_at
    assert (_parse_occurred_at(valeur) is None) is attendu_none


def test_formats_de_reponse_toleres():
    """Objet unique ou liste nue : on accepte, les modèles ne suivent pas tous la consigne."""
    from apps.worker.worker import _validate_extractions
    unique = {"type": "semantic", "subtype": "fact", "content": "un fait", "summary": "s"}
    assert len(_validate_extractions(unique, "brut")) == 1
    assert len(_validate_extractions([unique, dict(unique, content="autre")], "brut")) == 2


def test_faits_dupliques_ecartes():
    """Deux faits identiques entreraient en conflit sur (source_event_id, content_hash)."""
    from apps.worker.worker import _validate_extractions
    doublon = {"type": "semantic", "subtype": "fact", "content": "Caroline a un beagle", "summary": "s"}
    faits = _validate_extractions({"memories": [doublon, dict(doublon), doublon]}, "brut")
    assert len(faits) == 1


def test_nombre_de_faits_borne(monkeypatch):
    import apps.worker.worker as worker
    from apps.worker.worker import _validate_extractions
    monkeypatch.setattr(worker, "MAX_FACTS_PER_EVENT", 3)
    trop = [{"type": "semantic", "subtype": "fact", "content": f"fait {i}", "summary": "s"}
            for i in range(10)]
    assert len(_validate_extractions({"memories": trop}, "brut")) == 3


def test_sortie_vide_retombe_sur_heuristique():
    """Plutôt qu'un événement perdu, un fait heuristique."""
    from apps.worker.worker import _validate_extractions
    faits = _validate_extractions({"memories": []}, "Je préfère les réponses courtes")
    assert len(faits) == 1 and faits[0]["subtype"] == "preference"


# ─── Empreinte de contenu (idempotence du replay) ───

def test_hash_stable_et_insensible_a_la_mise_en_forme():
    from apps.worker.worker import content_hash
    assert content_hash("Caroline a un beagle") == content_hash("  caroline   a  un   BEAGLE  ")


def test_hash_distingue_des_faits_differents():
    from apps.worker.worker import content_hash
    assert content_hash("Caroline a un beagle") != content_hash("Melanie a un chat")
