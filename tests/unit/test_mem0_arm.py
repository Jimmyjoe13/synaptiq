"""Tests unitaires du bras mem0 du harness LOCOMO (benchmarks/mem0_arm.py).

Aucun de ces tests n'installe mem0 ni ne touche à PostgreSQL : ils verrouillent les trois
endroits où ce bras peut se tromper EN SILENCE, c'est-à-dire produire un chiffre publiable
et faux.

1. **La forme de réponse de `search()`.** mem0 a rendu tantôt une liste, tantôt
   `{"results": [...]}` selon la version. Une lecture ratée ne lève rien : elle rend un
   contexte vide, mem0 obtient 0 % et le rapport a l'air normal.
2. **Le budget de tokens.** Si un bras dispose de plus de contexte que l'autre, l'écart
   d'exactitude mesure la taille du contexte, pas la mémoire.
3. **La configuration injectée.** Les deux moteurs DOIVENT voir le même modèle
   d'embedding et le même LLM. Un défaut mem0 qui reprend le dessus (text-embedding-3-small,
   gpt-4o-mini) transformerait la comparaison en comparaison de fournisseurs.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "packages" / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

mem0_arm = pytest.importorskip("benchmarks.mem0_arm", reason="bras mem0 absent")
budget = pytest.importorskip("benchmarks.budget", reason="module budget absent")
from synaptiq_core.qem import estimate_tokens

DSN = "postgresql://synaptiq:mot%40passe@127.0.0.1:5435/synaptiq_dev"


def _config(**surcharges):
    base = dict(
        dsn=DSN, collection_name="mem0_test", embedding_provider="lmstudio",
        embedding_model="paraphrase-multilingual", embedding_base_url="http://localhost:1234/v1",
        embedding_api_key="", embedding_dim=384, llm_model="gpt-oss-120b-medium",
        llm_base_url="http://127.0.0.1:8899/v1", llm_api_key="",
    )
    base.update(surcharges)
    return mem0_arm.build_config(**base)


# ─── 1. Lecture de la réponse de search() ───

@pytest.mark.parametrize("brut,attendu", [
    # Forme v1.1 / v3 : enveloppe `results`.
    ({"results": [{"memory": "A"}, {"memory": "B"}]}, ["A", "B"]),
    # Forme historique : liste nue.
    ([{"memory": "A"}], ["A"]),
    # Certaines versions nomment le champ `text`.
    ([{"text": "A"}], ["A"]),
    # Liste de chaînes brutes.
    (["A", "B"], ["A", "B"]),
    # Les vides ne doivent pas occuper de ligne dans le contexte.
    ([{"memory": ""}, {"memory": "A"}, {}], ["A"]),
    # Absence totale de résultat : contexte vide, pas d'exception.
    ({"results": []}, []),
    ([], []),
    (None, []),
])
def test_lecture_des_resultats_mem0(brut, attendu):
    assert mem0_arm.extraire_contenus(brut) == attendu


def test_enveloppe_inattendue_ne_leve_pas():
    """Une forme inconnue rend une liste vide plutôt que de faire tomber le run entier."""
    assert mem0_arm.extraire_contenus({"donnees": [{"memory": "A"}]}) == []


# ─── 2. Budget de tokens, commun à tous les bras ───

def test_le_budget_saute_les_contenus_trop_gros_sans_s_arreter():
    """Règle de `collapse_by_utility` : un candidat trop lourd est sauté, pas terminal.

    S'arrêter au premier dépassement laisserait du budget inutilisé et ferait perdre des
    points au bras pour une raison d'implémentation, pas de mémoire.
    """
    longue = " ".join(["mot"] * 400)
    texte, tokens = budget.fit_to_budget([longue, "un souvenir court", longue, "et un autre"],
                                         max_tokens=50)
    assert tokens <= 50
    assert "un souvenir court" in texte
    assert "et un autre" in texte
    assert longue not in texte


def test_le_budget_utilise_l_estimateur_de_qem():
    contenus = ["premier souvenir de test", "second souvenir de test"]
    _, tokens = budget.fit_to_budget(contenus, max_tokens=10_000)
    assert tokens == sum(estimate_tokens(c) for c in contenus)


def test_le_budget_ignore_les_contenus_vides():
    texte, tokens = budget.fit_to_budget(["", None, "réel"], max_tokens=100)
    assert texte == "- réel"
    assert tokens == estimate_tokens("réel")


# ─── 3. Configuration injectée dans mem0 ───

def test_la_config_impose_les_modeles_de_synaptiq():
    """Sans cela, mem0 retomberait sur ses défauts et on comparerait deux fournisseurs."""
    config = _config()
    assert config["embedder"]["config"]["model"] == "paraphrase-multilingual"
    assert config["embedder"]["config"]["lmstudio_base_url"] == "http://localhost:1234/v1"
    assert config["llm"]["config"]["model"] == "gpt-oss-120b-medium"
    assert config["llm"]["config"]["openai_base_url"] == "http://127.0.0.1:8899/v1"


def test_la_temperature_d_extraction_est_nulle():
    """Un benchmark reproductible ne laisse pas l'extraction varier d'un run à l'autre."""
    assert _config()["llm"]["config"]["temperature"] == 0.0


def test_le_dsn_est_traduit_en_config_pgvector():
    """Les deux moteurs partagent le même PostgreSQL : même matériel, même configuration."""
    store = _config()["vector_store"]
    assert store["provider"] == "pgvector"
    assert store["config"]["dbname"] == "synaptiq_dev"
    assert store["config"]["user"] == "synaptiq"
    # Le mot de passe est encodé dans l'URL (%40 = @) : sans décodage, la connexion échoue
    # avec une erreur d'authentification difficile à rattacher à sa cause.
    assert store["config"]["password"] == "mot@passe"
    assert store["config"]["host"] == "127.0.0.1"
    assert store["config"]["port"] == 5435
    assert store["config"]["embedding_model_dims"] == 384
    # HNSW des deux côtés : SynaptiQ l'utilise (migration 20260729_perf_idx), comparer à un
    # mem0 en scan séquentiel mesurerait l'indexation, pas le rappel.
    assert store["config"]["hnsw"] is True


def test_un_dsn_invalide_est_refuse_immediatement():
    with pytest.raises(ValueError, match="DSN"):
        _config(dsn="pas-une-url")


def test_fournisseur_distant_bascule_sur_le_client_openai():
    """LM Studio a son fournisseur dédié ; tout le reste passe par le client OpenAI."""
    config = _config(embedding_provider="openai", embedding_api_key="sk-test")
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["openai_base_url"] == "http://localhost:1234/v1"
    assert config["embedder"]["config"]["api_key"] == "sk-test"


# ─── 4. L'environnement ne doit pas détourner le LLM de mem0 ───

def test_openrouter_dans_l_environnement_est_neutralisee(monkeypatch):
    """Le piège qui casserait la comparaison sans lever la moindre erreur.

    `mem0/llms/openai.py` teste `OPENROUTER_API_KEY` AVANT de lire la configuration : une
    variable présente pour un autre projet envoie l'extraction mem0 chez un fournisseur
    distant pendant que SynaptiQ extrait en local. Constaté sur cette machine — l'erreur
    n'a été visible que parce que le modèle local n'existait pas chez OpenRouter. Avec un
    nom présent des deux côtés, le run aboutirait et le rapport annoncerait « même LLM
    d'extraction » en toute bonne foi.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-peu-importe")
    neutralisees = mem0_arm.Mem0Arm._neutraliser_env_qui_detourne_le_llm(_config())

    assert "OPENROUTER_API_KEY" in neutralisees
    assert "OPENROUTER_API_KEY" not in os.environ


def test_l_url_du_llm_est_forcee_sur_celle_de_la_config(monkeypatch):
    """`OPENAI_BASE_URL` traînant dans l'environnement ne doit pas gagner sur le `.env`."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    neutralisees = mem0_arm.Mem0Arm._neutraliser_env_qui_detourne_le_llm(_config())

    assert "OPENAI_BASE_URL" in neutralisees
    assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8899/v1"


def test_environnement_propre_ne_neutralise_rien(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert mem0_arm.Mem0Arm._neutraliser_env_qui_detourne_le_llm(_config()) == []


# ─── 5. Capacités NLP annoncées dans le rapport ───

def test_les_capacites_nlp_sont_toujours_renseignees():
    """Le rapport doit pouvoir dire si mem0 tournait à pleine capacité, quel que soit l'env."""
    capacites = mem0_arm.capacites_nlp()
    assert set(capacites) == {"spacy", "en_core_web_sm", "full_capacity"}
    # La capacité complète exige les DEUX : spaCy seul ne suffit pas, `en_core_web_sm` est
    # ce qui active réellement la lemmatisation BM25 et la liaison d'entités.
    assert capacites["full_capacity"] is (capacites["spacy"] and capacites["en_core_web_sm"])


# ─── 6. Garde-fou du reset ───

def test_le_reset_refuse_un_prefixe_hors_perimetre_mem0():
    """`reset_collection` fait un DROP TABLE par préfixe : le préfixe est la seule barrière.

    Sans ce refus, `--mem0-collection memories` supprimerait la table du produit — et la
    vérification doit avoir lieu AVANT toute connexion, sinon un environnement sans base
    la contournerait par une erreur de connexion.
    """
    with pytest.raises(ValueError, match="mem0"):
        mem0_arm.reset_collection(DSN, "memories")


def test_le_reset_accepte_le_prefixe_dedie(monkeypatch):
    """Le nom par défaut doit passer la barrière (sinon le bras serait inutilisable)."""
    appels = []
    monkeypatch.setattr(mem0_arm.psycopg2, "connect",
                        lambda dsn: (_ for _ in ()).throw(RuntimeError("base absente")))
    with pytest.raises(RuntimeError, match="base absente"):
        mem0_arm.reset_collection(DSN, mem0_arm.DEFAULT_COLLECTION)
    assert appels == []   # la ValueError de préfixe n'a pas été levée : on est allé jusqu'à la base
