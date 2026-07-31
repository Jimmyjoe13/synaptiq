import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import redis
import requests
from dotenv import load_dotenv
from prometheus_client import Counter
from psycopg2 import pool as pg_pool

# Rendre le package partagé packages/core importable (dev local hors conteneur)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_root, os.path.join(_root, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synaptiq_core import get_embedder, handle_contradictions, link_supersedes, to_pgvector

# Registre des collections : l'intrication est décidée par collection, plus par instance
from synaptiq_core.collections import charger_registre
from synaptiq_core.embeddings import generate_mock_embedding  # noqa: F401 (compat rétro tests)
from synaptiq_core.observability import configure_logging, set_trace_id
from synaptiq_core.taxonomy import DEFAULT_SUBTYPE, VALID_SUBTYPES, normalize_extraction

# Configuration du logging
configure_logging("synaptiq-worker")
logger = logging.getLogger("synaptiq-worker")

EXTRACTION_DEGRADED_COUNTER = Counter(
    "synaptiq_extraction_degraded_total",
    "Nombre d'extractions de mémoire ayant replié sur les heuristiques regex suite à un échec LLM",
)

# Chargement des variables d'environnement depuis le .env RACINE (source unique)
load_dotenv(os.path.join(_root, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

# Pool de connexions PostgreSQL partagé par le process worker : évite d'ouvrir/fermer
# une connexion à chaque événement. Initialisé paresseusement au premier usage.
DB_POOL_MIN = int(os.getenv("WORKER_DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("WORKER_DB_POOL_MAX", "4"))
_db_pool: "pg_pool.ThreadedConnectionPool | None" = None


def get_db_pool() -> "pg_pool.ThreadedConnectionPool":
    """Retourne le pool de connexions (créé au premier appel)."""
    global _db_pool
    if _db_pool is None:
        _db_pool = pg_pool.ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, dsn=DATABASE_URL)
        logger.info("Pool PostgreSQL worker initialisé (%d–%d connexions).", DB_POOL_MIN, DB_POOL_MAX)
    return _db_pool

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
# Résilience aux erreurs transitoires (429 fréquent sur les modèles :free).
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF_S = float(os.getenv("LLM_RETRY_BACKOFF_S", "2.0"))

# Seuil de similarité cosinus au-delà duquel deux mémoires sont automatiquement intriquées.
QEM_ENTANGLE_THRESHOLD = float(os.getenv("QEM_ENTANGLE_THRESHOLD", "0.7"))

# Types de mémoire éligibles à l'intrication automatique (liste séparée par des virgules).
# Défaut 'procedural,semantic' : les souvenirs DURABLES (règles, erreurs, bonnes pratiques,
# faits, préférences) tissent le graphe exploité par le multi-hop de build_context.
# `episodic` est exclu par défaut — les épisodes bruts sont nombreux et peu discriminants,
# les intriquer densifie le graphe sans gain de pertinence. L'ajouter reste possible ici.
QEM_ENTANGLE_TYPES = {
    t.strip() for t in os.getenv("QEM_ENTANGLE_TYPES", "procedural,semantic").split(",") if t.strip()
}


def _is_entanglement_candidate(memory_data: dict, registry=None) -> bool:
    """Une mémoire doit-elle chercher des voisins à intriquer ?

    Historiquement restreint à `procedural` + `semantic/preference`, ce qui laissait le
    graphe VIDE dès que l'extraction retombait en `episodic/interaction` (cas du mode
    d'extraction heuristique) : sans arêtes, la propagation d'activation de build_context
    n'a rien à propager et Q-EM dégénère en simple top-k vectoriel.

    ## La décision passe de l'instance à la collection

    `QEM_ENTANGLE_TYPES` est un réglage d'INSTANCE : il vaut pour les quatre familles, donc
    pour tous les agents et tous leurs rayons à la fois. Exclure `episodic` globalement était
    le bon défaut — les épisodes bruts sont nombreux et peu discriminants — mais c'était
    aussi une décision impossible à nuancer : une collection d'épisodes réellement
    structurante (des comptes rendus de réunion, par exemple) ne pouvait pas tisser d'arête.

    Or le multi-hop est la seule dimension où Q-EM creuse nettement l'écart sur la baseline.
    Pouvoir densifier le graphe là où l'agent sait que ça compte est le gain mesurable de ce
    lot, et non un simple confort de rangement.

    Sans registre, on retombe exactement sur le comportement précédent.
    """
    if registry is not None:
        return registry.entangle_pour(memory_data.get("type"), memory_data.get("subtype"))
    return (
        memory_data.get("type") in QEM_ENTANGLE_TYPES
        or memory_data.get("subtype") == "preference"
    )


def _heuristic_extract(event_content: str) -> dict:
    """Extraction locale par heuristiques regex FR (hors-ligne / fallback).

    Fragile par nature (dépend de tournures françaises) : sert de repli quand le
    LLM est indisponible. La voie robuste est l'extraction LLM structurée.

    Ne produit qu'UN fait, sans date : c'est précisément ce que le repli fait perdre.
    """
    # Erreurs de code et résolutions
    error_match = re.search(
        r"(?:erreur|bug|exception|traceback|crash|failed|plantage|corrigé|résolu|warning)\s+([^.]+)",
        event_content, re.IGNORECASE,
    )
    if error_match:
        return {
            "extracted": True, "type": "procedural", "subtype": "code_error_resolution",
            "content": f"Résolution de bug/erreur détectée : {error_match.group(0).strip()}",
            "summary": "Résolution d'erreur de code", "confidence": 0.85, "importance": 0.7,
        }
    # Bonnes pratiques / playbooks
    best_practice_match = re.search(
        r"(?:bonne pratique|toujours|ne jamais|règle de conception|recommandation|best practice)\s+([^.]+)",
        event_content, re.IGNORECASE,
    )
    if best_practice_match:
        return {
            "extracted": True, "type": "procedural", "subtype": "coding_best_practices",
            "content": f"Directive de conception/code : {best_practice_match.group(0).strip()}",
            "summary": "Directive de conception de code", "confidence": 0.9, "importance": 0.8,
        }
    # Préférences utilisateur
    pref_match = re.search(
        r"(?:je préfère|je veux|ma préférence|utilise plutôt|ne fais pas|écris en)\s+([^.]+)",
        event_content, re.IGNORECASE,
    )
    if pref_match:
        return {
            "extracted": True, "type": "semantic", "subtype": "preference",
            "content": f"L'utilisateur a spécifié une préférence : {pref_match.group(1).strip()}",
            "summary": "Préférence utilisateur extraite", "confidence": 0.9, "importance": 0.8,
        }
    # Défaut : épisode générique
    return {
        "extracted": True, "type": "episodic", "subtype": "interaction",
        "content": f"Interaction : {event_content}",
        "summary": "Épisode d'interaction", "confidence": 0.8, "importance": 0.4,
    }


# Plafond de faits retenus par événement. Un tour de dialogue en énonce rarement plus ;
# la borne protège d'un modèle qui partirait en liste à rallonge.
MAX_FACTS_PER_EVENT = int(os.getenv("MAX_FACTS_PER_EVENT", "5"))


def content_hash(content: str) -> str:
    """Empreinte du contenu normalisé, clé de déduplication des faits d'un événement.

    Combinée à `source_event_id` dans un index unique, elle autorise plusieurs mémoires
    par événement tout en garantissant qu'un replay ne duplique rien : les mêmes faits
    produisent les mêmes empreintes.
    """
    return hashlib.sha256(" ".join(content.split()).strip().lower().encode("utf-8")).hexdigest()


# Taxonomie PARTAGÉE avec l'API (packages/core/synaptiq_core/taxonomy.py). Elle vivait ici,
# donc elle n'était appliquée que sur ce chemin d'extraction : l'écriture directe
# `POST /v1/memories` acceptait n'importe quel sous-type. Deux chemins, deux règles.
_VALID_SUBTYPES = VALID_SUBTYPES      # conservés comme alias : lisibilité locale du module
_DEFAULT_SUBTYPE = DEFAULT_SUBTYPE


def _parse_occurred_at(value) -> "datetime | None":
    """Interprète la date renvoyée par le LLM (ISO, éventuellement partielle).

    Accepte 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS' et les variantes avec 'Z'. Toute valeur
    inexploitable rend None : une date fausse serait pire que pas de date.
    """
    if not value or not isinstance(value, str):
        return None
    raw = value.strip().replace("Z", "+00:00")
    for candidate in (raw, raw[:10]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _validate_extraction(data: dict, event_content: str, registry=None) -> dict:
    """Normalise et valide UN fait de la sortie LLM ; corrige les incohérences.

    Ne lève jamais : un modèle qui hallucine un type doit produire une mémoire dégradée,
    pas faire perdre l'événement (cf. `taxonomy.normalize_extraction`).

    Le registre, quand il est fourni, préserve les collections que l'agent a déclarées :
    sans lui, l'extraction écrasait `clients_paca` en `fact` alors que l'écriture directe
    l'aurait conservé.
    """
    mtype, subtype = normalize_extraction(data.get("type"), data.get("subtype"), registry)

    def _clamp(value, default):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    return {
        "extracted": True,
        "type": mtype,
        "subtype": subtype,
        "content": (data.get("content") or event_content).strip(),
        "summary": (data.get("summary") or "Mémoire extraite").strip(),
        "confidence": _clamp(data.get("confidence"), 0.9),
        "importance": _clamp(data.get("importance"), 0.5),
        "occurred_at": _parse_occurred_at(data.get("occurred_at")),
    }


def _validate_extractions(payload, event_content: str, registry=None) -> list:
    """Normalise la sortie LLM en une LISTE de faits valides.

    Tolère les trois formes rencontrées : {"memories": [...]}, une liste nue, ou un objet
    unique (compatibilité avec les modèles qui ignorent la consigne multi-faits).
    Les doublons de contenu au sein d'un même événement sont écartés — ils créeraient des
    lignes en conflit sur (source_event_id, content_hash).
    """
    if isinstance(payload, dict):
        items = payload.get("memories")
        if not isinstance(items, list):
            items = [payload]          # objet unique -> un seul fait
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    faits, vus = [], set()
    for item in items[:MAX_FACTS_PER_EVENT]:
        if not isinstance(item, dict):
            continue
        fait = _validate_extraction(item, event_content, registry)
        cle = fait["content"].strip().lower()
        if not cle or cle in vus:
            continue
        vus.add(cle)
        faits.append(fait)

    # Aucun fait exploitable : on retombe sur l'heuristique plutôt que de perdre l'événement.
    return faits or [_heuristic_extract(event_content)]


# Schéma des mémoires extraites, pour les endpoints qui exigent `json_schema`.
_MEMORY_JSON_SCHEMA = {
    "name": "memories",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "subtype": {"type": "string"},
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "occurred_at": {"type": "string"},
                        "confidence": {"type": "number"},
                        "importance": {"type": "number"},
                    },
                    "required": ["type", "subtype", "content", "summary",
                                 "occurred_at", "confidence", "importance"],
                },
            },
        },
        "required": ["memories"],
    },
}

# Modes de sortie structurée, du plus au moins contraignant. Les endpoints
# « OpenAI-compatibles » divergent : OpenAI/Groq/OpenRouter acceptent `json_object`,
# tandis que LM Studio n'accepte QUE `json_schema` ou `text` et répond 400 sur
# `json_object`. Sans négociation, la configuration LM Studio — pourtant celle que le
# projet recommande par défaut — échouait à CHAQUE extraction et retombait
# silencieusement sur les heuristiques regex.
_RESPONSE_FORMAT_MODES = ("json_object", "json_schema", None)
_negotiated_format_mode: str | None = None
_format_negotiated = False


def _response_format_for(mode: str | None) -> dict | None:
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": _MEMORY_JSON_SCHEMA}
    return None


def call_llm_extractor(event_content: str, occurred_at: str | None = None,
                       registry=None) -> list:
    """Extrait les mémoires consolidées d'un événement brut. Retourne une LISTE de faits.

    - Sans clé LLM (ou LLM_PROVIDER=mock) : heuristiques regex locales (un seul fait).
    - Avec LLM : extraction structurée (JSON natif) validée ; repli sur les
      heuristiques en cas d'échec réseau/parse.

    `occurred_at` (ISO) est l'horodatage de l'événement : il sert de référence pour
    convertir les expressions relatives (« yesterday », « last week ») en dates absolues.
    """
    # Un endpoint LOCAL (LM Studio, Ollama…) n'exige aucune clé API. On ne retombe donc
    # sur l'heuristique que si : provider=mock, OU provider distant sans clé valide.
    _local_llm = any(h in LLM_BASE_URL for h in ("localhost", "127.0.0.1", "host.docker.internal"))
    _no_valid_key = (not LLM_API_KEY) or ("your_api_key" in LLM_API_KEY)
    if LLM_PROVIDER == "mock" or (_no_valid_key and not _local_llm):
        logger.info("Extraction heuristique locale (sans LLM).")
        return [_heuristic_extract(event_content)]

    logger.info("Appel LLM (%s : %s) pour l'extraction de mémoire.", LLM_PROVIDER, LLM_MODEL)
    # Prompt rédigé en ANGLAIS À DESSEIN. Un prompt en français conduit les modèles à
    # traduire le souvenir en français quelle que soit la langue de l'interaction : mesuré
    # à 2/6 de préservation de langue, contre 9/9 avec cette version. Une mémoire traduite
    # perd la formulation d'origine et dégrade le rappel sémantique face à des requêtes
    # posées dans la langue source.
    # Deux exigences portées par ce prompt, issues de l'analyse du run LOCOMO du 25/07 :
    #   - EXTRAIRE TOUS LES FAITS. Réclamer « une mémoire » poussait le modèle à résumer
    #     un tour par une généralité ; le fait vérifiable (« Caroline est allée à un groupe
    #     de soutien ») disparaissait, et 57 % des questions devenaient sans réponse.
    #   - DATER. L'horodatage de l'événement est fourni ici pour que les expressions
    #     relatives soient résolues en dates absolues : sans cela, 95 % des mémoires
    #     naissaient sans date et toute question « quand… » était perdue d'avance.
    reference = occurred_at or "unknown"
    prompt = (
        "Extract EVERY durable fact stated in the following interaction.\n\n"
        f"Interaction timestamp: {reference}\n"
        f"Interaction:\n\"{event_content}\"\n\n"
        "Rules:\n"
        f"1. Extract 1 to {MAX_FACTS_PER_EVENT} SEPARATE facts — one JSON entry per fact. "
        "A single turn often states several (an event, a preference, a relationship). "
        "Prefer CONCRETE, VERIFIABLE facts (who did what, when, where) over feelings, "
        "opinions or generalities. Do NOT merge unrelated facts into one entry.\n"
        "2. Resolve EVERY relative time expression ('yesterday', 'last week', "
        "'last Saturday', 'next month') into an ABSOLUTE date, computed from the "
        "interaction timestamp above. Put it in `occurred_at` as ISO 'YYYY-MM-DD'. "
        "Use an empty string only when the fact carries no date at all.\n"
        "3. Classify each fact:\n"
        "   - type 'procedural': subtype 'code_error_resolution' (errors/tracebacks and "
        "their fix) or 'coding_best_practices' (architecture rules, conventions).\n"
        "   - type 'semantic': subtype 'preference' (an explicit taste or choice) or "
        "'fact' (a stable fact about a person, entity or the world).\n"
        "   - type 'episodic': subtype 'interaction' (only when nothing durable is stated).\n"
        "4. Write `content` and `summary` IN THE SAME LANGUAGE as the interaction. "
        "Never translate.\n"
        "5. Write `content` in the third person as a self-contained sentence, "
        "understandable without the conversation (name people explicitly, never 'he'/'she').\n\n"
        "Reply with a SINGLE JSON object: {\"memories\": [{\"type\":..., \"subtype\":..., "
        "\"content\":..., \"summary\": \"short title\", \"occurred_at\": \"YYYY-MM-DD\", "
        "\"confidence\": float 0-1, \"importance\": float 0-1}, ...]}."
    )
    try:
        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY and "your_api_key" not in LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Precision memory extractor. Reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        def _post(mode: str | None):
            body = dict(payload)
            fmt = _response_format_for(mode)
            if fmt is not None:
                body["response_format"] = fmt
            return requests.post(f"{LLM_BASE_URL}/chat/completions",
                                 headers=headers, json=body, timeout=30)

        # Négociation du mode de sortie structurée, une seule fois par process : on retient
        # le premier mode que l'endpoint accepte. Un 400 signale un mode non supporté (et
        # non une panne), on passe donc au suivant sans consommer de tentative de retry.
        global _negotiated_format_mode, _format_negotiated
        if not _format_negotiated:
            for mode in _RESPONSE_FORMAT_MODES:
                probe = _post(mode)
                if probe.status_code != 400:
                    _negotiated_format_mode = mode
                    _format_negotiated = True
                    logger.info("Sortie structurée négociée avec %s : mode=%s.",
                                LLM_BASE_URL, mode or "aucun (texte libre)")
                    break
            else:
                _negotiated_format_mode = None
                _format_negotiated = True
                logger.warning("Aucun mode de sortie structurée accepté par %s : texte libre.",
                               LLM_BASE_URL)

        # Retry avec backoff sur les erreurs transitoires (429 rate-limit fréquent sur les
        # modèles :free d'OpenRouter ; 5xx). On respecte l'en-tête Retry-After si présent.
        response = None
        for attempt in range(LLM_MAX_RETRIES):
            response = _post(_negotiated_format_mode)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < LLM_MAX_RETRIES - 1:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if (retry_after or "").isdigit() else LLM_RETRY_BACKOFF_S * (2 ** attempt)
                logger.warning("LLM %s (tentative %d/%d), nouvel essai dans %.1fs.",
                               response.status_code, attempt + 1, LLM_MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            break
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip()
        # Tolérance : certains modèles encadrent le JSON en markdown malgré response_format
        if "```" in raw:
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
        faits = _validate_extractions(json.loads(raw), event_content, registry)
        logger.info("%d fait(s) extrait(s) de l'événement.", len(faits))
        return faits
    except Exception:
        logger.error("Échec de l'extraction LLM, repli sur les heuristiques regex.", exc_info=True)
        EXTRACTION_DEGRADED_COUNTER.inc()
        return [_heuristic_extract(event_content)]


_INSERT_MEMORY = """
    INSERT INTO memories (tenant_id, agent_id, type, subtype, content, summary, embedding,
                          confidence, importance, provenance, source_event_id,
                          occurred_at, content_hash)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (source_event_id, content_hash) WHERE source_event_id IS NOT NULL DO NOTHING
    RETURNING id;
"""


def _entangle(cur, tenant_id: str, agent_id: str, new_mem_id, fact: dict, embedding) -> None:
    """Relie un souvenir à ses plus proches voisins sémantiques (graphe Q-EM)."""
    embedding_str = to_pgvector(embedding)
    # `ORDER BY embedding <=> %s` et non `ORDER BY similarity DESC` : pgvector n'utilise
    # l'index HNSW que sur l'opérateur de distance. Trier sur l'alias forçait un scan
    # complet des mémoires de l'agent À CHAQUE fait extrait — le coût de l'intrication
    # croissait donc linéairement avec la taille de la mémoire.
    cur.execute("""
        SELECT id, type, subtype, (1 - (embedding <=> %s::vector)) AS similarity
        FROM memories
        WHERE tenant_id = %s AND agent_id = %s AND id != %s AND status = 'active'
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """, (embedding_str, tenant_id, agent_id, new_mem_id, embedding_str))

    for rel_row in cur.fetchall():
        similarity = float(rel_row[3] or 0.0)
        if similarity <= QEM_ENTANGLE_THRESHOLD:
            continue
        target_id, target_subtype = rel_row[0], rel_row[2]
        relation_type = "entangled_with"
        inverse = False
        # Une bonne pratique résout/remplace l'erreur associée (et réciproquement).
        if fact['subtype'] == 'coding_best_practices' and target_subtype == 'code_error_resolution':
            relation_type = "supersedes_by"
        elif fact['subtype'] == 'code_error_resolution' and target_subtype == 'coding_best_practices':
            relation_type, inverse = "supersedes_by", True

        insert_rel = """
            INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING;
        """
        pair = (target_id, new_mem_id) if inverse else (new_mem_id, target_id)
        cur.execute(insert_rel, (*pair, relation_type, similarity))
        logger.info("Intrication Q-EM : %s --(%s)--> %s (sim=%.2f)",
                    pair[0], relation_type, pair[1], similarity)


def process_event(event: dict) -> bool:
    """Traite un événement brut : extraction des faits, consolidation, intrication.

    Un événement produit désormais PLUSIEURS mémoires (un tour de dialogue énonce souvent
    plusieurs faits). Le tout est écrit dans une transaction unique : soit l'événement est
    entièrement consolidé, soit il sera rejoué intégralement.
    """
    tenant_id = event['tenant_id']
    agent_id = event['agent_id']
    content = event['content']
    event_id = event['id']
    # Horodatage de l'événement : référence pour résoudre les dates relatives. Fourni par
    # l'API dans le payload outbox ; absent, le LLM ne pourra simplement pas dater.
    event_time = event.get('created_at')

    # Corrélation : tous les logs de la consolidation de cet événement porteront le même
    # identifiant, y compris ceux émis depuis synaptiq_core (extraction, gouvernance).
    set_trace_id(f"event_{event_id}")
    logger.info("Traitement de l'événement %s pour l'agent %s.", event_id, agent_id,
                extra={"event_id": str(event_id), "agent_id": agent_id, "tenant_id": tenant_id})

    # 0. Registre des collections de cet agent, chargé AVANT l'extraction.
    #
    # Emprunt de connexion à part, volontairement court : les étapes 1 et 2 sont deux appels
    # RÉSEAU (LLM puis embeddings, jusqu'à 30 s chacun). Garder une connexion du pool
    # mobilisée pendant ce temps assècherait le pool sous charge — c'est précisément pour
    # cela que l'écriture n'emprunte qu'à l'étape 3.
    #
    # Le registre est nécessaire dès l'extraction : sans lui, `normalize_extraction` écrasait
    # toute collection non canonique par le défaut de sa famille. Un `clients_paca` proposé
    # par le LLM devenait `fact`, alors que l'écriture directe l'aurait conservé — deux
    # chemins, deux règles, la divergence que la taxonomie partagée avait déjà eu à corriger.
    pool = get_db_pool()
    conn_registre = pool.getconn()
    try:
        with conn_registre.cursor() as cur_reg:
            registre = charger_registre(cur_reg, tenant_id, agent_id)
        conn_registre.rollback()  # lecture seule
    finally:
        pool.putconn(conn_registre)

    # 1. Extraction : 1 à N faits, datés quand l'interaction le permet
    facts = call_llm_extractor(content, occurred_at=event_time, registry=registre)

    # 2. Embeddings en UN SEUL appel pour tous les faits (l'interface Embedder est batch)
    embeddings = get_embedder().embed([f['content'] for f in facts])

    # 3. Écriture en base : contradictions, insertion, intrication
    conn = pool.getconn()
    broken = False
    try:
        created = 0
        with conn.cursor() as cur:
            for fact, embedding in zip(facts, embeddings, strict=False):
                # Contradictions : archivage sur verdict EXPLICITE seulement (jamais sur la
                # seule similarité). Les ids retournés serviront à tisser l'arête de
                # supersession, une fois le nouvel id connu.
                superseded = handle_contradictions(cur, tenant_id, agent_id, fact, embedding)

                provenance = {"source": "event", "event_id": event_id}
                cur.execute(_INSERT_MEMORY, (
                    tenant_id, agent_id, fact['type'], fact['subtype'], fact['content'],
                    fact['summary'], embedding, fact['confidence'], fact['importance'],
                    json.dumps(provenance), event_id,
                    fact.get('occurred_at'), content_hash(fact['content']),
                ))
                row = cur.fetchone()
                if row is None:
                    # Ce fait précis a déjà été consolidé (replay) : rien à refaire.
                    continue
                created += 1
                new_mem_id = row[0]
                logger.info("Mémoire %s créée (%s/%s, date=%s).", new_mem_id,
                            fact['type'], fact['subtype'], fact.get('occurred_at') or "—")

                # 3bis. Traçabilité de la supersession : sans cette arête, un archivage
                # serait indiscernable d'une disparition (rien ne dirait par quoi la
                # préférence a été remplacée).
                if superseded:
                    link_supersedes(cur, new_mem_id, superseded)

                # 4. Graphe d'intrication : la collection décide, plus la variable globale.
                if _is_entanglement_candidate(fact, registre):
                    _entangle(cur, tenant_id, agent_id, new_mem_id, fact, embedding)

        conn.commit()
        if created == 0:
            logger.info("Événement %s déjà consolidé; ACK sans duplication.", event_id)
        return True

    except Exception:
        broken = True
        try:
            conn.rollback()
            broken = False  # rollback réussi → connexion réutilisable
        except Exception:
            # Le rollback lui-même a échoué : la connexion est inutilisable et sera fermée
            # par le `finally`. On journalise pour ne pas masquer une panne de connexion
            # derrière l'erreur SQL d'origine, mais on ne relance pas : c'est cette dernière
            # qui explique l'échec de l'événement.
            logger.warning("Rollback impossible : la connexion sera fermée.", exc_info=True)
        logger.error("Erreur SQL lors de l'enregistrement de la mémoire.", exc_info=True)
        return False
    finally:
        # Recycler la connexion ; la fermer si son état est douteux (rollback échoué).
        pool.putconn(conn, close=broken)

# ─── File d'événements : Redis Streams (consumer group + ACK + DLQ) ───
STREAM = os.getenv("EVENT_STREAM", "synaptiq:events")
GROUP = os.getenv("EVENT_GROUP", "synaptiq-workers")
DLQ = os.getenv("EVENT_DLQ", "synaptiq:events:dlq")
CONSUMER = f"worker-{os.getpid()}"
MAX_DELIVERIES = int(os.getenv("EVENT_MAX_DELIVERIES", "5"))
IDLE_RECLAIM_MS = int(os.getenv("EVENT_IDLE_RECLAIM_MS", "30000"))
# Intervalle max entre deux reclaim, même sous charge continue (retry qui avance toujours).
RECLAIM_INTERVAL_MS = int(os.getenv("EVENT_RECLAIM_INTERVAL_MS", "15000"))


def ensure_group(r) -> None:
    """Crée le consumer group (et le stream) s'il n'existe pas déjà."""
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        logger.info("Consumer group '%s' créé sur le stream '%s'.", GROUP, STREAM)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _to_dlq(r, msg_id: str, raw: str, reason: str, deliveries: int = 0) -> None:
    r.xadd(DLQ, {"data": raw or "", "error": reason, "deliveries": str(deliveries), "orig_id": msg_id})
    r.xack(STREAM, GROUP, msg_id)
    logger.error("Message %s envoyé en DLQ (%s).", msg_id, reason)


def _handle(r, msg_id: str, fields: dict) -> None:
    """Traite un message : ACK si OK, DLQ si empoisonné/dépassé, sinon laissé en pending."""
    raw = fields.get("data", "")
    # 1. Décodage : un message illisible est empoisonné -> DLQ direct (pas de boucle infinie)
    try:
        event = json.loads(raw)
    except Exception as e:
        _to_dlq(r, msg_id, raw, f"decode: {e}")
        return

    # 2. Traitement métier
    if process_event(event):
        r.xack(STREAM, GROUP, msg_id)
        return

    # 3. Échec : router en DLQ si le nombre de livraisons dépasse le plafond
    deliveries = 1
    pend = r.xpending_range(STREAM, GROUP, min=msg_id, max=msg_id, count=1)
    if pend:
        deliveries = pend[0].get("times_delivered", 1)
    if deliveries >= MAX_DELIVERIES:
        _to_dlq(r, msg_id, raw, "process_event failed", deliveries)
    else:
        logger.warning("Événement %s en échec (livraison %d/%d), re-livraison ultérieure.",
                       msg_id, deliveries, MAX_DELIVERIES)
        # Pas d'ACK : le message reste pending et sera repris par _reclaim()


def _reclaim(r) -> None:
    """Reprend les messages pending trop longtemps (worker mort, échec précédent)."""
    try:
        res = r.xautoclaim(STREAM, GROUP, CONSUMER, min_idle_time=IDLE_RECLAIM_MS,
                           start_id="0-0", count=10)
        # redis-py renvoie [next_cursor, messages] ou [next_cursor, messages, deleted]
        messages = res[1] if len(res) >= 2 else []
        for msg_id, fields in messages:
            if fields:
                _handle(r, msg_id, fields)
    except Exception as e:
        logger.debug("reclaim ignoré : %s", e)


# Cosinus minimal entre un vecteur stocké et le même contenu ré-embarqué par le modèle
# COURANT. Deux modèles différents de même dimension descendent très bas ; le même modèle
# rend 1.000 à la précision flottante près.
EMBEDDING_COHERENCE_MIN = float(os.getenv("EMBEDDING_COHERENCE_MIN", "0.999"))
EMBEDDING_COHERENCE_CHECK = os.getenv("EMBEDDING_COHERENCE_CHECK", "true").lower() in ("1", "true", "yes")


def verifier_coherence_embedding() -> None:
    """Refuse de démarrer si `EMBEDDING_MODEL` n'est pas celui qui a écrit les vecteurs.

    ## Pourquoi un contrôle de dimension ne suffit pas

    `EMBEDDING_DIM` protège du seul cas bruyant. Le cas dangereux est silencieux : deux
    modèles de MÊME dimension. L'instance de production a tourné avec un modèle anglophone
    (`all-minilm-l6-v2`, 384) sur une base écrite par un modèle multilingue
    (`paraphrase-multilingual-minilm-l12-v2`, 384 aussi). Aucune exception, aucun log, aucune
    métrique : les vecteurs cessent simplement d'être comparables et le rappel se dégrade EN
    SILENCE. Pour un moteur de mémoire, c'est indiscernable d'une mémoire pauvre — donc
    indébuggable de l'extérieur.

    Le seul test valide est empirique : ré-embarquer un contenu déjà stocké et comparer. Même
    modèle -> cosinus 1.000. Modèle différent -> nettement en dessous.

    Le worker QUITTE ici, contrairement au serveur MCP qui démarre dégradé. La différence est
    le lecteur : un worker est supervisé par un orchestrateur qui affiche ses sorties, alors
    qu'un client MCP jette stderr et se contente d'un « exit status 1 ». Et surtout, un worker
    qui continue ÉCRIT des vecteurs incompatibles — chaque événement traité aggrave les
    dégâts. Refuser de démarrer est ici la seule option non destructive.

    Base vierge ou vecteur illisible : on laisse passer. Il n'y a rien à contredire.
    """
    if not EMBEDDING_COHERENCE_CHECK:
        logger.warning("Contrôle de cohérence des embeddings DÉSACTIVÉ "
                       "(EMBEDDING_COHERENCE_CHECK=false).")
        return
    try:
        pool = get_db_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, embedding::text FROM memories "
                    "WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT 1"
                )
                ligne = cur.fetchone()
            conn.rollback()
        finally:
            pool.putconn(conn)
    except Exception:
        # Base injoignable au démarrage : ce n'est pas à ce contrôle de trancher. La boucle
        # principale a sa propre gestion de l'indisponibilité.
        logger.warning("Cohérence des embeddings non vérifiée (base injoignable).", exc_info=True)
        return

    if ligne is None or not ligne[1]:
        logger.info("Base sans vecteur : cohérence des embeddings sans objet.")
        return

    try:
        stocke = np.asarray(ligne[1].strip("[]").split(","), dtype=np.float64)
        recalcule = np.asarray(get_embedder().embed_one(ligne[0]), dtype=np.float64)
    except Exception:
        logger.warning("Cohérence des embeddings non vérifiable (embedder injoignable).",
                       exc_info=True)
        return

    if stocke.shape != recalcule.shape:
        raise SystemExit(
            f"EMBEDDING_MODEL={os.getenv('EMBEDDING_MODEL', '?')} produit des vecteurs de "
            f"dimension {recalcule.shape[0]}, la base en contient de {stocke.shape[0]}."
        )
    normes = float(np.linalg.norm(stocke) * np.linalg.norm(recalcule))
    cosinus = float(stocke @ recalcule / normes) if normes else 0.0
    if cosinus < EMBEDDING_COHERENCE_MIN:
        raise SystemExit(
            f"INCOHÉRENCE D'EMBEDDING : EMBEDDING_MODEL={os.getenv('EMBEDDING_MODEL', '?')} "
            f"ne correspond pas aux vecteurs déjà en base (cosinus {cosinus:.3f}, attendu "
            f">= {EMBEDDING_COHERENCE_MIN}). Les dimensions concordent, donc rien n'aurait "
            f"été signalé : le rappel se serait dégradé en silence. Restaurer le modèle "
            f"d'origine, ou ré-embarquer toute la base avec le nouveau."
        )
    logger.info("Cohérence des embeddings vérifiée (cosinus %.3f).", cosinus)


def main():
    logger.info("SynaptiQ Memory Worker démarré (consumer=%s)...", CONSUMER)
    verifier_coherence_embedding()
    r = None
    while r is None:
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            logger.info("Connecté à Redis avec succès.")
        except Exception as e:
            logger.warning(f"En attente de Redis... ({e})")
            time.sleep(2)

    ensure_group(r)

    # Reclaim périodique INDÉPENDANT du débit : sous charge continue, `xreadgroup`
    # ne retourne jamais vide, donc le reclaim déclenché sur inactivité seul ne passait
    # jamais -> les messages en échec restaient bloqués en pending sans être rejugés.
    # On force un reclaim au moins tous les RECLAIM_INTERVAL_MS quel que soit le trafic.
    last_reclaim = time.monotonic()
    reclaim_interval_s = RECLAIM_INTERVAL_MS / 1000.0

    # Boucle de consommation via consumer group (XREADGROUP bloquant, ACK explicite)
    while True:
        try:
            resp = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=10, block=5000)
            now = time.monotonic()
            if not resp or (now - last_reclaim) >= reclaim_interval_s:
                # Inactivité OU intervalle écoulé : reprendre les pending bloqués.
                _reclaim(r)
                last_reclaim = now
            if not resp:
                continue
            for _stream, messages in resp:
                for msg_id, fields in messages:
                    _handle(r, msg_id, fields)
        except KeyboardInterrupt:
            logger.info("Arrêt du worker par l'utilisateur.")
            break
        except Exception:
            logger.error("Erreur dans la boucle principale du worker.", exc_info=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
