import os
import sys

# Ajouter la racine du projet + packages/core au sys.path (imports monorepo, dev local)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (root_path, os.path.join(root_path, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hashlib
import json
import logging
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, model_validator

v1_router = APIRouter()
import redis
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

# Logique partagée (embeddings pluggables + gouvernance), plus d'import depuis le worker
from synaptiq_core import (
    content_hash,
    entangle,
    get_embedder,
    handle_contradictions,
    link_supersedes,
    to_pgvector,
)

# Registre des collections : le rangement est un objet que l'agent possède. Le chargement
# vit dans le cœur (comme `handle_contradictions`), afin que l'API et le worker voient
# forcément le MÊME registre — la taxonomie avait déjà divergé une fois entre les deux
# chemins d'écriture, elle ne doit pas recommencer.
from synaptiq_core.collections import charger_registre

# Orchestration des 4 phases Q-EM (sans SQL ni HTTP : testable en isolation)
from synaptiq_core.context_builder import RetrievalConfig, build_context_packet

# Journalisation structurée + corrélation par trace_id
from synaptiq_core.observability import configure_logging, set_trace_id

# Taxonomie partagée avec le worker : les DEUX chemins d'écriture appliquent la même règle
from synaptiq_core.qem import route_memory

# Fusion de classements pour la recherche hybride (fonctions pures, cf. retrieval.py)
from synaptiq_core.retrieval import DEFAULT_RRF_K, fuse_and_rank
from synaptiq_core.taxonomy import SubtypeMismatch, is_canonical, validate_subtype

# Configuration du logging
configure_logging("synaptiq-api")
logger = logging.getLogger("synaptiq-api")

# Chargement des variables d'environnement depuis le .env RACINE (source unique).
# NB : load_dotenv() sans argument remonterait depuis apps/api/ et chargerait un
# apps/api/.env résiduel — on force donc le .env de la racine du monorepo.
load_dotenv(os.path.join(root_path, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

# File d'événements (Redis Streams) + idempotence
EVENT_STREAM = os.getenv("EVENT_STREAM", "synaptiq:events")
IDEMPOTENCY_TTL = int(os.getenv("IDEMPOTENCY_TTL", "86400"))  # 24 h

# ─── Pools de connexions (thread-safe, initialisés au lifespan) ───
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))

# Seuils du moteur Q-EM (externalisés : ajustables sans redéploiement de code).
# Amortissement de la propagation d'activation le long des liens 'entangled_with'.
QEM_ENTANGLE_DAMPING = float(os.getenv("QEM_ENTANGLE_DAMPING", "0.5"))
# Nombre maximal de sauts de propagation d'activation (spreading activation multi-hop).
# 1 = comportement mono-saut historique ; 2 (défaut) ramène les souvenirs à 2 liens.
QEM_ENTANGLE_MAX_HOPS = int(os.getenv("QEM_ENTANGLE_MAX_HOPS", "2"))
# Au-delà de ce cosinus entre deux candidats, le moins prioritaire est filtré (redondance).
QEM_REDUNDANCY_THRESHOLD = float(os.getenv("QEM_REDUNDANCY_THRESHOLD", "0.75"))
# Décroissance temporelle : demi-vie (en jours) du score de récence. Une mémoire non
# ré-accédée voit sa pertinence divisée par 2 tous les N jours. 0 (ou négatif) = désactivé.
QEM_RECENCY_HALFLIFE_DAYS = float(os.getenv("QEM_RECENCY_HALFLIFE_DAYS", "90"))
# ─── Recherche hybride (vectoriel + plein texte) ───
# Le vecteur ramène le « sémantiquement proche », le plein texte les correspondances
# littérales (noms propres, dates, identifiants). Désactivable pour mesurer son apport.
RETRIEVAL_HYBRID = os.getenv("RETRIEVAL_HYBRID", "true").lower() in ("1", "true", "yes")
# Nombre de candidats ramenés PAR CHEMIN avant fusion.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "50"))
# Amortissement de la fusion par rang (RRF). 60 = valeur de référence.
RRF_K = int(os.getenv("RRF_K", str(DEFAULT_RRF_K)))
# Importance relative des deux chemins dans la fusion.
RRF_WEIGHT_VECTOR = float(os.getenv("RRF_WEIGHT_VECTOR", "1.0"))
RRF_WEIGHT_FTS = float(os.getenv("RRF_WEIGHT_FTS", "1.0"))

db_pool: pg_pool.ThreadedConnectionPool | None = None
redis_client = None
EVENTS_CAPTURED = Counter("synaptiq_events_captured_total", "Events persisted in the transactional outbox")
# `outcome=duplicate` compte les relances neutralisées. Sans ce compteur, l'idempotence est
# invérifiable de l'extérieur : un no-op et une création se ressemblent trop côté client.
MEMORY_WRITES = Counter("synaptiq_memory_writes_total", "Direct memory writes", ["outcome"])
CONTEXT_BUILDS = Counter("synaptiq_context_builds_total", "Context builds", ["outcome"])
CONTEXT_BUILD_SECONDS = Histogram("synaptiq_context_build_seconds", "Context build latency")

# ─── Jauges de santé du pipeline d'ingestion ───
# Les compteurs disent ce qui s'est passé ; ces deux jauges disent si le pipeline est en
# train de décrocher MAINTENANT. Ce sont les seules métriques qui préviennent l'incident :
#   - l'âge du plus vieil événement non publié révèle un relais mort (les compteurs, eux,
#     restent simplement plats, ce qui est indiscernable d'une absence de trafic) ;
#   - la profondeur de la DLQ révèle des événements empoisonnés qui ne seront jamais
#     consolidés — la donnée est acceptée côté client mais n'arrivera jamais en mémoire.
OUTBOX_PENDING = Gauge("synaptiq_outbox_pending", "Committed events not yet published to Redis")
OUTBOX_OLDEST_AGE = Gauge("synaptiq_outbox_oldest_age_seconds",
                          "Age of the oldest unpublished outbox entry")
DLQ_DEPTH = Gauge("synaptiq_dlq_depth", "Messages parked in the dead-letter queue")
# Densité du graphe Q-EM, par agent. Un graphe vide ne produit aucune erreur : la phase
# d'intrication tourne simplement à vide et le rappel perd le multi-hop, en silence. C'est
# resté invisible des semaines sur une instance réelle faute de cette métrique.
GRAPH_EDGES = Gauge("synaptiq_graph_edges", "entangled_with edges per agent", ["agent_id"])
GRAPH_EDGES_PER_MEMORY = Gauge("synaptiq_graph_edges_per_memory",
                               "entangled_with edges divided by active memories, per agent",
                               ["agent_id"])


@contextmanager
def get_conn():
    """Emprunte une connexion au pool et la restitue systématiquement.

    Remplace l'ancienne connexion globale unique (non thread-safe) : chaque
    requête obtient sa propre connexion, évitant les conditions de course sous
    charge (FastAPI sert les routes sync dans un threadpool).
    """
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


def get_redis_client():
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis non initialisé")
    return redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie applicatif (remplace @app.on_event('startup') déprécié)."""
    global db_pool, redis_client
    try:
        db_pool = pg_pool.ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, dsn=DATABASE_URL)
        logger.info("Pool PostgreSQL initialisé (%d–%d connexions).", DB_POOL_MIN, DB_POOL_MAX)
    except Exception as e:
        logger.error("Échec d'initialisation du pool PostgreSQL : %s", e, exc_info=True)
        db_pool = None
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Client Redis initialisé.")
    except Exception as e:
        logger.error("Échec d'initialisation de Redis : %s", e, exc_info=True)
        redis_client = None
    yield
    if db_pool is not None:
        db_pool.closeall()
    if redis_client is not None:
        redis_client.close()


app = FastAPI(title="SynaptiQ API", version="0.3.0", lifespan=lifespan)

# ─── Sécurité : CORS + rate limiting ───
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# CORS : par défaut AUCUNE origine navigateur autorisée (SynaptiQ est appelée
# serveur-à-serveur par le SDK/MCP, non soumis au CORS). Pour un front web,
# lister explicitement les origines dans CORS_ORIGINS.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
_cors_wildcard = CORS_ORIGINS == ["*"]
if _cors_wildcard:
    logger.warning("CORS_ORIGINS=* : credentials désactivés (combinaison non conforme). "
                   "Lister des origines explicites pour un front navigateur avec cookies.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # '*' est incompatible avec allow_credentials=True : on désactive alors les credentials.
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

with warnings.catch_warnings():
    # slowapi avertit qu'il n'a pas trouvé le fichier de configuration qu'on lui demande
    # justement d'ignorer : bruit pur, à chaque démarrage.
    warnings.filterwarnings("ignore", message="Config file '' not found",
                            category=UserWarning)
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[os.getenv("RATE_LIMIT", "120/minute")],
        # `config_filename=""` empêche slowapi de relire `.env` de son côté.
    #
        # Il l'ouvre via `starlette.config.Config`, SANS spécifier d'encodage : sur Windows
        # c'est donc cp1252, et un seul octet UTF-8 non représentable y fait planter l'API au
        # démarrage sur un `UnicodeDecodeError` opaque, très loin de sa cause. Un emoji suffit
        # (l'octet 0x8f du sélecteur de variante U+FE0F) — et le `.env.example` livré en
        # contient, donc le quickstart documenté `cp .env.example .env` suivi d'un lancement
        # local sur Windows échouait.
        #
        # Cette relecture est de toute façon redondante : SynaptiQ charge sa configuration
        # via `load_dotenv` (qui lit en UTF-8) et passe ici `default_limits` explicitement.
        config_filename="",
    )
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ─── Authentification par clé API + scoping tenant ───
AUTH_REQUIRED = os.getenv("SYNAPTIQ_AUTH_REQUIRED", "true").lower() in ("1", "true", "yes")


def _instance_tenant() -> str:
    """Tenant de l'instance auto-hébergée (un déploiement = un tenant).

    Lu dynamiquement (pas figé à l'import) pour rester testable et reconfigurable.
    N'est jamais fourni par l'appelant : le périmètre est décidé par le serveur.
    """
    return os.getenv("SYNAPTIQ_TENANT", "default")


class AuthContext:
    """Ce qu'une clé API autorise : un tenant, des permissions, un périmètre d'agents."""

    def __init__(self, tenant_id: str, scopes: list[str] | None = None,
                 agent_scope: list[str] | None = None, actor: str = "") -> None:
        self.tenant_id = tenant_id
        # Défaut lecture+écriture : aligné sur le DEFAULT de la colonne (clés antérieures
        # aux scopes). L'absence de 'admin' est volontaire — la purge doit être explicite.
        self.scopes = list(scopes) if scopes else ["read", "write"]
        # None = tous les agents du tenant (comportement historique). Une liste restreint.
        self.agent_scope = list(agent_scope) if agent_scope else None
        # Préfixe du hash de clé : identifie l'appelant dans l'audit sans stocker de secret.
        self.actor = actor


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─── Cache d'authentification ───
# Chaque requête faisait un SELECT + un UPDATE + un COMMIT sur `api_keys`. Toutes les
# requêtes d'une même clé écrivaient donc sur la MÊME ligne : verrou disputé, WAL inutile,
# et une latence plancher doublée sur l'endpoint le plus chaud du produit.
#
# Le cache accepte une fenêtre de révocation bornée (`AUTH_CACHE_TTL`, 60 s par défaut) :
# une clé désactivée reste acceptée au maximum ce délai. Mettre 0 désactive le cache pour
# une révocation immédiate. `last_used_at` n'est plus rafraîchi qu'une fois par fenêtre —
# c'est un indicateur d'usage, pas un journal d'accès (l'audit, lui, est dans `audit_log`).
AUTH_CACHE_TTL = float(os.getenv("AUTH_CACHE_TTL", "60"))
AUTH_CACHE_MAX = int(os.getenv("AUTH_CACHE_MAX", "1024"))
_auth_cache: "dict[str, tuple[float, tuple]]" = {}
_auth_cache_lock = threading.Lock()
AUTH_CACHE_HITS = Counter("synaptiq_auth_cache_total", "API key resolutions", ["outcome"])


def _auth_cache_get(key_hash: str):
    """Entrée de cache encore valide, sinon None. Les clés invalides ne sont pas mises en cache."""
    if AUTH_CACHE_TTL <= 0:
        return None
    with _auth_cache_lock:
        entree = _auth_cache.get(key_hash)
        if entree is None:
            return None
        expiration, valeur = entree
        if time.monotonic() >= expiration:
            _auth_cache.pop(key_hash, None)
            return None
        return valeur


def _auth_cache_put(key_hash: str, valeur: tuple) -> None:
    if AUTH_CACHE_TTL <= 0:
        return
    with _auth_cache_lock:
        # Éviction grossière mais suffisante : une instance a une poignée de clés, et le
        # plafond ne sert qu'à empêcher une croissance non bornée sous flot de clés invalides.
        if len(_auth_cache) >= AUTH_CACHE_MAX:
            _auth_cache.clear()
        _auth_cache[key_hash] = (time.monotonic() + AUTH_CACHE_TTL, valeur)


def invalidate_auth_cache() -> None:
    """Vide le cache des clés API (révocation immédiate, et isolation entre tests)."""
    with _auth_cache_lock:
        _auth_cache.clear()


def get_auth(authorization: str | None = Header(default=None)) -> AuthContext | None:
    """Résout la clé API (header Bearer) vers un tenant, des scopes et un périmètre d'agents.

    - Aucune clé + auth désactivée  -> None (mode dev, tous droits, pas d'isolation).
    - Aucune clé + auth requise      -> 401.
    - Clé fournie                    -> validée en base, sinon 401.
    """
    if not authorization:
        if AUTH_REQUIRED:
            raise HTTPException(status_code=401, detail="Clé API requise (Authorization: Bearer <clé>)")
        return None
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Format attendu : Authorization: Bearer <clé>")
    raw = authorization.split(" ", 1)[1].strip()
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    key_hash = _hash_key(raw)

    cachee = _auth_cache_get(key_hash)
    if cachee is not None:
        AUTH_CACHE_HITS.labels("hit").inc()
        return AuthContext(tenant_id=cachee[0], scopes=cachee[1], agent_scope=cachee[2],
                           actor=key_hash[:8])

    # Initialisé AVANT le try : une exception SQL laissait auparavant `row` non liée,
    # transformant une panne base en NameError opaque au lieu d'un 401/503 propre.
    row = None
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id, scopes, agent_scope FROM api_keys "
                "WHERE key_hash = %s AND active = true",
                (key_hash,),
            )
            row = cur.fetchone()
            if row:
                # Rafraîchi au plus une fois par fenêtre de cache (l'entrée qui suit fait
                # que les requêtes suivantes ne repassent plus ici).
                cur.execute(
                    "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE key_hash = %s",
                    (key_hash,),
                )
                conn.commit()
    finally:
        db_pool.putconn(conn)
    if not row:
        # Une clé invalide n'est JAMAIS mise en cache : sinon un attaquant pourrait remplir
        # le cache de hachages arbitraires, et une clé réactivée resterait rejetée.
        AUTH_CACHE_HITS.labels("invalid").inc()
        raise HTTPException(status_code=401, detail="Clé API invalide ou révoquée")
    AUTH_CACHE_HITS.labels("miss").inc()
    _auth_cache_put(key_hash, (row[0], row[1], row[2]))
    return AuthContext(tenant_id=row[0], scopes=row[1], agent_scope=row[2],
                       actor=key_hash[:8])


def resolve_tenant(auth: AuthContext | None) -> str:
    """Résout le tenant effectif de la requête.

    - Clé API valide -> tenant porté par la clé.
    - Sans auth (instance auto-hébergée) -> tenant d'instance (SYNAPTIQ_TENANT).

    Le tenant n'est plus jamais transmis par l'appelant : impossible de lire ou
    d'écrire dans un autre périmètre en trafiquant le body.
    """
    return auth.tenant_id if auth else _instance_tenant()


def require_scope(auth: AuthContext | None, scope: str) -> None:
    """Exige une permission portée par la clé.

    Trois permissions : `read` (retrieve, context/build), `write` (events, memories),
    `admin` (purge). Une clé de lecture ne doit pas pouvoir écrire, et surtout aucune clé
    d'agent ne doit pouvoir vider l'instance.

    Sans auth (`SYNAPTIQ_AUTH_REQUIRED=false`), `read` et `write` passent — c'est le mode
    d'une instance de confiance sur localhost. **`admin` ne passe JAMAIS sans clé**, et cette
    exception est le cœur de la fonction.

    Mesuré sur l'instance de production le 30/07 : `DELETE /v1/memories?confirm=<mauvais>`
    répondait `400`, et non `401`. Aucune clé n'était fournie. L'enchaînement était complet —
    `get_auth()` renvoie None, cette fonction retournait immédiatement, donc les trois
    garde-fous de la purge (permission `admin`, `?confirm=<tenant>`, ligne d'audit) se
    réduisaient à la connaissance du nom du tenant… que le message d'erreur 400 livre
    lui-même. Un seul appel local suffisait à vider la mémoire de l'instance.

    Un booléen de confort ne doit pas ouvrir un endpoint irréversible : la commodité du mode
    sans auth vaut pour la lecture et l'écriture, pas pour la destruction.
    """
    if auth is None:
        if scope == "admin":
            raise HTTPException(
                status_code=403,
                detail="La purge exige une clé API portant la permission 'admin' "
                       "(scripts/create_api_key.py --scopes read write admin), "
                       "même lorsque SYNAPTIQ_AUTH_REQUIRED=false.",
            )
        return
    if scope not in auth.scopes:
        raise HTTPException(
            status_code=403,
            detail=f"Permission '{scope}' absente de cette clé API (scopes : {', '.join(auth.scopes)}).",
        )


def resolve_agent(auth: AuthContext | None, requested_agent_id: str) -> str:
    """Vérifie que la clé a le droit d'agir au nom de `requested_agent_id`.

    `agent_id` arrive du corps de la requête — et, côté MCP, il était jusqu'au 29/07 un
    paramètre d'outil, donc une valeur que le LLM lui-même choisissait. Sans ce contrôle,
    l'isolation entre agents d'une même instance n'était qu'une convention : il suffisait
    de changer une chaîne pour lire la mémoire d'un autre agent.

    Une clé sans `agent_scope` (NULL en base) conserve l'accès à tous les agents de son
    tenant : c'est le comportement historique, et le cas normal d'une instance mono-agent.
    """
    if auth is not None and auth.agent_scope and requested_agent_id not in auth.agent_scope:
        raise HTTPException(
            status_code=403,
            detail=f"Cette clé API n'est pas autorisée pour l'agent '{requested_agent_id}' "
                   f"(périmètre : {', '.join(auth.agent_scope)}).",
        )
    return requested_agent_id


def audit(cur, tenant_id: str, action: str, auth: AuthContext | None,
          agent_id: str | None = None, **details) -> None:
    """Journalise une opération sensible. Compteurs et paramètres UNIQUEMENT.

    La table survit à la purge RGPD : elle ne doit donc jamais porter de contenu mémoire.
    """
    cur.execute(
        "INSERT INTO audit_log (tenant_id, agent_id, action, actor, details) "
        "VALUES (%s, %s, %s, %s, %s)",
        (tenant_id, agent_id, action, (auth.actor if auth else "no-auth"), json.dumps(details)),
    )

def _fetch_candidates(cur, vector_str: str, query_text: str, tenant: str,
                      agent_id: str, memory_types: list[str],
                      collections: list[str] | None = None) -> list[dict]:
    """Ramène les candidats par similarité vectorielle ET, si activé, par plein texte.

    Une seule requête à deux CTE plutôt que deux allers-retours : chaque ligne porte son
    rang dans chaque chemin (`rank_vec`, `rank_fts`, NULL quand le chemin ne l'a pas
    trouvée), ce qui permet la fusion RRF côté Python sur des fonctions pures testables.

    `websearch_to_tsquery` est utilisé plutôt que `plainto_tsquery` : il tolère une requête
    en langage naturel sans lever d'erreur de syntaxe, ce qui est indispensable ici où la
    requête vient d'un agent et n'est jamais échappée à la main.

    ⚠️ Chaque chemin attaque `memories` DIRECTEMENT, et le filtre tenant/agent y est répété.
    C'est délibéré et non négociable : la version précédente factorisait ce filtre dans une
    CTE `filtre` référencée trois fois. Or PostgreSQL n'inline une CTE que si elle est
    référencée UNE seule fois — au-delà elle est matérialisée, le tri par distance ne porte
    donc plus sur la table mais sur un résultat intermédiaire, et l'index HNSW devient
    inutilisable. Mesuré sur 5 000 mémoires (`scripts/explain_retrieval.py`) :
    `Seq Scan` + tri de 5 000 lignes en **64,3 ms**, contre un `Index Scan using
    idx_memories_embedding_hnsw` en **6,2 ms** avec la forme ci-dessous. L'ancien plan est
    linéaire en volume : l'écart croît avec le corpus. Ne pas « simplifier » en refactorisant
    le filtre commun dans une CTE.
    """
    champs = ("id", "type", "subtype", "content", "confidence", "importance",
              "last_accessed_at", "created_at", "occurred_at", "embedding::text")
    colonnes = ", ".join(champs)
    # Le SELECT final joint `memories` (alias m) et les rangs : sans préfixe, PostgreSQL
    # refuse la requête ("column reference id is ambiguous").
    colonnes_m = ", ".join(f"m.{c}" for c in champs)

    # Filtrage FIN par collection. Le fragment est CHOISI par le serveur (deux formes
    # possibles, sans aucune donnee d'appelant) ; les noms de collection passent par un
    # parametre lie, comme le reste. Meme motif que `type_filter` dans `retrieve_memories`.
    #
    # ⚠️ Le fragment est place APRES `type = ANY(...)` dans chaque WHERE, et les parametres
    # suivent le meme ordre : ces requetes sont positionnelles, un decalage y serait muet
    # (les types concordent tous) et fausserait le filtre au lieu de lever.
    filtre_col = "AND subtype = ANY(%s)" if collections else ""
    filtre_col_m = "AND m.subtype = ANY(%s)" if collections else ""
    p_col: list = [collections] if collections else []

    # Interpolation limitee a `colonnes`/`colonnes_m` (listes de colonnes CONSTANTES,
    # definies juste au-dessus) : aucune valeur d'appelant n'entre dans le SQL, toutes
    # passent par des parametres lies (cf. l'exemption S608 justifiee dans ruff.toml).
    if not RETRIEVAL_HYBRID:
        # `ORDER BY embedding <=> x` (et non `ORDER BY similarity DESC`) : pgvector
        # n'utilise l'index HNSW que sur l'opérateur de distance, jamais sur un alias.
        cur.execute(f"""
            SELECT {colonnes},
                   (1 - (embedding <=> %s::vector)) AS similarity,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_accessed_at)) AS age_seconds,
                   row_number() OVER (ORDER BY embedding <=> %s::vector) AS rank_vec,
                   NULL::bigint AS rank_fts
            FROM memories
            WHERE tenant_id = %s AND agent_id = %s AND type = ANY(%s) {filtre_col}
              AND status = 'active'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_str, vector_str, tenant, agent_id, memory_types, *p_col, vector_str,
              RETRIEVAL_CANDIDATES))
        return cur.fetchall()

    cur.execute(f"""
        WITH vectoriel AS (
            SELECT id, row_number() OVER (ORDER BY distance) AS rank_vec
            FROM (
                SELECT id, embedding <=> %s::vector AS distance
                FROM memories
                WHERE tenant_id = %s AND agent_id = %s AND type = ANY(%s) {filtre_col}
                  AND status = 'active'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ) v
        ),
        plein_texte AS (
            SELECT id, row_number() OVER (ORDER BY score DESC) AS rank_fts
            FROM (
                SELECT m.id, ts_rank(m.content_tsv, q.query) AS score
                FROM memories m, websearch_to_tsquery('simple', %s) AS q(query)
                WHERE m.tenant_id = %s AND m.agent_id = %s AND m.type = ANY(%s) {filtre_col_m}
                  AND m.status = 'active' AND m.content_tsv @@ q.query
                ORDER BY ts_rank(m.content_tsv, q.query) DESC
                LIMIT %s
            ) t
        ),
        -- Union des deux chemins : une mémoire trouvée par les deux porte ses deux rangs.
        retenus AS (
            SELECT id, min(rank_vec) AS rank_vec, min(rank_fts) AS rank_fts
            FROM (
                SELECT id, rank_vec, NULL::bigint AS rank_fts FROM vectoriel
                UNION ALL
                SELECT id, NULL::bigint AS rank_vec, rank_fts FROM plein_texte
            ) u
            GROUP BY id
        )
        SELECT {colonnes_m},
               (1 - (m.embedding <=> %s::vector)) AS similarity,
               EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - m.last_accessed_at)) AS age_seconds,
               r.rank_vec, r.rank_fts
        FROM retenus r JOIN memories m ON m.id = r.id;
    """, (vector_str, tenant, agent_id, memory_types, *p_col, vector_str, RETRIEVAL_CANDIDATES,
          query_text, tenant, agent_id, memory_types, *p_col, RETRIEVAL_CANDIDATES,
          vector_str))
    return cur.fetchall()


def parse_embedding(val):
    """Désérialise un vecteur pgvector renvoyé en texte ('[0.1,0.2,...]').

    Conversion vectorisée : une compréhension `[float(x) for x in ...]` faisait 384 appels
    à `float()` par candidat, soit ~19 000 conversions Python par construction de contexte
    (50 candidats). `np.asarray(..., dtype=float64)` fait la même chose en une passe C.
    Le tableau retourné alimente directement le produit matriciel de `filter_redundancy`.
    """
    if isinstance(val, str):
        val = val.strip('[]').strip()
        if not val:
            return []
        return np.asarray(val.split(','), dtype=np.float64)
    if val is None:
        return []
    # Déjà une séquence (liste ou tableau) : rien à convertir.
    return val


class PostgresMemoryStore:
    """Implémentation `MemoryStore` sur psycopg2, bornée à un (tenant, agent).

    Le périmètre est fixé À LA CONSTRUCTION et aucune méthode ne prend de `tenant_id` ni
    d'`agent_id`. C'est ce qui rend la fuite d'isolation F1 structurellement impossible à
    reproduire : la traversée du graphe ne peut plus « oublier » un filtre qu'elle n'a même
    pas la possibilité d'exprimer.
    """

    def __init__(self, cur, tenant_id: str, agent_id: str,
                 collections: list[str] | None = None) -> None:
        self._cur = cur
        self._tenant = tenant_id
        self._agent = agent_id
        # Restriction optionnelle à certaines collections. Fixée à la construction, comme le
        # périmètre : le cœur n'a pas à savoir qu'un filtre de rangement existe.
        self._collections = collections

    @staticmethod
    def _normaliser(ligne) -> dict:
        """Désérialise le vecteur ; le cœur ne connaît pas le format texte de pgvector."""
        valeur = dict(ligne)
        valeur["embedding"] = parse_embedding(valeur.get("embedding"))
        return valeur

    def fetch_candidates(self, query_vector, query_text: str, memory_types: list[str]) -> list[dict]:
        lignes = _fetch_candidates(self._cur, to_pgvector(query_vector), query_text,
                                   self._tenant, self._agent, memory_types,
                                   self._collections)
        return [self._normaliser(ligne) for ligne in lignes]

    def fetch_relationships(self, memory_ids: list[str]) -> list[dict]:
        self._cur.execute(
            """
            SELECT source_memory_id, target_memory_id, relation_type, weight
            FROM relationships
            WHERE source_memory_id = ANY(%s::uuid[])
               OR target_memory_id = ANY(%s::uuid[]);
            """,
            (memory_ids, memory_ids),
        )
        return self._cur.fetchall()

    def fetch_by_ids(self, memory_ids: list[str]) -> list[dict]:
        self._cur.execute(
            """
            SELECT id, type, subtype, content, confidence, importance,
                   last_accessed_at, created_at, occurred_at, embedding::text
            FROM memories
            WHERE id = ANY(%s::uuid[])
              AND tenant_id = %s AND agent_id = %s
              AND status = 'active';
            """,
            (memory_ids, self._tenant, self._agent),
        )
        return [self._normaliser(ligne) for ligne in self._cur.fetchall()]

    def mark_accessed(self, memory_ids: list[str]) -> None:
        self._cur.execute(
            """
            UPDATE memories
            SET access_count = access_count + 1, last_accessed_at = CURRENT_TIMESTAMP
            WHERE id = ANY(%s::uuid[]);
            """,
            (memory_ids,),
        )


def retrieval_config() -> RetrievalConfig:
    """Assemble la configuration du moteur depuis les constantes de module.

    Lue à chaque requête (et non figée à l'import) pour que les tests et le harness de
    benchmark puissent faire varier une phase par simple `monkeypatch` du module.
    """
    return RetrievalConfig(
        hybrid=RETRIEVAL_HYBRID,
        candidates=RETRIEVAL_CANDIDATES,
        rrf_k=RRF_K,
        weight_vector=RRF_WEIGHT_VECTOR,
        weight_fts=RRF_WEIGHT_FTS,
        entangle_damping=QEM_ENTANGLE_DAMPING,
        entangle_max_hops=QEM_ENTANGLE_MAX_HOPS,
        redundancy_threshold=QEM_REDUNDANCY_THRESHOLD,
        recency_halflife_days=QEM_RECENCY_HALFLIFE_DAYS,
    )


# Modèles Pydantic
MemoryType = Literal["semantic", "episodic", "procedural", "working"]


class EventInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    session_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "sess_abc"})
    content: str = Field(..., min_length=1, max_length=12000, json_schema_extra={"example": "L'utilisateur demande à rédiger un email pro."})
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=100)
    # Clé de déduplication optionnelle : deux appels avec la même clé (même tenant)
    # ne créent qu'un seul événement.
    idempotency_key: str | None = Field(default=None, max_length=128, json_schema_extra={"example": "evt-2026-07-15-001"})

class ContextConstraints(BaseModel):
    max_tokens: int = Field(default=1200, ge=1, le=8000)
    # Familles cognitives. Le plafond de 4 n'est pas arbitraire : c'est le nombre TOTAL de
    # familles, et elles restent fermées (chacune porte un comportement du moteur).
    memory_types: list[MemoryType] = Field(default=["semantic", "episodic", "procedural", "working"], min_length=1, max_length=4)
    # Filtrage FIN par collection (`memories.subtype`). C'est ici que la granularité s'ouvre :
    # un agent qui a déclaré `clients_paca` peut viser ce seul rayon au lieu de ratisser tout
    # `semantic`. Moins de candidats en entrée de Q-EM, donc moins de bruit à budget égal.
    # None = toutes les collections des familles retenues (comportement historique).
    collections: list[str] | None = Field(default=None, max_length=32,
                                          json_schema_extra={"example": ["clients_paca"]})

class ContextRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    session_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "sess_abc"})
    task: str = Field(..., min_length=1, max_length=4000, json_schema_extra={"example": "Rédiger un email de suivi"})
    query: str = Field(..., min_length=1, max_length=8000, json_schema_extra={"example": "Style d'écriture concis de Jimmy"})
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)
    explain: bool = False

# Âge maximal toléré du plus vieil événement non publié avant de déclarer l'ingestion en
# panne. 300 s : très au-delà du cycle du relais (OUTBOX_POLL_SECONDS=0.5), donc aucun faux
# positif sous charge, mais un relais mort est signalé en cinq minutes.
HEALTH_OUTBOX_MAX_AGE_S = float(os.getenv("HEALTH_OUTBOX_MAX_AGE_S", "300"))


def _etat_ingestion() -> str:
    """`healthy`, `stalled` (relais mort) ou `unknown` (base injoignable).

    ## Pourquoi cette sonde vit dans /health et pas seulement dans /metrics

    Le pipeline d'ingestion peut être ENTIÈREMENT mort sans qu'aucun appelant ne s'en
    aperçoive : `/events` écrit dans l'outbox et répond `201 captured`. C'est le relais
    (`apps/relay/relay.py`) qui publie ensuite vers Redis. Sans lui, l'événement est accepté,
    persisté, et jamais consolidé — l'appelant reçoit un succès pour une donnée qui n'arrivera
    jamais en mémoire.

    Constaté sur l'instance de production le 30/07 : `synaptiq-relay` était éteint depuis deux
    jours. Les jauges qui détectent exactement ça existaient déjà (`synaptiq_outbox_pending`,
    `synaptiq_outbox_oldest_age_seconds`), mais rien ne scrutait `/metrics` : une supervision
    non branchée est une supervision absente. `/health` est le seul endpoint que quelqu'un
    regarde vraiment — et le seul que les orchestrateurs interrogent.

    C'est l'ÂGE et non le nombre qui est la bonne mesure : un outbox à 500 entrées qui se vide
    est sain, un outbox à 1 entrée immobile depuis une heure ne l'est pas.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - min(created_at))), 0) "
                    "FROM event_outbox WHERE published_at IS NULL"
                )
                age = float(cur.fetchone()[0] or 0)
            conn.rollback()  # lecture seule : ne pas laisser de transaction ouverte
    except Exception:
        # Postgres est déjà signalé `unhealthy` par ailleurs : ne pas le rapporter deux fois
        # sous un nom qui ferait chercher au mauvais endroit.
        return "unknown"
    return "healthy" if age < HEALTH_OUTBOX_MAX_AGE_S else "stalled"


@v1_router.get("/health")
def health_check():
    db_status = "healthy"
    redis_status = "healthy"

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        db_status = "unhealthy"

    try:
        r = get_redis_client()
        r.ping()
    except Exception:
        redis_status = "unhealthy"

    ingestion = _etat_ingestion() if db_status == "healthy" else "unknown"

    return {
        "status": "ok" if (db_status == "healthy" and redis_status == "healthy"
                           and ingestion == "healthy") else "degraded",
        "services": {
            "postgres": db_status,
            "redis": redis_status,
            # `stalled` = des événements acceptés attendent un relais qui ne vient pas.
            "ingestion": ingestion,
        }
    }


def _refresh_pipeline_gauges() -> None:
    """Réévalue les jauges de santé au moment du scrape.

    Chaque source est isolée dans son propre try : une panne Redis ne doit pas priver
    l'exploitant de la métrique Postgres, et réciproquement. Une source injoignable laisse
    sa jauge à sa dernière valeur plutôt que de faire échouer `/metrics` — un endpoint de
    métriques en erreur, c'est la supervision qui s'éteint pendant l'incident.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*), "
                    "COALESCE(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - min(created_at))), 0) "
                    "FROM event_outbox WHERE published_at IS NULL"
                )
                en_attente, age = cur.fetchone()
            conn.rollback()  # lecture seule : ne pas laisser de transaction ouverte
        OUTBOX_PENDING.set(en_attente or 0)
        OUTBOX_OLDEST_AGE.set(float(age or 0))
    except Exception:
        logger.warning("Jauges outbox non rafraîchies (PostgreSQL injoignable).", exc_info=True)

    try:
        DLQ_DEPTH.set(get_redis_client().xlen(os.getenv("EVENT_DLQ", "synaptiq:events:dlq")))
    except Exception:
        logger.warning("Jauge DLQ non rafraîchie (Redis injoignable).", exc_info=True)

    # Densité du graphe d'intrication, PAR AGENT. C'est la métrique qui manquait : un graphe
    # vide ne lève aucune erreur et dégrade le rappel en silence (la phase 2 de Q-EM tourne
    # simplement à vide). Un agent resté à 0 alors qu'il compte des centaines de souvenirs
    # signale soit une instance antérieure au 01/08, soit un `QEM_ENTANGLE_THRESHOLD` trop
    # haut pour sa langue — voir `scripts/rebuild_entanglement.py`.
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.agent_id,
                           count(DISTINCT s.id) AS souvenirs,
                           count(r.source_memory_id) AS aretes
                    FROM memories s
                    LEFT JOIN relationships r
                           ON r.source_memory_id = s.id AND r.relation_type = 'entangled_with'
                    WHERE s.tenant_id = %s AND s.status = 'active'
                    GROUP BY 1
                """, (_instance_tenant(),))
                lignes = cur.fetchall()
            conn.rollback()  # lecture seule
        for agent_id, souvenirs, aretes in lignes:
            GRAPH_EDGES.labels(agent_id).set(aretes or 0)
            GRAPH_EDGES_PER_MEMORY.labels(agent_id).set(
                (aretes or 0) / souvenirs if souvenirs else 0.0)
    except Exception:
        logger.warning("Jauges du graphe non rafraîchies (PostgreSQL injoignable).",
                       exc_info=True)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics. The reference Compose profile binds this endpoint to localhost."""
    _refresh_pipeline_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@v1_router.post("/events", status_code=201)
def capture_event(event: EventInput, auth: AuthContext | None = Depends(get_auth)):
    """
    Enregistre un événement brut et le publie dans le stream Redis (traitement asynchrone).
    Idempotent si `idempotency_key` est fourni.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "write")
    resolve_agent(auth, event.agent_id)
    try:
        with get_conn() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO events (tenant_id, agent_id, session_id, content, metadata, idempotency_key)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
                        RETURNING id, created_at;
                        """,
                        (tenant, event.agent_id, event.session_id,
                         event.content, json.dumps(event.metadata), event.idempotency_key),
                    )
                    result = cur.fetchone()
                    if result is None:
                        cur.execute(
                            "SELECT id, created_at FROM events WHERE tenant_id = %s AND idempotency_key = %s",
                            (tenant, event.idempotency_key),
                        )
                        result = cur.fetchone()
                        conn.commit()
                        return {"status": "duplicate", "event_id": str(result["id"]),
                                "created_at": result["created_at"].isoformat()}

                    event_id = str(result['id'])
                    created_at = result['created_at'].isoformat()
                    payload = {
                        "id": event_id, "tenant_id": tenant, "agent_id": event.agent_id,
                        "session_id": event.session_id, "content": event.content,
                        "metadata": json.dumps(event.metadata), "created_at": created_at,
                    }
                    cur.execute(
                        "INSERT INTO event_outbox (event_id, payload) VALUES (%s, %s) "
                        "ON CONFLICT (event_id) DO NOTHING",
                        (event_id, json.dumps(payload)),
                    )
                    conn.commit()
            except Exception:
                conn.rollback()
                raise

        logger.info("Événement %s capturé dans l'outbox.", event_id)
        EVENTS_CAPTURED.inc()
        return {"status": "captured", "event_id": event_id, "created_at": created_at}

    except HTTPException:
        # Sans cette clause, le `except Exception` ci-dessous ravalait le 503 levé par
        # get_conn() (base indisponible) et le renvoyait en 500 : le client ne pouvait plus
        # distinguer « réessaie plus tard » de « bug serveur ».
        raise
    except Exception:
        logger.error("Erreur lors de la capture de l'événement.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None

@v1_router.post("/context/build")
def build_context(request: ContextRequest, auth: AuthContext | None = Depends(get_auth)):
    """
    Assemble un paquet de contexte compact pour le LLM en fonction de la tache.
    Implemente le moteur Q-EM (Quantum Entanglement Memory) :
    1. Superposition : recherche hybride (vectoriel pgvector + plein texte, fusion RRF).
    2. Intrication   : propagation d'activation via les liaisons 'entangled_with'.
    3. Interference  : filtrage destructif des contradictions et des redondances.
    4. Mesure        : collapse glouton par densite d'utilite sous budget de tokens.

    L'orchestration des 4 phases vit dans `synaptiq_core.context_builder` : ce handler ne
    fait plus que resoudre le perimetre, ouvrir une transaction et fournir un magasin.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "read")
    resolve_agent(auth, request.agent_id)
    start_time = time.perf_counter()
    # Identifiant de correlation unique par requete. Il etait derive d'un horodatage a la
    # seconde : deux requetes concurrentes recevaient donc le meme, ce qui le rendait
    # inutilisable pour correler quoi que ce soit.
    trace_id = f"trace_{uuid.uuid4().hex}"
    # Rend le trace_id visible dans TOUS les logs emis pendant cette requete, y compris
    # ceux de synaptiq_core. Il est aussi retourne au client, qui peut donc citer un
    # identifiant retrouvable dans les journaux du serveur.
    set_trace_id(trace_id)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialise")
    conn = db_pool.getconn()
    try:
        query_vector = get_embedder().embed_one(request.query)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Le registre décide des SECTIONS du paquet : une collection déclarée par cet
            # agent y apparaît, même vide. Chargé ici, avant l'assemblage, pour que la
            # réponse ait la même forme qu'il y ait des souvenirs ou non.
            registre = charger_registre(cur, tenant, request.agent_id)
            resultat = build_context_packet(
                store=PostgresMemoryStore(cur, tenant, request.agent_id,
                                          request.constraints.collections),
                query_vector=query_vector,
                query_text=request.query,
                memory_types=request.constraints.memory_types,
                max_tokens=request.constraints.max_tokens,
                config=retrieval_config(),
                trace_id=trace_id,
                explain=request.explain,
                registry=registre,
            )
            # `mark_accessed` a ecrit dans la transaction : la valider.
            conn.commit()

        CONTEXT_BUILDS.labels("success" if resultat["selected_memory_ids"] else "empty").inc()
        return resultat

    except HTTPException:
        conn.rollback()
        CONTEXT_BUILDS.labels("error").inc()
        raise
    except Exception:
        conn.rollback()
        CONTEXT_BUILDS.labels("error").inc()
        logger.error("Echec de la construction du contexte (trace=%s).", trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        CONTEXT_BUILD_SECONDS.observe(time.perf_counter() - start_time)
        db_pool.putconn(conn)


class MemoryInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    type: MemoryType = Field(..., json_schema_extra={"example": "semantic"})
    subtype: str | None = Field(None, max_length=50, json_schema_extra={"example": "preference"})
    content: str = Field(..., min_length=1, max_length=12000, json_schema_extra={"example": "Jimmy préfère les e-mails courts."})
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # Idempotence EXPLICITE, en complément de la déduplication par contenu (voir
    # `create_memory`). Réservée aux appelants qui possèdent une clé réellement stable —
    # l'identifiant de la ligne source d'un import, par exemple. Un agent conversationnel
    # n'en a pas : c'est pourquoi elle ne peut pas être le mécanisme principal, une clé
    # régénérée à chaque tentative ne protégeant de rien.
    idempotency_key: str | None = Field(default=None, max_length=128,
                                        json_schema_extra={"example": "crm-row-4711"})

    @model_validator(mode="after")
    def _verifier_la_taxonomie(self):
        """Refuse un sous-type canonique rattaché au mauvais type.

        Jusqu'au 29/07, la taxonomie n'était appliquée que sur le chemin d'extraction du
        worker : cet endpoint acceptait n'importe quel sous-type. Un sous-type LIBRE reste
        accepté (le routage retombe sur la collection du type, et un libellé métier précis
        est légitime), mais `type='semantic'` avec `subtype='coding_best_practices'` est une
        erreur démontrable de l'appelant : la mémoire irait dans `facts` alors qu'elle visait
        `best_practices`. Voir `synaptiq_core.taxonomy`.
        """
        try:
            validate_subtype(self.type, self.subtype)
        except SubtypeMismatch as e:
            raise ValueError(str(e)) from e
        return self

def _memoire_existante(cur, tenant: str, agent_id: str, empreinte: str,
                       cle_idempotence: str | None):
    """Cherche une mémoire ACTIVE déjà écrite en direct pour ce contenu (ou cette clé).

    Bornée à `status = 'active'` et à `source_event_id IS NULL`, exactement comme les index
    uniques qu'elle double :

      - `active` : archiver un fait puis le ré-affirmer plus tard est un cas LÉGITIME (une
        décision revient, une préférence redevient vraie). Contraindre sur toutes les lignes
        interdirait ce mouvement et rendrait un archivage définitif.
      - `source_event_id IS NULL` : les mémoires issues du worker ont leur propre
        déduplication, par événement. Deux événements distincts qui énoncent le même fait
        sont deux souvenirs, et ce n'est pas à cet endpoint d'en juger.
    """
    if cle_idempotence:
        cur.execute(
            "SELECT id FROM memories "
            "WHERE tenant_id = %s AND agent_id = %s AND idempotency_key = %s "
            "AND status = 'active' AND source_event_id IS NULL LIMIT 1",
            (tenant, agent_id, cle_idempotence),
        )
        ligne = cur.fetchone()
        if ligne is not None:
            return ligne[0]
    cur.execute(
        "SELECT id FROM memories "
        "WHERE tenant_id = %s AND agent_id = %s AND content_hash = %s "
        "AND status = 'active' AND source_event_id IS NULL LIMIT 1",
        (tenant, agent_id, empreinte),
    )
    ligne = cur.fetchone()
    return ligne[0] if ligne is not None else None


def _reponse_memoire(memory: "MemoryInput", memory_id: str, statut: str, registre):
    """Corps de réponse de `POST /v1/memories`, identique quel que soit le statut.

    `duplicate` rend la MÊME forme que `created`, avec l'identifiant de la ligne déjà en
    base : un appelant qui relance après timeout obtient un identifiant exploitable, et la
    seule différence visible est `status`. Deux formes de réponse pour un même endpoint
    obligeraient chaque client à traiter le cas dégradé séparément — et donc à l'oublier.

    Le registre est PASSÉ et non rechargé : le chemin de création en a déjà besoin pour
    décider de l'intrication, et le relire ici doublerait la lecture sur le chemin chaud.
    """
    return {
        "status": statut,
        "memory_id": memory_id,
        # Collection du context_packet où ce souvenir sera servi. Rendue explicite
        # parce qu'un sous-type libre retombe sur la collection du TYPE : sans cette
        # information, l'appelant ne peut pas savoir que son libellé métier n'a pas
        # produit le routage fin qu'il imaginait.
        # Routage résolu par le REGISTRE de cet agent, et non plus par la cascade
        # de `if` : une collection qu'il a déclarée lui-même est donc honorée ici.
        "collection": route_memory(memory.type, memory.subtype, registre),
        "canonical_subtype": is_canonical(memory.type, memory.subtype),
    }


@v1_router.post("/memories", status_code=201)
def create_memory(memory: MemoryInput, auth: AuthContext | None = Depends(get_auth)):
    """Enregistre directement un souvenir consolidé, sans passer par l'extraction.

    **Idempotent sur le contenu.** Jusqu'au 01/08 c'était un `INSERT` nu : un client qui
    relançait l'appel après un timeout perçu — alors que le premier avait abouti côté
    serveur — créait une SECONDE ligne, sans erreur ni trace. Le coût n'était pas surtout
    dans le rappel (la phase 3 de Q-EM annule les redondances au-dessus de 0,75, et deux
    copies ont un cosinus de 1,0) mais dans le GRAPHE : un clone est fatalement le premier
    des 3 voisins retenus par l'intrication, et cette arête-là ne porte aucune information.
    Mesuré sur le corpus de benchmark : 67 arêtes sur 1420 reliaient une mémoire à un clone
    d'elle-même. Rien ne l'annule, le graphe étant persistant.

    Deux tentatives identiques renvoient donc désormais le même `memory_id`, la seconde avec
    `status: "duplicate"`. Le code HTTP reste `201` : `/v1/events` a établi cette convention
    (no-op signalé par le champ `status`), et en ouvrir une seconde dans la même API coûterait
    plus cher en confusion que la rigueur HTTP n'y gagnerait.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "write")
    resolve_agent(auth, memory.agent_id)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    empreinte = content_hash(memory.content)
    try:
        with conn.cursor() as cur:
            # ── Pré-contrôle de doublon, AVANT l'embedding ────────────────────────────────
            # Placé ici et pas après, pour deux raisons. D'abord la panne visée est un
            # TIMEOUT côté client, et le temps est presque toujours passé dans l'embedding
            # (mesuré au-delà de 5 s à froid) : la relance doit donc court-circuiter
            # justement l'appel qui a expiré, sinon elle expire à son tour. Ensuite
            # `handle_contradictions` archive : sur une relance, rien ne doit être archivé
            # une seconde fois.
            existant = _memoire_existante(cur, tenant, memory.agent_id, empreinte,
                                          memory.idempotency_key)
            if existant is not None:
                # Réponse construite AVANT le rollback : `_reponse_memoire` lit le registre
                # de collections, donc il lui faut la transaction encore ouverte.
                reponse = _reponse_memoire(
                    memory, str(existant), "duplicate",
                    charger_registre(cur, tenant, memory.agent_id))
                conn.rollback()      # lecture seule : ne rien laisser en transaction
                MEMORY_WRITES.labels("duplicate").inc()
                logger.info("Écriture directe déjà présente : relance traitée en no-op.",
                            extra={"agent_id": memory.agent_id, "memory_id": str(existant)})
                return reponse

            embedding = get_embedder().embed_one(memory.content)

            # Gestion des contradictions
            new_mem_dict = {
                "type": memory.type,
                "subtype": memory.subtype,
                "content": memory.content
            }
            # Archivage sur verdict EXPLICITE de contradiction seulement (cf. governance).
            superseded = handle_contradictions(cur, tenant, memory.agent_id, new_mem_dict, embedding)

            # Insertion. `ON CONFLICT DO NOTHING` SANS cible : deux index uniques partiels
            # couvrent cette table (contenu et clé d'idempotence) et une clause `ON CONFLICT`
            # ne peut en nommer qu'un. Sans cible, PostgreSQL neutralise l'insertion sur
            # n'importe quelle violation d'unicité — donc sur les deux.
            # Ce filet ne remplace pas le pré-contrôle ci-dessus : il rattrape la course
            # entre deux requêtes concurrentes portant le même contenu, fenêtre que le
            # SELECT seul laisse ouverte.
            query = """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding,
                                      confidence, importance, status, content_hash, idempotency_key)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id;
            """
            cur.execute(query, (
                tenant,
                memory.agent_id,
                memory.type,
                memory.subtype,
                memory.content,
                embedding,
                memory.confidence,
                memory.importance,
                empreinte,
                memory.idempotency_key,
            ))
            ligne = cur.fetchone()
            if ligne is None:
                # Course perdue : une requête concurrente a inséré le même contenu entre
                # notre SELECT et notre INSERT. Le doublon est refusé, on rend la ligne
                # gagnante — l'appelant obtient un identifiant utilisable, pas une erreur.
                # `handle_contradictions` a pu archiver juste avant : on COMMIT, parce que
                # ce verdict-là est valide indépendamment de qui a gagné la course.
                gagnante = _memoire_existante(cur, tenant, memory.agent_id, empreinte,
                                              memory.idempotency_key)
                reponse = _reponse_memoire(
                    memory, str(gagnante) if gagnante else "", "duplicate",
                    charger_registre(cur, tenant, memory.agent_id))
                conn.commit()
                MEMORY_WRITES.labels("duplicate").inc()
                logger.info("Course d'insertion perdue sur un contenu identique : no-op.",
                            extra={"agent_id": memory.agent_id,
                                   "memory_id": str(gagnante) if gagnante else None})
                return reponse

            new_id = ligne[0]
            # Traçabilité : relier la nouvelle mémoire aux préférences qu'elle remplace.
            if superseded:
                link_supersedes(cur, new_id, superseded)

            # ── Graphe d'intrication ──────────────────────────────────────────────────────
            # Tissé ICI depuis le 01/08. Auparavant `_entangle` n'existait que dans le worker,
            # donc une écriture directe ne construisait AUCUNE arête : la phase 2 de Q-EM
            # (propagation d'activation) tournait sur un graphe vide pour tout agent écrivant
            # par `store_memory`, sans erreur ni log. Mesuré sur une instance réelle : 28
            # souvenirs, 0 arête, après des semaines d'usage.
            # Dans la même transaction que l'insertion : une arête sans son souvenir n'a pas
            # de sens, et l'inverse non plus.
            registre = charger_registre(cur, tenant, memory.agent_id)
            if registre.entangle_pour(memory.type, memory.subtype):
                entangle(cur, tenant, memory.agent_id, new_id, memory.subtype, embedding)

            # Réponse assemblée dans la MÊME transaction que l'insertion : `charger_registre`
            # lit en base, et le faire après le commit ouvrirait une transaction de plus.
            reponse = _reponse_memoire(memory, str(new_id), "created", registre)
            conn.commit()
            MEMORY_WRITES.labels("created").inc()

            logger.info("Mémoire créée en direct par l'agent : %s", new_id,
                        extra={"agent_id": memory.agent_id, "memory_type": memory.type,
                               "subtype": memory.subtype})
            return reponse
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur lors de la création de la mémoire.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        db_pool.putconn(conn)

class RetrieveRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    query: str = Field(..., min_length=1, max_length=8000)
    limit: int = Field(default=5, ge=1, le=100)
    memory_type: MemoryType | None = None
    # Symétrique de `ContextConstraints.collections` : viser un rayon plutôt qu'une famille.
    collections: list[str] | None = Field(default=None, max_length=32)

@v1_router.post("/retrieve")
def retrieve_memories(request: RetrieveRequest, auth: AuthContext | None = Depends(get_auth)):
    """
    Recherche HYBRIDE : similarité vectorielle (pgvector) fusionnée par RRF avec une
    recherche plein texte. Le vecteur ramène le sémantiquement proche, le plein texte les
    correspondances littérales (noms propres, dates, identifiants) qu'il manque.
    `RETRIEVAL_HYBRID=false` revient au classement purement vectoriel.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "read")
    resolve_agent(auth, request.agent_id)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        query_vector = get_embedder().embed_one(request.query)
        vector_str = to_pgvector(query_vector)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Préfixé `m.` : le chemin hybride aliase `memories` en m dans les deux branches.
            # `type_filter` est un fragment CHOISI par le serveur (deux valeurs
            # possibles), jamais une chaine fournie par l'appelant : le type de memoire
            # est valide en amont par Pydantic (Literal, cf. exemption S608 dans ruff.toml).
            # Le fragment et `filtre_params` sont construits ENSEMBLE et dans le même ordre :
            # les trois requêtes ci-dessous sont positionnelles, et un décalage y serait muet
            # (les types concordent) — il fausserait le filtre au lieu de lever.
            type_filter = "AND m.type = %s" if request.memory_type else ""
            filtre_params: list = [tenant, request.agent_id]
            if request.memory_type:
                filtre_params.append(request.memory_type)
            if request.collections:
                type_filter += " AND m.subtype = ANY(%s)"
                filtre_params.append(request.collections)

            if not RETRIEVAL_HYBRID:
                cur.execute(f"""
                    SELECT m.id, m.type, m.subtype, m.content, m.confidence, m.importance,
                           m.last_accessed_at, m.occurred_at,
                           (1 - (m.embedding <=> %s::vector)) AS similarity
                    FROM memories m
                    WHERE m.tenant_id = %s AND m.agent_id = %s {type_filter} AND m.status = 'active'
                    -- Trier sur l'OPÉRATEUR de distance, pas sur l'alias `similarity` :
                    -- pgvector n'utilise l'index HNSW que sur `ORDER BY embedding <=> x` ASC.
                    ORDER BY m.embedding <=> %s::vector
                    LIMIT %s;
                """, (vector_str, *filtre_params, vector_str, request.limit))
                return {"memories": cur.fetchall()}

            # Sur-échantillonner chaque chemin avant fusion : la bonne réponse peut être
            # 12e en vectoriel et 2e en plein texte, il faut donc regarder au-delà de
            # `limit` dans les deux avant de trancher.
            pool_size = max(request.limit * 4, RETRIEVAL_CANDIDATES)
            # Même forme que `_fetch_candidates` : chaque chemin attaque `memories`
            # directement, filtre répété. Une CTE commune référencée plusieurs fois serait
            # matérialisée par PostgreSQL et neutraliserait l'index HNSW (cf. la note de
            # `_fetch_candidates` et `scripts/explain_retrieval.py`).
            cur.execute(f"""
                WITH vectoriel AS (
                    SELECT id, row_number() OVER (ORDER BY distance) AS rank_vec
                    FROM (
                        SELECT m.id, m.embedding <=> %s::vector AS distance
                        FROM memories m
                        WHERE m.tenant_id = %s AND m.agent_id = %s {type_filter}
                          AND m.status = 'active'
                        ORDER BY m.embedding <=> %s::vector
                        LIMIT %s
                    ) v
                ),
                plein_texte AS (
                    SELECT id, row_number() OVER (ORDER BY score DESC) AS rank_fts
                    FROM (
                        SELECT m.id, ts_rank(m.content_tsv, q.query) AS score
                        FROM memories m, websearch_to_tsquery('simple', %s) AS q(query)
                        WHERE m.tenant_id = %s AND m.agent_id = %s {type_filter}
                          AND m.status = 'active' AND m.content_tsv @@ q.query
                        ORDER BY ts_rank(m.content_tsv, q.query) DESC
                        LIMIT %s
                    ) t
                ),
                retenus AS (
                    SELECT id, min(rank_vec) AS rank_vec, min(rank_fts) AS rank_fts
                    FROM (
                        SELECT id, rank_vec, NULL::bigint AS rank_fts FROM vectoriel
                        UNION ALL
                        SELECT id, NULL::bigint AS rank_vec, rank_fts FROM plein_texte
                    ) u
                    GROUP BY id
                )
                SELECT m.id, m.type, m.subtype, m.content, m.confidence, m.importance,
                       m.last_accessed_at, m.occurred_at,
                       (1 - (m.embedding <=> %s::vector)) AS similarity,
                       r.rank_vec, r.rank_fts
                FROM retenus r JOIN memories m ON m.id = r.id;
            """, (vector_str, *filtre_params, vector_str, pool_size,
                  request.query, *filtre_params, pool_size,
                  vector_str))
            rows = cur.fetchall()

            par_id = {str(r["id"]): r for r in rows}
            rang_vec = [str(r["id"]) for r in sorted(
                (r for r in rows if r["rank_vec"] is not None), key=lambda r: r["rank_vec"])]
            rang_fts = [str(r["id"]) for r in sorted(
                (r for r in rows if r["rank_fts"] is not None), key=lambda r: r["rank_fts"])]
            ordre = fuse_and_rank([rang_vec, rang_fts], k=RRF_K,
                                  weights=[RRF_WEIGHT_VECTOR, RRF_WEIGHT_FTS],
                                  limit=request.limit)
            # `rank_vec`/`rank_fts` sont des détails d'implémentation : hors contrat public.
            memoires = []
            for mem_id in ordre:
                row = dict(par_id[mem_id])
                row.pop("rank_vec", None)
                row.pop("rank_fts", None)
                memoires.append(row)
            return {"memories": memoires}
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur de recherche de souvenirs.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        db_pool.putconn(conn)


# Plafond du nombre de collections qu'un agent peut se créer. Ce n'est pas une limite
# technique : c'est un garde-fou de LISIBILITÉ. Un LLM à qui l'on donne le droit de créer
# une catégorie en crée une à chaque nouveauté ; quarante rayons produisent un paquet de
# contexte éclaté en quarante sections dont trente-cinq vides — plus dur à exploiter pour le
# modèle que les sept d'origine. Le contrôle anti-doublon SÉMANTIQUE (comparaison des
# descriptions) et la fusion viendront compléter ce plafond ; il est volontairement en place
# dès l'ouverture de la création, pour qu'aucune fenêtre ne laisse la taxonomie s'emballer.
MAX_COLLECTIONS_PER_AGENT = int(os.getenv("MAX_COLLECTIONS_PER_AGENT", "50"))

# Cosinus au-delà duquel deux descriptions de collection décrivent la même chose. 0,85 est
# volontairement plus haut que le seuil de redondance des souvenirs (0,75) : refuser à tort
# une collection légitime est plus coûteux que d'en laisser passer une proche — l'agent peut
# fusionner après coup, mais un refus le laisse sans rayon où ranger.
COLLECTION_DUP_THRESHOLD = float(os.getenv("COLLECTION_DUP_THRESHOLD", "0.85"))

# Jours sans écriture au-delà desquels une collection vide est signalée comme dormante.
# Une collection créée puis jamais utilisée est le premier symptôme d'une taxonomie qui
# part en morceaux ; la signaler est ce qui permet de la fusionner ou de la supprimer.
COLLECTION_STALE_DAYS = int(os.getenv("COLLECTION_STALE_DAYS", "14"))


def _cosinus(a, b) -> float:
    """Cosinus entre deux vecteurs. 0.0 si l'un est nul (jamais une division par zéro)."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    normes = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / normes) if normes else 0.0


def _collection_trop_proche(cur, tenant: str, agent_id: str, description: str,
                            vecteur) -> tuple[str, float] | None:
    """Une collection existante décrit-elle déjà la même chose ?

    ## Le moteur retourné contre sa propre dérive

    Un LLM à qui l'on donne le droit de créer une catégorie en crée une à chaque nuance :
    `clients_paca`, puis `clients_region_paca`, puis `prospects_paca`. Aucune n'est fausse,
    et le résultat est une mémoire éparpillée où plus rien ne se retrouve. Un contrôle sur
    le NOM n'y peut rien — ces trois-là sont trois chaînes distinctes.

    La description, elle, est du texte : on la compare avec l'outil que le produit sait
    déjà faire, la similarité vectorielle. C'est le même mécanisme que le filtre de
    redondance des souvenirs, appliqué à la taxonomie.

    Les descriptions sans vecteur en base (collections système, et celles reprises par la
    migration) sont embarquées À LA VOLÉE, en un seul appel groupé. Sans cela, la protection
    serait inopérante précisément contre les doublons des sept rayons livrés — le cas le
    plus probable pour un agent qui débute.
    """
    cur.execute(
        "SELECT name, description, description_embedding::text FROM memory_collections "
        "WHERE created_by = 'system' OR (tenant_id = %s AND agent_id = %s)",
        (tenant, agent_id),
    )
    lignes = [dict(li) if isinstance(li, dict) else
              {"name": li[0], "description": li[1], "description_embedding": li[2]}
              for li in cur.fetchall()]

    connus: list[tuple[str, Any]] = []
    a_embarquer: list[tuple[str, str]] = []
    for li in lignes:
        if not (li["description"] or "").strip():
            continue                      # rien à comparer
        if li["description_embedding"]:
            connus.append((li["name"], parse_embedding(li["description_embedding"])))
        else:
            a_embarquer.append((li["name"], li["description"]))

    if a_embarquer:
        # Un seul appel groupé : l'interface Embedder est batch, et la création de
        # collection est une opération rare.
        vecteurs = get_embedder().embed([d for _, d in a_embarquer])
        connus.extend((nom, vec) for (nom, _), vec in zip(a_embarquer, vecteurs, strict=False))

    meilleur: tuple[str, float] | None = None
    for nom, autre in connus:
        score = _cosinus(vecteur, autre)
        if score >= COLLECTION_DUP_THRESHOLD and (meilleur is None or score > meilleur[1]):
            meilleur = (nom, score)
    return meilleur


class CollectionInput(BaseModel):
    """Déclaration d'un nouveau rayon par l'agent lui-même."""

    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    # Devient la valeur de `memories.subtype`. Snake_case minuscule : ce nom voyage jusque
    # dans les clés du context_packet, il doit rester lisible et stable.
    name: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$",
                      json_schema_extra={"example": "clients_paca"})
    family: Literal["semantic", "episodic", "procedural", "working"] = Field(
        ..., json_schema_extra={"example": "semantic"})
    # OBLIGATOIRE, et ce n'est pas de la bureaucratie : elle sera vectorisée pour refuser
    # une collection qui en double une autre, et c'est aussi ce que l'agent relira dans
    # `list_collections` pour décider s'il doit créer ou réutiliser.
    description: str = Field(..., min_length=10, max_length=500)
    entangle: bool = Field(
        default=True,
        description="Cette collection tisse-t-elle des liens 'entangled_with' ? "
                    "Vrai pour un savoir structurant, faux pour du volumineux peu "
                    "discriminant (brouillons, journaux bruts).")
    # Plusieurs collections peuvent alimenter une même section du paquet. Par défaut chacune
    # a la sienne, ce qui est le cas attendu.
    packet_key: str | None = Field(default=None, min_length=2, max_length=64,
                                   pattern=r"^[a-z0-9_]+$")


@v1_router.post("/collections", status_code=201)
def create_collection(payload: CollectionInput, auth: AuthContext | None = Depends(get_auth)):
    """Déclare une collection. C'est l'acte par lequel l'agent structure sa propre mémoire.

    ## Pourquoi la création est EXPLICITE et non implicite

    Écrire dans une collection inexistante ne la crée pas. Un rangement auto-créé à la
    volée serait indiscernable d'une faute de frappe : `clients_paca`, `client_paca` et
    `clientspaca` cohabiteraient, chacun avec sa section dans le paquet, et personne ne
    saurait laquelle fait foi. Un acte délibéré produit une taxonomie qu'on peut relire.

    ## Ce qui est refusé

    - un nom CANONIQUE (`fact`, `preference`, `rule`…). Le registre sait techniquement faire
      primer une collection d'agent sur son homonyme système, mais ouvrir ça ici laisserait
      un agent rerouter en silence tout ce qui est déjà rangé sous ce nom ;
    - un doublon exact pour ce même agent ;
    - le dépassement de `MAX_COLLECTIONS_PER_AGENT`.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "write")
    resolve_agent(auth, payload.agent_id)

    if is_canonical(payload.family, payload.name):
        raise HTTPException(
            status_code=422,
            detail=f"'{payload.name}' est une collection système de la famille "
                   f"'{payload.family}' : elle existe déjà et sert tous les agents. "
                   f"Choisir un autre nom, ou écrire directement dedans.",
        )

    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT count(*) AS n FROM memory_collections "
                "WHERE tenant_id = %s AND agent_id = %s AND created_by = 'agent'",
                (tenant, payload.agent_id),
            )
            existantes = cur.fetchone()["n"]
            if existantes >= MAX_COLLECTIONS_PER_AGENT:
                conn.rollback()
                raise HTTPException(
                    status_code=409,
                    detail=f"Plafond atteint ({MAX_COLLECTIONS_PER_AGENT} collections). "
                           f"Une taxonomie qui grossit sans fin devient illisible : "
                           f"réutiliser une collection existante, ou en fusionner deux.",
                )

            # Anti-doublon SÉMANTIQUE. Le contrôle d'unicité du nom ne protège de rien ici :
            # `clients_paca` et `clients_region_paca` sont deux chaînes distinctes qui
            # désignent le même rayon.
            vecteur = get_embedder().embed_one(payload.description)
            proche = _collection_trop_proche(cur, tenant, payload.agent_id,
                                             payload.description, vecteur)
            if proche is not None:
                conn.rollback()
                nom_proche, score = proche
                raise HTTPException(
                    status_code=409,
                    detail=f"'{nom_proche}' décrit déjà la même chose (similarité "
                           f"{score:.2f}). Y ranger ce souvenir plutôt que de créer un "
                           f"rayon de plus. Si les deux sont réellement distincts, "
                           f"reformuler la description pour dire en quoi.",
                )

            cur.execute(
                """
                INSERT INTO memory_collections
                    (tenant_id, agent_id, name, family, packet_key, entangle, description,
                     description_embedding, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'agent')
                ON CONFLICT DO NOTHING
                RETURNING id;
                """,
                (tenant, payload.agent_id, payload.name, payload.family,
                 payload.packet_key or payload.name, payload.entangle, payload.description,
                 to_pgvector(vecteur)),
            )
            ligne = cur.fetchone()
            if ligne is None:
                conn.rollback()
                raise HTTPException(
                    status_code=409,
                    detail=f"La collection '{payload.name}' existe déjà pour cet agent.",
                )
            audit(cur, tenant, "create_collection", auth, agent_id=payload.agent_id,
                  name=payload.name, family=payload.family, entangle=payload.entangle)
            conn.commit()

        logger.info("Collection '%s' déclarée par l'agent %s (famille %s, intrication %s).",
                    payload.name, payload.agent_id, payload.family, payload.entangle)
        return {
            "status": "created",
            "name": payload.name,
            "family": payload.family,
            "packet_key": payload.packet_key or payload.name,
            "entangle": payload.entangle,
            # Ce qu'il faut passer à `store_memory` pour écrire dedans : sans ce rappel,
            # l'agent doit deviner que « collection » se traduit par (type, subtype).
            "usage": {"type": payload.family, "subtype": payload.name},
        }
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur lors de la création de la collection.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        db_pool.putconn(conn)


class MergeInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    source: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    target: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")


@v1_router.post("/collections/merge")
def merge_collections(payload: MergeInput, auth: AuthContext | None = Depends(get_auth)):
    """Verse les souvenirs de `source` dans `target`, puis supprime `source`.

    ## Pourquoi cette opération est indispensable

    Sans elle, une taxonomie ne peut que grossir. Le plafond et l'anti-doublon ralentissent
    la dérive, ils ne la corrigent pas : dès que deux rayons proches ont été créés, seule
    la fusion permet de revenir en arrière. C'est le ramasse-miettes de la structure.

    ## Deux refus, pour deux raisons distinctes

    - **`source` doit appartenir à l'agent.** Une collection système sert tous les agents :
      la fusionner depuis une requête d'agent la retirerait à tout le monde.
    - **Même famille des deux côtés.** La famille porte un COMPORTEMENT (intrication,
      décroissance, section de repli). Déplacer un souvenir de `episodic` vers `semantic`
      changerait donc la manière dont le moteur le traite, sans que personne ne l'ait
      demandé. Un rangement ne doit pas modifier une sémantique.

    Aucun souvenir n'est détruit : seule l'étiquette change. L'opération est journalisée.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "write")
    resolve_agent(auth, payload.agent_id)

    if payload.source == payload.target:
        raise HTTPException(status_code=422, detail="Source et cible identiques.")
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")

    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT name, family, created_by FROM memory_collections "
                "WHERE name = ANY(%s) AND (created_by = 'system' "
                "   OR (tenant_id = %s AND agent_id = %s))",
                ([payload.source, payload.target], tenant, payload.agent_id),
            )
            par_nom = {li["name"]: li for li in cur.fetchall()}
            source, cible = par_nom.get(payload.source), par_nom.get(payload.target)

            if source is None:
                raise HTTPException(status_code=404,
                                    detail=f"Collection source '{payload.source}' inconnue.")
            if cible is None:
                raise HTTPException(status_code=404,
                                    detail=f"Collection cible '{payload.target}' inconnue.")
            if source["created_by"] == "system":
                raise HTTPException(
                    status_code=422,
                    detail=f"'{payload.source}' est une collection système : elle sert tous "
                           f"les agents et ne peut pas être fusionnée.")
            if source["family"] != cible["family"]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Familles différentes ('{source['family']}' vers "
                           f"'{cible['family']}'). La famille décide de l'intrication et de "
                           f"la décroissance : la changer modifierait le traitement des "
                           f"souvenirs déplacés, pas seulement leur rangement.")

            cur.execute(
                "UPDATE memories SET subtype = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE tenant_id = %s AND agent_id = %s AND type = %s AND subtype = %s",
                (payload.target, tenant, payload.agent_id, source["family"], payload.source),
            )
            deplacees = cur.rowcount
            cur.execute(
                "DELETE FROM memory_collections WHERE tenant_id = %s AND agent_id = %s "
                "AND name = %s AND created_by = 'agent'",
                (tenant, payload.agent_id, payload.source),
            )
            audit(cur, tenant, "merge_collections", auth, agent_id=payload.agent_id,
                  source=payload.source, target=payload.target, moved=deplacees)
            conn.commit()

        logger.info("Fusion de collections : %s -> %s (%d souvenirs déplacés).",
                    payload.source, payload.target, deplacees)
        return {"status": "merged", "source": payload.source, "target": payload.target,
                "moved_memories": deplacees}
    except HTTPException:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur lors de la fusion de collections.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        db_pool.putconn(conn)


@v1_router.get("/collections")
def list_collections(agent_id: str, auth: AuthContext | None = Depends(get_auth)):
    """Registre des collections visibles par un agent : les système et les siennes.

    Un agent ne peut pas structurer ce qu'il ne peut pas consulter. C'est cet endpoint qui
    rend la collection réellement observable — et donc de premier ordre. L'outil MCP
    `list_collections` s'y branchera au lot 3, avec `create_collection`.

    `memory_count` est calculé ici plutôt que maintenu dans une colonne : une collection
    déclarée mais vide est une information utile (l'agent l'a créée puis n'en a rien fait),
    et un compteur dénormalisé dériverait au premier écart d'écriture.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "read")
    resolve_agent(auth, agent_id)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            registre = charger_registre(cur, tenant, agent_id)
            cur.execute(
                "SELECT type AS family, subtype AS name, count(*) AS n "
                "FROM memories WHERE tenant_id = %s AND agent_id = %s AND status = 'active' "
                "GROUP BY 1, 2",
                (tenant, agent_id),
            )
            comptes = {(ligne["family"], ligne["name"]): ligne["n"]
                       for ligne in cur.fetchall()}
            # Âge des collections de l'agent : une collection déclarée puis jamais utilisée
            # est le premier symptôme d'une taxonomie qui se disperse. La signaler est ce
            # qui rend la fusion possible — un défaut qu'on ne voit pas ne se corrige pas.
            cur.execute(
                "SELECT name, family, "
                "EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400 AS jours "
                "FROM memory_collections WHERE tenant_id = %s AND agent_id = %s",
                (tenant, agent_id),
            )
            ages = {(li["family"], li["name"]): float(li["jours"] or 0)
                    for li in cur.fetchall()}
            conn.rollback()  # lecture seule

        collections = []
        for col in sorted(registre.collections,
                          key=lambda c: (c.created_by, c.family, c.name)):
            nombre = comptes.get((col.family, col.name), 0)
            age = ages.get((col.family, col.name))
            collections.append({
                "name": col.name,
                "family": col.family,
                "packet_key": col.packet_key,
                "description": col.description,
                "entangle": col.entangle,
                "created_by": col.created_by,
                "memory_count": nombre,
                # Vide ET ancienne. Une collection créée il y a dix minutes et encore vide
                # est normale : l'agent est en train de la remplir.
                "stale": bool(col.created_by == "agent" and nombre == 0
                              and age is not None and age >= COLLECTION_STALE_DAYS),
            })
        return {
            "agent_id": agent_id,
            "collections": collections,
            "packet_keys": list(registre.packet_keys()),
            "limits": {
                "max_collections": MAX_COLLECTIONS_PER_AGENT,
                "used": sum(1 for c in collections if c["created_by"] == "agent"),
            },
        }
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur de lecture du registre de collections.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        db_pool.putconn(conn)


@v1_router.delete("/memories")
def purge_memories(
    agent_id: str | None = None,
    confirm: str | None = Query(
        default=None,
        description="Doit valoir l'identifiant du tenant de l'instance. Garde-fou anti-appel accidentel.",
    ),
    auth: AuthContext | None = Depends(get_auth),
):
    """
    Purge RGPD : supprime les mémoires (et événements) de l'instance.
    Scopé au tenant résolu côté serveur ; les relationships sont supprimées en cascade (FK).
    Filtre optionnel par `agent_id`.

    Trois garde-fous, ajoutés le 29/07 : cet endpoint pouvait auparavant vider l'intégralité
    d'un tenant avec la même clé qui sert à lire, sans confirmation et sans laisser de trace.
      1. permission `admin` explicite (les clés d'agent ne l'ont pas par défaut) ;
      2. `confirm` doit valoir l'identifiant du tenant — un appel involontaire échoue ;
      3. une ligne d'`audit_log` est écrite dans la MÊME transaction que la suppression.

    La suppression reste physique et immédiate : c'est un endpoint RGPD, un effacement
    différé ou réversible irait à l'encontre de sa raison d'être.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "admin")
    if agent_id:
        resolve_agent(auth, agent_id)
    if confirm != tenant:
        # Le tenant est nommé dans le message : ce n'est pas un secret (l'appelant est déjà
        # authentifié sur ce périmètre), et un message opaque ne ferait que provoquer des
        # tâtonnements sur un endpoint destructeur.
        raise HTTPException(
            status_code=400,
            detail=f"Purge non confirmée. Rappeler avec ?confirm={tenant} pour valider la suppression.",
        )
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            if agent_id:
                cur.execute("DELETE FROM memories WHERE tenant_id = %s AND agent_id = %s", (tenant, agent_id))
                deleted_mem = cur.rowcount
                cur.execute("DELETE FROM events WHERE tenant_id = %s AND agent_id = %s", (tenant, agent_id))
                deleted_evt = cur.rowcount
            else:
                cur.execute("DELETE FROM memories WHERE tenant_id = %s", (tenant,))
                deleted_mem = cur.rowcount
                cur.execute("DELETE FROM events WHERE tenant_id = %s", (tenant,))
                deleted_evt = cur.rowcount
            # Même transaction que la suppression : pas de purge sans trace, pas de trace
            # sans purge.
            audit(cur, tenant, "purge_memories", auth, agent_id=agent_id,
                  deleted_memories=deleted_mem, deleted_events=deleted_evt,
                  scope="agent" if agent_id else "tenant")
            conn.commit()
        logger.info("Purge RGPD tenant=%s agent=%s : %d mémoires, %d événements.",
                    tenant, agent_id, deleted_mem, deleted_evt)
        return {
            "status": "purged",
            "tenant_id": tenant,
            "agent_id": agent_id,
            "deleted_memories": deleted_mem,
            "deleted_events": deleted_evt,
        }
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur lors de la purge RGPD.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        db_pool.putconn(conn)


app.include_router(v1_router, prefix="/v1", tags=["v1"])
app.include_router(v1_router, include_in_schema=False)


