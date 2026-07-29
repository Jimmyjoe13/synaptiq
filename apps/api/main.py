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
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

v1_router = APIRouter()
import redis
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

# Logique partagée (embeddings pluggables + gouvernance), plus d'import depuis le worker
from synaptiq_core import get_embedder, handle_contradictions, link_supersedes, to_pgvector

# Orchestration des 4 phases Q-EM (sans SQL ni HTTP : testable en isolation)
from synaptiq_core.context_builder import RetrievalConfig, build_context_packet

# Journalisation structurée + corrélation par trace_id
from synaptiq_core.observability import configure_logging, set_trace_id

# Fusion de classements pour la recherche hybride (fonctions pures, cf. retrieval.py)
from synaptiq_core.retrieval import DEFAULT_RRF_K, fuse_and_rank

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


app = FastAPI(title="SynaptiQ API", version="0.2.0", lifespan=lifespan)

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

limiter = Limiter(key_func=get_remote_address, default_limits=[os.getenv("RATE_LIMIT", "120/minute")])
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
    """Exige une permission portée par la clé. Sans auth (mode dev), tout est permis.

    Trois permissions : `read` (retrieve, context/build), `write` (events, memories),
    `admin` (purge). Une clé de lecture ne doit pas pouvoir écrire, et surtout aucune clé
    d'agent ne doit pouvoir vider l'instance.
    """
    if auth is None:
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
                      agent_id: str, memory_types: list[str]) -> list[dict]:
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
            WHERE tenant_id = %s AND agent_id = %s AND type = ANY(%s) AND status = 'active'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_str, vector_str, tenant, agent_id, memory_types, vector_str,
              RETRIEVAL_CANDIDATES))
        return cur.fetchall()

    cur.execute(f"""
        WITH vectoriel AS (
            SELECT id, row_number() OVER (ORDER BY distance) AS rank_vec
            FROM (
                SELECT id, embedding <=> %s::vector AS distance
                FROM memories
                WHERE tenant_id = %s AND agent_id = %s AND type = ANY(%s) AND status = 'active'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ) v
        ),
        plein_texte AS (
            SELECT id, row_number() OVER (ORDER BY score DESC) AS rank_fts
            FROM (
                SELECT m.id, ts_rank(m.content_tsv, q.query) AS score
                FROM memories m, websearch_to_tsquery('simple', %s) AS q(query)
                WHERE m.tenant_id = %s AND m.agent_id = %s AND m.type = ANY(%s)
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
    """, (vector_str, tenant, agent_id, memory_types, vector_str, RETRIEVAL_CANDIDATES,
          query_text, tenant, agent_id, memory_types, RETRIEVAL_CANDIDATES,
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

    def __init__(self, cur, tenant_id: str, agent_id: str) -> None:
        self._cur = cur
        self._tenant = tenant_id
        self._agent = agent_id

    @staticmethod
    def _normaliser(ligne) -> dict:
        """Désérialise le vecteur ; le cœur ne connaît pas le format texte de pgvector."""
        valeur = dict(ligne)
        valeur["embedding"] = parse_embedding(valeur.get("embedding"))
        return valeur

    def fetch_candidates(self, query_vector, query_text: str, memory_types: list[str]) -> list[dict]:
        lignes = _fetch_candidates(self._cur, to_pgvector(query_vector), query_text,
                                   self._tenant, self._agent, memory_types)
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
    memory_types: list[MemoryType] = Field(default=["semantic", "episodic", "procedural", "working"], min_length=1, max_length=4)

class ContextRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    session_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "sess_abc"})
    task: str = Field(..., min_length=1, max_length=4000, json_schema_extra={"example": "Rédiger un email de suivi"})
    query: str = Field(..., min_length=1, max_length=8000, json_schema_extra={"example": "Style d'écriture concis de Jimmy"})
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)
    explain: bool = False

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

    return {
        "status": "ok" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        "services": {
            "postgres": db_status,
            "redis": redis_status
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
            resultat = build_context_packet(
                store=PostgresMemoryStore(cur, tenant, request.agent_id),
                query_vector=query_vector,
                query_text=request.query,
                memory_types=request.constraints.memory_types,
                max_tokens=request.constraints.max_tokens,
                config=retrieval_config(),
                trace_id=trace_id,
                explain=request.explain,
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

@v1_router.post("/memories", status_code=201)
def create_memory(memory: MemoryInput, auth: AuthContext | None = Depends(get_auth)):
    """
    Permet à un agent IA d'enregistrer directement un souvenir consolidé.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "write")
    resolve_agent(auth, memory.agent_id)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        embedding = get_embedder().embed_one(memory.content)
        with conn.cursor() as cur:
            # Gestion des contradictions
            new_mem_dict = {
                "type": memory.type,
                "subtype": memory.subtype,
                "content": memory.content
            }
            # Archivage sur verdict EXPLICITE de contradiction seulement (cf. governance).
            superseded = handle_contradictions(cur, tenant, memory.agent_id, new_mem_dict, embedding)

            # Insertion
            query = """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active')
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
                memory.importance
            ))
            new_id = cur.fetchone()[0]
            # Traçabilité : relier la nouvelle mémoire aux préférences qu'elle remplace.
            if superseded:
                link_supersedes(cur, new_id, superseded)
            conn.commit()

            logger.info(f"Mémoire créée en direct par l'agent : {new_id}")
            return {
                "status": "created",
                "memory_id": str(new_id)
            }
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
            type_filter = "AND m.type = %s" if request.memory_type else ""
            filtre_params = [tenant, request.agent_id]
            if request.memory_type:
                filtre_params.append(request.memory_type)

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


