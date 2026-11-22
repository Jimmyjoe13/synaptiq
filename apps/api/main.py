import os
import sys

# Ajouter la racine du projet + packages/core au sys.path (imports monorepo, dev local)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (root_path, os.path.join(root_path, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
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
from pydantic import BaseModel, Field, field_validator, model_validator

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
from synaptiq_core.belief_guard import CroyanceRefusee, valider_croyance

# Compartiments (v0.4) : 9 régions fixes, propriété du moteur. Définis dans le cœur, comme
# le reste de la taxonomie, pour que l'API et le worker rangent selon les MÊMES règles.
from synaptiq_core.compartments import (
    ALIAS,
    COMPARTIMENT_CROYANCES,
    NOMS,
    PAR_NOM,
    Rangement,
    catalogue,
    prototypes,
    resoudre,
)

# Orchestration des 4 phases Q-EM (sans SQL ni HTTP : testable en isolation)
from synaptiq_core.context_builder import RetrievalConfig, build_context_packet

# Journalisation structurée + corrélation par trace_id
from synaptiq_core.observability import configure_logging, set_trace_id

# Taxonomie partagée avec le worker : les DEUX chemins d'écriture appliquent la même règle
from synaptiq_core.project import (
    activer_scan_iteratif,
    filtre_projet,
    mode_scan_iteratif,
    normaliser_projet,
)

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
# Agrégation des apports reçus de plusieurs parents : "max" (défaut, borne l'activation
# d'un nœud très connecté) ou "sum" (historique : les hubs dépassaient les seeds).
QEM_ENTANGLE_AGGREGATION = os.getenv("QEM_ENTANGLE_AGGREGATION", "max")
# Au-delà de ce cosinus entre deux candidats, le moins bien classé est filtré (redondance).
# 0.75 -> 0.90 le 26/09 : avec le modèle multilingue, deux souvenirs DISTINCTS du même
# auteur montent couramment à 0,76-0,86 ; les vrais quasi-doublons sont au-dessus de 0,95.
QEM_REDUNDANCY_THRESHOLD = float(os.getenv("QEM_REDUNDANCY_THRESHOLD", "0.90"))
# Collapse : nombre de meilleurs candidats (par score) admis avant la densité d'utilité,
# et exposant β de la densité score/tokens**β (1 = historique, <1 pénalise moins la longueur).
QEM_COLLAPSE_TOP_K = int(os.getenv("QEM_COLLAPSE_TOP_K", "3"))
QEM_DENSITY_EXPONENT = float(os.getenv("QEM_DENSITY_EXPONENT", "0.5"))
# Part maximale du budget de tokens accordée aux croyances (famille `reflective`, lot C).
# Parts max du budget par compartiment plafonné. `QEM_REFLECTIVE_MAX_SHARE` reste lu en
# repli : c'était le nom du réglage des croyances avant la v0.4.
QEM_CROYANCES_MAX_SHARE = float(os.getenv(
    "QEM_CROYANCES_MAX_SHARE", os.getenv("QEM_REFLECTIVE_MAX_SHARE", "0.15")))
QEM_REFLEXES_MAX_SHARE = float(os.getenv("QEM_REFLEXES_MAX_SHARE", "0.20"))
# Nombre max de réflexes injectés à chaque contexte (tronc cérébral). 0 = désactivé.
QEM_REFLEXES_INJECTES = int(os.getenv("QEM_REFLEXES_INJECTES", "8"))
# Décroissance temporelle : demi-vie (en jours) du score de récence. Une mémoire non
# ré-accédée voit sa pertinence divisée par 2 tous les N jours. 0 (ou négatif) = désactivé.
QEM_RECENCY_HALFLIFE_DAYS = float(os.getenv("QEM_RECENCY_HALFLIFE_DAYS", "14"))
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


_pool_lock = threading.Lock()
_redis_lock = threading.Lock()


def get_db_pool() -> pg_pool.ThreadedConnectionPool:
    """Retourne le pool PostgreSQL en le réinitialisant si nécessaire.

    Auto-cicatrisant : si le pool n'a pas pu être créé au démarrage (ex:
    PostgreSQL en cours de boot), il est créé à la volée au premier appel
    qui en a besoin sans exiger de redémarrage manuel du conteneur.
    """
    global db_pool
    if db_pool is not None and not getattr(db_pool, "closed", False):
        return db_pool
    with _pool_lock:
        if db_pool is None or getattr(db_pool, "closed", False):
            try:
                db_pool = pg_pool.ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, dsn=DATABASE_URL)
                logger.info("Pool PostgreSQL auto-initialisé (%d–%d connexions).", DB_POOL_MIN, DB_POOL_MAX)
            except Exception as e:
                logger.warning("Tentative d'initialisation du pool PostgreSQL échouée : %s", e)
                db_pool = None
                raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé") from None
    return db_pool


@contextmanager
def get_conn():
    """Emprunte une connexion au pool et la restitue systématiquement.

    Remplace l'ancienne connexion globale unique (non thread-safe) : chaque
    requête obtient sa propre connexion, évitant les conditions de course sous
    charge (FastAPI sert les routes sync dans un threadpool).

    Auto-cicatrisant : si le pool n'a pas pu être initialisé au boot, il est
    créé à la volée dès que la base répond. Si une coupure survient, la connexion
    morte est fermée au putconn au lieu d'être remise en circulation.

    Paramètres HNSW appliqués à chaque session :
    - `ef_search = 150` : le défaut (40) ne donne qu'un recall@10 d'environ 50%.
      100-200 est la plage recommandée pour 95%+ de recall (benchmarks pgvector).
    - `iterative_scan = relaxed_order` : permet à pgvector de continuer le scan
      après filtrage, améliorant la latence des requêtes filtrées de 5-9x.
    """
    pool = get_db_pool()
    conn = pool.getconn()
    casse = False
    try:
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = %s", (os.getenv("HNSW_EF_SEARCH", "150"),))
            cur.execute("SET hnsw.iterative_scan = %s", (os.getenv("HNSW_ITERATIVE_SCAN", "relaxed_order"),))
        yield conn
    except Exception:
        casse = True
        try:
            conn.rollback()
            casse = False
        except Exception:
            logger.warning("Rollback impossible : la connexion sera fermée.", exc_info=True)
        raise
    finally:
        pool.putconn(conn, close=casse)


def get_redis_client():
    global redis_client
    if redis_client is not None:
        return redis_client
    with _redis_lock:
        if redis_client is None:
            try:
                client = redis.from_url(REDIS_URL, decode_responses=True)
                client.ping()
                redis_client = client
                logger.info("Client Redis auto-initialisé.")
            except Exception as e:
                logger.warning("Échec d'initialisation de Redis : %s", e)
                raise HTTPException(status_code=503, detail="Redis non initialisé") from None
        return redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie applicatif (remplace @app.on_event('startup') déprécié)."""
    global db_pool, redis_client
    retries = int(os.getenv("DB_STARTUP_RETRIES", "5"))
    for attempt in range(1, retries + 1):
        try:
            get_db_pool()
            break
        except Exception as e:
            if attempt < retries:
                logger.info("PostgreSQL en attente de démarrage (tentative %d/%d)...", attempt, retries)
                await asyncio.sleep(1)
            else:
                logger.warning("PostgreSQL non prêt au démarrage après %d tentatives (%s) — le pool sera auto-initialisé à la première requête.", retries, e)
    try:
        get_redis_client()
    except Exception as e:
        logger.warning("Redis non prêt au démarrage (%s) — le client sera initialisé à la première requête.", e)
    yield
    if db_pool is not None:
        db_pool.closeall()
    if redis_client is not None:
        redis_client.close()


app = FastAPI(title="SynaptiQ API", version="0.4.0", lifespan=lifespan)

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
    pool = get_db_pool()
    key_hash = _hash_key(raw)

    cachee = _auth_cache_get(key_hash)
    if cachee is not None:
        AUTH_CACHE_HITS.labels("hit").inc()
        return AuthContext(tenant_id=cachee[0], scopes=cachee[1], agent_scope=cachee[2],
                           actor=key_hash[:8])

    # Initialisé AVANT le try : une exception SQL laissait auparavant `row` non liée,
    # transformant une panne base en NameError opaque au lieu d'un 401/503 propre.
    row = None
    conn = pool.getconn()
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
        pool.putconn(conn)
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
                      agent_id: str, compartments: list[str],
                      project: str | None = None,
                      include_global: bool = True) -> list[dict]:
    """Ramène les candidats par similarité vectorielle ET, si activé, par plein texte.

    Une seule requête à deux CTE plutôt que deux allers-retours : chaque ligne porte son
    rang dans chaque chemin (`rank_vec`, `rank_fts`, NULL quand le chemin ne l'a pas
    trouvée), ce qui permet la fusion RRF côté Python sur des fonctions pures testables.

    `websearch_to_tsquery` a été abandonné le 11/08 : il produisait un ET de tous les
    termes, mots vides compris, sur une config 'simple' qui ne les retire pas — la
    question « quel port pour le Postgres de SynaptiQ » ne remontait littéralement rien
    (0/12 sur le banc de l'audit). Le tsquery est désormais un OU PONDÉRÉ de lexèmes :
    construit depuis `to_tsvector('french', …)`, il est débarrassé des mots vides et
    normalisé (stemming) ; `to_tsquery('french', 'a | b | c')` matche tout souvenir qui
    porte AU MOINS un terme, et `ts_rank` classe plus haut ceux qui en portent plusieurs.
    La question vient d'un agent et n'est jamais échappée à la main, mais il n'y a plus
    rien à échapper : les lexèmes sortent de la normalisation PostgreSQL elle-même, ils
    ne peuvent pas transporter de syntaxe de requête.

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
    champs = ("id", "type", "compartment", "content", "confidence", "importance",
              "last_accessed_at", "created_at", "occurred_at", "embedding::text")
    colonnes = ", ".join(champs)
    # Le SELECT final joint `memories` (alias m) et les rangs : sans préfixe, PostgreSQL
    # refuse la requête ("column reference id is ambiguous").
    colonnes_m = ", ".join(f"m.{c}" for c in champs)

    # Filtre PROJET (lot B) : fragment CHOISI par le serveur parmi trois formes fixes, le
    # nom du projet passe en paramètre lié. Placé APRÈS `compartment = ANY(...)` dans chaque
    # WHERE, paramètres dans le même ordre : ces requêtes sont positionnelles, un décalage
    # y serait muet (les types concordent) et fausserait le filtre au lieu de lever.
    frag_projet, p_projet = filtre_projet(project, include_global)
    frag_projet_m, _ = filtre_projet(project, include_global, alias="m.")
    filtre_col = frag_projet
    filtre_col_m = frag_projet_m
    p_col: list = list(p_projet)
    if frag_projet:
        # Filtre sélectif : sans scan itératif, HNSW rendrait moins que le LIMIT.
        activer_scan_iteratif(cur, mode_scan_iteratif())

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
            WHERE tenant_id = %s AND agent_id = %s AND compartment = ANY(%s) {filtre_col}
              AND status = 'active'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_str, vector_str, tenant, agent_id, compartments, *p_col, vector_str,
              RETRIEVAL_CANDIDATES))
        return cur.fetchall()

    cur.execute(f"""
        WITH vectoriel AS (
            SELECT id, row_number() OVER (ORDER BY distance) AS rank_vec
            FROM (
                SELECT id, embedding <=> %s::vector AS distance
                FROM memories
                WHERE tenant_id = %s AND agent_id = %s AND compartment = ANY(%s) {filtre_col}
                  AND status = 'active'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ) v
        ),
        plein_texte AS (
            SELECT id, row_number() OVER (ORDER BY score DESC) AS rank_fts
            FROM (
                SELECT m.id, ts_rank(m.content_tsv, q.query) AS score
                FROM memories m, (
                    -- OU pondéré de lexèmes (cf. la note de `_fetch_candidates`) : les
                    -- mots vides ont déjà été retirés par la config 'french', et un
                    -- seul terme suffit à matcher.
                    SELECT to_tsquery('french', string_agg(lexeme, ' | ')) AS query
                    FROM unnest(to_tsvector('french', %s))
                ) AS q
                WHERE m.tenant_id = %s AND m.agent_id = %s AND m.compartment = ANY(%s) {filtre_col_m}
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
    """, (vector_str, tenant, agent_id, compartments, *p_col, vector_str, RETRIEVAL_CANDIDATES,
          query_text, tenant, agent_id, compartments, *p_col, RETRIEVAL_CANDIDATES,
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
                 project: str | None = None, include_global: bool = True) -> None:
        self._cur = cur
        self._tenant = tenant_id
        self._agent = agent_id
        # Idem pour le projet (lot B) : il borne AUSSI la complétion du graphe
        # (`fetch_by_ids`), sans quoi une arête ramènerait un souvenir d'un autre projet.
        self._project = project
        self._include_global = include_global

    @staticmethod
    def _normaliser(ligne) -> dict:
        """Désérialise le vecteur ; le cœur ne connaît pas le format texte de pgvector."""
        valeur = dict(ligne)
        valeur["embedding"] = parse_embedding(valeur.get("embedding"))
        return valeur

    def fetch_candidates(self, query_vector, query_text: str, compartments: list[str]) -> list[dict]:
        lignes = _fetch_candidates(self._cur, to_pgvector(query_vector), query_text,
                                   self._tenant, self._agent, compartments,
                                   self._project, self._include_global)
        return [self._normaliser(ligne) for ligne in lignes]

    def fetch_ancres(self, compartment: str, limit: int) -> list[dict]:
        """Réflexes injectés d'office : les plus importants, puis les plus récents.

        Même périmètre que la recherche, projet compris (filtre répété ici comme dans
        chaque chemin SQL, jamais factorisé en CTE — cf. CLAUDE.md §3).
        """
        frag_projet, p_projet = filtre_projet(self._project, self._include_global)
        self._cur.execute(
            f"""
            SELECT id, type, compartment, content, confidence, importance,
                   last_accessed_at, created_at, occurred_at, embedding::text,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_accessed_at)) AS age_seconds
            FROM memories
            WHERE tenant_id = %s AND agent_id = %s AND compartment = %s
              AND status = 'active' {frag_projet}
            ORDER BY importance DESC, created_at DESC
            LIMIT %s;
            """,
            (self._tenant, self._agent, compartment, *p_projet, limit),
        )
        return [self._normaliser(ligne) for ligne in self._cur.fetchall()]

    def fetch_relationships(self, memory_ids: list[str]) -> list[dict]:
        # UNION ALL au lieu d'OR : PostgreSQL utilise les index efficacement.
        # Avec OR, le planner fait souvent un scan séquentiel. UNION ALL permet
        # d'utiliser idx_relationships_source_type et idx_relationships_target.
        # La condition `source_memory_id <> ALL(%s)` évite les doublons quand une
        # arête a ses deux extrémités dans la liste.
        self._cur.execute(
            """
            SELECT source_memory_id, target_memory_id, relation_type, weight
            FROM relationships
            WHERE source_memory_id = ANY(%s::uuid[])
            UNION ALL
            SELECT source_memory_id, target_memory_id, relation_type, weight
            FROM relationships
            WHERE target_memory_id = ANY(%s::uuid[])
              AND source_memory_id <> ALL(%s::uuid[]);
            """,
            (memory_ids, memory_ids, memory_ids),
        )
        return self._cur.fetchall()

    def fetch_by_ids(self, memory_ids: list[str]) -> list[dict]:
        frag_projet, p_projet = filtre_projet(self._project, self._include_global)
        # Fragment choisi par `filtre_projet` parmi trois formes fixes (aucune donnée
        # d'appelant dans le SQL, le nom du projet passe en paramètre lié).
        self._cur.execute(
            f"""
            SELECT id, type, compartment, content, confidence, importance,
                   last_accessed_at, created_at, occurred_at, embedding::text
            FROM memories
            WHERE id = ANY(%s::uuid[])
              AND tenant_id = %s AND agent_id = %s
              AND status = 'active' {frag_projet};
            """,
            (memory_ids, self._tenant, self._agent, *p_projet),
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
        entangle_aggregation=QEM_ENTANGLE_AGGREGATION,
        redundancy_threshold=QEM_REDUNDANCY_THRESHOLD,
        recency_halflife_days=QEM_RECENCY_HALFLIFE_DAYS,
        collapse_top_k=QEM_COLLAPSE_TOP_K,
        density_exponent=QEM_DENSITY_EXPONENT,
        croyances_max_share=QEM_CROYANCES_MAX_SHARE,
        reflexes_max_share=QEM_REFLEXES_MAX_SHARE,
        reflexes_injectes=QEM_REFLEXES_INJECTES,
    )


def prototypes_compartiments() -> dict[str, list[float]]:
    """Prototypes des compartiments pour l'embedder de l'instance (cache du cœur)."""
    return prototypes(get_embedder())


# Modèles Pydantic
# Nom de compartiment STRICT, pour les FILTRES de lecture : viser un compartiment qui
# n'existe pas est une erreur de l'appelant (422), pas une recherche vide silencieuse.
# À l'écriture, au contraire, un nom inconnu est classé par le moteur (cf. MemoryInput).
CompartmentName = Literal[NOMS]  # type: ignore[valid-type]


class EventInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    session_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "sess_abc"})
    content: str = Field(..., min_length=1, max_length=12000, json_schema_extra={"example": "L'utilisateur demande à rédiger un email pro."})
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=100)
    # Clé de déduplication optionnelle : deux appels avec la même clé (même tenant)
    # ne créent qu'un seul événement.
    idempotency_key: str | None = Field(default=None, max_length=128, json_schema_extra={"example": "evt-2026-07-15-001"})
    # Projet du souvenir (lot B). None = global. Normalisé (minuscules) et validé par
    # `synaptiq_core.project.normaliser_projet` : un nom invalide est refusé en 422.
    project: str | None = Field(default=None, max_length=64,
                                json_schema_extra={"example": "synaptiq"})

    @field_validator("project")
    @classmethod
    def _normaliser_projet(cls, valeur):
        return normaliser_projet(valeur)

class ContextConstraints(BaseModel):
    max_tokens: int = Field(default=1200, ge=1, le=8000)
    # Compartiments interrogés. Par défaut les 9 : `croyances` nuance le paquet sous son
    # quota, `reflexes` y est injecté d'office. Viser un sous-ensemble réduit le bruit à
    # budget égal (ex. ["alertes", "savoir_faire"] pour un débogage).
    compartments: list[CompartmentName] = Field(  # type: ignore[valid-type]
        default_factory=lambda: list(NOMS), min_length=1, max_length=len(NOMS),
        json_schema_extra={"example": ["savoir_faire", "alertes"]})
    # Filtre projet (lot B). None = tous projets confondus (historique). Avec un projet,
    # `include_global` ajoute les souvenirs globaux (préférences, conventions générales).
    project: str | None = Field(default=None, max_length=64,
                                json_schema_extra={"example": "synaptiq"})
    include_global: bool = True

    @field_validator("project")
    @classmethod
    def _normaliser_projet(cls, valeur):
        return normaliser_projet(valeur)

class ContextRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    session_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "sess_abc"})
    task: str = Field(..., min_length=1, max_length=4000, json_schema_extra={"example": "Rédiger un email de suivi"})
    query: str = Field(..., min_length=1, max_length=8000, json_schema_extra={"example": "Style d'écriture concis de Jimmy"})
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)
    explain: bool = False
    # "selected" : trace des retenus (historique) ; "all" : tous les candidats, avec le
    # motif d'exclusion (redondance et par qui, contradiction, hors budget).
    explain_level: Literal["selected", "all"] = "selected"
    # Rafraîchir last_accessed_at / access_count des retenus. Toujours désactivé quand
    # `explain` est vrai : un diagnostic ne doit pas modifier la récence qu'il observe.
    record_access: bool = True

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


@app.get("/v1/admin/slow-queries")
def slow_queries(auth: AuthContext | None = Depends(get_auth), limit: int = Query(default=10, ge=1, le=100)):
    """Endpoint admin : requêtes les plus lentes via pg_stat_statements.

    Nécessite une clé API avec scope 'admin' (même si SYNAPTIQ_AUTH_REQUIRED=false).
    Utile pour identifier les requêtes SQL qui ralentissent l'instance en production.
    """
    require_scope(auth, "admin")
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT query, calls,
                           round(total_exec_time::numeric, 2) AS total_ms,
                           round(mean_exec_time::numeric, 2) AS mean_ms,
                           round(max_exec_time::numeric, 2) AS max_ms
                    FROM pg_stat_statements
                    WHERE query NOT LIKE '%%pg_stat_statements%%'
                    ORDER BY total_exec_time DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                conn.rollback()
            except Exception as e:
                conn.rollback()
                raise HTTPException(
                    status_code=500,
                    detail=f"pg_stat_statements indisponible (extension non activée ?) : {e}",
                ) from None
    return {"slow_queries": [dict(row) for row in rows]}


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
                        INSERT INTO events (tenant_id, agent_id, session_id, content, metadata,
                                            idempotency_key, project)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
                        RETURNING id, created_at;
                        """,
                        (tenant, event.agent_id, event.session_id,
                         event.content, json.dumps(event.metadata), event.idempotency_key,
                         event.project),
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
                        # Absent (None) pour un événement global : le worker le relit tel quel.
                        "project": event.project,
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

# ─── Cache Redis pour /context/build ─────────────────────────────────────────
# Les requêtes d'un agent sont souvent répétées (même question, même contexte).
# Le cache évite de refaire tout le calcul Q-EM (recherche hybride + intrication +
# interférence + collapse) pour une requête identique dans la fenêtre TTL.
#
# Clé : `ctx_cache:{tenant}:{agent}:{hash(query + compartments + max_tokens + project)}`
# TTL : 60s (défaut, configurable via CONTEXT_CACHE_TTL)
# Sérialisation : JSON (le résultat est déjà un dict sérialisable)
#
# Le cache est invalidé automatiquement par le TTL. Une invalidation proactive
# (purge à l'écriture) serait plus complexe et le TTL court suffit pour un usage
# d'agent où les mémoires changent peu entre deux requêtes rapprochées.
CONTEXT_CACHE_TTL = int(os.getenv("CONTEXT_CACHE_TTL", "60"))
CONTEXT_CACHE_ENABLED = os.getenv("CONTEXT_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CONTEXT_CACHE_HITS = Counter("synaptiq_context_cache_total", "Context build cache", ["outcome"])


def _context_cache_key(tenant: str, agent_id: str, request: ContextRequest) -> str:
    """Clé de cache unique pour une requête de contexte.

    Inclut tous les paramètres qui influencent le résultat : query, compartments,
    max_tokens, project, include_global.
    """
    c = request.constraints
    key_data = f"{request.query}|{','.join(c.compartments)}|{c.max_tokens}|{c.project}|{c.include_global}"
    key_hash = hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:16]
    return f"ctx_cache:{tenant}:{agent_id}:{key_hash}"


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

    Cache Redis : les requêtes identiques dans la fenêtre TTL (60s par défaut) sont
    servies depuis le cache, évitant de refaire tout le calcul Q-EM.
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
    pool = get_db_pool()

    # ── Cache Redis ──
    cache_key = _context_cache_key(tenant, request.agent_id, request)
    if CONTEXT_CACHE_ENABLED and redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                CONTEXT_CACHE_HITS.labels("hit").inc()
                resultat = json.loads(cached)
                # La trace_id doit rester unique par requête, même en cas de cache hit.
                # Elle sert à corréler les logs à une requête spécifique.
                resultat["trace_id"] = trace_id
                logger.info("Cache hit pour %s (trace=%s).", cache_key, trace_id)
                return resultat
        except Exception:
            logger.warning("Cache Redis indisponible pour la lecture.", exc_info=True)

    conn = pool.getconn()
    try:
        query_vector = get_embedder().embed_one(request.query)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Les sections du paquet sont les 9 compartiments, toujours présents : la forme
            # de la réponse ne dépend ni de l'agent ni de ce qu'il a déjà rangé.
            resultat = build_context_packet(
                store=PostgresMemoryStore(cur, tenant, request.agent_id,
                                          request.constraints.project,
                                          request.constraints.include_global),
                query_vector=query_vector,
                query_text=request.query,
                compartments=list(request.constraints.compartments),
                max_tokens=request.constraints.max_tokens,
                config=retrieval_config(),
                trace_id=trace_id,
                explain=request.explain,
                explain_level=request.explain_level,
                record_access=request.record_access and not request.explain,
            )
            # `mark_accessed` a ecrit dans la transaction : la valider.
            conn.commit()

        # ── Mise en cache ──
        if CONTEXT_CACHE_ENABLED and redis_client is not None:
            try:
                redis_client.setex(cache_key, CONTEXT_CACHE_TTL, json.dumps(resultat, default=str))
            except Exception:
                logger.warning("Cache Redis indisponible pour l'écriture.", exc_info=True)

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
        pool.putconn(conn)


class MemoryInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    # Compartiment où ranger le souvenir. C'est le SEUL rangement que l'agent fournit, et il
    # est facultatif : absent ou inconnu, le moteur classe par similarité avec les
    # prototypes des compartiments, et la réponse le dit (`compartment_origin`).
    compartment: str | None = Field(None, max_length=64,
                                    json_schema_extra={"example": "savoir_faire"})
    # ⚠️ DÉPRÉCIÉS (v0.3). Encore lus pour ne pas casser les clients existants le jour de
    # la bascule : un ancien nom de collection est converti par `compartments.ALIAS`
    # (origine `alias`, journalisée). Seront retirés en v0.5.
    type: str | None = Field(None, max_length=20, description="Déprécié : utiliser compartment.")
    subtype: str | None = Field(None, max_length=64, description="Déprécié : utiliser compartment.")
    # Sujet d'une croyance (compartiment `croyances` seulement) : l'utilisateur, ou les
    # humains / tiers en général.
    about: Literal["user", "humans"] | None = None
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
    # Projet du souvenir (lot B). None = global. Normalisé (minuscules) et validé par
    # `synaptiq_core.project.normaliser_projet` : un nom invalide est refusé en 422.
    project: str | None = Field(default=None, max_length=64,
                                json_schema_extra={"example": "synaptiq"})

    @field_validator("project")
    @classmethod
    def _normaliser_projet(cls, valeur):
        return normaliser_projet(valeur)

    # Croyances (lot C) : souvenirs à l'appui, et croyance remplacée (révision par
    # supersession plutôt que par accumulation). Refusés hors du compartiment `croyances`.
    evidence: list[str] | None = Field(default=None, max_length=20)
    replaces: uuid.UUID | None = None

    @model_validator(mode="after")
    def _verifier_la_croyance(self):
        """Ramène les anciennes formes de croyance au compartiment, puis applique
        `belief_guard`.

        Dans le validateur (et non dans le handler) pour que le refus sorte en 422 avec le
        motif, comme toute autre erreur de saisie : l'agent peut reformuler.
        """
        # Formes v0.3 d'une croyance : type='reflective', ou sous-type user_model /
        # human_insights. Le sujet se déduit du sous-type.
        ancien = self.subtype if self.compartment is None else self.compartment
        if self.compartment is None and (self.type == "reflective"
                                         or ancien in ("user_model", "human_insights")):
            self.compartment = COMPARTIMENT_CROYANCES
        if ancien in ("user_model", "human_insights") and self.about is None:
            self.about = "humans" if ancien == "human_insights" else "user"
        if self.compartment in ("user_model", "human_insights"):
            self.compartment = COMPARTIMENT_CROYANCES

        if self.compartment != COMPARTIMENT_CROYANCES:
            if self.evidence or self.replaces or self.about:
                raise ValueError("`evidence`, `replaces` et `about` ne s'appliquent qu'aux "
                                 "croyances (compartment='croyances').")
            return self
        self.about = self.about or "user"
        try:
            validee = valider_croyance(self.content, self.confidence, self.evidence)
        except CroyanceRefusee as e:
            raise ValueError(str(e)) from e
        self.confidence = validee.confidence
        self.evidence = list(validee.evidence)
        return self

    @property
    def est_croyance(self) -> bool:
        return self.compartment == COMPARTIMENT_CROYANCES

    def rangement(self, vecteur) -> Rangement:
        """Compartiment retenu : demandé, converti depuis un ancien nom, ou classé."""
        if self.est_croyance:
            return Rangement(COMPARTIMENT_CROYANCES, "agent", demande=self.compartment)
        demande = self.compartment if self.compartment is not None else self.subtype
        return resoudre(demande, vecteur, prototypes_compartiments(), famille=self.type)


def _croyance_contestee(cur, tenant: str, agent_id: str, empreinte: str) -> bool:
    """Une croyance de même contenu a-t-elle été contestée par l'utilisateur ?"""
    cur.execute(
        "SELECT 1 FROM memories WHERE tenant_id = %s AND agent_id = %s "
        "AND content_hash = %s AND status = 'contested' LIMIT 1",
        (tenant, agent_id, empreinte))
    return cur.fetchone() is not None


def _archiver_croyance_remplacee(cur, tenant: str, agent_id: str, ancienne: str) -> list[str]:
    """Archive la croyance remplacée ; 404 si elle n'est pas une croyance active de l'agent.

    Borné au (tenant, agent) et au compartiment : `replaces` ne doit pas devenir un moyen
    d'archiver n'importe quel souvenir par identifiant.
    """
    cur.execute(
        "UPDATE memories SET status = 'archived', updated_at = CURRENT_TIMESTAMP "
        "WHERE id = %s AND tenant_id = %s AND agent_id = %s AND compartment = %s "
        "AND status = 'active' RETURNING id",
        (ancienne, tenant, agent_id, COMPARTIMENT_CROYANCES))
    if cur.fetchone() is None:
        raise HTTPException(status_code=404,
                            detail=f"Aucune croyance active {ancienne} à remplacer.")
    return [ancienne]


def _memoire_existante(cur, tenant: str, agent_id: str, empreinte: str,
                       cle_idempotence: str | None):
    """Cherche une mémoire ACTIVE déjà écrite en direct pour ce contenu (ou cette clé).

    Renvoie `(id, compartiment)` ou None.

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
            "SELECT id, compartment FROM memories "
            "WHERE tenant_id = %s AND agent_id = %s AND idempotency_key = %s "
            "AND status = 'active' AND source_event_id IS NULL LIMIT 1",
            (tenant, agent_id, cle_idempotence),
        )
        ligne = cur.fetchone()
        if ligne is not None:
            return ligne[0], ligne[1]
    cur.execute(
        "SELECT id, compartment FROM memories "
        "WHERE tenant_id = %s AND agent_id = %s AND content_hash = %s "
        "AND status = 'active' AND source_event_id IS NULL LIMIT 1",
        (tenant, agent_id, empreinte),
    )
    ligne = cur.fetchone()
    return (ligne[0], ligne[1]) if ligne is not None else None


def _reponse_memoire(memory: "MemoryInput", memory_id: str, statut: str,
                     rangement: Rangement) -> dict:
    """Corps de réponse de `POST /v1/memories`, identique quel que soit le statut.

    `duplicate` rend la MÊME forme que `created`, avec l'identifiant de la ligne déjà en
    base : un appelant qui relance après timeout obtient un identifiant exploitable, et la
    seule différence visible est `status`. Deux formes de réponse pour un même endpoint
    obligeraient chaque client à traiter le cas dégradé séparément — et donc à l'oublier.

    Le compartiment est TOUJOURS rendu, avec qui l'a choisi : un agent qui n'a rien
    précisé, ou qui a donné un nom inconnu, doit voir où son souvenir a été rangé pour
    pouvoir le corriger (v0.3 : un libellé métier retombait dans `facts` sans le dire).
    """
    compartiment = PAR_NOM[rangement.compartiment]
    return {
        "status": statut,
        "memory_id": memory_id,
        "compartment": compartiment.nom,
        "region": compartiment.region,
        # `agent` : nom valide fourni ; `alias` : ancien nom de collection converti ;
        # `moteur` : classé par similarité ; `existant` : doublon, rangement d'origine.
        "compartment_origin": rangement.origine,
        "classification_similarity": rangement.similarite,
        "requested": rangement.demande,
        # Projet effectivement enregistré (normalisé) : None = souvenir global.
        "project": memory.project,
    }


def _provenance(memory: "MemoryInput", rangement: Rangement) -> dict:
    """Provenance d'un souvenir écrit en direct."""
    if memory.est_croyance:
        provenance: dict[str, Any] = {"about": memory.about}
        if memory.evidence:
            provenance["evidence"] = memory.evidence
        return provenance
    if rangement.origine in ("alias", "moteur") and rangement.demande:
        return {"requested_compartment": rangement.demande}
    return {}


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
    pool = get_db_pool()
    conn = pool.getconn()
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
                conn.rollback()      # lecture seule : ne rien laisser en transaction
                MEMORY_WRITES.labels("duplicate").inc()
                logger.info("Écriture directe déjà présente : relance traitée en no-op.",
                            extra={"agent_id": memory.agent_id, "memory_id": str(existant[0])})
                # Rangement d'ORIGINE : un doublon ne se reclasse pas.
                return _reponse_memoire(memory, str(existant[0]), "duplicate",
                                        Rangement(existant[1], "existant",
                                                  demande=memory.compartment))

            # Une croyance CONTESTÉE par l'utilisateur ne se réécrit pas à l'identique : sans
            # ce contrôle, l'agent la renoterait à la session suivante et la contestation
            # n'aurait servi à rien. Reformuler (avec de nouveaux indices) reste possible.
            if memory.est_croyance and _croyance_contestee(
                    cur, tenant, memory.agent_id, empreinte):
                conn.rollback()
                raise HTTPException(
                    status_code=409,
                    detail="Cette croyance a été contestée par l'utilisateur : elle ne peut "
                           "pas être réenregistrée telle quelle.")

            embedding = get_embedder().embed_one(memory.content)
            # Le compartiment se décide APRÈS l'embedding : le classement par similarité
            # réutilise le vecteur du souvenir, aucun appel d'embedding supplémentaire.
            rangement = memory.rangement(embedding)
            compartiment = PAR_NOM[rangement.compartiment]

            # Gestion des contradictions
            new_mem_dict = {
                "compartment": compartiment.nom,
                "content": memory.content
            }
            # Archivage sur verdict EXPLICITE de contradiction seulement (cf. governance).
            superseded = handle_contradictions(cur, tenant, memory.agent_id, new_mem_dict,
                                               embedding, project=memory.project)

            # Insertion. `ON CONFLICT DO NOTHING` SANS cible : deux index uniques partiels
            # couvrent cette table (contenu et clé d'idempotence) et une clause `ON CONFLICT`
            # ne peut en nommer qu'un. Sans cible, PostgreSQL neutralise l'insertion sur
            # n'importe quelle violation d'unicité — donc sur les deux.
            # Ce filet ne remplace pas le pré-contrôle ci-dessus : il rattrape la course
            # entre deux requêtes concurrentes portant le même contenu, fenêtre que le
            # SELECT seul laisse ouverte.
            query = """
                INSERT INTO memories (tenant_id, agent_id, type, compartment, content, embedding,
                                      confidence, importance, status, content_hash, idempotency_key,
                                      project, provenance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id;
            """
            cur.execute(query, (
                tenant,
                memory.agent_id,
                # Famille DÉDUITE du compartiment : plus personne ne la fournit.
                compartiment.famille,
                compartiment.nom,
                memory.content,
                embedding,
                memory.confidence,
                memory.importance,
                empreinte,
                memory.idempotency_key,
                memory.project,
                # Croyance : indices (ids de souvenirs, c'est ce qui la rend contrôlable) et
                # sujet. Autre souvenir : nom demandé quand le moteur a dû trancher.
                json.dumps(_provenance(memory, rangement)),
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
                    memory, str(gagnante[0]) if gagnante else "", "duplicate",
                    Rangement(gagnante[1], "existant", demande=memory.compartment)
                    if gagnante else rangement)
                conn.commit()
                MEMORY_WRITES.labels("duplicate").inc()
                logger.info("Course d'insertion perdue sur un contenu identique : no-op.",
                            extra={"agent_id": memory.agent_id,
                                   "memory_id": str(gagnante[0]) if gagnante else None})
                return reponse

            new_id = ligne[0]
            # Révision d'une croyance : l'ancienne est archivée et reliée à la nouvelle par
            # `supersedes_by` (même mécanisme que les préférences contredites).
            if memory.replaces is not None:
                superseded = [*superseded, *_archiver_croyance_remplacee(
                    cur, tenant, memory.agent_id, str(memory.replaces))]
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
            # Seuls les compartiments qui intriquent tissent des liens (les épisodes et
            # les observations, nombreux et peu discriminants, densifieraient le graphe
            # sans gain ; les réflexes, présents partout, deviendraient des hubs).
            if compartiment.intrique:
                entangle(cur, tenant, memory.agent_id, new_id, compartiment.nom, embedding,
                         project=memory.project)

            reponse = _reponse_memoire(memory, str(new_id), "created", rangement)
            conn.commit()
            MEMORY_WRITES.labels("created").inc()

            logger.info("Mémoire créée en direct par l'agent : %s", new_id,
                        extra={"agent_id": memory.agent_id, "compartment": compartiment.nom,
                               "compartment_origin": rangement.origine})
            return reponse
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.error("Erreur lors de la création de la mémoire.", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from None
    finally:
        pool.putconn(conn)

class RetrieveRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    query: str = Field(..., min_length=1, max_length=8000)
    limit: int = Field(default=5, ge=1, le=100)
    # Restreint la recherche à certains compartiments. None = tous.
    compartments: list[CompartmentName] | None = Field(  # type: ignore[valid-type]
        default=None, max_length=len(NOMS))
    # Filtre projet (lot B). None = tous projets confondus (historique). Avec un projet,
    # `include_global` ajoute les souvenirs globaux (préférences, conventions générales).
    project: str | None = Field(default=None, max_length=64,
                                json_schema_extra={"example": "synaptiq"})
    include_global: bool = True

    @field_validator("project")
    @classmethod
    def _normaliser_projet(cls, valeur):
        return normaliser_projet(valeur)

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
    pool = get_db_pool()
    conn = pool.getconn()
    try:
        query_vector = get_embedder().embed_one(request.query)
        vector_str = to_pgvector(query_vector)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Préfixé `m.` : le chemin hybride aliase `memories` en m dans les deux branches.
            # `type_filter` est un fragment CHOISI par le serveur (formes fixes), jamais
            # une chaine fournie par l'appelant : les compartiments passent en paramètre
            # lié et sont validés en amont par Pydantic (Literal, cf. S608 dans ruff.toml).
            # Le fragment et `filtre_params` sont construits ENSEMBLE et dans le même ordre :
            # les trois requêtes ci-dessous sont positionnelles, et un décalage y serait muet
            # (les types concordent) — il fausserait le filtre au lieu de lever.
            type_filter = "AND m.compartment = ANY(%s)" if request.compartments else ""
            filtre_params: list = [tenant, request.agent_id]
            if request.compartments:
                filtre_params.append(list(request.compartments))
            # Projet (lot B) : même fragment que `_fetch_candidates`, ajouté en dernier.
            frag_projet, p_projet = filtre_projet(request.project, request.include_global,
                                                  alias="m.")
            if frag_projet:
                type_filter += f" {frag_projet}"
                filtre_params.extend(p_projet)
                activer_scan_iteratif(cur, mode_scan_iteratif())

            if not RETRIEVAL_HYBRID:
                cur.execute(f"""
                    SELECT m.id, m.compartment, m.content, m.confidence, m.importance,
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
                        FROM memories m, (
                            -- OU pondéré de lexèmes, même forme que `_fetch_candidates` :
                            -- la config 'french' retire les mots vides, un seul terme
                            -- suffit à matcher, `ts_rank` récompense les multi-hits.
                            SELECT to_tsquery('french', string_agg(lexeme, ' | ')) AS query
                            FROM unnest(to_tsvector('french', %s))
                        ) AS q
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
                SELECT m.id, m.compartment, m.content, m.confidence, m.importance,
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
        pool.putconn(conn)


# ─── Croyances : ce que l'agent pense (compartiment `croyances`, lot C) ─────
# Lecture et contestation exposées À PART de /retrieve : l'utilisateur doit pouvoir voir,
# d'un seul appel, TOUT ce que l'agent pense de lui — et le contester. C'est la condition
# pour qu'un agent ait le droit de se forger une opinion sur la personne qu'il sert.

class ContestInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    # Motif libre, journalisé en longueur seulement (audit_log ne porte jamais de contenu).
    reason: str | None = Field(default=None, max_length=500)


@v1_router.get("/beliefs")
def list_beliefs(agent_id: str, about: Literal["user", "humans"] | None = None,
                 auth: AuthContext | None = Depends(get_auth)):
    """Toutes les croyances ACTIVES de l'agent, la plus assurée d'abord."""
    tenant = resolve_tenant(auth)
    require_scope(auth, "read")
    resolve_agent(auth, agent_id)
    sujets = [about] if about else ["user", "humans"]
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, COALESCE(provenance->>'about', 'user') AS about, content, confidence,
                   provenance->'evidence' AS evidence, project, created_at
            FROM memories
            WHERE tenant_id = %s AND agent_id = %s AND compartment = %s AND status = 'active'
              AND COALESCE(provenance->>'about', 'user') = ANY(%s)
            ORDER BY confidence DESC, created_at DESC
        """, (tenant, agent_id, COMPARTIMENT_CROYANCES, sujets))
        lignes = cur.fetchall()
    return {"beliefs": [
        {"id": str(r["id"]),
         "about": r["about"],
         "content": r["content"], "confidence": r["confidence"],
         "evidence": r["evidence"] or [], "project": r["project"],
         "created_at": r["created_at"].isoformat()}
        for r in lignes]}


@v1_router.post("/beliefs/{belief_id}/contest")
def contest_belief(belief_id: uuid.UUID, payload: ContestInput,
                   auth: AuthContext | None = Depends(get_auth)):
    """Conteste une croyance : elle sort du contexte et ne peut plus être réécrite telle quelle.

    Statut `contested` (et non suppression) : la ligne sert de témoin pour refuser la
    réécriture à l'identique (cf. `create_memory`), et la contestation reste traçable.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "write")
    resolve_agent(auth, payload.agent_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE memories SET status = 'contested', updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND tenant_id = %s AND agent_id = %s AND compartment = %s
                  AND status = 'active'
                RETURNING id
            """, (str(belief_id), tenant, payload.agent_id, COMPARTIMENT_CROYANCES))
            if cur.fetchone() is None:
                conn.rollback()
                raise HTTPException(status_code=404,
                                    detail=f"Aucune croyance active {belief_id}.")
            audit(cur, tenant, "belief_contested", auth, payload.agent_id,
                  belief_id=str(belief_id), reason_length=len(payload.reason or ""))
        conn.commit()
    return {"status": "contested", "belief_id": str(belief_id)}


@v1_router.get("/compartments")
def list_compartments(agent_id: str, auth: AuthContext | None = Depends(get_auth)):
    """Les 9 compartiments, avec leur fonction et le nombre de souvenirs de l'agent.

    C'est ce que l'agent consulte pour juger où ranger. La liste est FIXE (propriété du
    moteur) : il n'existe ni création ni fusion, l'agent choisit, il ne crée pas.
    `memory_count` est calculé ici plutôt que maintenu : un compteur dénormalisé dériverait
    au premier écart d'écriture.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "read")
    resolve_agent(auth, agent_id)
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT compartment, count(*) AS n FROM memories "
            "WHERE tenant_id = %s AND agent_id = %s AND status = 'active' GROUP BY 1",
            (tenant, agent_id),
        )
        comptes = {ligne["compartment"]: ligne["n"] for ligne in cur.fetchall()}
        conn.rollback()  # lecture seule
    compartiments = []
    for fiche in catalogue():
        fiche["memory_count"] = comptes.get(fiche["name"], 0)
        compartiments.append(fiche)
    return {"agent_id": agent_id, "compartments": compartiments,
            # Anciens noms acceptés à l'écriture pendant la transition (v0.4 → v0.5).
            "legacy_aliases": dict(sorted(ALIAS.items()))}


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

    Quatrième garde-fou (11/08) : le périmètre d'agents de la clé n'était consulté QUE si
    l'appelant fournissait `agent_id`. Une clé `admin` bornée à un seul agent qui omettait
    le paramètre tombait donc dans la branche « tout le tenant » et détruisait
    physiquement la mémoire de tous les autres agents — son périmètre n'ayant jamais été
    lu. Omettre un filtre ÉLARGISSAIT l'effet : exactement l'inverse de ce qu'une
    restriction doit faire.

    La suppression reste physique et immédiate : c'est un endpoint RGPD, un effacement
    différé ou réversible irait à l'encontre de sa raison d'être.
    """
    tenant = resolve_tenant(auth)
    require_scope(auth, "admin")
    if agent_id:
        resolve_agent(auth, agent_id)
    elif auth is not None and auth.agent_scope:
        # Purge globale demandée par une clé qui n'a pas le droit de parler pour tous les
        # agents. Refusée AVANT le contrôle de `confirm` : c'est une question
        # d'autorisation, pas d'intention. Le message indique la seule forme d'appel
        # possible pour cette clé, faute de quoi l'appelant relancerait à l'identique.
        raise HTTPException(
            status_code=403,
            detail="Cette clé API est restreinte aux agents "
                   f"{', '.join(auth.agent_scope)} : la purge globale du tenant lui est "
                   "interdite. Préciser ?agent_id=<agent> pour purger un agent de son "
                   "périmètre.",
        )
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


