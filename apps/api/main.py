import sys
import os

# Ajouter la racine du projet + packages/core au sys.path (imports monorepo, dev local)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (root_path, os.path.join(root_path, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import hashlib
import time
from contextlib import contextmanager, asynccontextmanager
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header, Response, APIRouter
from pydantic import BaseModel, Field

v1_router = APIRouter()
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
import redis
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from dotenv import load_dotenv

# Logique partagée (embeddings pluggables + gouvernance), plus d'import depuis le worker
from synaptiq_core import get_embedder, to_pgvector, handle_contradictions
# Cœur algorithmique Q-EM (fonctions pures : superposition -> intrication -> interférence -> mesure)
from synaptiq_core.qem import (
    compute_recency_factor,
    initial_score,
    propagate_entanglement,
    apply_contradictions,
    filter_redundancy,
    collapse_by_utility,
)
# Fusion de classements pour la recherche hybride (fonctions pures, cf. retrieval.py)
from synaptiq_core.retrieval import DEFAULT_RRF_K, fuse_and_rank, reciprocal_rank_fusion

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

db_pool: Optional[pg_pool.ThreadedConnectionPool] = None
redis_client = None
EVENTS_CAPTURED = Counter("synaptiq_events_captured_total", "Events persisted in the transactional outbox")
CONTEXT_BUILDS = Counter("synaptiq_context_builds_total", "Context builds", ["outcome"])
CONTEXT_BUILD_SECONDS = Histogram("synaptiq_context_build_seconds", "Context build latency")


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
        logger.error("Échec d'initialisation du pool PostgreSQL : %s", e)
        db_pool = None
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Client Redis initialisé.")
    except Exception as e:
        logger.error("Échec d'initialisation de Redis : %s", e)
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
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

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
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_auth(authorization: Optional[str] = Header(default=None)) -> Optional[AuthContext]:
    """Résout la clé API (header Bearer) vers un tenant.

    - Aucune clé + auth désactivée  -> None (mode dev, pas d'isolation).
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
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id FROM api_keys WHERE key_hash = %s AND active = true",
                (_hash_key(raw),),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE key_hash = %s",
                    (_hash_key(raw),),
                )
                conn.commit()
    finally:
        db_pool.putconn(conn)
    if not row:
        raise HTTPException(status_code=401, detail="Clé API invalide ou révoquée")
    return AuthContext(tenant_id=row[0])


def resolve_tenant(auth: Optional[AuthContext]) -> str:
    """Résout le tenant effectif de la requête.

    - Clé API valide -> tenant porté par la clé.
    - Sans auth (instance auto-hébergée) -> tenant d'instance (SYNAPTIQ_TENANT).

    Le tenant n'est plus jamais transmis par l'appelant : impossible de lire ou
    d'écrire dans un autre périmètre en trafiquant le body.
    """
    return auth.tenant_id if auth else _instance_tenant()

def _fetch_candidates(cur, vector_str: str, query_text: str, tenant: str,
                      agent_id: str, memory_types: List[str]) -> List[dict]:
    """Ramène les candidats par similarité vectorielle ET, si activé, par plein texte.

    Une seule requête à deux CTE plutôt que deux allers-retours : chaque ligne porte son
    rang dans chaque chemin (`rank_vec`, `rank_fts`, NULL quand le chemin ne l'a pas
    trouvée), ce qui permet la fusion RRF côté Python sur des fonctions pures testables.

    `websearch_to_tsquery` est utilisé plutôt que `plainto_tsquery` : il tolère une requête
    en langage naturel sans lever d'erreur de syntaxe, ce qui est indispensable ici où la
    requête vient d'un agent et n'est jamais échappée à la main.
    """
    champs = ("id", "type", "subtype", "content", "confidence", "importance",
              "last_accessed_at", "created_at", "occurred_at", "embedding::text")
    colonnes = ", ".join(champs)
    # Le SELECT final joint trois CTE qui portent toutes une colonne `id` : sans préfixe,
    # PostgreSQL refuse la requête ("column reference id is ambiguous").
    colonnes_filtre = ", ".join(f"f.{c}" for c in champs)

    if not RETRIEVAL_HYBRID:
        cur.execute(f"""
            SELECT {colonnes},
                   (1 - (embedding <=> %s::vector)) AS similarity,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_accessed_at)) AS age_seconds,
                   row_number() OVER (ORDER BY embedding <=> %s::vector) AS rank_vec,
                   NULL::bigint AS rank_fts
            FROM memories
            WHERE tenant_id = %s AND agent_id = %s AND type = ANY(%s) AND status = 'active'
            ORDER BY similarity DESC
            LIMIT %s;
        """, (vector_str, vector_str, tenant, agent_id, memory_types, RETRIEVAL_CANDIDATES))
        return cur.fetchall()

    cur.execute(f"""
        WITH filtre AS (
            SELECT * FROM memories
            WHERE tenant_id = %s AND agent_id = %s AND type = ANY(%s) AND status = 'active'
        ),
        vectoriel AS (
            SELECT id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS rank_vec
            FROM filtre ORDER BY embedding <=> %s::vector LIMIT %s
        ),
        plein_texte AS (
            SELECT f.id,
                   row_number() OVER (ORDER BY ts_rank(f.content_tsv, q.query) DESC) AS rank_fts
            FROM filtre f, websearch_to_tsquery('simple', %s) AS q(query)
            WHERE f.content_tsv @@ q.query
            ORDER BY ts_rank(f.content_tsv, q.query) DESC
            LIMIT %s
        )
        SELECT {colonnes_filtre},
               (1 - (f.embedding <=> %s::vector)) AS similarity,
               EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - f.last_accessed_at)) AS age_seconds,
               v.rank_vec, t.rank_fts
        FROM filtre f
        LEFT JOIN vectoriel v ON v.id = f.id
        LEFT JOIN plein_texte t ON t.id = f.id
        WHERE v.rank_vec IS NOT NULL OR t.rank_fts IS NOT NULL;
    """, (tenant, agent_id, memory_types,
          vector_str, vector_str, RETRIEVAL_CANDIDATES,
          query_text, RETRIEVAL_CANDIDATES,
          vector_str))
    return cur.fetchall()


def parse_embedding(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip('[]')
        if not val.strip():
            return []
        return [float(x) for x in val.split(',')]
    return []

# Modèles Pydantic
MemoryType = Literal["semantic", "episodic", "procedural", "working"]


class EventInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    session_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "sess_abc"})
    content: str = Field(..., min_length=1, max_length=12000, json_schema_extra={"example": "L'utilisateur demande à rédiger un email pro."})
    metadata: Dict[str, Any] = Field(default_factory=dict, max_length=100)
    # Clé de déduplication optionnelle : deux appels avec la même clé (même tenant)
    # ne créent qu'un seul événement.
    idempotency_key: Optional[str] = Field(default=None, max_length=128, json_schema_extra={"example": "evt-2026-07-15-001"})

class ContextConstraints(BaseModel):
    max_tokens: int = Field(default=1200, ge=1, le=8000)
    memory_types: List[MemoryType] = Field(default=["semantic", "episodic", "procedural", "working"], min_length=1, max_length=4)

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


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics. The reference Compose profile binds this endpoint to localhost."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@v1_router.post("/events", status_code=201)
def capture_event(event: EventInput, auth: Optional[AuthContext] = Depends(get_auth)):
    """
    Enregistre un événement brut et le publie dans le stream Redis (traitement asynchrone).
    Idempotent si `idempotency_key` est fourni.
    """
    tenant = resolve_tenant(auth)
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

    except Exception as e:
        logger.error(f"Erreur lors de la capture de l'événement : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@v1_router.post("/context/build")
def build_context(request: ContextRequest, auth: Optional[AuthContext] = Depends(get_auth)):
    """
    Assemble un paquet de contexte compact pour le LLM en fonction de la tâche.
    Implémente le module Q-EM (Quantum Entanglement Memory) :
    1. Superposition : Recherche sémantique par similarité vectorielle (pgvector).
    2. Intrication : Propagation d'activation via les liaisons 'entangled_with'.
    3. Interférence : Filtrage destructif des contradictions et redondances.
    4. Mesure : Collapse par densité de tokens pour maximiser l'utilité sous budget de tokens.
    """
    tenant = resolve_tenant(auth)
    start_time = time.perf_counter()
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Génération de l'embedding de la requête (fournisseur réel)
            query_vector = get_embedder().embed_one(request.query)
            vector_str = to_pgvector(query_vector)

            # 2. Superposition — recherche des candidats.
            # Deux chemins complémentaires : la similarité vectorielle ramène le
            # « sémantiquement proche », le plein texte rattrape les correspondances
            # LITTÉRALES (noms propres, dates, identifiants) que l'embedding manque.
            # Mesuré sur LOCOMO : 47 % des questions échouaient faute de rappel, quelle
            # que soit la stratégie de classement en aval.
            rows = _fetch_candidates(
                cur, vector_str, request.query, tenant,
                request.agent_id, request.constraints.memory_types,
            )

            # Rangs par chemin -> score de fusion RRF (indépendant des échelles de score).
            rrf_scores = {}
            if RETRIEVAL_HYBRID:
                rang_vectoriel = [str(r['id']) for r in sorted(
                    (r for r in rows if r['rank_vec'] is not None), key=lambda r: r['rank_vec'])]
                rang_plein_texte = [str(r['id']) for r in sorted(
                    (r for r in rows if r['rank_fts'] is not None), key=lambda r: r['rank_fts'])]
                rrf_scores = reciprocal_rank_fusion(
                    [rang_vectoriel, rang_plein_texte],
                    k=RRF_K, weights=[RRF_WEIGHT_VECTOR, RRF_WEIGHT_FTS],
                )
            meilleur_rrf = max(rrf_scores.values()) if rrf_scores else 0.0

            # Initialiser la structure des candidats
            candidates = {}
            for row in rows:
                mem_id = str(row['id'])
                sim = float(row['similarity'] or 0.0)
                sim_clipped = max(0.0, sim)
                # Décroissance temporelle : atténue la pertinence des mémoires anciennes
                # (demi-vie configurable, réactivée à chaque accès via last_accessed_at).
                # Seuil externalisé lu ici, calcul délégué au cœur pur (qem.py).
                recency_factor = compute_recency_factor(row['age_seconds'], QEM_RECENCY_HALFLIFE_DAYS)
                candidates[mem_id] = {
                    "id": mem_id,
                    "type": row['type'],
                    "subtype": row['subtype'],
                    "content": row['content'],
                    "confidence": float(row['confidence'] or 1.0),
                    "importance": float(row['importance'] or 0.5),
                    "last_accessed_at": row['last_accessed_at'],
                    "created_at": row['created_at'],
                    # Date du FAIT (≠ created_at) : préfixée au contenu par le collapse,
                    # sans quoi le LLM ne peut répondre à aucune question « quand… ».
                    "occurred_at": row['occurred_at'],
                    "embedding": parse_embedding(row['embedding']),
                    "similarity": sim_clipped,
                    "recency_factor": recency_factor,
                    # Pertinence de départ. En hybride, elle vient du rang FUSIONNÉ
                    # (normalisé sur le meilleur candidat) et non du seul cosinus : sans
                    # cela, un souvenir trouvé uniquement par le plein texte entrerait avec
                    # un score faible et serait éliminé par le collapse — le rappel gagné
                    # serait aussitôt reperdu.
                    "score": initial_score(
                        rrf_scores[mem_id] / meilleur_rrf if meilleur_rrf else sim_clipped,
                        recency_factor,
                    ) if (RETRIEVAL_HYBRID and mem_id in rrf_scores)
                    else initial_score(sim_clipped, recency_factor),
                }

            if not candidates:
                # Schéma complet (7 clés) même à vide, pour un contrat stable côté consommateur.
                CONTEXT_BUILDS.labels("empty").inc()
                return {
                    "context_packet": {"facts": [], "preferences": [], "episodes": [],
                                       "rules": [], "best_practices": [], "errors": [], "examples": []},
                    "token_estimate": 0,
                    "selected_memory_ids": [],
                    "trace_id": f"trace_{int(datetime.utcnow().timestamp())}",
                    "retrieval_trace": [] if request.explain else None,
                }

            # 3. Récupération des relations d'intrication et de contradiction
            candidate_ids = list(candidates.keys())
            rel_query = """
                SELECT source_memory_id, target_memory_id, relation_type, weight
                FROM relationships
                WHERE source_memory_id = ANY(%s::uuid[])
                   OR target_memory_id = ANY(%s::uuid[]);
            """
            cur.execute(rel_query, (candidate_ids, candidate_ids))
            relationships = cur.fetchall()

            # Récupérer les mémoires intriquées manquantes du graphe
            missing_ids = []
            for rel in relationships:
                src = str(rel['source_memory_id'])
                tgt = str(rel['target_memory_id'])
                if src in candidates and tgt not in candidates and tgt not in missing_ids:
                    missing_ids.append(tgt)
                elif tgt in candidates and src not in candidates and src not in missing_ids:
                    missing_ids.append(src)

            if missing_ids:
                cur.execute("""
                    SELECT id, type, subtype, content, confidence, importance, last_accessed_at, created_at, occurred_at, embedding::text
                    FROM memories
                    WHERE id = ANY(%s::uuid[]) AND status = 'active';
                """, (missing_ids,))
                for row in cur.fetchall():
                    mem_id = str(row['id'])
                    candidates[mem_id] = {
                        "id": mem_id,
                        "type": row['type'],
                        "subtype": row['subtype'],
                        "content": row['content'],
                        "confidence": float(row['confidence'] or 1.0),
                        "importance": float(row['importance'] or 0.5),
                        "last_accessed_at": row['last_accessed_at'],
                        "created_at": row['created_at'],
                        "occurred_at": row['occurred_at'],
                        "embedding": parse_embedding(row['embedding']),
                        "similarity": 0.0,
                        "score": 0.0
                    }

            # ── Algorithme Q-EM délégué au cœur pur (packages/core/synaptiq_core/qem.py) ──
            # Les seuils QEM_* restent lus côté API (os.getenv) et sont passés en paramètres.

            # 4. Intrication : propagation d'activation amortie ('entangled_with')
            propagate_entanglement(candidates, relationships, QEM_ENTANGLE_DAMPING, QEM_ENTANGLE_MAX_HOPS)

            # 5. Interférences destructives
            #    A. Contradictions / supersession (annule la plus ancienne)
            apply_contradictions(candidates, relationships)
            #    B. Redondances sémantiques (cosinus des embeddings > seuil)
            filter_redundancy(candidates, QEM_REDUNDANCY_THRESHOLD)

            # 6. Mesure : collapse glouton par densité d'utilité/token + routage 7 clés.
            context_packet, selected_ids, token_count = collapse_by_utility(
                candidates, request.constraints.max_tokens
            )
            max_tokens = request.constraints.max_tokens

            # 7. Enregistrement des statistiques d'accès
            if selected_ids:
                update_query = """
                    UPDATE memories 
                    SET access_count = access_count + 1, 
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ANY(%s::uuid[]);
                """
                cur.execute(update_query, (selected_ids,))
                conn.commit()

            # `context_packet` (7 clés) est déjà assemblé par collapse_by_utility.
            logger.info(f"Q-EM: Mesure achevée. {len(selected_ids)} mémoires sélectionnées. Tokens: {token_count}/{max_tokens}")
            
            CONTEXT_BUILDS.labels("success").inc()
            return {
                "context_packet": context_packet,
                "token_estimate": token_count,
                "selected_memory_ids": selected_ids,
                "trace_id": f"trace_{int(datetime.utcnow().timestamp())}",
                "retrieval_trace": [
                    {
                        "memory_id": memory_id,
                        "similarity": candidates[memory_id]["similarity"],
                        "recency_factor": candidates[memory_id].get("recency_factor", 0.0),
                        "score": candidates[memory_id]["score"],
                        "selection_reason": "selected_by_utility_under_token_budget",
                    }
                    for memory_id in selected_ids
                ] if request.explain else None,
            }
            
    except HTTPException:
        CONTEXT_BUILDS.labels("error").inc()
        raise
    except Exception as e:
        conn.rollback()
        CONTEXT_BUILDS.labels("error").inc()
        logger.error(f"Erreur lors de la construction du contexte : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
    finally:
        CONTEXT_BUILD_SECONDS.observe(time.perf_counter() - start_time)
        db_pool.putconn(conn)

class MemoryInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$", json_schema_extra={"example": "agent_sales_01"})
    type: MemoryType = Field(..., json_schema_extra={"example": "semantic"})
    subtype: Optional[str] = Field(None, max_length=50, json_schema_extra={"example": "preference"})
    content: str = Field(..., min_length=1, max_length=12000, json_schema_extra={"example": "Jimmy préfère les e-mails courts."})
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

@v1_router.post("/memories", status_code=201)
def create_memory(memory: MemoryInput, auth: Optional[AuthContext] = Depends(get_auth)):
    """
    Permet à un agent IA d'enregistrer directement un souvenir consolidé.
    """
    tenant = resolve_tenant(auth)
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
            handle_contradictions(cur, tenant, memory.agent_id, new_mem_dict, embedding)
            
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
            conn.commit()
            
            logger.info(f"Mémoire créée en direct par l'agent : {new_id}")
            return {
                "status": "created",
                "memory_id": str(new_id)
            }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la création de la mémoire : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
    finally:
        db_pool.putconn(conn)

class RetrieveRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    query: str = Field(..., min_length=1, max_length=8000)
    limit: int = Field(default=5, ge=1, le=100)
    memory_type: Optional[MemoryType] = None

@v1_router.post("/retrieve")
def retrieve_memories(request: RetrieveRequest, auth: Optional[AuthContext] = Depends(get_auth)):
    """
    Recherche HYBRIDE : similarité vectorielle (pgvector) fusionnée par RRF avec une
    recherche plein texte. Le vecteur ramène le sémantiquement proche, le plein texte les
    correspondances littérales (noms propres, dates, identifiants) qu'il manque.
    `RETRIEVAL_HYBRID=false` revient au classement purement vectoriel.
    """
    tenant = resolve_tenant(auth)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        query_vector = get_embedder().embed_one(request.query)
        vector_str = to_pgvector(query_vector)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            type_filter = "AND type = %s" if request.memory_type else ""
            filtre_params = [tenant, request.agent_id]
            if request.memory_type:
                filtre_params.append(request.memory_type)

            if not RETRIEVAL_HYBRID:
                cur.execute(f"""
                    SELECT id, type, subtype, content, confidence, importance, last_accessed_at,
                           occurred_at, (1 - (embedding <=> %s::vector)) AS similarity
                    FROM memories
                    WHERE tenant_id = %s AND agent_id = %s {type_filter} AND status = 'active'
                    ORDER BY similarity DESC
                    LIMIT %s;
                """, (vector_str, *filtre_params, request.limit))
                return {"memories": cur.fetchall()}

            # Sur-échantillonner chaque chemin avant fusion : la bonne réponse peut être
            # 12e en vectoriel et 2e en plein texte, il faut donc regarder au-delà de
            # `limit` dans les deux avant de trancher.
            pool_size = max(request.limit * 4, RETRIEVAL_CANDIDATES)
            cur.execute(f"""
                WITH filtre AS (
                    SELECT * FROM memories
                    WHERE tenant_id = %s AND agent_id = %s {type_filter} AND status = 'active'
                ),
                vectoriel AS (
                    SELECT id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS rank_vec
                    FROM filtre ORDER BY embedding <=> %s::vector LIMIT %s
                ),
                plein_texte AS (
                    SELECT f.id,
                           row_number() OVER (ORDER BY ts_rank(f.content_tsv, q.query) DESC) AS rank_fts
                    FROM filtre f, websearch_to_tsquery('simple', %s) AS q(query)
                    WHERE f.content_tsv @@ q.query
                    ORDER BY ts_rank(f.content_tsv, q.query) DESC
                    LIMIT %s
                )
                SELECT f.id, f.type, f.subtype, f.content, f.confidence, f.importance,
                       f.last_accessed_at, f.occurred_at,
                       (1 - (f.embedding <=> %s::vector)) AS similarity,
                       v.rank_vec, t.rank_fts
                FROM filtre f
                LEFT JOIN vectoriel v ON v.id = f.id
                LEFT JOIN plein_texte t ON t.id = f.id
                WHERE v.rank_vec IS NOT NULL OR t.rank_fts IS NOT NULL;
            """, (*filtre_params, vector_str, vector_str, pool_size,
                  request.query, pool_size, vector_str))
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
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur de recherche de souvenirs : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
    finally:
        db_pool.putconn(conn)


@v1_router.delete("/memories")
def purge_memories(
    agent_id: Optional[str] = None,
    auth: Optional[AuthContext] = Depends(get_auth),
):
    """
    Purge RGPD : supprime les mémoires (et événements) de l'instance.
    Scopé au tenant résolu côté serveur ; les relationships sont supprimées en cascade (FK).
    Filtre optionnel par `agent_id`.
    """
    tenant = resolve_tenant(auth)
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
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la purge RGPD : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
    finally:
        db_pool.putconn(conn)


app.include_router(v1_router, prefix="/v1", tags=["v1"])
app.include_router(v1_router, include_in_schema=False)


