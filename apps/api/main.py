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
from contextlib import contextmanager, asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# Logique partagée (embeddings pluggables + gouvernance), plus d'import depuis le worker
from synaptiq_core import get_embedder, to_pgvector, handle_contradictions

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
# Au-delà de ce cosinus entre deux candidats, le moins prioritaire est filtré (redondance).
QEM_REDUNDANCY_THRESHOLD = float(os.getenv("QEM_REDUNDANCY_THRESHOLD", "0.75"))
# Décroissance temporelle : demi-vie (en jours) du score de récence. Une mémoire non
# ré-accédée voit sa pertinence divisée par 2 tous les N jours. 0 (ou négatif) = désactivé.
QEM_RECENCY_HALFLIFE_DAYS = float(os.getenv("QEM_RECENCY_HALFLIFE_DAYS", "90"))

db_pool: Optional[pg_pool.ThreadedConnectionPool] = None
redis_client = None


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


app = FastAPI(title="SynaptiQ API", version="0.1.0", lifespan=lifespan)

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
AUTH_REQUIRED = os.getenv("SYNAPTIQ_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")


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
class EventInput(BaseModel):
    agent_id: str = Field(..., example="agent_sales_01")
    session_id: str = Field(..., example="sess_abc")
    content: str = Field(..., example="L'utilisateur demande à rédiger un email pro.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Clé de déduplication optionnelle : deux appels avec la même clé (même tenant)
    # ne créent qu'un seul événement.
    idempotency_key: Optional[str] = Field(default=None, example="evt-2026-07-15-001")

class ContextConstraints(BaseModel):
    max_tokens: int = Field(default=1200)
    memory_types: List[str] = Field(default=["semantic", "episodic", "procedural", "working"])

class ContextRequest(BaseModel):
    agent_id: str = Field(..., example="agent_sales_01")
    session_id: str = Field(..., example="sess_abc")
    task: str = Field(..., example="Rédiger un email de suivi")
    query: str = Field(..., example="Style d'écriture concis de Jimmy")
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)

@app.get("/health")
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

@app.post("/events", status_code=201)
def capture_event(event: EventInput, auth: Optional[AuthContext] = Depends(get_auth)):
    """
    Enregistre un événement brut et le publie dans le stream Redis (traitement asynchrone).
    Idempotent si `idempotency_key` est fourni.
    """
    tenant = resolve_tenant(auth)
    r = get_redis_client()

    # Garde d'idempotence : SET NX pose un verrou ; si la clé existe, c'est un doublon.
    idem_k = None
    if event.idempotency_key:
        idem_k = f"synaptiq:idem:{tenant}:{event.idempotency_key}"
        if not r.set(idem_k, "pending", nx=True, ex=IDEMPOTENCY_TTL):
            existing = r.get(idem_k)
            logger.info("Événement idempotent ignoré (clé=%s).", event.idempotency_key)
            return {"status": "duplicate", "event_id": existing}

    try:
        with get_conn() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO events (tenant_id, agent_id, session_id, content, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id, created_at;
                        """,
                        (tenant, event.agent_id, event.session_id,
                         event.content, json.dumps(event.metadata)),
                    )
                    result = cur.fetchone()
                    conn.commit()
            except Exception:
                conn.rollback()
                raise

        event_id = str(result['id'])
        created_at = result['created_at'].isoformat()

        # Publication dans le stream Redis (consommé par le worker via consumer group)
        payload = {
            "id": event_id,
            "tenant_id": tenant,
            "agent_id": event.agent_id,
            "session_id": event.session_id,
            "content": event.content,
            "metadata": json.dumps(event.metadata),
            "created_at": created_at,
        }
        r.xadd(EVENT_STREAM, {"data": json.dumps(payload)})

        if idem_k:
            r.set(idem_k, event_id, ex=IDEMPOTENCY_TTL)

        logger.info(f"Événement {event_id} capturé et publié dans le stream.")
        return {"status": "captured", "event_id": event_id, "created_at": created_at}

    except Exception as e:
        # Libérer le verrou d'idempotence pour permettre un nouvel essai
        if idem_k:
            r.delete(idem_k)
        logger.error(f"Erreur lors de la capture de l'événement : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.post("/context/build")
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
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Génération de l'embedding de la requête (fournisseur réel)
            query_vector = get_embedder().embed_one(request.query)
            vector_str = to_pgvector(query_vector)
            
            # 2. Superposition (Recherche sémantique des candidats)
            # Tri par similarité cosinus décroissante ( pgvector <=> distance cosinus )
            query = """
                SELECT id, type, subtype, content, confidence, importance, last_accessed_at, created_at, embedding::text,
                       (1 - (embedding <=> %s::vector)) AS similarity,
                       EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_accessed_at)) AS age_seconds
                FROM memories
                WHERE tenant_id = %s
                  AND agent_id = %s
                  AND type = ANY(%s)
                  AND status = 'active'
                ORDER BY similarity DESC
                LIMIT 50;
            """
            
            cur.execute(query, (
                vector_str,
                tenant,
                request.agent_id,
                request.constraints.memory_types
            ))
            
            rows = cur.fetchall()
            
            # Initialiser la structure des candidats
            candidates = {}
            for row in rows:
                mem_id = str(row['id'])
                sim = float(row['similarity'] or 0.0)
                sim_clipped = max(0.0, sim)
                # Décroissance temporelle : atténue la pertinence des mémoires anciennes
                # (demi-vie configurable). Réactivée à chaque accès (last_accessed_at).
                recency_factor = 1.0
                if QEM_RECENCY_HALFLIFE_DAYS > 0:
                    age_days = float(row['age_seconds'] or 0.0) / 86400.0
                    recency_factor = 0.5 ** (age_days / QEM_RECENCY_HALFLIFE_DAYS)
                candidates[mem_id] = {
                    "id": mem_id,
                    "type": row['type'],
                    "subtype": row['subtype'],
                    "content": row['content'],
                    "confidence": float(row['confidence'] or 1.0),
                    "importance": float(row['importance'] or 0.5),
                    "last_accessed_at": row['last_accessed_at'],
                    "created_at": row['created_at'],
                    "embedding": parse_embedding(row['embedding']),
                    "similarity": sim_clipped,
                    "recency_factor": recency_factor,
                    # Le score de départ pondère la similarité par la récence.
                    "score": sim_clipped * recency_factor
                }

            if not candidates:
                # Schéma complet (7 clés) même à vide, pour un contrat stable côté consommateur.
                return {
                    "context_packet": {"facts": [], "preferences": [], "episodes": [],
                                       "rules": [], "best_practices": [], "errors": [], "examples": []},
                    "token_estimate": 0,
                    "selected_memory_ids": [],
                    "trace_id": f"trace_{int(datetime.utcnow().timestamp())}"
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
                    SELECT id, type, subtype, content, confidence, importance, last_accessed_at, created_at, embedding::text
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
                        "embedding": parse_embedding(row['embedding']),
                        "similarity": 0.0,
                        "score": 0.0
                    }

            # 4. Propagation de l'activation (Intrication)
            for rel in relationships:
                if rel['relation_type'] == 'entangled_with':
                    src = str(rel['source_memory_id'])
                    tgt = str(rel['target_memory_id'])
                    weight = float(rel['weight'] or 1.0)
                    
                    # Propagation bidirectionnelle amortie (QEM_ENTANGLE_DAMPING)
                    if src in candidates and tgt in candidates:
                        candidates[tgt]['score'] += candidates[src]['similarity'] * weight * QEM_ENTANGLE_DAMPING
                        candidates[src]['score'] += candidates[tgt]['similarity'] * weight * QEM_ENTANGLE_DAMPING

            # 5. Interférences Quantiques (Filtre destructif)
            # A. Contradictions et Remplacements (supersedes)
            for rel in relationships:
                if rel['relation_type'] in ('contradicts', 'supersedes_by'):
                    src = str(rel['source_memory_id'])
                    tgt = str(rel['target_memory_id'])
                    if src in candidates and tgt in candidates:
                        # Annuler le score de la plus ancienne ou de moindre confiance
                        c_src = candidates[src]
                        c_tgt = candidates[tgt]
                        if c_src['created_at'] < c_tgt['created_at']:
                            c_src['score'] = 0.0
                            logger.info(f"Q-EM: Interférence destructive (contradiction) : {src} annulé par {tgt}")
                        else:
                            c_tgt['score'] = 0.0
                            logger.info(f"Q-EM: Interférence destructive (contradiction) : {tgt} annulé par {src}")

            # B. Redondances sémantiques (similarité cosinus des embeddings > 0.75)
            active_ids = [cid for cid, c in candidates.items() if c['score'] > 0.0]
            # Trier pour conserver les plus importants ou récents en priorité
            active_ids.sort(key=lambda cid: (candidates[cid]['importance'], candidates[cid]['created_at']), reverse=True)
            
            for i in range(len(active_ids)):
                id_i = active_ids[i]
                if candidates[id_i]['score'] == 0.0:
                    continue
                emb_i = candidates[id_i]['embedding']
                
                for j in range(i + 1, len(active_ids)):
                    id_j = active_ids[j]
                    if candidates[id_j]['score'] == 0.0:
                        continue
                    emb_j = candidates[id_j]['embedding']
                    
                    if emb_i and emb_j:
                        # Similarité cosinus via produit scalaire (puisque normalisés)
                        cosine_sim = sum(x * y for x, y in zip(emb_i, emb_j))
                        if cosine_sim > QEM_REDUNDANCY_THRESHOLD:
                            candidates[id_j]['score'] = 0.0
                            logger.info(f"Q-EM: Interférence destructive (redondance sim={cosine_sim:.2f}) : {id_j} annulé au profit de {id_i}")

            # 6. Mesure (Collapse du contexte par densité d'utilité/token)
            collapsed_candidates = []
            for mem_id, c in candidates.items():
                if c['score'] > 0.0:
                    tokens = max(1, int(len(c['content'].split()) * 1.3))
                    utility_density = c['score'] / tokens
                    collapsed_candidates.append({
                        "id": mem_id,
                        "type": c['type'],
                        "content": c['content'],
                        "tokens": tokens,
                        "utility_density": utility_density
                    })

            # Trier par densité d'utilité par token décroissante
            collapsed_candidates.sort(key=lambda x: x['utility_density'], reverse=True)

            facts = []
            preferences = []
            episodes = []
            rules = []
            best_practices = []
            errors = []
            examples = []
            selected_ids = []
            token_count = 0
            max_tokens = request.constraints.max_tokens

            # Collapse glouton sous contrainte de jetons
            for c in collapsed_candidates:
                if token_count + c['tokens'] <= max_tokens:
                    selected_ids.append(c['id'])
                    token_count += c['tokens']
                    
                    m_type = c['type']
                    m_subtype = c.get('subtype')
                    content = c['content']
                    
                    if m_type == 'semantic':
                        if m_subtype == 'preference':
                            preferences.append(content)
                        else:
                            facts.append(content)
                    elif m_type == 'episodic':
                        episodes.append(content)
                    elif m_type == 'procedural':
                        if m_subtype == 'coding_best_practices':
                            best_practices.append(content)
                        elif m_subtype == 'code_error_resolution':
                            errors.append(content)
                        else:
                            rules.append(content)
                    elif m_type == 'working':
                        examples.append(content)
                else:
                    logger.debug(f"Q-EM: Hors budget pour {c['id']} (tokens={c['tokens']}, restant={max_tokens - token_count})")

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

            # Construction finale du paquet
            context_packet = {
                "facts": facts,
                "preferences": preferences,
                "episodes": episodes,
                "rules": rules,
                "best_practices": best_practices,
                "errors": errors,
                "examples": examples
            }
            
            logger.info(f"Q-EM: Mesure achevée. {len(selected_ids)} mémoires sélectionnées. Tokens: {token_count}/{max_tokens}")
            
            return {
                "context_packet": context_packet,
                "token_estimate": token_count,
                "selected_memory_ids": selected_ids,
                "trace_id": f"trace_{int(datetime.utcnow().timestamp())}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la construction du contexte : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
    finally:
        db_pool.putconn(conn)

class MemoryInput(BaseModel):
    agent_id: str = Field(..., example="agent_sales_01")
    type: str = Field(..., example="semantic")
    subtype: Optional[str] = Field(None, example="preference")
    content: str = Field(..., example="Jimmy préfère les e-mails courts.")
    confidence: float = Field(default=1.0)
    importance: float = Field(default=0.5)

@app.post("/memories", status_code=201)
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
    agent_id: str
    query: str
    limit: int = 5
    memory_type: Optional[str] = None

@app.post("/retrieve")
def retrieve_memories(request: RetrieveRequest, auth: Optional[AuthContext] = Depends(get_auth)):
    """
    Recherche SÉMANTIQUE vectorielle (pgvector) : embed la requête puis trie les
    souvenirs actifs par similarité cosinus décroissante (opérateur <=>).
    Le paramètre `query` pilote désormais réellement le classement.
    """
    tenant = resolve_tenant(auth)
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Pool PostgreSQL non initialisé")
    conn = db_pool.getconn()
    try:
        query_vector = get_embedder().embed_one(request.query)
        vector_str = to_pgvector(query_vector)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            params: list = [vector_str, tenant, request.agent_id]
            type_filter = ""
            if request.memory_type:
                type_filter = "AND type = %s"
                params.append(request.memory_type)
            params.append(request.limit)

            query = f"""
                SELECT id, type, subtype, content, confidence, importance, last_accessed_at,
                       (1 - (embedding <=> %s::vector)) AS similarity
                FROM memories
                WHERE tenant_id = %s
                  AND agent_id = %s
                  {type_filter}
                  AND status = 'active'
                ORDER BY similarity DESC
                LIMIT %s;
            """
            cur.execute(query, tuple(params))
            results = cur.fetchall()
            return {"memories": results}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur de recherche de souvenirs : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
    finally:
        db_pool.putconn(conn)


@app.delete("/memories")
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

