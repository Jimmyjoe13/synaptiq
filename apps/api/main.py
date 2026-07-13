import sys
import os

# Ajouter la racine du projet au sys.path pour résoudre les imports absolus du monorepo
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("synaptiq-api")

# Chargement des variables d'environnement
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

app = FastAPI(title="SynaptiQ API", version="0.1.0")

# Connexions aux services
db_conn = None
redis_client = None

def get_db_connection():
    global db_conn
    if db_conn is None or db_conn.closed != 0:
        try:
            db_conn = psycopg2.connect(DATABASE_URL)
            logger.info("Connexion établie avec PostgreSQL.")
        except Exception as e:
            logger.error(f"Erreur de connexion PostgreSQL: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")
    return db_conn

def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            logger.info("Connexion établie avec Redis.")
        except Exception as e:
            logger.error(f"Erreur de connexion Redis: {e}")
            raise HTTPException(status_code=500, detail="Redis connection failed")
    return redis_client

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
    tenant_id: str = Field(..., example="org_01")
    agent_id: str = Field(..., example="agent_sales_01")
    session_id: str = Field(..., example="sess_abc")
    content: str = Field(..., example="L'utilisateur demande à rédiger un email pro.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ContextConstraints(BaseModel):
    max_tokens: int = Field(default=1200)
    memory_types: List[str] = Field(default=["semantic", "episodic", "procedural", "working"])

class ContextRequest(BaseModel):
    tenant_id: str = Field(..., example="org_01")
    agent_id: str = Field(..., example="agent_sales_01")
    session_id: str = Field(..., example="sess_abc")
    task: str = Field(..., example="Rédiger un email de suivi")
    query: str = Field(..., example="Style d'écriture concis de Jimmy")
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)

@app.on_event("startup")
def startup_event():
    # S'assurer que les connexions sont prêtes au démarrage
    try:
        get_db_connection()
        get_redis_client()
    except Exception as e:
        logger.warning(f"Impossible d'établir toutes les connexions au démarrage : {e}")

@app.get("/health")
def health_check():
    db_status = "healthy"
    redis_status = "healthy"
    
    try:
        conn = get_db_connection()
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
def capture_event(event: EventInput):
    """
    Enregistre un événement brut de l'agent et le publie dans Redis pour traitement asynchrone.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Insertion dans PostgreSQL
            query = """
                INSERT INTO events (tenant_id, agent_id, session_id, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, created_at;
            """
            cur.execute(query, (
                event.tenant_id,
                event.agent_id,
                event.session_id,
                event.content,
                json.dumps(event.metadata)
            ))
            result = cur.fetchone()
            conn.commit()
            
            event_id = str(result['id'])
            created_at = result['created_at'].isoformat()
            
            # Publication dans la queue Redis pour le worker
            r = get_redis_client()
            event_payload = {
                "id": event_id,
                "tenant_id": event.tenant_id,
                "agent_id": event.agent_id,
                "session_id": event.session_id,
                "content": event.content,
                "metadata": json.dumps(event.metadata),
                "created_at": created_at
            }
            # Utilisation d'une liste Redis simple comme FIFO Queue pour la v0
            r.rpush("synaptiq:event_queue", json.dumps(event_payload))
            
            logger.info(f"Événement {event_id} capturé et empilé dans Redis.")
            return {
                "status": "captured",
                "event_id": event_id,
                "created_at": created_at
            }
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la capture de l'événement : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/build")
def build_context(request: ContextRequest):
    """
    Assemble un paquet de contexte compact pour le LLM en fonction de la tâche.
    Implémente le module Q-EM (Quantum Entanglement Memory) :
    1. Superposition : Recherche sémantique par similarité vectorielle (pgvector).
    2. Intrication : Propagation d'activation via les liaisons 'entangled_with'.
    3. Interférence : Filtrage destructif des contradictions et redondances.
    4. Mesure : Collapse par densité de tokens pour maximiser l'utilité sous budget de tokens.
    """
    from apps.worker.worker import generate_mock_embedding
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Génération de l'embedding de la requête
            query_vector = generate_mock_embedding(request.query)
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
            # 2. Superposition (Recherche sémantique des candidats)
            # Tri par similarité cosinus décroissante ( pgvector <=> distance cosinus )
            query = """
                SELECT id, type, subtype, content, confidence, importance, last_accessed_at, created_at, embedding::text,
                       (1 - (embedding <=> %s::vector)) AS similarity
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
                request.tenant_id,
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
                    "score": sim_clipped
                }

            if not candidates:
                return {
                    "context_packet": {"facts": [], "episodes": [], "rules": [], "examples": []},
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
                    
                    # Propagation bidirectionnelle amortie (0.5)
                    if src in candidates and tgt in candidates:
                        candidates[tgt]['score'] += candidates[src]['similarity'] * weight * 0.5
                    if tgt in candidates and src in candidates:
                        candidates[src]['score'] += candidates[tgt]['similarity'] * weight * 0.5

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
                        if cosine_sim > 0.75:
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
            
    except Exception as e:
        logger.error(f"Erreur lors de la construction du contexte : {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MemoryInput(BaseModel):
    tenant_id: str = Field(..., example="org_01")
    agent_id: str = Field(..., example="agent_sales_01")
    type: str = Field(..., example="semantic")
    subtype: Optional[str] = Field(None, example="preference")
    content: str = Field(..., example="Jimmy préfère les e-mails courts.")
    confidence: float = Field(default=1.0)
    importance: float = Field(default=0.5)

@app.post("/memories", status_code=201)
def create_memory(memory: MemoryInput):
    """
    Permet à un agent IA d'enregistrer directement un souvenir consolidé.
    """
    from apps.worker.worker import generate_mock_embedding, handle_contradictions
    conn = get_db_connection()
    try:
        embedding = generate_mock_embedding(memory.content)
        with conn.cursor() as cur:
            # Gestion des contradictions
            new_mem_dict = {
                "type": memory.type,
                "subtype": memory.subtype,
                "content": memory.content
            }
            handle_contradictions(cur, memory.tenant_id, memory.agent_id, new_mem_dict)
            
            # Insertion
            query = """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active')
                RETURNING id;
            """
            cur.execute(query, (
                memory.tenant_id,
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
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la création de la mémoire : {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RetrieveRequest(BaseModel):
    tenant_id: str
    agent_id: str
    query: str
    limit: int = 5
    memory_type: Optional[str] = None

@app.post("/retrieve")
def retrieve_memories(request: RetrieveRequest):
    """
    Permet à un agent IA de rechercher sémantiquement dans ses souvenirs.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if request.memory_type:
                query = """
                    SELECT id, type, subtype, content, confidence, importance, last_accessed_at
                    FROM memories
                    WHERE tenant_id = %s
                      AND agent_id = %s
                      AND type = %s
                      AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                cur.execute(query, (request.tenant_id, request.agent_id, request.memory_type, request.limit))
            else:
                query = """
                    SELECT id, type, subtype, content, confidence, importance, last_accessed_at
                    FROM memories
                    WHERE tenant_id = %s
                      AND agent_id = %s
                      AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                cur.execute(query, (request.tenant_id, request.agent_id, request.limit))
                
            results = cur.fetchall()
            return {"memories": results}
    except Exception as e:
        logger.error(f"Erreur de recherche de souvenirs : {e}")
        raise HTTPException(status_code=500, detail=str(e))

