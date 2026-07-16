import os
import sys
import json
import time
import logging
import re
import psycopg2
import redis
import requests
from dotenv import load_dotenv

# Rendre le package partagé packages/core importable (dev local hors conteneur)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_root, os.path.join(_root, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synaptiq_core import get_embedder, to_pgvector, handle_contradictions
from synaptiq_core.embeddings import generate_mock_embedding  # noqa: F401 (compat rétro tests)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("synaptiq-worker")

# Chargement des variables d'environnement depuis le .env RACINE (source unique)
load_dotenv(os.path.join(_root, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free")

def call_llm_extractor(event_content: str) -> dict:
    """
    Analyse l'événement brut pour en extraire des mémoires consolidées.
    Supporte l'extraction par LLM (OpenRouter) et une heuristique locale (mock) si aucune clé n'est fournie.
    """
    # 1. Règle d'extraction locale par défaut (Heuristique/Mock)
    # Très utile pour le test hors-ligne ou si aucune clé n'est fournie
    if LLM_PROVIDER == "mock" or not LLM_API_KEY or "your_api_key" in LLM_API_KEY:
        logger.info("Utilisation de l'extracteur heuristique local (sans LLM).")
        
        # Détection des erreurs de code et des résolutions
        error_match = re.search(
            r"(?:erreur|bug|exception|traceback|crash|failed|plantage|corrigé|résolu|warning)\s+([^.]+)",
            event_content,
            re.IGNORECASE
        )
        if error_match:
            return {
                "extracted": True,
                "type": "procedural",
                "subtype": "code_error_resolution",
                "content": f"Résolution de bug/erreur détectée : {error_match.group(0).strip()}",
                "summary": "Résolution d'erreur de code",
                "confidence": 0.85,
                "importance": 0.7
            }
            
        # Détection des bonnes pratiques et playbooks
        best_practice_match = re.search(
            r"(?:bonne pratique|toujours|ne jamais|règle de conception|recommandation|best practice)\s+([^.]+)",
            event_content,
            re.IGNORECASE
        )
        if best_practice_match:
            return {
                "extracted": True,
                "type": "procedural",
                "subtype": "coding_best_practices",
                "content": f"Directive de conception/code : {best_practice_match.group(0).strip()}",
                "summary": "Directive de conception de code",
                "confidence": 0.9,
                "importance": 0.8
            }
            
        # Détection basique des préférences de l'utilisateur
        pref_match = re.search(
            r"(?:je préfère|je veux|ma préférence|utilise plutôt|ne fais pas|écris en)\s+([^.]+)", 
            event_content, 
            re.IGNORECASE
        )
        if pref_match:
            extracted_fact = f"L'utilisateur a spécifié une préférence : {pref_match.group(1).strip()}"
            return {
                "extracted": True,
                "type": "semantic",
                "subtype": "preference",
                "content": extracted_fact,
                "summary": "Préférence utilisateur extraite",
                "confidence": 0.9,
                "importance": 0.8
            }
            
        # Si aucun pattern, on extrait comme un épisode générique
        return {
            "extracted": True,
            "type": "episodic",
            "subtype": "interaction",
            "content": f"Interaction : {event_content}",
            "summary": "Épisode d'interaction",
            "confidence": 0.8,
            "importance": 0.4
        }

    # 2. Extraction via LLM (OpenRouter / APIs)
    logger.info(f"Appel du LLM ({LLM_PROVIDER} : {LLM_MODEL}) pour l'extraction de mémoire.")
    
    prompt = f"""
    Tu es le module d'extraction de mémoire de SynaptiQ. Ton rôle est d'analyser l'interaction suivante et d'extraire des éléments de mémoire durable pour l'agent.
    
    Interaction à analyser :
    "{event_content}"
    
    Tu devez classifier cette mémoire selon l'un des types suivants :
    1. "procedural" (sous-type "code_error_resolution" pour des erreurs système, tracebacks et leurs résolutions, ou "coding_best_practices" pour des playbooks, règles d'architecture et bonnes pratiques de programmation).
    2. "semantic" (sous-type "preference" pour les choix explicites de l'utilisateur, ou "fact" pour des faits généraux stables).
    3. "episodic" (sous-type "interaction" pour un résumé historique d'une action menée à bien ou d'une étape projet importante).
    
    Tu devez renvoyer UN UNIQUE objet JSON contenant :
    {{
      "type": "semantic", "episodic" ou "procedural",
      "subtype": "code_error_resolution", "coding_best_practices", "preference", "fact" ou "interaction",
      "content": "Le souvenir extrait rédigé de manière claire, concise et affirmative à la troisième personne (ex: 'L'agent ne doit pas importer de librairies réseau au niveau global sous Windows')",
      "summary": "Un titre ou résumé très court du souvenir (ex: 'Windows Multiprocessing Fix')",
      "confidence": un float entre 0.0 et 1.0 (degré de certitude),
      "importance": un float entre 0.0 et 1.0 (importance opérationnelle pour l'agent)
    }}
    
    Renvoie UNIQUEMENT le JSON brut, aucun autre texte.
    """
    
    try:
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Tu es un extracteur de mémoire de précision qui répond uniquement en JSON."},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        response_json = response.json()
        raw_content = response_json["choices"][0]["message"]["content"].strip()
        
        # Nettoyer les balises de code markdown si le LLM en a rajouté
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()
            
        data = json.loads(raw_content)
        return {
            "extracted": True,
            "type": data.get("type", "semantic"),
            "subtype": data.get("subtype", "fact"),
            "content": data.get("content", event_content),
            "summary": data.get("summary", "Mémoire extraite"),
            "confidence": float(data.get("confidence", 0.9)),
            "importance": float(data.get("importance", 0.5))
        }
    except Exception as e:
        logger.error(f"Échec de l'extraction LLM : {e}. Utilisation du fallback heuristique.")
        # Fallback local en cas d'erreur API
        return {
            "extracted": True,
            "type": "episodic",
            "subtype": "interaction",
            "content": f"Interaction (brute suite erreur LLM) : {event_content}",
            "summary": "Épisode d'interaction",
            "confidence": 0.5,
            "importance": 0.3
        }

def process_event(event: dict) -> bool:
    """
    Traite un événement brut extrait de Redis.
    """
    tenant_id = event['tenant_id']
    agent_id = event['agent_id']
    content = event['content']
    event_id = event['id']
    
    logger.info(f"Traitement de l'événement {event_id} pour l'agent {agent_id}...")
    
    # 1. Extraction de la mémoire
    memory_data = call_llm_extractor(content)
    
    # 2. Génération d'embedding (fournisseur réel configuré : LM Studio par défaut)
    embedding = get_embedder().embed_one(memory_data['content'])
    
    # 3. Écriture en base de données avec gestion des contradictions et des intrications
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Gestion des contradictions (archivage scopé sémantiquement)
            handle_contradictions(cur, tenant_id, agent_id, memory_data, embedding)
            
            # Insertion de la nouvelle mémoire
            insert_query = """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, summary, embedding, confidence, importance, provenance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            provenance = {
                "source": "event",
                "event_id": event_id
            }
            
            cur.execute(insert_query, (
                tenant_id,
                agent_id,
                memory_data['type'],
                memory_data['subtype'],
                memory_data['content'],
                memory_data['summary'],
                embedding,
                memory_data['confidence'],
                memory_data['importance'],
                json.dumps(provenance)
            ))
            
            new_mem_id = cur.fetchone()[0]
            logger.info(f"Nouvelle mémoire consolidée créée avec l'ID {new_mem_id} ({memory_data['type']}/{memory_data['subtype']}).")
            
            # 4. Graphe d'intrication sémantique automatique (Q-EM)
            # Si c'est une règle, une erreur ou une bonne pratique, on cherche sémantiquement des souvenirs associés pour créer des liaisons
            if memory_data['type'] == 'procedural' or memory_data['subtype'] == 'preference':
                embedding_str = to_pgvector(embedding)
                find_rel_query = """
                    SELECT id, type, subtype, (1 - (embedding <=> %s::vector)) AS similarity
                    FROM memories
                    WHERE tenant_id = %s
                      AND agent_id = %s
                      AND id != %s
                      AND status = 'active'
                    ORDER BY similarity DESC
                    LIMIT 3;
                """
                cur.execute(find_rel_query, (
                    embedding_str,
                    tenant_id,
                    agent_id,
                    new_mem_id
                ))
                related_rows = cur.fetchall()
                for rel_row in related_rows:
                    similarity = float(rel_row[3] or 0.0)
                    if similarity > 0.7:  # Seuil d'intrication sémantique
                        target_id = rel_row[0]
                        relation_type = "entangled_with"
                        
                        # Règle d'intrication : une bonne pratique résout/remplace une erreur associée
                        if memory_data['subtype'] == 'coding_best_practices' and rel_row[2] == 'code_error_resolution':
                            relation_type = "supersedes_by"
                        elif memory_data['subtype'] == 'code_error_resolution' and rel_row[2] == 'coding_best_practices':
                            # Inverser la relation : la cible remplace la source
                            relation_type = "supersedes_by"
                            
                        # Insérer la relation
                        insert_rel_query = """
                            INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING;
                        """
                        # Si relation inversée, inverser les arguments
                        if relation_type == "supersedes_by" and memory_data['subtype'] == 'code_error_resolution':
                            cur.execute(insert_rel_query, (target_id, new_mem_id, relation_type, similarity))
                        else:
                            cur.execute(insert_rel_query, (new_mem_id, target_id, relation_type, similarity))
                        
                        logger.info(f"Intrication Q-EM établie : {new_mem_id} --({relation_type})--> {target_id} (sim={similarity:.2f})")
            
            conn.commit()
            return True
            
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
        logger.error(f"Erreur SQL lors de l'enregistrement de la mémoire : {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# ─── File d'événements : Redis Streams (consumer group + ACK + DLQ) ───
STREAM = os.getenv("EVENT_STREAM", "synaptiq:events")
GROUP = os.getenv("EVENT_GROUP", "synaptiq-workers")
DLQ = os.getenv("EVENT_DLQ", "synaptiq:events:dlq")
CONSUMER = f"worker-{os.getpid()}"
MAX_DELIVERIES = int(os.getenv("EVENT_MAX_DELIVERIES", "5"))
IDLE_RECLAIM_MS = int(os.getenv("EVENT_IDLE_RECLAIM_MS", "30000"))


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


def main():
    logger.info("SynaptiQ Memory Worker démarré (consumer=%s)...", CONSUMER)
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

    # Boucle de consommation via consumer group (XREADGROUP bloquant, ACK explicite)
    while True:
        try:
            resp = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=10, block=5000)
            if not resp:
                # Aucun nouveau message : on tente de reprendre les pending bloqués
                _reclaim(r)
                continue
            for _stream, messages in resp:
                for msg_id, fields in messages:
                    _handle(r, msg_id, fields)
        except KeyboardInterrupt:
            logger.info("Arrêt du worker par l'utilisateur.")
            break
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale du worker : {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
