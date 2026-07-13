import os
import json
import time
import logging
import re
import hashlib
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import requests
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("synaptiq-worker")

# Chargement du fichier .env (on cherche d'abord localement, puis à côté dans apps/api/)
load_dotenv()
if not os.getenv("DATABASE_URL"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../api/.env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free")

# Fonction de mock d'embedding (dimension 384) pour all-MiniLM-L6-v2
def generate_mock_embedding(text: str) -> list:
    """
    Génère un vecteur déterministe normalisé de 384 floats basé sur le hash du texte.
    Pratique pour le test et le fonctionnement v0 hors-ligne.
    """
    sha = hashlib.sha256(text.encode('utf-8')).digest()
    vector = []
    for i in range(384):
        # Utiliser les octets du hash de manière circulaire
        val = sha[i % 32]
        # Mapper l'octet (0-255) vers une valeur entre -1.0 et 1.0
        val_normalized = (val / 127.5) - 1.0
        vector.append(val_normalized)
        
    # Normalisation du vecteur pour la similarité cosinus
    norm = sum(x**2 for x in vector)**0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector

def call_llm_extractor(event_content: str) -> dict:
    """
    Analyse l'événement brut pour en extraire des mémoires consolidées.
    Supporte l'extraction par LLM (OpenRouter) et une heuristique locale (mock) si aucune clé n'est fournie.
    """
    # 1. Règle d'extraction locale par défaut (Heuristique/Mock)
    # Très utile pour le test hors-ligne ou si aucune clé n'est fournie
    if LLM_PROVIDER == "mock" or not LLM_API_KEY or "your_api_key" in LLM_API_KEY:
        logger.info("Utilisation de l'extracteur heuristique local (sans LLM).")
        
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
            
        # Si aucun pattern de préférence, on extrait comme un épisode générique
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
    
    Tu dois renvoyer UN UNIQUE objet JSON contenant :
    {{
      "type": "semantic" (pour des faits, préférences ou connaissances stables) ou "episodic" (pour un résumé d'action/résultat) ou "procedural" (règles et playbooks),
      "subtype": "preference", "fact", "rule" ou "interaction",
      "content": "Le fait extrait rédigé de manière claire et concise à la troisième personne",
      "summary": "Un titre ou résumé très court du fait",
      "confidence": un float entre 0.0 et 1.0 (degré de certitude),
      "importance": un float entre 0.0 et 1.0 (importance opérationnelle)
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

def handle_contradictions(cur, tenant_id: str, agent_id: str, new_memory: dict) -> None:
    """
    Gère la détection et le traitement des contradictions.
    Si la nouvelle mémoire est une préférence ('preference') ou une règle ('rule'),
    on archive les anciennes de même type/subtype pour éviter les conflits directs.
    """
    if new_memory['type'] == 'semantic' and new_memory['subtype'] == 'preference':
        logger.info(f"Analyse des contradictions pour la préférence : {new_memory['content']}")
        # Archiver les anciennes préférences
        archive_query = """
            UPDATE memories
            SET status = 'archived', updated_at = CURRENT_TIMESTAMP
            WHERE tenant_id = %s
              AND agent_id = %s
              AND type = 'semantic'
              AND subtype = 'preference'
              AND status = 'active';
        """
        cur.execute(archive_query, (tenant_id, agent_id))
        logger.info("Anciennes préférences archivées avec succès.")

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
    
    # 2. Génération d'embedding
    embedding = generate_mock_embedding(memory_data['content'])
    
    # 3. Écriture en base de données avec gestion des contradictions
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Gestion des contradictions (Archivage + Confiance)
            handle_contradictions(cur, tenant_id, agent_id, memory_data)
            
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
            
            # Insertion d'une relation d'intrication (relation entre l'événement source et la mémoire créée)
            # Pour la v0, on peut par exemple créer une relation s'il y a lieu.
            
            conn.commit()
            logger.info(f"Nouvelle mémoire consolidée créée avec l'ID {new_mem_id}.")
            return True
            
    except Exception as e:
        logger.error(f"Erreur SQL lors de l'enregistrement de la mémoire : {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def main():
    logger.info("SynaptiQ Memory Worker démarré...")
    r = None
    while r is None:
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            logger.info("Connecté à Redis avec succès.")
        except Exception as e:
            logger.warning(f"En attente de Redis... ({e})")
            time.sleep(2)
            
    # Boucle de consommation d'événements
    while True:
        try:
            # Récupération non bloquante de la liste avec un court sleep s'il n'y a rien (évite les Socket Timeouts de Redis sous Windows)
            raw_payload = r.lpop("synaptiq:event_queue")
            if not raw_payload:
                time.sleep(1)
                continue
            
            event = json.loads(raw_payload)
            success = process_event(event)
            
            if not success:
                logger.warning(f"Échec du traitement de l'événement {event.get('id')}. Réintroduction dans la queue.")
                # Optionnel : réintroduire dans la queue pour re-tentative en v0
                r.rpush("synaptiq:event_queue", raw_payload)
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Arrêt du worker par l'utilisateur.")
            break
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale du worker : {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
