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
    
    # 3. Écriture en base de données avec gestion des contradictions et des intrications
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
            logger.info(f"Nouvelle mémoire consolidée créée avec l'ID {new_mem_id} ({memory_data['type']}/{memory_data['subtype']}).")
            
            # 4. Graphe d'intrication sémantique automatique (Q-EM)
            # Si c'est une règle, une erreur ou une bonne pratique, on cherche sémantiquement des souvenirs associés pour créer des liaisons
            if memory_data['type'] == 'procedural' or memory_data['subtype'] == 'preference':
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
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
