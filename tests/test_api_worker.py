import os
import sys
import json
import time
import unittest
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

# Ajouter les chemins au sys.path pour pouvoir importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from apps.api.main import app as fastapi_app
from apps.worker.worker import process_event, generate_mock_embedding

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

class TestSynaptiqIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Le tenant est désormais résolu côté serveur (plus dans le body) : on aligne
        # le tenant d'instance sur celui utilisé par les insertions directes du test.
        os.environ["SYNAPTIQ_TENANT"] = "test_tenant"
        # S'assurer que les connexions de test fonctionnent
        try:
            cls.db_conn = psycopg2.connect(DATABASE_URL)
            cls.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        except Exception as e:
            print(f"\n[ERROR] ERREUR DE CONNEXION DANS SETUPCLASS : {e}")
            raise e

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'db_conn') and cls.db_conn:
            cls.db_conn.close()

    def setUp(self):
        # Nettoyer les tables de test avant chaque test
        with self.db_conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE relationships CASCADE;")
            cur.execute("TRUNCATE TABLE memories CASCADE;")
            cur.execute("TRUNCATE TABLE events CASCADE;")
            self.db_conn.commit()
        # Vider la file d'attente Redis
        self.redis_client.delete("synaptiq:event_queue")

    def test_end_to_end_flow(self):
        tenant_id = "test_tenant"
        agent_id = "test_agent"
        session_id = "test_session"
        
        # --- Étape 1 : Simulation de l'appel API POST /events ---
        # Au lieu de démarrer uvicorn, nous testons directement la logique de main.py
        # en appelant la fonction ou via TestClient. Pour rester simple, nous insérons
        # l'événement et testons le traitement du worker.
        
        event_content = "Je préfère les e-mails courts et concis de moins de 100 mots."
        
        # Insertion manuelle comme le ferait l'API
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO events (tenant_id, agent_id, session_id, content)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at;
                """,
                (tenant_id, agent_id, session_id, event_content)
            )
            event_row = cur.fetchone()
            self.db_conn.commit()
            
            event_id = str(event_row['id'])
            created_at = event_row['created_at'].isoformat()
            
        event_payload = {
            "id": event_id,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "content": event_content,
            "metadata": json.dumps({}),
            "created_at": created_at
        }
        
        # --- Étape 2 : Traitement par le Worker ---
        # Exécution synchrone de la fonction de traitement du worker
        success = process_event(event_payload)
        self.assertTrue(success, "Le traitement de l'événement par le worker a échoué.")
        
        # --- Étape 3 : Vérification de la création de la mémoire ---
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM memories WHERE tenant_id = %s AND agent_id = %s;",
                (tenant_id, agent_id)
            )
            memories = cur.fetchall()
            
            self.assertEqual(len(memories), 1, "La mémoire aurait dû être créée.")
            memory = memories[0]
            self.assertEqual(memory['type'], 'semantic')
            self.assertEqual(memory['subtype'], 'preference')
            self.assertEqual(memory['status'], 'active')
            self.assertIn("courts et concis", memory['content'])
            
        # --- Étape 4 : Gestion des Contradictions ---
        # Deuxième événement modifiant la préférence
        contradictory_content = "En fait, je préfère les e-mails très détaillés et formels à l'avenir."
        
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO events (tenant_id, agent_id, session_id, content)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at;
                """,
                (tenant_id, agent_id, session_id, contradictory_content)
            )
            event_row_2 = cur.fetchone()
            self.db_conn.commit()
            event_id_2 = str(event_row_2['id'])
            created_at_2 = event_row_2['created_at'].isoformat()
            
        event_payload_2 = {
            "id": event_id_2,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "content": contradictory_content,
            "metadata": json.dumps({}),
            "created_at": created_at_2
        }
        
        # Traitement du deuxième événement par le worker
        success_2 = process_event(event_payload_2)
        self.assertTrue(success_2)
        
        # Vérification de l'archivage de l'ancienne mémoire et activation de la nouvelle
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM memories WHERE tenant_id = %s AND agent_id = %s ORDER BY created_at ASC;",
                (tenant_id, agent_id)
            )
            memories_after = cur.fetchall()
            
            self.assertEqual(len(memories_after), 2, "Il devrait y avoir deux mémoires en base.")
            
            # Première mémoire (doit être archivée)
            mem_1 = memories_after[0]
            self.assertEqual(mem_1['status'], 'archived', "L'ancienne mémoire de préférence aurait dû être archivée.")
            self.assertIn("courts et concis", mem_1['content'])
            
            # Deuxième mémoire (doit être active)
            mem_2 = memories_after[1]
            self.assertEqual(mem_2['status'], 'active', "La nouvelle mémoire de préférence doit être active.")
            self.assertIn("détaillés et formels", mem_2['content'])

        # --- Étape 5 : Récupération du contexte (Retrieval / Collapse) ---
        # Appel de l'API de build de contexte via la fonction de FastAPI main
        from fastapi.testclient import TestClient
        client = TestClient(fastapi_app)
        
        response = client.post("/context/build", json={
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "task": "Rédiger un e-mail commercial",
            "query": "Quel style utiliser ?",
            "constraints": {
                "max_tokens": 1000,
                "memory_types": ["semantic"]
            }
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        context_packet = data['context_packet']
        # Le paquet de contexte ne doit contenir QUE le fait actif ("très détaillés et formels")
        self.assertEqual(len(context_packet['facts']), 1)
        self.assertIn("détaillés et formels", context_packet['facts'][0])
        self.assertNotIn("courts et concis", context_packet['facts'][0])
        
        print("\n[SUCCESS] Test d'integration de bout en bout reussi avec succes !")

if __name__ == '__main__':
    unittest.main()
