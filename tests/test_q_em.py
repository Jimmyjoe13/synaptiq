import os
import sys
import json
import unittest
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from fastapi.testclient import TestClient

# Ajouter les chemins au sys.path pour pouvoir importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from apps.api.main import app as fastapi_app
from apps.worker.worker import generate_mock_embedding

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")

class TestQuantumEntanglementMemory(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            cls.db_conn = psycopg2.connect(DATABASE_URL)
            cls.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            cls.client = TestClient(fastapi_app)
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

    def test_q_em_entanglement_propagation(self):
        """
        Test de l'Intrication Quantique (Propagation d'activation) :
        Une mémoire M2 (procédurale/règle) est intriquée avec une mémoire M1 (sémantique).
        M1 correspond sémantiquement à la requête, mais pas M2.
        Grâce à l'intrication, M2 doit être ramenée dans le contexte final.
        """
        tenant_id = "tenant_quantum"
        agent_id = "agent_quantum"
        session_id = "session_quantum"
        
        query_text = "Quelles sont les préférences de Jimmy pour le dev Python ?"
        query_embedding = generate_mock_embedding(query_text)
        
        # 1. Insertion de M1 : Mémoire sémantique directement liée à la requête (partage son embedding)
        m1_content = "L'utilisateur Jimmy adore programmer des jeux vidéo en Python."
        m1_embedding = query_embedding
        
        # 2. Insertion de M2 : Règle procédurale intriquée (qui ne mentionne pas Python ni Jimmy)
        m2_content = "Consigne de style : Toujours nommer les fonctions en snake_case et documenter avec des docstrings."
        m2_embedding = generate_mock_embedding(m2_content) # Embedding sémantiquement distant
        
        with self.db_conn.cursor() as cur:
            # Insertion M1
            cur.execute(
                """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status)
                VALUES (%s, %s, 'semantic', 'fact', %s, %s, 1.0, 0.8, 'active')
                RETURNING id;
                """,
                (tenant_id, agent_id, m1_content, m1_embedding)
            )
            m1_id = cur.fetchone()[0]
            
            # Insertion M2
            cur.execute(
                """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status)
                VALUES (%s, %s, 'procedural', 'rule', %s, %s, 1.0, 0.9, 'active')
                RETURNING id;
                """,
                (tenant_id, agent_id, m2_content, m2_embedding)
            )
            m2_id = cur.fetchone()[0]
            
            # Création de la relation d'intrication quantique
            cur.execute(
                """
                INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
                VALUES (%s, %s, 'entangled_with', 1.0);
                """,
                (m1_id, m2_id)
            )
            self.db_conn.commit()

        # 3. Appel de build_context avec une requête ciblant M1
        response = self.client.post("/context/build", json={
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "task": "Écrire un script",
            "query": "Quelles sont les préférences de Jimmy pour le dev Python ?",
            "constraints": {
                "max_tokens": 1000,
                "memory_types": ["semantic", "procedural"]
            }
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        context_packet = data['context_packet']
        selected_ids = data['selected_memory_ids']
        
        # M1 doit être présente
        self.assertIn(m1_content, context_packet['facts'])
        # M2 doit avoir été intriquée et donc ramenée dans les règles, malgré l'absence de similarité directe !
        self.assertIn(m2_content, context_packet['rules'])
        self.assertEqual(len(selected_ids), 2)
        print("\n[SUCCESS] Intrication quantique et propagation d'activation vérifiées !")

    def test_q_em_destructive_interference_redundancy(self):
        """
        Test de l'Interférence Destructive de Redondance :
        Deux mémoires sémantiques sont très proches sémantiquement (redondantes).
        Le filtre d'interférence destructive doit en éliminer une pour économiser des tokens.
        """
        tenant_id = "tenant_quantum"
        agent_id = "agent_quantum"
        session_id = "session_quantum"
        
        # Créer deux mémoires avec des embeddings identiques (redondance maximale de 1.0)
        m1_content = "Jimmy boit du thé vert sencha tous les matins au réveil."
        m2_content = "Jimmy prend une tasse de thé vert sencha au réveil chaque matin."
        # Pour forcer une similarité de 1.0 dans le test, on va leur donner le même embedding
        embedding = generate_mock_embedding(m1_content)
        
        with self.db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status)
                VALUES (%s, %s, 'semantic', 'preference', %s, %s, 1.0, 0.8, 'active');
                """,
                (tenant_id, agent_id, m1_content, embedding)
            )
            cur.execute(
                """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status)
                VALUES (%s, %s, 'semantic', 'preference', %s, %s, 1.0, 0.5, 'active'); -- importance plus faible
                """,
                (tenant_id, agent_id, m2_content, embedding)
            )
            self.db_conn.commit()

        # Appel build_context
        response = self.client.post("/context/build", json={
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "task": "Préparer le petit déjeuner",
            "query": "Que boit Jimmy le matin ?",
            "constraints": {
                "max_tokens": 1000,
                "memory_types": ["semantic"]
            }
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        context_packet = data['context_packet']
        selected_ids = data['selected_memory_ids']
        
        # Le contexte ne doit contenir qu'UNE SEULE des deux mémoires (celle de plus forte importance, M1)
        self.assertEqual(len(selected_ids), 1, "L'interférence destructive de redondance aurait dû éliminer la mémoire en doublon.")
        self.assertIn(m1_content, context_packet['facts'])
        self.assertNotIn(m2_content, context_packet['facts'])
        print("[SUCCESS] Interférence destructive de redondance vérifiée !")

    def test_q_em_destructive_interference_contradiction(self):
        """
        Test de l'Interférence Destructive de Contradiction :
        Deux mémoires sémantiques sont déclarées contradictoires dans la table 'relationships'.
        Le filtre d'interférence destructive doit éliminer la plus ancienne.
        """
        tenant_id = "tenant_quantum"
        agent_id = "agent_quantum"
        session_id = "session_quantum"
        
        m1_content = "Le bureau de Paris ferme à 18h00."
        m2_content = "Le bureau de Paris ferme désormais à 19h30 à partir d'aujourd'hui."
        
        emb1 = generate_mock_embedding(m1_content)
        emb2 = generate_mock_embedding(m2_content)
        
        with self.db_conn.cursor() as cur:
            # Insérer M1 (créée en premier)
            cur.execute(
                """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status, created_at)
                VALUES (%s, %s, 'semantic', 'fact', %s, %s, 1.0, 0.7, 'active', '2026-07-09T08:00:00Z')
                RETURNING id;
                """,
                (tenant_id, agent_id, m1_content, emb1)
            )
            m1_id = cur.fetchone()[0]
            
            # Insérer M2 (créée après)
            cur.execute(
                """
                INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, confidence, importance, status, created_at)
                VALUES (%s, %s, 'semantic', 'fact', %s, %s, 1.0, 0.7, 'active', '2026-07-09T09:00:00Z')
                RETURNING id;
                """,
                (tenant_id, agent_id, m2_content, emb2)
            )
            m2_id = cur.fetchone()[0]
            
            # Lier les deux mémoires par une relation de contradiction
            cur.execute(
                """
                INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
                VALUES (%s, %s, 'contradicts', 1.0);
                """,
                (m1_id, m2_id)
            )
            self.db_conn.commit()

        # Appel build_context
        response = self.client.post("/context/build", json={
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "task": "Planifier une réunion",
            "query": "Quelles sont les heures d'ouverture à Paris ?",
            "constraints": {
                "max_tokens": 1000,
                "memory_types": ["semantic"]
            }
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        context_packet = data['context_packet']
        selected_ids = data['selected_memory_ids']
        
        # Le contexte ne doit contenir que M2 (la plus récente)
        self.assertEqual(len(selected_ids), 1, "La contradiction aurait dû éliminer la mémoire obsolète.")
        self.assertIn(m2_content, context_packet['facts'])
        self.assertNotIn(m1_content, context_packet['facts'])
        print("[SUCCESS] Interférence destructive de contradiction vérifiée !")

if __name__ == '__main__':
    unittest.main()
