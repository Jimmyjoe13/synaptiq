"""Tests d'intégration de la fiabilité v0.2 : outbox transactionnel, idempotence
API, publication relay et déduplication worker au replay.

Ces invariants sont le cœur de la garantie « at-least-once sans duplication » :
- l'API écrit event + outbox dans une transaction (jamais directement dans Redis) ;
- une même idempotency_key ne crée qu'un seul événement ;
- le relay publie l'outbox vers Redis puis marque published_at (rejeu sûr) ;
- le worker déduplique par memories.source_event_id (rejeu = pas de doublon).

Exige Postgres + Redis (marqué integration via conftest). Auth désactivée.
"""
import os
import sys
import json
import unittest

import psycopg2
import redis
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from apps.api.main import app as fastapi_app  # noqa: E402
from apps.relay.relay import publish_pending  # noqa: E402

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")
EVENT_STREAM = os.getenv("EVENT_STREAM", "synaptiq:events")


class TestReliability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["SYNAPTIQ_TENANT"] = "test_tenant"
        os.environ["SYNAPTIQ_AUTH_REQUIRED"] = "false"
        cls.db_conn = psycopg2.connect(DATABASE_URL)
        cls.db_conn.autocommit = True
        cls.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        cls.db_pool = pg_pool.ThreadedConnectionPool(1, 4, dsn=DATABASE_URL)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "db_pool", None):
            cls.db_pool.closeall()
        if getattr(cls, "db_conn", None):
            cls.db_conn.close()

    def setUp(self):
        with self.db_conn.cursor() as cur:
            # CASCADE : purge memories + event_outbox qui référencent events.
            cur.execute("TRUNCATE TABLE relationships CASCADE;")
            cur.execute("TRUNCATE TABLE memories CASCADE;")
            cur.execute("TRUNCATE TABLE events CASCADE;")
        self.redis_client.delete(EVENT_STREAM)

    def _count(self, sql, params=()):
        with self.db_conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    def test_capture_writes_to_outbox_not_redis(self):
        """POST /events persiste event + outbox dans une transaction, sans publier."""
        with TestClient(fastapi_app) as client:
            resp = client.post("/events", json={
                "agent_id": "agent_rel",
                "session_id": "sess_rel",
                "content": "Un événement de test pour l'outbox.",
            })
        self.assertLess(resp.status_code, 300, resp.text)
        self.assertEqual(resp.json()["status"], "captured")

        self.assertEqual(self._count("SELECT count(*) FROM events"), 1)
        self.assertEqual(
            self._count("SELECT count(*) FROM event_outbox WHERE published_at IS NULL"), 1
        )
        # Rien n'a été publié directement dans Redis par l'API.
        self.assertEqual(self.redis_client.xlen(EVENT_STREAM), 0)

    def test_idempotency_key_dedupes_at_api(self):
        """Deux POST avec la même idempotency_key => un seul événement + un seul outbox."""
        payload = {
            "agent_id": "agent_rel",
            "session_id": "sess_rel",
            "content": "Événement idempotent.",
            "idempotency_key": "evt-key-001",
        }
        with TestClient(fastapi_app) as client:
            first = client.post("/events", json=payload)
            second = client.post("/events", json=payload)

        self.assertLess(first.status_code, 300, first.text)
        self.assertLess(second.status_code, 300, second.text)
        self.assertEqual(first.json()["status"], "captured")
        self.assertEqual(second.json()["status"], "duplicate")
        self.assertEqual(first.json()["event_id"], second.json()["event_id"])

        self.assertEqual(self._count("SELECT count(*) FROM events"), 1)
        self.assertEqual(self._count("SELECT count(*) FROM event_outbox"), 1)

    def test_relay_publishes_pending_once(self):
        """Le relay publie l'outbox vers Redis puis marque published_at (rejeu sûr)."""
        with TestClient(fastapi_app) as client:
            client.post("/events", json={
                "agent_id": "agent_rel",
                "session_id": "sess_rel",
                "content": "Événement à publier par le relay.",
            })

        published = publish_pending(self.db_pool, self.redis_client)
        self.assertEqual(published, 1)
        self.assertEqual(self.redis_client.xlen(EVENT_STREAM), 1)
        self.assertEqual(
            self._count("SELECT count(*) FROM event_outbox WHERE published_at IS NOT NULL"), 1
        )

        # Second passage : plus rien en attente, pas de nouvelle publication.
        published_again = publish_pending(self.db_pool, self.redis_client)
        self.assertEqual(published_again, 0)
        self.assertEqual(self.redis_client.xlen(EVENT_STREAM), 1)

        # Le message publié contient bien l'id de l'événement.
        entries = self.redis_client.xrange(EVENT_STREAM)
        payload = json.loads(entries[0][1]["data"])
        self.assertIn("id", payload)
        self.assertEqual(
            payload["id"],
            self._count("SELECT id::text FROM events LIMIT 1"),
        )

    def test_worker_dedupes_replayed_event(self):
        """Rejouer le même événement (même source_event_id) ne crée qu'une mémoire."""
        from unittest.mock import patch
        from apps.worker.worker import process_event, generate_mock_embedding

        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO events (tenant_id, agent_id, session_id, content) "
                "VALUES (%s, %s, %s, %s) RETURNING id, created_at",
                ("test_tenant", "agent_rel", "sess_rel", "Je préfère les réponses concises."),
            )
            row = cur.fetchone()

        event_payload = {
            "id": str(row["id"]),
            "tenant_id": "test_tenant",
            "agent_id": "agent_rel",
            "session_id": "sess_rel",
            "content": "Je préfère les réponses concises.",
            "metadata": json.dumps({}),
            "created_at": row["created_at"].isoformat(),
        }

        const_vec = generate_mock_embedding("const")

        class _ConstEmbedder:
            dim = 384

            def embed(self, texts):
                return [const_vec for _ in texts]

            def embed_one(self, text):
                return const_vec

        with patch("apps.worker.worker.get_embedder", return_value=_ConstEmbedder()):
            self.assertTrue(process_event(event_payload))
            # Rejeu du même événement (crash entre INSERT et ACK simulé).
            self.assertTrue(process_event(event_payload))

        self.assertEqual(
            self._count("SELECT count(*) FROM memories WHERE source_event_id = %s", (row["id"],)),
            1,
        )


if __name__ == "__main__":
    unittest.main()
