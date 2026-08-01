"""Tests d'intégration de la fiabilité v0.2 : outbox transactionnel, idempotence
API, publication relay et déduplication worker au replay.

Ces invariants sont le cœur de la garantie « at-least-once sans duplication » :
- l'API écrit event + outbox dans une transaction (jamais directement dans Redis) ;
- une même idempotency_key ne crée qu'un seul événement ;
- le relay publie l'outbox vers Redis puis marque published_at (rejeu sûr) ;
- le worker déduplique par memories.source_event_id (rejeu = pas de doublon).

Exige Postgres + Redis (marqué integration via conftest). Auth désactivée.
"""
import json
import os
import sys
import unittest

import psycopg2
import redis
from fastapi.testclient import TestClient
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conftest import purge_tenants

from apps.api.main import app as fastapi_app
from apps.relay.relay import publish_pending

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/1")
EVENT_STREAM = os.getenv("EVENT_STREAM", "synaptiq:events")

# Les comptes doivent être bornés au tenant du test : la purge ne vide plus toute la base
# (elle détruirait les données réelles de l'instance), d'autres périmètres coexistent donc.
_EVENTS_DU_TENANT = "SELECT count(*) FROM events WHERE tenant_id = 'test_tenant'"
_OUTBOX_DU_TENANT = ("SELECT count(*) FROM event_outbox o JOIN events e ON e.id = o.event_id "
                     "WHERE e.tenant_id = 'test_tenant'")


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
        # Purge bornée au tenant du test : un TRUNCATE global effacerait aussi les
        # données réelles de l'instance (cf. conftest.purge_tenants).
        purge_tenants(self.db_conn, "test_tenant")
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

        self.assertEqual(self._count(_EVENTS_DU_TENANT), 1)
        self.assertEqual(
            self._count(_OUTBOX_DU_TENANT + " AND o.published_at IS NULL"), 1
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

        self.assertEqual(self._count(_EVENTS_DU_TENANT), 1)
        self.assertEqual(self._count(_OUTBOX_DU_TENANT), 1)

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
            self._count(_OUTBOX_DU_TENANT + " AND o.published_at IS NOT NULL"), 1
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
            self._count("SELECT id::text FROM events WHERE tenant_id = 'test_tenant' LIMIT 1"),
        )

    def test_health_signale_un_relais_mort(self):
        """RÉGRESSION 30/07 : /health doit dénoncer une ingestion à l'arrêt.

        Le relais de l'instance de production était éteint depuis deux jours. `/events`
        continuait de répondre `201 captured` et rien ne l'indiquait : les événements
        s'empilaient dans l'outbox sans jamais devenir des mémoires. Les jauges qui le
        détectaient existaient, mais rien ne scrutait /metrics.

        On vieillit artificiellement l'entrée d'outbox plutôt que d'attendre 300 s.
        """
        import apps.api.main as main

        with TestClient(fastapi_app) as client:
            client.post("/events", json={
                "agent_id": "agent_rel",
                "session_id": "sess_rel",
                "content": "Evenement qui va rester en attente.",
            })

            # Outbox frais, personne ne l'a encore publié : c'est normal, pas une panne.
            sain = client.get("/health").json()
            self.assertEqual(sain["services"]["ingestion"], "healthy")

            with self.db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE event_outbox SET created_at = CURRENT_TIMESTAMP - interval '1 hour' "
                    "WHERE event_id IN (SELECT id FROM events WHERE tenant_id = 'test_tenant')"
                )

            malade = client.get("/health").json()

        self.assertEqual(malade["services"]["ingestion"], "stalled")
        # Et le statut global doit basculer : un `ok` avec une ingestion morte serait pire
        # que pas de sonde du tout.
        self.assertEqual(malade["status"], "degraded")
        self.assertEqual(malade["services"]["postgres"], "healthy")

        # Une fois publié, l'outbox se vide et la santé revient.
        publish_pending(self.db_pool, self.redis_client)
        with TestClient(fastapi_app) as client:
            self.assertEqual(client.get("/health").json()["services"]["ingestion"], "healthy")
        self.assertGreater(main.HEALTH_OUTBOX_MAX_AGE_S, 0)

    def test_worker_dedupes_replayed_event(self):
        """Rejouer le même événement (même source_event_id) ne crée qu'une mémoire."""
        from unittest.mock import patch

        from apps.worker.worker import generate_mock_embedding, process_event

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

    def test_worker_dedupes_replayed_multifact_event(self):
        """Un événement multi-faits crée N mémoires, et son rejeu n'en duplique aucune.

        Depuis l'extraction multi-faits, l'unicité ne porte plus sur `source_event_id`
        seul mais sur `(source_event_id, content_hash)` : c'est ce couple qui garantit
        l'idempotence tout en autorisant plusieurs faits par événement.
        """
        from unittest.mock import patch

        from apps.worker.worker import generate_mock_embedding, process_event

        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO events (tenant_id, agent_id, session_id, content) "
                "VALUES (%s, %s, %s, %s) RETURNING id, created_at",
                ("test_tenant", "agent_multi", "sess_multi",
                 "Caroline a adopté un beagle et préfère courir le matin."),
            )
            row = cur.fetchone()

        event_payload = {
            "id": str(row["id"]),
            "tenant_id": "test_tenant",
            "agent_id": "agent_multi",
            "session_id": "sess_multi",
            "content": "Caroline a adopté un beagle et préfère courir le matin.",
            "metadata": json.dumps({}),
            "created_at": row["created_at"].isoformat(),
        }

        faits = [
            {"extracted": True, "type": "semantic", "subtype": "fact",
             "content": "Caroline a adopté un beagle", "summary": "Adoption",
             "confidence": 1.0, "importance": 0.8, "occurred_at": None},
            {"extracted": True, "type": "semantic", "subtype": "preference",
             "content": "Caroline préfère courir le matin", "summary": "Préférence",
             "confidence": 0.9, "importance": 0.6, "occurred_at": None},
        ]

        # Embeddings DISTINCTS par fait : un vecteur constant déclencherait l'archivage
        # de la préférence par `handle_contradictions` et fausserait le décompte.
        vecs = {f["content"]: generate_mock_embedding(f["content"]) for f in faits}

        class _PerContentEmbedder:
            dim = 384

            def embed(self, texts):
                return [vecs.get(t, generate_mock_embedding(t)) for t in texts]

            def embed_one(self, text):
                return self.embed([text])[0]

        with patch("apps.worker.worker.get_embedder", return_value=_PerContentEmbedder()), \
             patch("apps.worker.worker.call_llm_extractor", return_value=faits):
            self.assertTrue(process_event(event_payload))
            self.assertTrue(process_event(event_payload))   # rejeu

        self.assertEqual(
            self._count("SELECT count(*) FROM memories WHERE source_event_id = %s", (row["id"],)),
            len(faits),
            "Le rejeu doit laisser exactement un exemplaire de chaque fait.",
        )


if __name__ == "__main__":
    unittest.main()
