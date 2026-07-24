"""Publie de façon fiable les événements validés vers Redis Streams."""
import json
import logging
import os
import time

import redis
from dotenv import load_dotenv
from psycopg2 import pool as pg_pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6399/0")
EVENT_STREAM = os.getenv("EVENT_STREAM", "synaptiq:events")
POLL_SECONDS = float(os.getenv("OUTBOX_POLL_SECONDS", "0.5"))
logger = logging.getLogger("synaptiq-relay")


def publish_pending(db_pool, redis_client) -> int:
    """Publie un lot. Les doublons sont sûrs: le worker déduplique source_event_id."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, payload FROM event_outbox WHERE published_at IS NULL
                   ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 100"""
            )
            rows = cur.fetchall()
            for outbox_id, payload in rows:
                redis_client.xadd(EVENT_STREAM, {"data": json.dumps(payload)})
                cur.execute(
                    "UPDATE event_outbox SET published_at = CURRENT_TIMESTAMP, attempts = attempts + 1, last_error = NULL WHERE id = %s",
                    (outbox_id,),
                )
            conn.commit()
            return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db_pool = pg_pool.ThreadedConnectionPool(1, 4, dsn=DATABASE_URL)
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    while True:
        try:
            published = publish_pending(db_pool, redis_client)
            if not published:
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            return
        except Exception as exc:
            logger.exception("Publication outbox échouée: %s", exc)
            time.sleep(2)


if __name__ == "__main__":
    main()
