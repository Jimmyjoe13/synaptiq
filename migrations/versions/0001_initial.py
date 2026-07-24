"""Schéma initial SynaptiQ : provisionne une base VIERGE.

Miroir idempotent de infra/postgres/init.sql. Corrige le trou où `alembic upgrade head`
échouait sur une base neuve (l'ancienne unique migration `20260724_v02` faisait des
ALTER TABLE sur des tables inexistantes). Tout est en IF NOT EXISTS : sur une base legacy
déjà provisionnée par init.sql, cette migration ne fait rien de destructif.
"""
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- events ---
    op.execute("""CREATE TABLE IF NOT EXISTS events (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id VARCHAR(50) NOT NULL,
        agent_id VARCHAR(50) NOT NULL,
        session_id VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )""")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_tenant_agent_session ON events(tenant_id, agent_id, session_id)")

    # --- memories ---
    op.execute("""CREATE TABLE IF NOT EXISTS memories (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id VARCHAR(50) NOT NULL,
        agent_id VARCHAR(50) NOT NULL,
        type VARCHAR(20) NOT NULL,
        subtype VARCHAR(50),
        content TEXT NOT NULL,
        summary TEXT,
        embedding VECTOR(384) NOT NULL,
        confidence DOUBLE PRECISION DEFAULT 1.0,
        importance DOUBLE PRECISION DEFAULT 0.5,
        recency_score DOUBLE PRECISION DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'active',
        version INTEGER DEFAULT 1,
        provenance JSONB DEFAULT '{}'::jsonb
    )""")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(tenant_id, agent_id, type, status)")

    # --- api_keys ---
    op.execute("""CREATE TABLE IF NOT EXISTS api_keys (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        key_hash VARCHAR(64) NOT NULL UNIQUE,
        tenant_id VARCHAR(50) NOT NULL,
        name VARCHAR(100),
        active BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        last_used_at TIMESTAMP WITH TIME ZONE
    )""")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash) WHERE active")

    # --- relationships (graphe Q-EM) ---
    op.execute("""CREATE TABLE IF NOT EXISTS relationships (
        source_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
        target_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
        relation_type VARCHAR(50) NOT NULL,
        weight DOUBLE PRECISION DEFAULT 1.0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (source_memory_id, target_memory_id)
    )""")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: initial schema migration")
