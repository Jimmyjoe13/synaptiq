"""Add transactional outbox and event-to-memory idempotency."""
from alembic import op

revision = "20260724_v02"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_events_tenant_idempotency ON events(tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL")
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_event_id UUID")
    op.execute("""DO $$ BEGIN
        ALTER TABLE memories ADD CONSTRAINT fk_memories_source_event
        FOREIGN KEY (source_event_id) REFERENCES events(id) ON DELETE SET NULL;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_source_event ON memories(source_event_id) WHERE source_event_id IS NOT NULL")
    op.execute("""CREATE TABLE IF NOT EXISTS event_outbox (
        id BIGSERIAL PRIMARY KEY, event_id UUID NOT NULL UNIQUE REFERENCES events(id) ON DELETE CASCADE,
        payload JSONB NOT NULL, created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        published_at TIMESTAMP WITH TIME ZONE, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT)""")
    op.execute("CREATE INDEX IF NOT EXISTS idx_event_outbox_pending ON event_outbox(created_at) WHERE published_at IS NULL")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: data-bearing reliability migration")
