"""Extraction multi-faits et datation des souvenirs.

Deux verrous levés ici, mesurés sur le run LOCOMO du 25/07 :

1. `UNIQUE (source_event_id)` imposait UNE mémoire par événement. Or un tour de dialogue
   énonce souvent 2-3 faits : le modèle devait choisir, et privilégiait la généralité au
   fait vérifiable (57 % des questions échouaient faute d'information en base). L'unicité
   passe donc sur `(source_event_id, content_hash)` : plusieurs faits par événement, mais
   un replay du même événement reste dédupliqué — l'idempotence est préservée.

2. `created_at` datait l'ÉCRITURE de la ligne, jamais le fait lui-même. Les mémoires
   perdaient l'horodatage de leur événement source (5 % seulement en portaient un), rendant
   les questions temporelles insolubles. `occurred_at` porte désormais la date à laquelle
   le fait s'est produit.
"""
from alembic import op

revision = "20260725_multifact"
down_revision = "20260724_v02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Quand le fait a eu lieu (≠ created_at, qui date l'insertion en base).
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMP WITH TIME ZONE")
    # SHA-256 du contenu normalisé : discrimine les faits issus d'un même événement.
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)")

    # Rétro-compatibilité : les lignes existantes n'ont pas de hash. On leur en attribue un
    # AVANT de créer l'index unique, sinon deux mémoires du même événement (impossible
    # aujourd'hui, mais prudence) entreraient en collision sur NULL.
    op.execute("UPDATE memories SET content_hash = encode(sha256(content::bytea), 'hex') "
               "WHERE content_hash IS NULL")

    # L'ancien index unique bloquait le multi-faits : on le remplace.
    op.execute("DROP INDEX IF EXISTS idx_memories_source_event")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_event_fact "
               "ON memories(source_event_id, content_hash) WHERE source_event_id IS NOT NULL")

    # Filtrage et tri temporels (questions « quand… », fenêtres de dates).
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_occurred "
               "ON memories(tenant_id, agent_id, occurred_at) WHERE occurred_at IS NOT NULL")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: data-bearing schema migration")
