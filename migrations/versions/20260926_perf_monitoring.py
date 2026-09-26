"""Phase 4 : monitoring pg_stat_statements + table de pré-calcul 2-hop.

Active pg_stat_statements pour identifier les requêtes lentes en production.
Crée la table entanglement_2hop pour le pré-calcul des voisins à 2 sauts.
"""
from alembic import op

revision = "20260926_perf_monitoring"
down_revision = "20260927_reflective_family"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Activer pg_stat_statements (nécessite un rechargement de la config PostgreSQL)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")

    # Table de pré-calcul des voisins à 2 sauts (Phase 4)
    # Remplie lors de l'intrication (entanglement.py), lue par propagate_entanglement.
    op.execute("""
        CREATE TABLE IF NOT EXISTS entanglement_2hop (
            memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            neighbor_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            hop INTEGER NOT NULL DEFAULT 2,
            weight DOUBLE PRECISION DEFAULT 1.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (memory_id, neighbor_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_entanglement_2hop_neighbor
        ON entanglement_2hop(neighbor_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS entanglement_2hop")
    op.execute("DROP EXTENSION IF EXISTS pg_stat_statements")
