"""v0.5.0 (lot 2) — archives, récompense, plasticité.

  1. La contrainte CHECK des compartiments accepte `archives` (mémoire dormante, remplie
     par le sommeil : décision de Jimmy du 23/11, AUCUNE suppression de données).
  2. `memories.reward` : récompense explicite (outil `reinforce`), bornée à ±5 par l'API.
  3. `relationships.last_reinforced_at` : date du dernier renforcement hebbien d'un lien,
     lue par le sommeil pour affaiblir les liens que rien n'a renforcés (jamais supprimés).
  4. Index partiel des liens d'intrication, pour la phase de plasticité du sommeil.

Downgrade non supporté (convention du dépôt pour une migration porteuse de données).
"""
from alembic import op

revision = "20261123_sommeil"
down_revision = "20261122_compartments"
branch_labels = None
depends_on = None

# Copie FIGÉE au 23/11/2026 de synaptiq_core.compartments.NOMS.
COMPARTIMENTS = ("reflexes", "decisions", "savoir_faire", "alertes", "environnement",
                 "savoirs", "episodes", "observations", "croyances", "archives")


def upgrade() -> None:
    noms = ", ".join(f"'{c}'" for c in COMPARTIMENTS)
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS chk_memories_compartment")
    op.execute("ALTER TABLE memories ADD CONSTRAINT chk_memories_compartment "
               f"CHECK (compartment IN ({noms}))")
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS reward REAL NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE relationships "
               "ADD COLUMN IF NOT EXISTS last_reinforced_at TIMESTAMP WITH TIME ZONE")
    op.execute("CREATE INDEX IF NOT EXISTS idx_relationships_intrication "
               "ON relationships(source_memory_id) WHERE relation_type = 'entangled_with'")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: data-bearing schema migration.")
