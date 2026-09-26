"""Dimension PROJET sur les souvenirs (lot B de l'audit du 26/09).

Constat : un agent généraliste (Claude Code, Codex…) travaille sur plusieurs projets, et
rien dans le schéma ne les distinguait. `subtype` porte une catégorie cognitive
(`decisions`, `infra_access`…) partagée par tous les projets ; `provenance` était vide sur
100 % des souvenirs réels. Mesuré sur l'agent `claude_code_orchestrator` : 39 % de
souvenirs Émile, 15 % SynaptiQ, 9 % elu-scraper… mêlés sans filtre possible. Un rappel sur
« état du projet SynaptiQ » ramenait majoritairement d'autres projets.

  - `memories.project` : NULL = souvenir GLOBAL (valable pour tous les projets :
    préférences de l'utilisateur, conventions générales). Une valeur = souvenir propre à
    ce projet. Nom libre normalisé côté API (`synaptiq_core.project`), pas de table :
    une liste fermée obligerait à déclarer chaque projet avant d'écrire.
  - `events.project` : le chemin `/v1/events` doit pouvoir porter la même information
    jusqu'au worker.
  - Index partiel `(tenant_id, agent_id, project)` sur les actifs : le filtre projet est
    appliqué dans CHAQUE chemin de recherche (vectoriel, plein texte), à côté du filtre
    tenant/agent qu'il prolonge.

Les colonnes sont nullables, sans défaut : la migration est instantanée (pas de réécriture
de table) et tous les souvenirs existants deviennent globaux, ce qui préserve exactement le
comportement actuel tant qu'aucun filtre projet n'est demandé.
"""
from alembic import op

revision = "20260926_memory_project"
down_revision = "20260811_fts_french"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS project VARCHAR(64)")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS project VARCHAR(64)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_project
        ON memories (tenant_id, agent_id, project) WHERE status = 'active'
    """)


def downgrade() -> None:
    # Réversible sans perte de souvenir : seul le rattachement au projet disparaît.
    op.execute("DROP INDEX IF EXISTS idx_memories_project")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS project")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS project")
