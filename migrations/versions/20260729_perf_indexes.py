"""Index manquants sur le chemin critique de `build_context` (audit F12).

`relationships` a pour clé primaire `(source_memory_id, target_memory_id)`. Un index composite
ne sert PAS une recherche sur sa deuxième colonne seule : la requête de `build_context`

    WHERE source_memory_id = ANY(...) OR target_memory_id = ANY(...)

faisait donc un parcours séquentiel complet de la table des relations à chaque construction
de contexte — c'est-à-dire à chaque appel de l'endpoint le plus chaud du produit.

L'index sur `audit_log(action)` sert la consultation des opérations sensibles (« quelles
purges ont eu lieu ? »), qui n'a pas de chemin d'accès aujourd'hui.
"""
from alembic import op

revision = "20260729_perf_idx"
down_revision = "20260729_key_scopes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target "
               "ON relationships(target_memory_id)")
    # Le graphe est parcouru par type de relation ('entangled_with' vs 'supersedes_by') :
    # l'index couvre la source ET le type pour éviter un filtre sur le tas.
    op.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source_type "
               "ON relationships(source_memory_id, relation_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_action "
               "ON audit_log(action, created_at DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_relationships_target")
    op.execute("DROP INDEX IF EXISTS idx_relationships_source_type")
    op.execute("DROP INDEX IF EXISTS idx_audit_log_action")
