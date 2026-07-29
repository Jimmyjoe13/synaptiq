"""Scopes de clé API (permissions + périmètre d'agents) et journal d'audit.

Deux trous fermés ici, relevés à l'audit du 28/07 :

1. **Une clé API valait tous les droits sur tous les agents du tenant.** `agent_id`
   arrivait dans le body de la requête (et, côté MCP, était un paramètre d'outil donc
   contrôlé par le LLM lui-même) : n'importe quel agent pouvait lire ou écrire la mémoire
   d'un autre, et n'importe quelle clé pouvait appeler la purge RGPD. `scopes` porte
   désormais les permissions (`read` / `write` / `admin`) et `agent_scope` restreint la
   clé à une liste d'agents (NULL = tous, comportement historique).

   Le défaut `{read,write}` est délibéré : les clés EXISTANTES gardent lecture et
   écriture, mais perdent le droit de purge. C'est une rupture assumée — la purge
   supprimait l'intégralité d'un tenant sans confirmation ni trace.

2. **Aucune trace des opérations destructrices.** `audit_log` conserve qui a purgé quoi
   et quand. Volontairement SANS contenu mémoire (des compteurs seulement) : la table
   survit à la purge RGPD, elle ne doit donc jamais porter de données personnelles.
"""
from alembic import op

revision = "20260729_key_scopes"
down_revision = "20260726_hybrid_fts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Permissions portées par la clé. NOT NULL + défaut : les clés existantes restent
    # fonctionnelles en lecture/écriture sans intervention manuelle.
    op.execute(
        "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scopes TEXT[] "
        "NOT NULL DEFAULT ARRAY['read','write']::text[]"
    )
    # Périmètre d'agents autorisés. NULL (et non tableau vide) = tous les agents, afin de
    # conserver exactement le comportement antérieur pour les clés déjà émises.
    op.execute("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS agent_scope TEXT[]")

    op.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id BIGSERIAL PRIMARY KEY,
        tenant_id VARCHAR(50) NOT NULL,
        agent_id VARCHAR(50),
        action VARCHAR(50) NOT NULL,
        -- Préfixe du hash de la clé appelante (8 caractères) : identifie l'auteur sans
        -- jamais stocker de secret réutilisable.
        actor VARCHAR(64),
        -- Compteurs et paramètres d'appel uniquement, jamais de contenu mémoire.
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_log_tenant_created "
        "ON audit_log(tenant_id, created_at DESC)"
    )


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: security-bearing migration")
