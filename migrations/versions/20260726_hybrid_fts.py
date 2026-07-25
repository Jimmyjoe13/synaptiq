"""Recherche hybride : index full-text pour compléter la similarité vectorielle.

Constat du benchmark LOCOMO du 25/07 : 47 % des questions échouaient QUELLE QUE SOIT la
stratégie de ranking (Q-EM comme top-k vectoriel). Le goulot n'est pas le classement mais
le rappel — l'embedding ramène du « sémantiquement proche » et manque les correspondances
LITTÉRALES : noms propres, dates, identifiants, références techniques. C'est précisément
ce que la recherche plein texte retrouve.

Configuration 'simple' délibérément, et non 'english'/'french' :
  - le corpus d'une instance est souvent multilingue (SynaptiQ préserve la langue source
    de chaque souvenir depuis le 25/07) ; une config par langue ferait des faux négatifs
    sur les autres ;
  - le rôle de ce chemin est de rattraper les termes EXACTS, pas de généraliser — le
    stemming n'y apporte rien, la similarité sémantique est déjà couverte par le vecteur.

Colonne générée plutôt qu'index d'expression : le tsvector est calculé une fois à
l'écriture et reste visible pour le débogage.
"""
from alembic import op

revision = "20260726_hybrid_fts"
down_revision = "20260725_multifact"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_content_tsv "
               "ON memories USING GIN (content_tsv)")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: schema migration")
