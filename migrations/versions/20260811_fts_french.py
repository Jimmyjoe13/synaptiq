"""Plein texte : configuration 'french' — le chemin remontait 0 résultat sur 12 (11/08).

Constat de l'audit du 11/08 : le plein texte (moitié du RRF hybride) ne remontait RIEN
sur les douze questions réelles du banc d'essai. Cause double :

  - `websearch_to_tsquery('simple', …)` produit un ET de tous les termes, mots vides
    compris (« quel », « pour », « le »…) : une question en langage naturel n'a
    pratiquement aucune chance de matcher un souvenir entier ;
  - une config 'simple' ne normalise ni ne filtre rien : pas de retrait des mots vides,
    pas de stemming.

La configuration 'simple' était un choix argumenté (corpus multilingue, rattraper des
termes EXACTS). Le résultat mesuré est un chemin mort — et un chemin mort ne rattrape
rien du tout. On passe en 'french' : les mots vides disparaissent à la normalisation,
le stemming rapproche les variantes. Le côté multilingue est couvert par le chemin
vectoriel, qui reste le chemin principal ; la requête elle-même est désormais un OU
pondéré de lexèmes (côté code, cf. `_fetch_candidates`), un seul terme suffit à
remonter un souvenir et `ts_rank` récompense ceux qui en portent plusieurs.

La colonne générée est recréée (le tsvector est calculé une fois à l'écriture, jamais à
la lecture) ; l'index GIN est recréé par la même occasion.
"""
from alembic import op

revision = "20260811_fts_french"
down_revision = "20260801_memory_idem"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS content_tsv")
    op.execute("""
        ALTER TABLE memories ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('french', coalesce(content, ''))) STORED
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_content_tsv "
               "ON memories USING GIN (content_tsv)")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: schema migration")
