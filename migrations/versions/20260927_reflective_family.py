"""Famille cognitive `reflective` : les croyances de l'agent (lot C, 26/09).

Décision de Jimmy du 26/09 : un compartiment pour ce que l'agent PENSE de l'utilisateur et
des humains. Ce sont des croyances, pas des faits — confiance, indices, révision par
remplacement, pas de décroissance, rendu « hypothèse » — donc une FAMILLE (un comportement
du moteur) et non deux collections de `semantic` qui se seraient comportées comme des faits.

  - `chk_collection_family` accepte `reflective` : les deux collections système doivent
    pouvoir figurer dans `memory_collections` (elles apparaissent dans `list_collections`).
    Les agents, eux, ne peuvent toujours pas créer de collection dans cette famille :
    l'API ne l'expose pas (`CollectionInput.family`).
  - Amorçage des collections système `user_model` et `human_insights`.

`memories.type` n'a pas de contrainte CHECK (constaté) : rien à modifier de ce côté. Le
statut `contested` (croyance contestée par l'utilisateur) tient dans `status VARCHAR(20)`,
lui aussi sans contrainte.
"""
import sqlalchemy as sa
from alembic import op

revision = "20260927_reflective_family"
down_revision = "20260926_memory_project"
branch_labels = None
depends_on = None

COLLECTIONS = (
    ("user_model", "user_model",
     "Ce que l'agent pense de l'utilisateur : hypotheses sur sa facon de travailler, ses "
     "attentes, ses reactions (avec confiance)."),
    ("human_insights", "human_insights",
     "Reflexions de l'agent sur les humains en general et sur les personnes qu'il cotoie "
     "(avec confiance)."),
)


def upgrade() -> None:
    op.execute("ALTER TABLE memory_collections DROP CONSTRAINT IF EXISTS chk_collection_family")
    op.execute("""
        ALTER TABLE memory_collections ADD CONSTRAINT chk_collection_family
        CHECK (family IN ('semantic','episodic','procedural','working','reflective'))
    """)
    # Paramètres liés : les descriptions portent des apostrophes.
    insertion = sa.text(
        "INSERT INTO memory_collections "
        "(name, family, packet_key, entangle, description, created_by) "
        "VALUES (:nom, 'reflective', :cle, false, :description, 'system') "
        "ON CONFLICT DO NOTHING")
    for nom, cle, description in COLLECTIONS:
        op.get_bind().execute(insertion, {"nom": nom, "cle": cle, "description": description})


def downgrade() -> None:
    # Refus explicite s'il existe des croyances : les faire disparaître en silence de la
    # taxonomie laisserait des souvenirs d'une famille que le code ne connaîtrait plus.
    n = op.get_bind().execute(
        sa.text("SELECT count(*) FROM memories WHERE type = 'reflective'")).scalar()
    if n:
        raise RuntimeError(f"{n} souvenir(s) 'reflective' en base : downgrade refusé.")
    op.execute("DELETE FROM memory_collections WHERE family = 'reflective'")
    op.execute("ALTER TABLE memory_collections DROP CONSTRAINT IF EXISTS chk_collection_family")
    op.execute("""
        ALTER TABLE memory_collections ADD CONSTRAINT chk_collection_family
        CHECK (family IN ('semantic','episodic','procedural','working'))
    """)
