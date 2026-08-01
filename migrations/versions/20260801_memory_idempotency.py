"""Idempotence de l'écriture directe `POST /v1/memories`.

Le chemin direct était un `INSERT` nu. Une relance client après timeout perçu — alors que le
premier appel avait abouti côté serveur — créait une seconde ligne, sans erreur ni trace.
L'index unique existant, `idx_memories_event_fact`, ne pouvait rien y faire : il est PARTIEL
sur `WHERE source_event_id IS NOT NULL`, et une écriture directe a `source_event_id` nul.

Deux index partiels sont ajoutés, tous deux bornés à `status = 'active'` et
`source_event_id IS NULL` :

  - `(tenant_id, agent_id, content_hash)` — mécanisme PRINCIPAL. La déduplication porte sur
    le contenu parce que l'appelant typique est un modèle invoquant un outil MCP : il n'a
    aucune clé stable à fournir, et une clé régénérée à chaque tentative ne protège de rien.
  - `(tenant_id, agent_id, idempotency_key)` — complément pour les appelants qui possèdent
    une vraie clé stable (identifiant de ligne source d'un import, par exemple).

Le `status = 'active'` n'est pas cosmétique : archiver un fait puis le ré-affirmer plus tard
est légitime. Contraindre sur TOUTES les lignes rendrait tout archivage définitif.

⚠️ LE BACKFILL EST EN PYTHON, PAS EN SQL — et c'est le point délicat de cette migration.
`content_hash` normalise via `str.split()`, qui découpe sur tout blanc Unicode, U+00A0 et
U+202F comprises. Le `\\s` de PostgreSQL est ASCII-only et ne les reconnaît pas : un backfill
écrit en SQL produirait donc une empreinte DIFFÉRENTE de celle que calcule le code, sans
lever la moindre erreur, et l'index unique cesserait de protéger précisément les lignes qu'il
croit couvrir. Vérifié sur l'instance de référence : 65 mémoires sur 1015 contiennent U+202F
(« 10:37 U+202F am », « MySQL U+202F 8.0 ») et divergent entre les deux formules. La
migration `20260725_multifact` avait d'ailleurs déjà backfillé en SQL brut, sans casse à
l'époque parce que son index portait aussi sur `source_event_id`.
"""
import os
import sys

from alembic import op
from sqlalchemy import text

revision = "20260801_memory_idem"
down_revision = "20260731_collections"
branch_labels = None
depends_on = None

# `packages/core` importable comme dans les services (même repli `sys.path`).
_racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _chemin in (_racine, os.path.join(_racine, "packages", "core")):
    if _chemin not in sys.path:
        sys.path.insert(0, _chemin)

from synaptiq_core.hashing import content_hash

# Les lignes sont traitées par paquets : une instance peuplée peut en compter beaucoup, et un
# seul UPDATE par ligne dans une transaction unique tiendrait la table trop longtemps.
TAILLE_LOT = 500


def upgrade() -> None:
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128)")

    connexion = op.get_bind()

    # 1. Backfill des écritures directes dépourvues d'empreinte. Le chemin worker renseigne
    #    déjà la colonne ; seul le chemin direct l'ignorait.
    while True:
        lignes = connexion.execute(
            text("SELECT id, content FROM memories "
                 "WHERE content_hash IS NULL AND source_event_id IS NULL "
                 "LIMIT :lot"),
            {"lot": TAILLE_LOT},
        ).fetchall()
        if not lignes:
            break
        for identifiant, contenu in lignes:
            connexion.execute(
                text("UPDATE memories SET content_hash = :h WHERE id = :i"),
                {"h": content_hash(contenu), "i": identifiant},
            )

    # 2. Dédoublonnage préalable : un index unique refuse de se créer sur des collisions
    #    existantes. Les doublons déjà en base sont ARCHIVÉS, jamais supprimés — on garde la
    #    ligne la plus ancienne active, cohérent avec le fait que l'API rend maintenant la
    #    ligne gagnante. Un `DELETE` ferait perdre les arêtes du graphe en cascade.
    op.execute("""
        UPDATE memories SET status = 'archived'
        WHERE id IN (
            SELECT id FROM (
                SELECT id, row_number() OVER (
                    PARTITION BY tenant_id, agent_id, content_hash ORDER BY created_at
                ) AS rang
                FROM memories
                WHERE status = 'active' AND source_event_id IS NULL AND content_hash IS NOT NULL
            ) classe WHERE rang > 1
        )
    """)

    # 3. Les deux index. `create_memory` insère avec un `ON CONFLICT DO NOTHING` SANS cible,
    #    justement parce qu'ils sont deux et qu'une clause ne peut en nommer qu'un.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_direct_content
        ON memories(tenant_id, agent_id, content_hash)
        WHERE source_event_id IS NULL AND status = 'active' AND content_hash IS NOT NULL
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_direct_idempotency
        ON memories(tenant_id, agent_id, idempotency_key)
        WHERE source_event_id IS NULL AND status = 'active' AND idempotency_key IS NOT NULL
    """)


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: data-bearing schema migration")
