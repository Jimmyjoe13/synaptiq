"""Les collections logiques deviennent des objets, que l'agent peut créer lui-même.

## Ce que ce lot change

Une « collection » n'existait nulle part : c'était le résultat d'un `if/elif` dans
`qem.route_memory`. Un agent ne pouvait donc ni la consulter, ni en créer, ni décider
comment ses souvenirs sont rangés — alors que SynaptiQ est censé être SA mémoire.

`memory_collections` matérialise l'objet. Les sept sections historiques du context_packet
y sont semées comme collections `system`, valables pour tous les agents.

## Pourquoi aucune donnée n'est déplacée

Le découpage retenu épouse le schéma existant :

  - `memories.type`    -> la FAMILLE cognitive (semantic / episodic / procedural / working).
    Fermée, propriété du moteur : elle décide de l'intrication, de la décroissance et de la
    section de repli. Ce n'est pas une catégorie de rangement, c'est un comportement.
  - `memories.subtype` -> le NOM de la collection. Champ libre, propriété de l'agent.

Conséquence heureuse : les sous-types déjà écrits deviennent rétroactivement de vraies
collections. Le backfill ci-dessous les déclare, y compris ceux que l'agent avait inventés
de lui-même (`nana_intelligence_lead_webhook`, `seo_audit_july_2026`…) et qui étaient
jusqu'ici servis dans `facts` sans que personne ne le lui dise.

## Isolation

Une collection `agent` porte son `tenant_id` et son `agent_id` : elle n'est jamais visible
depuis la mémoire d'un autre agent. Une collection `system` les a à NULL et vaut partout.
Deux index uniques PARTIELS plutôt qu'une seule contrainte : dans un index unique
ordinaire, PostgreSQL considère deux NULL comme distincts, ce qui autoriserait plusieurs
collections système homonymes.
"""
import sqlalchemy as sa
from alembic import op

revision = "20260731_collections"
down_revision = "20260729_perf_idx"
branch_labels = None
depends_on = None

# (nom, famille, clé de paquet, intrication, description)
# `entangle` reproduit le défaut historique `QEM_ENTANGLE_TYPES=procedural,semantic` : les
# épisodes bruts sont nombreux et peu discriminants. La nouveauté n'est pas la valeur, c'est
# qu'elle devient décidable collection par collection (câblé au worker au lot 2).
COLLECTIONS_SYSTEME = [
    ("fact", "semantic", "facts", True,
     "Faits stables sur une personne, une entite ou le monde."),
    ("preference", "semantic", "preferences", True,
     "Gouts, choix et preferences explicites de l'utilisateur."),
    ("interaction", "episodic", "episodes", False,
     "Episodes bruts d'interaction, quand rien de durable n'est enonce."),
    ("rule", "procedural", "rules", True,
     "Regles de conduite et procedures a appliquer."),
    ("coding_best_practices", "procedural", "best_practices", True,
     "Regles d'architecture, conventions et bonnes pratiques de code."),
    ("code_error_resolution", "procedural", "errors", True,
     "Erreurs rencontrees et leur resolution."),
    ("scratch", "working", "examples", False,
     "Memoire de travail volatile, exemples ponctuels."),
]

# Section de repli par famille, pour une collection libre. Identique au comportement
# historique de `route_memory` sur un sous-type inconnu.
REPLI_PAR_FAMILLE = {
    "semantic": "facts",
    "episodic": "episodes",
    "procedural": "rules",
    "working": "examples",
}


def upgrade() -> None:
    # `subtype` porte désormais un nom choisi par l'agent : 50 caractères devient étroit
    # pour un libellé métier explicite. Élargissement seul, aucune donnée touchée.
    op.execute("ALTER TABLE memories ALTER COLUMN subtype TYPE VARCHAR(64)")

    op.execute("""CREATE TABLE IF NOT EXISTS memory_collections (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        -- NULL sur les deux = collection systeme, valable pour tous les agents.
        tenant_id VARCHAR(50),
        agent_id VARCHAR(50),
        -- Correspond a memories.subtype.
        name VARCHAR(64) NOT NULL,
        -- Correspond a memories.type. Ferme aux 4 familles cognitives.
        family VARCHAR(20) NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        -- Vecteur de la description. Alimente le controle anti-doublon semantique du
        -- lot 4 : a la creation, on compare la description proposee a celles existantes
        -- pour refuser une collection qui en double une autre. Nullable tant que le lot 4
        -- n'est pas fait.
        description_embedding VECTOR(384),
        -- Cette collection tisse-t-elle des aretes entangled_with ? Remplacera la variable
        -- globale QEM_ENTANGLE_TYPES au lot 2.
        entangle BOOLEAN NOT NULL DEFAULT true,
        -- Section du context_packet ou ces souvenirs sont servis.
        packet_key VARCHAR(64) NOT NULL,
        created_by VARCHAR(10) NOT NULL DEFAULT 'agent',
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT chk_collection_family
            CHECK (family IN ('semantic','episodic','procedural','working')),
        CONSTRAINT chk_collection_origin
            CHECK (created_by IN ('system','agent')),
        -- Une collection d'agent DOIT etre attribuee ; une collection systeme ne doit
        -- jamais l'etre. Sans cette contrainte, une ligne mal formee deviendrait soit
        -- invisible, soit visible par tous.
        CONSTRAINT chk_collection_scope CHECK (
            (created_by = 'system' AND tenant_id IS NULL AND agent_id IS NULL)
            OR (created_by = 'agent' AND tenant_id IS NOT NULL AND agent_id IS NOT NULL)
        )
    )""")

    # Index uniques PARTIELS : un index ordinaire sur (tenant_id, agent_id, name) laisserait
    # passer plusieurs collections systeme homonymes, PostgreSQL traitant chaque NULL comme
    # une valeur distincte.
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_collection_systeme "
               "ON memory_collections(name, family) WHERE created_by = 'system'")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_collection_agent "
               "ON memory_collections(tenant_id, agent_id, name, family) "
               "WHERE created_by = 'agent'")
    # Chemin de lecture du registre : toutes les collections d'un (tenant, agent).
    op.execute("CREATE INDEX IF NOT EXISTS idx_collections_perimetre "
               "ON memory_collections(tenant_id, agent_id)")

    # Paramètres LIÉS et non interpolés : les descriptions contiennent des apostrophes
    # françaises (« d'interaction », « de l'utilisateur ») qui fermeraient le littéral SQL.
    insertion = sa.text(
        "INSERT INTO memory_collections "
        "(name, family, packet_key, entangle, description, created_by) "
        "VALUES (:nom, :famille, :cle, :entangle, :description, 'system') "
        "ON CONFLICT DO NOTHING"
    )
    for nom, famille, cle, entangle, description in COLLECTIONS_SYSTEME:
        op.get_bind().execute(insertion, {
            "nom": nom, "famille": famille, "cle": cle,
            "entangle": entangle, "description": description,
        })

    # ── Backfill : les sous-types deja ecrits deviennent de vraies collections ──
    # Elles etaient jusqu'ici servies dans la section de repli de leur famille sans que
    # l'agent en soit informe. On les declare avec ce meme routage : le comportement ne
    # change pas, mais la collection devient VISIBLE et modifiable par son proprietaire.
    reprise = sa.text("""
        INSERT INTO memory_collections
            (tenant_id, agent_id, name, family, packet_key, entangle, description,
             created_by)
        SELECT DISTINCT m.tenant_id, m.agent_id, m.subtype, :famille, :repli, :entangle,
               'Collection reprise automatiquement des memoires existantes.', 'agent'
        FROM memories m
        WHERE m.type = :famille
          AND m.subtype IS NOT NULL AND m.subtype <> ''
          AND NOT EXISTS (
              SELECT 1 FROM memory_collections c
              WHERE c.created_by = 'system'
                AND c.name = m.subtype AND c.family = m.type
          )
        ON CONFLICT DO NOTHING
    """)
    for famille, repli in REPLI_PAR_FAMILLE.items():
        op.get_bind().execute(reprise, {
            "famille": famille, "repli": repli,
            "entangle": famille in ("procedural", "semantic"),
        })


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory_collections")
    # `subtype` n'est PAS ramene a VARCHAR(50) : des noms plus longs peuvent avoir ete
    # ecrits entre-temps, et les tronquer detruirait le rangement de l'agent.
