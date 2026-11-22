"""v0.4.0 — les collections deviennent des COMPARTIMENTS fixes (modèle du cerveau).

Contexte : audit du 22/11/2026. Les collections créées librement par les agents avaient
dégénéré (80 collections, ~25 à un seul souvenir, quasi-doublons, une même collection
étalée sur 3 ou 4 familles donc sans comportement homogène). Elles sont remplacées par
9 compartiments fixes, propriété du moteur (cf. `synaptiq_core.compartments`).

Ce que fait cette migration, dans une seule transaction :
  1. ajoute `memories.compartment` et le remplit à partir de (type, subtype), avec une
     COPIE FIGÉE de la table de correspondance du cœur — une migration ne doit pas
     changer de sens parce que le code a évolué depuis ;
  2. recalcule `memories.type` à partir du compartiment (la famille est désormais
     DÉDUITE ; `working` disparaît) ;
  3. reporte le sujet des croyances (`user_model` / `human_insights`) dans
     `provenance->>'about'` ;
  4. renomme `subtype` en `legacy_subtype` (traçabilité, et matière première de
     `scripts/reclassify_compartments.py` qui reclasse par similarité les souvenirs que
     seul le repli par famille a rangés) ;
  5. pose la contrainte CHECK sur les 9 noms et l'index de lecture ;
  6. supprime `memory_collections`.

Downgrade non supporté (migration porteuse de données, convention du dépôt) :
`legacy_subtype` conserve de quoi revenir en arrière à la main.
"""
from alembic import op

revision = "20261122_compartments"
down_revision = "20260926_perf_monitoring"
branch_labels = None
depends_on = None

# ─── Copie FIGÉE au 22/11/2026 de synaptiq_core.compartments ─────────────────
COMPARTIMENTS = ("reflexes", "decisions", "savoir_faire", "alertes", "environnement",
                 "savoirs", "episodes", "observations", "croyances")

FAMILLE = {
    "reflexes": "procedural", "decisions": "semantic", "savoir_faire": "procedural",
    "alertes": "procedural", "environnement": "semantic", "savoirs": "semantic",
    "episodes": "episodic", "observations": "episodic", "croyances": "reflective",
}

ALIAS_PAR_FAMILLE = {
    ("episodic", "project_state"): "episodes",
    ("semantic", "project_state"): "savoirs",
    ("procedural", "project_state"): "savoirs",
    ("working", "project_state"): "observations",
}

ALIAS = {
    "preference": "reflexes", "preferences": "reflexes", "rule": "reflexes",
    "rules": "reflexes", "jimmy_workflow": "reflexes",
    "decisions": "decisions", "decision": "decisions",
    "architecture-decision": "decisions", "architecture_decision": "decisions",
    "config-decision": "decisions",
    "coding_best_practices": "savoir_faire", "best_practices": "savoir_faire",
    "deploy_playbooks": "savoir_faire", "deploy_playbook": "savoir_faire",
    "harness_procedures": "savoir_faire",
    "code_error_resolution": "alertes", "errors": "alertes", "error_resolution": "alertes",
    "resolution-erreur": "alertes", "bugfix": "alertes", "harness_errors": "alertes",
    "infra_access": "environnement",
    "fact": "savoirs", "facts": "savoirs", "project_info": "savoirs",
    "harness_facts": "savoirs", "reflexions_humaines": "savoirs",
    "interaction": "episodes", "session_log": "episodes", "milestone": "episodes",
    "harness_episodes": "episodes",
    "scratch": "observations", "examples": "observations", "audit_reports": "observations",
    "user_model": "croyances", "human_insights": "croyances",
}

REPLI_FAMILLE = {
    "semantic": "savoirs", "episodic": "episodes", "procedural": "savoir_faire",
    "working": "observations", "reflective": "croyances",
}


def _lit(valeur: str) -> str:
    """Littéral SQL. Les valeurs sont des constantes de CE fichier, jamais des données."""
    return "'" + valeur.replace("'", "''") + "'"


def _case_compartiment() -> str:
    """CASE SQL : (famille, nom) exact > nom > repli par famille > savoirs."""
    branches = []
    for (famille, nom), cible in ALIAS_PAR_FAMILLE.items():
        branches.append(f"WHEN type = {_lit(famille)} AND subtype = {_lit(nom)} "
                        f"THEN {_lit(cible)}")
    for nom in COMPARTIMENTS:  # un sous-type qui porte déjà un nom de compartiment
        branches.append(f"WHEN subtype = {_lit(nom)} THEN {_lit(nom)}")
    for nom, cible in ALIAS.items():
        branches.append(f"WHEN subtype = {_lit(nom)} THEN {_lit(cible)}")
    for famille, cible in REPLI_FAMILLE.items():
        branches.append(f"WHEN type = {_lit(famille)} THEN {_lit(cible)}")
    return "CASE " + " ".join(branches) + " ELSE 'savoirs' END"


def upgrade() -> None:
    # 1. Colonne + remplissage.
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS compartment VARCHAR(20)")
    # Littéraux issus des constantes de CE fichier (cf. `_lit`), aucune donnée externe.
    op.execute(f"UPDATE memories SET compartment = {_case_compartiment()} "  # noqa: S608
               "WHERE compartment IS NULL")

    # 3. Sujet des croyances AVANT de toucher au sous-type (il est lu ici).
    op.execute("""
        UPDATE memories
        SET provenance = COALESCE(provenance, '{}'::jsonb)
            || jsonb_build_object('about',
                 CASE WHEN subtype = 'human_insights' THEN 'humans' ELSE 'user' END)
        WHERE compartment = 'croyances'
    """)

    # 2. Famille déduite du compartiment.
    case_famille = "CASE " + " ".join(
        f"WHEN compartment = {_lit(c)} THEN {_lit(f)}" for c, f in FAMILLE.items()
    ) + " END"
    op.execute(f"UPDATE memories SET type = {case_famille}")  # noqa: S608

    # 4. L'ancien nom de collection est conservé, mais ne pilote plus rien.
    op.execute("ALTER TABLE memories RENAME COLUMN subtype TO legacy_subtype")

    # 5. Contrainte + index de lecture (même forme que idx_memories_lookup sur `type`).
    op.execute("ALTER TABLE memories ALTER COLUMN compartment SET NOT NULL")
    noms = ", ".join(_lit(c) for c in COMPARTIMENTS)
    op.execute(f"ALTER TABLE memories ADD CONSTRAINT chk_memories_compartment "
               f"CHECK (compartment IN ({noms}))")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_compartment "
               "ON memories(tenant_id, agent_id, compartment, status)")

    # 6. Le registre des collections n'a plus d'objet : les compartiments vivent dans le code.
    op.execute("DROP TABLE IF EXISTS memory_collections")


def downgrade() -> None:
    raise RuntimeError("Downgrade intentionally unsupported: data-bearing schema migration "
                       "(memories.legacy_subtype conserve l'ancien rangement).")
