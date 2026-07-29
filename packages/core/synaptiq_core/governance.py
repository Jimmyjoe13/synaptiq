"""SynaptiQ — gouvernance mémoire (contradictions, supersession).

Partagé par l'API et le worker.
"""
from __future__ import annotations

import logging
import os

from synaptiq_core.contradiction import ContradictionJudge, get_contradiction_judge
from synaptiq_core.embeddings import to_pgvector

logger = logging.getLogger("synaptiq-core.governance")


def _default_threshold() -> float:
    """Seuil de PRÉ-FILTRAGE sémantique, lu à l'appel (et non à l'import).

    Lu dynamiquement pour rester reconfigurable et testable, comme les seuils Q-EM.
    """
    return float(os.getenv("CONTRADICTION_SIM_THRESHOLD", "0.8"))


# Conservé pour la compatibilité des imports existants ; préférer `_default_threshold()`.
CONTRADICTION_SIM_THRESHOLD = _default_threshold()


def handle_contradictions(
    cur,
    tenant_id: str,
    agent_id: str,
    new_memory: dict,
    new_embedding: list[float] | None = None,
    threshold: float | None = None,
    judge: ContradictionJudge | None = None,
) -> list[str]:
    """Archive les préférences que la nouvelle **contredit explicitement**.

    Retourne la liste des ids archivés (vide si aucun).

    ## Ce qui a changé le 29/07 (et pourquoi)

    L'implémentation précédente archivait en un seul UPDATE toute préférence active dont le
    cosinus dépassait le seuil — elle assimilait « proche » à « contradictoire » et
    détruisait donc des préférences parfaitement compatibles (« mails courts » vs « mails
    en français », ~0,85 de cosinus). Silencieusement, à chaque écriture.

    Le déroulé est maintenant en deux temps :

      1. **Pré-filtre sémantique** (SELECT, non destructif) : ne remonter que les
         préférences actives assez proches pour qu'un conflit soit plausible. Sert
         uniquement à borner le nombre d'appels au juge.
      2. **Verdict explicite** : chaque candidate est soumise au juge
         (`packages/core/synaptiq_core/contradiction.py`). Seules celles pour lesquelles il
         répond « contradiction » sont archivées.

    Sans juge configuré (aucun LLM disponible), le verdict est toujours négatif : **rien
    n'est archivé**. Les deux préférences coexistent, ce qui dégrade au pire la
    déduplication — jamais la donnée.

    À appeler AVANT d'insérer la nouvelle mémoire (sinon elle s'archiverait elle-même).
    Les ids retournés servent à tisser les arêtes `supersedes_by` une fois le nouvel id
    connu, ce qui rend la décision traçable et explicable.
    """
    if new_memory.get("type") != "semantic" or new_memory.get("subtype") != "preference":
        return []

    # Repli de sûreté : sans embedding, pas de pré-filtre possible -> on ne touche à rien.
    if not new_embedding:
        logger.warning("handle_contradictions sans embedding : aucun archivage (scoping impossible).")
        return []

    seuil = _default_threshold() if threshold is None else threshold
    verdict = judge if judge is not None else get_contradiction_judge()
    nouveau_contenu = (new_memory.get("content") or "").strip()

    # 1. Pré-filtre sémantique — lecture seule.
    cur.execute(
        """
        SELECT id, content
        FROM memories
        WHERE tenant_id = %s
          AND agent_id = %s
          AND type = 'semantic'
          AND subtype = 'preference'
          AND status = 'active'
          AND (1 - (embedding <=> %s::vector)) >= %s
        """,
        (tenant_id, agent_id, to_pgvector(new_embedding), seuil),
    )
    candidates = cur.fetchall()
    if not candidates:
        return []

    logger.info("%d préférence(s) proche(s) de « %s » : soumission au juge de contradiction.",
                len(candidates), nouveau_contenu[:60])

    # 2. Verdict explicite, préférence par préférence.
    a_archiver = [
        str(mem_id) for mem_id, contenu in candidates
        if verdict(contenu, nouveau_contenu)
    ]

    if not a_archiver:
        logger.info("Aucune contradiction constatée : les %d préférence(s) proche(s) sont conservées.",
                    len(candidates))
        return []

    cur.execute(
        "UPDATE memories SET status = 'archived', updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ANY(%s::uuid[])",
        (a_archiver,),
    )
    logger.info("%d préférence(s) archivée(s) sur verdict de contradiction explicite.", len(a_archiver))
    return a_archiver


def link_supersedes(cur, new_memory_id, superseded_ids: list[str]) -> None:
    """Trace la supersession dans le graphe : `nouveau --(supersedes_by)--> ancien`.

    Sans cette arête, un archivage serait indiscernable d'une disparition : rien en base ne
    dirait POURQUOI la préférence est passée en `archived`, ni laquelle l'a remplacée.
    `qem.apply_contradictions` exploite le même type de relation à la lecture.
    """
    for ancien in superseded_ids:
        cur.execute(
            """
            INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
            VALUES (%s, %s, 'supersedes_by', 1.0)
            ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING
            """,
            (new_memory_id, ancien),
        )
