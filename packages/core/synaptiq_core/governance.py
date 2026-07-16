"""SynaptiQ — gouvernance mémoire (contradictions, supersession).

Partagé par l'API et le worker.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from synaptiq_core.embeddings import to_pgvector

logger = logging.getLogger("synaptiq-core.governance")

# Seuil de similarité cosinus au-delà duquel deux préférences sont jugées « sur le même sujet »
CONTRADICTION_SIM_THRESHOLD = float(os.getenv("CONTRADICTION_SIM_THRESHOLD", "0.8"))


def handle_contradictions(
    cur,
    tenant_id: str,
    agent_id: str,
    new_memory: dict,
    new_embedding: Optional[List[float]] = None,
    threshold: float = CONTRADICTION_SIM_THRESHOLD,
) -> int:
    """Archive les préférences actives EN CONFLIT lors de l'arrivée d'une nouvelle préférence.

    Contrairement à l'ancien comportement (qui archivait TOUTES les préférences
    actives de l'agent, quel que soit le sujet), on ne cible désormais que les
    préférences **sémantiquement proches** de la nouvelle (similarité cosinus
    >= `threshold`). Deux préférences sur des sujets différents coexistent donc.

    À appeler AVANT d'insérer la nouvelle mémoire (sinon elle s'archiverait elle-même).
    Retourne le nombre de préférences archivées.
    """
    if new_memory.get("type") != "semantic" or new_memory.get("subtype") != "preference":
        return 0

    # Repli de sûreté : sans embedding, on ne peut pas scoper -> on n'archive rien
    # (préférable à un archivage en masse destructeur de données).
    if not new_embedding:
        logger.warning("handle_contradictions sans embedding : aucun archivage (scoping impossible).")
        return 0

    logger.info("Contradictions (scopé sémantiquement) pour : %s", new_memory.get("content"))
    vec = to_pgvector(new_embedding)
    archive_query = """
        UPDATE memories
        SET status = 'archived', updated_at = CURRENT_TIMESTAMP
        WHERE id IN (
            SELECT id FROM memories
            WHERE tenant_id = %s
              AND agent_id = %s
              AND type = 'semantic'
              AND subtype = 'preference'
              AND status = 'active'
              AND (1 - (embedding <=> %s::vector)) >= %s
        );
    """
    cur.execute(archive_query, (tenant_id, agent_id, vec, threshold))
    archived = cur.rowcount if cur.rowcount is not None else 0
    logger.info("%d préférence(s) proche(s) archivée(s) (seuil=%.2f).", archived, threshold)
    return archived
