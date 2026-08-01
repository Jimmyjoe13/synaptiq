"""Construction du graphe d'intrication — PARTAGÉE par l'API et le worker.

Cette fonction vivait dans `apps/worker/worker.py`, donc seul le chemin `/v1/events` tissait
des arêtes. Conséquence, invisible et durable : un agent qui écrit uniquement par
`store_memory` (donc `POST /v1/memories`) ne construisait **aucun** graphe, et la phase 2 de
Q-EM — la propagation d'activation — tournait sur un ensemble vide. Aucune erreur, aucun log :
le rappel retombait simplement sur la recherche hybride et l'interférence.

Mesuré avant correctif sur une instance réelle : l'agent `antigravity_orchestrator` comptait
28 souvenirs et **0 arête**, après des semaines d'usage.

C'est le troisième cas du même motif dans ce dépôt, après la taxonomie et `content_hash` :
une règle définie dans un seul des deux chemins d'écriture. D'où ce module.

## Ce que le graphe fait, et ne fait pas

`propagate_entanglement` (cf. `qem.py`) n'active un lien que si **les deux extrémités sont
déjà présentes dans le vivier de candidats**. Le graphe RECLASSE donc ce vivier, il ne
l'élargit pas : il promeut un souvenir qui a matché faiblement, il ne peut pas ramener un
souvenir que la recherche hybride n'a pas sorti du tout. Utile pour calibrer les attentes.

Le stockage est dirigé (`nouveau -> voisin`) mais la lecture est bidirectionnelle
(`fetch_relationships` cherche en source OU cible, et l'adjacence de `propagate_entanglement`
l'est aussi). **Une seule arête par paire suffit donc** — ne pas la doubler.
"""
import logging
import os

from synaptiq_core.embeddings import to_pgvector

logger = logging.getLogger(__name__)

__all__ = ["entangle", "seuil_intrication"]

# Nombre de voisins examinés par souvenir. Volontairement bas : au-delà, le graphe se densifie
# sans gain de pertinence et la propagation diffuse l'activation vers du bruit plausible.
VOISINS_EXAMINES = 3

# `LIMIT %s` en paramètre lié plutôt qu'interpolé : la valeur est une constante de ce module,
# mais une requête sans concaténation ne se relit pas pour vérifier qu'elle est sûre.
_SQL_VOISINS = """
    SELECT id, type, subtype, (1 - (embedding <=> %s::vector)) AS similarity
    FROM memories
    WHERE tenant_id = %s AND agent_id = %s AND id != %s AND status = 'active'
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
"""

_SQL_ARETE = """
    INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING;
"""


def seuil_intrication() -> float:
    """Seuil de similarité au-delà duquel deux souvenirs sont intriqués.

    Lu à CHAQUE appel, et non figé à l'import : c'est la convention du dépôt, elle permet à une
    étude d'ablation ou à un test de faire varier une phase sans redéploiement.

    ⚠️ Le défaut `0.7` est calibré sur des corpus anglophones et **ne se transpose pas**. Mesuré
    sur 55 souvenirs français courts avec `paraphrase-multilingual-MiniLM-L12-v2` : 8 arêtes à
    0.70 contre 52 à 0.62, les plus proches voisins plafonnant vers 0.68. Un graphe quasi vide
    ne lève aucune erreur — d'où la jauge `synaptiq_graph_edges_per_memory` et
    `scripts/rebuild_entanglement.py` pour reconstruire après un changement de seuil.
    """
    return float(os.getenv("QEM_ENTANGLE_THRESHOLD", "0.7"))


def entangle(cur, tenant_id: str, agent_id: str, new_mem_id, subtype: str | None,
             embedding, threshold: float | None = None) -> int:
    """Relie un souvenir à ses plus proches voisins sémantiques. Retourne le nombre d'arêtes.

    À appeler APRÈS l'insertion du souvenir (le `id != %s` l'exclut de ses propres voisins),
    dans la MÊME transaction : une arête sans son souvenir n'a pas de sens, et le contraire
    non plus.

    `subtype` porte la seule règle de typage : une bonne pratique résout l'erreur qu'elle
    remplace, et réciproquement — cette paire-là produit un `supersedes_by`, pas une simple
    intrication. Ne jamais rejouer cette règle en MASSE sur des données existantes : la phase
    d'interférence annule la cible d'un `supersedes_by`, donc une reprise en bloc supprimerait
    des souvenirs encore valides. C'est un effet d'ÉCRITURE, pas de maintenance.
    """
    if threshold is None:
        threshold = seuil_intrication()

    embedding_str = to_pgvector(embedding)
    # `ORDER BY embedding <=> %s` et non `ORDER BY similarity DESC` : pgvector n'utilise
    # l'index HNSW que sur l'opérateur de distance. Trier sur l'alias forçait un scan
    # complet des mémoires de l'agent À CHAQUE fait extrait — le coût de l'intrication
    # croissait donc linéairement avec la taille de la mémoire.
    cur.execute(_SQL_VOISINS, (embedding_str, tenant_id, agent_id, new_mem_id, embedding_str,
                               VOISINS_EXAMINES))

    aretes = 0
    for rel_row in cur.fetchall():
        similarity = float(rel_row[3] or 0.0)
        if similarity <= threshold:
            continue
        target_id, target_subtype = rel_row[0], rel_row[2]
        relation_type = "entangled_with"
        inverse = False
        # Une bonne pratique résout/remplace l'erreur associée (et réciproquement).
        if subtype == 'coding_best_practices' and target_subtype == 'code_error_resolution':
            relation_type = "supersedes_by"
        elif subtype == 'code_error_resolution' and target_subtype == 'coding_best_practices':
            relation_type, inverse = "supersedes_by", True

        pair = (target_id, new_mem_id) if inverse else (new_mem_id, target_id)
        cur.execute(_SQL_ARETE, (*pair, relation_type, similarity))
        aretes += 1
        logger.info("Intrication Q-EM : %s --(%s)--> %s (sim=%.2f)",
                    pair[0], relation_type, pair[1], similarity)
    return aretes
