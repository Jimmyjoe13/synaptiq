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

## Ce module n'émet QUE des arêtes `entangled_with` (corrigé le 11/08)

Il en émettait une seconde sorte : `supersedes_by`, dès qu'une `coding_best_practices` et une
`code_error_resolution` dépassaient le seuil de cosinus. C'était le motif « similaire ⇒
contradictoire » que le lot F5 a précisément banni de `governance.handle_contradictions`, où
une supersession exige désormais le verdict EXPLICITE d'un juge fail-closed
(`synaptiq_core.contradiction`). Ici, aucun juge : le cosinus décidait seul d'une destruction
persistante, et `apply_contradictions` annulait ensuite le souvenir le plus ANCIEN du couple
— donc la bonne pratique, à chaque `build_context` où les deux remontaient, sans un log
d'alerte. Le `inverse=True` censé l'éviter n'avait aucun effet : l'interférence ignorait le
sens de l'arête.

Une bonne pratique et le journal d'erreur qui l'a motivée ne sont d'ailleurs pas
contradictoires : ils sont **complémentaires**. Les relier par `entangled_with` est le bon
geste — et le seul utile, puisque `propagate_entanglement` ne lit que ce type d'arête : la
supersession les excluait de la propagation en plus d'en détruire une.

Toute supersession passe donc par `governance` (juge explicite + trace `link_supersedes`).
C'est aussi ce que `scripts/rebuild_entanglement.py` fait depuis toujours ; les deux chemins
sont enfin cohérents.
"""
import logging
import os

from synaptiq_core.embeddings import to_pgvector
from synaptiq_core.project import activer_scan_iteratif, filtre_projet, mode_scan_iteratif

logger = logging.getLogger(__name__)

__all__ = ["RELATION_INTRICATION", "entangle", "seuil_intrication"]

# Nombre de voisins examinés par souvenir. Volontairement bas : au-delà, le graphe se densifie
# sans gain de pertinence et la propagation diffuse l'activation vers du bruit plausible.
VOISINS_EXAMINES = 3

# Le SEUL type d'arête que le tissage produit. Nommé plutôt que répété en littéral : c'est
# aussi le seul type que `propagate_entanglement` sait lire.
RELATION_INTRICATION = "entangled_with"

# `LIMIT %s` en paramètre lié plutôt qu'interpolé : la valeur est une constante de ce module,
# mais une requête sans concaténation ne se relit pas pour vérifier qu'elle est sûre.
# `{filtre_projet}` : fragment fixe de `project.filtre_projet` (vide pour un souvenir global).
_SQL_VOISINS = """
    SELECT id, type, subtype, (1 - (embedding <=> %s::vector)) AS similarity
    FROM memories
    WHERE tenant_id = %s AND agent_id = %s AND id != %s AND status = 'active' {filtre_projet}
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
"""

_SQL_ARETE = """
    INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING;
"""

# Version batch pour execute_values : un seul aller-retour SQL pour tous les voisins.
# `%s` dans le template est remplacé par execute_values avec les tuples de valeurs.
_SQL_ARETE_BATCH = """
    INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
    VALUES %s
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
             embedding, threshold: float | None = None, project: str | None = None) -> int:
    """Relie un souvenir à ses plus proches voisins sémantiques. Retourne le nombre d'arêtes.

    À appeler APRÈS l'insertion du souvenir (le `id != %s` l'exclut de ses propres voisins),
    dans la MÊME transaction : une arête sans son souvenir n'a pas de sens, et le contraire
    non plus.

    **Toutes les arêtes produites ici sont des `entangled_with`** : le tissage constate une
    proximité sémantique, il ne prononce aucun jugement de valeur entre deux souvenirs. Une
    supersession détruit de la donnée à la lecture ; elle exige un verdict explicite et relève
    de `governance` (cf. l'en-tête du module). Ne pas réintroduire de règle de typage ici.

    `subtype` reste au contrat d'appel (les deux chemins d'écriture le passent) et n'est plus
    utilisé que pour la journalisation : le garder évite de toucher `apps/` et laisse la porte
    ouverte à un pré-filtre PAR TYPE — mais un pré-filtre ne serait toujours pas un verdict.

    `project` (lot B, 26/09) : un souvenir de projet n'est relié qu'aux souvenirs du MÊME
    projet et aux globaux. Sans cela, la propagation d'activation ferait passer le rappel
    d'un projet à l'autre par le graphe, annulant le filtre posé à la recherche. Un
    souvenir global (`project` None) peut, lui, être relié à tout projet.
    """
    if threshold is None:
        threshold = seuil_intrication()

    embedding_str = to_pgvector(embedding)
    # `ORDER BY embedding <=> %s` et non `ORDER BY similarity DESC` : pgvector n'utilise
    # l'index HNSW que sur l'opérateur de distance. Trier sur l'alias forçait un scan
    # complet des mémoires de l'agent À CHAQUE fait extrait — le coût de l'intrication
    # croissait donc linéairement avec la taille de la mémoire.
    fragment, p_projet = filtre_projet(project, include_global=True)
    if fragment:
        activer_scan_iteratif(cur, mode_scan_iteratif())
    # Fragment choisi parmi trois formes fixes par `filtre_projet` : aucune donnée d'appelant.
    cur.execute(_SQL_VOISINS.format(filtre_projet=fragment),
                (embedding_str, tenant_id, agent_id, new_mem_id, *p_projet, embedding_str,
                 VOISINS_EXAMINES))

    aretes = 0
    voisins = []
    for rel_row in cur.fetchall():
        similarity = float(rel_row[3] or 0.0)
        if similarity <= threshold:
            continue
        target_id, target_subtype = rel_row[0], rel_row[2]
        voisins.append((new_mem_id, target_id, RELATION_INTRICATION, similarity))
        logger.info("Intrication Q-EM : %s (%s) --(%s)--> %s (%s, sim=%.2f)",
                    new_mem_id, subtype, RELATION_INTRICATION, target_id, target_subtype,
                    similarity)

    # Phase 4 : pré-calcul des voisins à 2 sauts.
    # La propagation multi-hop (propagate_entanglement) explore le graphe à 2 sauts
    # à chaque build_context. Avec un graphe dense (900 arêtes pour 255 souvenirs),
    # chaque saut coûte un scan SQL + un chargement en mémoire.
    #
    # Le pré-calcul stocke les voisins à 2 sauts dans une table dédiée
    # (`entanglement_2hop`), remplie lors de l'intrication. À la lecture,
    # `propagate_entanglement` peut utiliser cette table au lieu de
    # parcourir le graphe.
    #
    # Le pré-calcul est fait dans la même transaction que l'intrication :
    # si l'intrication échoue, le pré-calcul est annulé aussi.
    if voisins and len(voisins) > 0:
        # Récupérer les voisins à 2 sauts : pour chaque voisin direct,
        # chercher ses propres voisins (à l'exception du nouveau souvenir
        # et des voisins directs déjà connus).
        voisins_directs = [v[1] for v in voisins]
        if voisins_directs:
            # CTE récursive pour trouver les voisins à 2 sauts.
            # On limite à 2 sauts pour éviter l'explosion combinatoire.
            cur.execute(
                """
                WITH RECURSIVE voisins_2hop AS (
                    -- Niveau 1 : voisins directs du nouveau souvenir
                    SELECT r.target_memory_id AS mem_id, 1 AS hop
                    FROM relationships r
                    WHERE r.source_memory_id = ANY(%s::uuid[])
                      AND r.relation_type = 'entangled_with'
                      AND r.target_memory_id != %s
                    UNION
                    -- Niveau 2 : voisins des voisins directs
                    SELECT r.target_memory_id AS mem_id, 2 AS hop
                    FROM relationships r
                    INNER JOIN voisins_2hop v ON r.source_memory_id = v.mem_id
                    WHERE r.relation_type = 'entangled_with'
                      AND r.target_memory_id != %s
                )
                SELECT mem_id, hop FROM voisins_2hop WHERE hop = 2
                LIMIT 20;
                """,
                (voisins_directs, new_mem_id, new_mem_id),
            )
            voisins_2hop = cur.fetchall()
            if voisins_2hop:
                # Stocker les voisins à 2 sauts dans la table de pré-calcul.
                # ON CONFLICT DO NOTHING : si le voisin est déjà connu, ne pas dupliquer.
                # Les curseurs de test (_CurseurDouble) n'ont pas d'attribut `connection` :
                # on retombe sur des INSERTs séquentiels dans ce cas.
                try:
                    from psycopg2.extras import execute_values
                    execute_values(
                        cur,
                        """
                        INSERT INTO entanglement_2hop (memory_id, neighbor_id, hop, weight)
                        VALUES %s
                        ON CONFLICT (memory_id, neighbor_id) DO NOTHING;
                        """,
                        [(new_mem_id, str(row[0]), row[1], 1.0) for row in voisins_2hop],
                    )
                except AttributeError:
                    # Curseur de test ou wrapper sans `connection` : INSERTs séquentiels.
                    for row in voisins_2hop:
                        cur.execute(
                            """
                            INSERT INTO entanglement_2hop (memory_id, neighbor_id, hop, weight)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (memory_id, neighbor_id) DO NOTHING;
                            """,
                            (new_mem_id, str(row[0]), row[1], 1.0),
                        )
                logger.info("Pré-calcul 2-hop : %d voisins à 2 sauts pour %s.",
                            len(voisins_2hop), new_mem_id)

    # Batch INSERT avec execute_values : un seul aller-retour SQL pour tous les voisins.
    # Avec INSERTs séquentiels, chaque voisin coûtait un round-trip réseau + parsing.
    # Les curseurs de test (_CurseurDouble) n'ont pas d'attribut `connection` :
    # on retombe sur des INSERTs séquentiels dans ce cas.
    if voisins:
        try:
            from psycopg2.extras import execute_values
            execute_values(cur, _SQL_ARETE_BATCH, voisins)
        except AttributeError:
            # Curseur de test ou wrapper sans `connection` : INSERTs séquentiels.
            for voisin in voisins:
                cur.execute(_SQL_ARETE, voisin)
        aretes = len(voisins)

    return aretes
