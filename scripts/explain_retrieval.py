#!/usr/bin/env python3
"""Mesure l'usage réel de l'index vectoriel par les requêtes de retrieval.

Pourquoi ce script existe : l'audit du 28/07 soupçonne la requête hybride de neutraliser
l'index HNSW. La CTE `filtre` y est référencée trois fois ; or PostgreSQL n'inline une CTE
que si elle est référencée UNE seule fois — au-delà, elle est matérialisée, et le tri par
distance ne porte donc plus sur la table mais sur un résultat intermédiaire. L'index
devient inutilisable. Un soupçon ne suffit pas : ce script le tranche avec EXPLAIN.

Usage :
    python scripts/explain_retrieval.py --seed 5000     # peuple puis mesure
    python scripts/explain_retrieval.py                 # mesure seulement
    python scripts/explain_retrieval.py --cleanup       # retire le jeu de mesure

Le jeu de mesure vit sous un tenant dédié (`__explain_bench__`) : il n'entre jamais en
collision avec des données réelles, et `--cleanup` le retire intégralement.
"""
import argparse
import os
import random
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "packages", "core"))
load_dotenv(os.path.join(ROOT, ".env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db"
)
TENANT = "__explain_bench__"
AGENT = "bench_agent"
DIM = int(os.getenv("EMBEDDING_DIM", "384"))
# Graine fixe : deux exécutions comparent les mêmes plans sur les mêmes données.
SEED = 20260729

MOTS = ["postgres", "redis", "migration", "vecteur", "agent", "memoire", "index",
        "requete", "latence", "embedding", "contexte", "graphe", "erreur", "regle"]


def _vecteur_unitaire(rng) -> str:
    vec = [rng.gauss(0, 1) for _ in range(DIM)]
    norme = sum(x * x for x in vec) ** 0.5
    return "[" + ",".join(str(x / norme) for x in vec) + "]"


def seed(conn, n: int) -> None:
    rng = random.Random(SEED)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE tenant_id = %s", (TENANT,))
        for i in range(n):
            contenu = " ".join(rng.sample(MOTS, 6)) + f" enregistrement {i}"
            cur.execute(
                "INSERT INTO memories (tenant_id, agent_id, type, subtype, content, embedding, "
                "confidence, importance, status) "
                "VALUES (%s, %s, 'semantic', 'fact', %s, %s, 1.0, 0.5, 'active')",
                (TENANT, AGENT, contenu, _vecteur_unitaire(rng)),
            )
        conn.commit()
    with conn.cursor() as cur:
        cur.execute("ANALYZE memories")
        conn.commit()
    print(f"[seed] {n} mémoires sous tenant '{TENANT}', ANALYZE fait.")


# ─── Les deux formes de requête comparées ────────────────────────────────────

# AVANT : CTE `filtre` référencée 3 fois (vectoriel, plein_texte, SELECT final).
REQUETE_AVANT = """
    WITH filtre AS (
        SELECT * FROM memories
        WHERE tenant_id = %(tenant)s AND agent_id = %(agent)s
          AND type = ANY(%(types)s) AND status = 'active'
    ),
    vectoriel AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank_vec
        FROM filtre ORDER BY embedding <=> %(vec)s::vector LIMIT %(k)s
    ),
    plein_texte AS (
        SELECT f.id, row_number() OVER (ORDER BY ts_rank(f.content_tsv, q.query) DESC) AS rank_fts
        FROM filtre f, websearch_to_tsquery('simple', %(q)s) AS q(query)
        WHERE f.content_tsv @@ q.query
        ORDER BY ts_rank(f.content_tsv, q.query) DESC LIMIT %(k)s
    )
    SELECT f.id, (1 - (f.embedding <=> %(vec)s::vector)) AS similarity, v.rank_vec, t.rank_fts
    FROM filtre f
    LEFT JOIN vectoriel v ON v.id = f.id
    LEFT JOIN plein_texte t ON t.id = f.id
    WHERE v.rank_vec IS NOT NULL OR t.rank_fts IS NOT NULL;
"""

# APRÈS : forme réellement en production (`_fetch_candidates`, apps/api/main.py). Chaque
# chemin attaque `memories` directement et répète le filtre tenant/agent ; l'ORDER BY porte
# donc sur la table, et le planificateur peut choisir l'index HNSW.
REQUETE_APRES = """
    WITH vectoriel AS (
        SELECT id, row_number() OVER (ORDER BY distance) AS rank_vec
        FROM (
            SELECT id, embedding <=> %(vec)s::vector AS distance
            FROM memories
            WHERE tenant_id = %(tenant)s AND agent_id = %(agent)s
              AND type = ANY(%(types)s) AND status = 'active'
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT %(k)s
        ) v
    ),
    plein_texte AS (
        SELECT id, row_number() OVER (ORDER BY score DESC) AS rank_fts
        FROM (
            SELECT m.id, ts_rank(m.content_tsv, q.query) AS score
            FROM memories m, websearch_to_tsquery('simple', %(q)s) AS q(query)
            WHERE m.tenant_id = %(tenant)s AND m.agent_id = %(agent)s
              AND m.type = ANY(%(types)s) AND m.status = 'active'
              AND m.content_tsv @@ q.query
            ORDER BY ts_rank(m.content_tsv, q.query) DESC
            LIMIT %(k)s
        ) t
    ),
    retenus AS (
        SELECT id, min(rank_vec) AS rank_vec, min(rank_fts) AS rank_fts
        FROM (
            SELECT id, rank_vec, NULL::bigint AS rank_fts FROM vectoriel
            UNION ALL
            SELECT id, NULL::bigint AS rank_vec, rank_fts FROM plein_texte
        ) u
        GROUP BY id
    )
    SELECT m.id, (1 - (m.embedding <=> %(vec)s::vector)) AS similarity,
           r.rank_vec, r.rank_fts
    FROM retenus r JOIN memories m ON m.id = r.id;
"""


def explain(conn, titre: str, requete: str, params: dict) -> str:
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + requete, params)
        plan = "\n".join(ligne[0] for ligne in cur.fetchall())
    utilise_hnsw = "idx_memories_embedding_hnsw" in plan
    seq_scan = "Seq Scan on memories" in plan
    duree = [ligne for ligne in plan.splitlines() if "Execution Time" in ligne]
    print(f"\n{'=' * 72}\n{titre}\n{'=' * 72}")
    print(plan)
    print(f"\n  -> index HNSW utilisé : {'OUI' if utilise_hnsw else 'NON'}")
    print(f"  -> Seq Scan sur memories : {'OUI' if seq_scan else 'non'}")
    if duree:
        print(f"  -> {duree[0].strip()}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Peupler N mémoires de mesure")
    parser.add_argument("--cleanup", action="store_true", help="Supprimer le jeu de mesure")
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        if args.cleanup:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE tenant_id = %s", (TENANT,))
                conn.commit()
            print(f"[cleanup] jeu de mesure '{TENANT}' supprimé.")
            return

        if args.seed:
            seed(conn, args.seed)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM memories WHERE tenant_id = %s", (TENANT,))
            total = cur.fetchone()[0]
        if not total:
            print("Aucune donnée de mesure. Relancer avec --seed 5000.")
            return
        print(f"[mesure] {total} mémoires sous '{TENANT}', dimension {DIM}.")

        rng = random.Random(SEED + 1)
        params = {
            "tenant": TENANT, "agent": AGENT,
            "types": ["semantic", "episodic", "procedural", "working"],
            "vec": _vecteur_unitaire(rng),
            "q": "latence requete index",
            "k": int(os.getenv("RETRIEVAL_CANDIDATES", "50")),
        }

        explain(conn, "AVANT — CTE `filtre` référencée 3 fois (état au 28/07)",
                REQUETE_AVANT, params)
        explain(conn, "APRÈS — chaque chemin attaque `memories` directement",
                REQUETE_APRES, params)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
