"""Banc de RAPPEL reproductible : /v1/retrieve contre /v1/context/build, sur la base de DEV.

## Ce qu'il mesure

Pour chaque requête dont les souvenirs attendus (gold) sont connus :
  - `/retrieve` (top 10)        : rang du premier gold -> recall@1/3/5/10, MRR ;
  - `/context/build` (budgets)  : le gold est-il DANS le paquet servi au LLM ? à quelle
    position ? — c'est ce que l'agent reçoit réellement ;
  - fréquence des « hubs » : part des paquets qui contiennent au moins un des 5 souvenirs
    les plus souvent sélectionnés, toutes requêtes confondues. Un hub présent partout est
    un souvenir générique qui vole le budget des souvenirs pertinents.

## Pourquoi un banc séparé de la prod

Le premier banc (26/09) tapait l'instance de production : l'API y rafraîchit
`last_accessed_at` à chaque sélection, donc le banc a écrit en prod ET s'est contaminé
lui-même (la récence des souvenirs testés remontait à 1.0 au fil des requêtes). Ici :
  - la fixture (cf. `scripts/bench_export.py`) est réimportée dans `synaptiq_dev`, sous un
    tenant dédié, avec des horodatages décalés pour que les ÂGES soient ceux de l'export ;
  - l'état d'accès est restauré après CHAQUE requête : toutes partent du même état ;
  - les vecteurs de requête sont pré-calculés : ni LM Studio ni la prod ne sont sollicités.

Refuse de tourner si `DATABASE_URL` désigne la base de production.

Usage :
  python benchmarks/recall_bench.py --fixture C:/Users/jimmy/synaptiq-bench-private/recall_claude_code_orchestrator.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

# Réglages de LECTURE alignés sur l'instance de production (lus à l'import de l'API) :
# rejouer avec d'autres réglages mesurerait un autre moteur. Surchargeables par --env.
REGLAGES_PROD = {
    "QEM_RECENCY_HALFLIFE_DAYS": "14",
    "QEM_ENTANGLE_THRESHOLD": "0.68",
    "CONTRADICTION_JUDGE": "off",   # lecture seule : le juge ne sert qu'à l'écriture
}

TENANT_BANC = "bench_recall"
TOP_RETRIEVE = 10
N_HUBS = 5

log = logging.getLogger("recall_bench")


def _parse_ts(v):
    return dt.datetime.fromisoformat(v) if v else None


def importer_fixture(conn, fixture: dict) -> dict:
    """Réimporte la fixture sous TENANT_BANC. Retourne l'état d'accès initial par id.

    Les horodatages sont décalés de (maintenant - export_now) : un souvenir qui avait
    10 jours à l'export en a 10 aujourd'hui, donc la récence calculée est la même.
    """
    import psycopg2.extras

    agent = fixture["source"]["agent"]
    decalage = dt.datetime.now(dt.UTC) - _parse_ts(fixture["export_now"])

    def dec(v):
        t = _parse_ts(v)
        return t + decalage if t else None

    with conn.cursor() as cur:
        # Périmètre du banc UNIQUEMENT (les arêtes partent en cascade avec les mémoires).
        cur.execute("DELETE FROM memories WHERE tenant_id = %s", (TENANT_BANC,))
        cur.execute("DELETE FROM memory_collections WHERE tenant_id = %s", (TENANT_BANC,))

        lignes = [(
            m["id"], TENANT_BANC, agent, m["type"], m["subtype"], m["content"], m["summary"],
            m["embedding"], m["confidence"], m["importance"], m["recency_score"],
            m["access_count"], dec(m["last_accessed_at"]), dec(m["created_at"]),
            dec(m["updated_at"]), m["status"], m["version"], json.dumps(m["provenance"] or {}),
            dec(m["occurred_at"]), m["content_hash"],
        ) for m in fixture["memories"]]
        psycopg2.extras.execute_values(cur, """
            INSERT INTO memories (id, tenant_id, agent_id, type, subtype, content, summary,
                embedding, confidence, importance, recency_score, access_count,
                last_accessed_at, created_at, updated_at, status, version, provenance,
                occurred_at, content_hash)
            VALUES %s""", lignes,
            template="(%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)")

        psycopg2.extras.execute_values(cur, """
            INSERT INTO relationships (source_memory_id, target_memory_id, relation_type,
                weight, created_at) VALUES %s ON CONFLICT DO NOTHING""",
            [(a["source_memory_id"], a["target_memory_id"], a["relation_type"], a["weight"],
              dec(a["created_at"])) for a in fixture["relationships"]])

        for c in fixture["collections"]:
            cur.execute("""
                INSERT INTO memory_collections (tenant_id, agent_id, name, family, description,
                    description_embedding, entangle, packet_key, created_by)
                VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s,'agent')""",
                (TENANT_BANC, agent, c["name"], c["family"], c["description"],
                 c["description_embedding"], c["entangle"], c["packet_key"]))
    conn.commit()
    return {m["id"]: (m["access_count"], dec(m["last_accessed_at"])) for m in fixture["memories"]}


def restaurer_acces(conn, etat: dict) -> None:
    """Remet access_count / last_accessed_at à l'état importé (chaque requête part du même)."""
    import psycopg2.extras

    with conn.cursor() as cur:
        # Tenant écrit en dur dans la requête (constante du module, pas une entrée) :
        # execute_values ne gère qu'un seul %s, celui de VALUES.
        psycopg2.extras.execute_values(cur, f"""
            UPDATE memories m SET access_count = v.ac, last_accessed_at = v.la
            FROM (VALUES %s) AS v(id, ac, la)
            WHERE m.id = v.id::uuid AND m.tenant_id = '{TENANT_BANC}'""",  # noqa: S608
            [(i, ac, la) for i, (ac, la) in etat.items()],
            template="(%s, %s::int, %s::timestamptz)")
    conn.commit()


def appliquer_projets(conn, chemin_csv: str) -> None:
    """Rattache les souvenirs du banc à leur projet d'après un CSV de backfill."""
    import csv

    with open(chemin_csv, encoding="utf-8", newline="") as f:
        props = [(r["projet_propose"].strip(), r["id"]) for r in csv.DictReader(f)
                 if r["projet_propose"].strip()]
    with conn.cursor() as cur:
        cur.executemany(
            f"UPDATE memories SET project = %s WHERE id = %s AND tenant_id = '{TENANT_BANC}'",  # noqa: S608
            props)
    conn.commit()
    log.info("%d souvenirs rattachés à un projet.", len(props))


def rang(ids: list[str], gold: set[str]) -> int | None:
    """Rang (1-based) du premier gold dans la liste, None s'il est absent."""
    for i, mid in enumerate(ids, 1):
        if mid in gold:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Banc de rappel retrieve vs context/build")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--budgets", default="1200,8000")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL (surcharge un réglage)")
    ap.add_argument("--out", default=None, help="JSON de résultats (défaut : à côté de la fixture)")
    ap.add_argument("--quiet", action="store_true")
    # Lot B : rattacher les souvenirs à leur projet (CSV de scripts/backfill_project.py)
    # puis filtrer chaque requête par son projet (JSON {qid: projet}). Mesure le gain du
    # filtre projet à corpus identique.
    ap.add_argument("--projects-csv", default=None)
    ap.add_argument("--query-projects", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    # Garde-fou : ce banc ÉCRIT (import, restauration). Jamais sur la prod.
    dsn = os.getenv("DATABASE_URL", "")
    if "synaptiq_dev" not in dsn:
        print("REFUS : DATABASE_URL ne désigne pas synaptiq_dev.")
        return 2

    reglages = dict(REGLAGES_PROD)
    for kv in args.env:
        k, _, v = kv.partition("=")
        reglages[k] = v
    os.environ.update(reglages)
    os.environ["SYNAPTIQ_TENANT"] = TENANT_BANC

    with open(args.fixture, encoding="utf-8") as f:
        fixture = json.load(f)
    agent = fixture["source"]["agent"]
    requetes = [q for q in fixture["queries"] if q.get("gold")]
    vecteurs = {q["query"]: q["vector"] for q in fixture["queries"]}

    import psycopg2
    from fastapi.testclient import TestClient

    from apps.api import main as api

    # Embedder de banc : sert les vecteurs pré-calculés, lève sur toute requête inconnue
    # (un vecteur inventé fausserait la mesure sans bruit).
    class EmbedderFige:
        dim = 384

        def embed_one(self, texte):
            return vecteurs[texte]

        def embed(self, textes):
            return [vecteurs[t] for t in textes]

    api.get_embedder = lambda: EmbedderFige()
    api.AUTH_REQUIRED = False
    logging.getLogger("synaptiq-api").setLevel(logging.WARNING)

    conn = psycopg2.connect(dsn)
    etat = importer_fixture(conn, fixture)
    if args.projects_csv:
        appliquer_projets(conn, args.projects_csv)
    projets_requetes: dict[str, str] = {}
    if args.query_projects:
        with open(args.query_projects, encoding="utf-8") as f:
            projets_requetes = json.load(f)
    log.info("Fixture importée : %d souvenirs (%s) sous tenant %s. Réglages : %s",
             len(etat), agent, TENANT_BANC, reglages)

    budgets = [int(b) for b in args.budgets.split(",")]
    resultats = []
    with TestClient(api.app) as client:
        for q in requetes:
            gold = set(q["gold"])
            ligne = {"qid": q["qid"], "query": q["query"], "gold": q["gold"]}

            t0 = time.perf_counter()
            projet = projets_requetes.get(q["qid"])
            filtre = {"project": projet} if projet else {}
            ligne["project"] = projet
            r = client.post("/v1/retrieve", json={"agent_id": agent, "query": q["query"],
                                                   "limit": TOP_RETRIEVE, **filtre})
            ligne["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            r.raise_for_status()
            ids = [str(m["id"]) for m in r.json()["memories"]]
            ligne["retrieve_rank"] = rang(ids, gold)
            restaurer_acces(conn, etat)

            for b in budgets:
                t0 = time.perf_counter()
                r = client.post("/v1/context/build", json={
                    "agent_id": agent, "session_id": "bench", "task": q["query"],
                    "query": q["query"], "constraints": {"max_tokens": b, **filtre},
                    "explain": True,
                })
                ms = round((time.perf_counter() - t0) * 1000, 1)
                r.raise_for_status()
                corps = r.json()
                sel = [str(i) for i in corps["selected_memory_ids"]]
                ligne[f"ctx{b}"] = {"rank": rang(sel, gold), "n": len(sel), "ms": ms,
                                    "tokens": corps.get("token_estimate"), "selected": sel}
                restaurer_acces(conn, etat)
            resultats.append(ligne)
            log.info("%-4s retrieve=%-4s %s", q["qid"], ligne["retrieve_rank"],
                     "  ".join(f"ctx{b}={ligne[f'ctx{b}']['rank']}" for b in budgets))
    conn.close()

    # ── Agrégats ────────────────────────────────────────────────────────────────
    n = len(resultats)

    def recall(rangs, k):
        return sum(1 for x in rangs if x is not None and x <= k) / n

    def mrr(rangs):
        return sum(1 / x for x in rangs if x) / n

    agr = {"n": n}
    rr = [x["retrieve_rank"] for x in resultats]
    agr["retrieve"] = {f"recall@{k}": round(recall(rr, k), 3) for k in (1, 3, 5, 10)}
    agr["retrieve"]["mrr"] = round(mrr(rr), 3)
    for b in budgets:
        rc = [x[f"ctx{b}"]["rank"] for x in resultats]
        # Fréquence des hubs : les N_HUBS ids les plus sélectionnés, et la part des paquets
        # qui en contiennent au moins un.
        compte = collections.Counter(i for x in resultats for i in x[f"ctx{b}"]["selected"])
        hubs = {i for i, _ in compte.most_common(N_HUBS)}
        freq_hub = sum(1 for x in resultats if hubs & set(x[f"ctx{b}"]["selected"])) / n
        agr[f"ctx{b}"] = {
            "gold_dans_paquet": round(sum(1 for x in rc if x) / n, 3),
            "recall@5": round(recall(rc, 5), 3),
            "mrr": round(mrr(rc), 3),
            "freq_hub": round(freq_hub, 3),
            "hub_max_share": round(compte.most_common(1)[0][1] / n, 3) if compte else 0,
            "n_moyen": round(sum(x[f"ctx{b}"]["n"] for x in resultats) / n, 1),
        }

    sortie = args.out or os.path.join(os.path.dirname(os.path.abspath(args.fixture)),
                                      "recall_bench_results.json")
    with open(sortie, "w", encoding="utf-8") as f:
        json.dump({"reglages": reglages, "agregats": agr, "requetes": resultats}, f,
                  ensure_ascii=False, indent=1)
    print(json.dumps(agr, ensure_ascii=False, indent=1))
    print(f"-> {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
