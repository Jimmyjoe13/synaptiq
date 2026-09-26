"""Exporte le corpus d'UN agent vers une fixture de banc de rappel, en LECTURE SEULE.

Pourquoi : le banc de rappel du 26/09 a été joué directement sur l'instance de
production. Il y a modifié `last_accessed_at` (l'API rafraîchit les accès, même en
`explain`), il s'est donc contaminé lui-même, et il n'est pas rejouable. Ce script fige
une photographie de l'agent (souvenirs + vecteurs + arêtes + collections) que
`benchmarks/recall_bench.py` réimporte ensuite dans la base de DEV.

Garanties :
  - Connexion ouverte avec `default_transaction_read_only=on` : PostgreSQL refuse toute
    écriture, même par erreur de ce script.
  - La sortie contient du contenu PRIVÉ (topologie, adresses, décisions). Elle doit vivre
    HORS du dépôt, qui est open source : le script refuse un dossier de sortie situé dans
    le dépôt.

Les requêtes du banc (question + ids gold) sont fournies à part (`--queries`, JSON) ; le
script leur ajoute leur VECTEUR, calculé avec l'embedder configuré (même modèle que les
souvenirs, vérifié : le cosinus entre un vecteur stocké et son recalcul doit valoir 1.000).
Le rejeu n'a ainsi besoin ni de LM Studio ni de la prod.

Usage :
  python scripts/bench_export.py --dsn "postgresql://.../synaptiq_db" \
      --agent claude_code_orchestrator --queries queries.json \
      --out C:/Users/jimmy/synaptiq-bench-private
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import psycopg2
import psycopg2.extras

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

from synaptiq_core.embeddings import get_embedder  # noqa: E402

# Toutes les colonnes utiles au rejeu. `content_tsv` est GÉNÉRÉE (recalculée à l'import),
# `embedding` part en texte pgvector ('[0.1,...]') pour rester du JSON pur.
_COLONNES_MEMOIRE = (
    "id, type, subtype, content, summary, embedding::text AS embedding, confidence, "
    "importance, recency_score, access_count, last_accessed_at, created_at, updated_at, "
    "status, version, provenance, occurred_at, content_hash"
)


def _json_defaut(obj):
    """Sérialise dates et UUID (psycopg2 les rend en objets Python)."""
    if isinstance(obj, dt.datetime | dt.date):
        return obj.isoformat()
    return str(obj)


def _cosinus(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", required=True, help="DSN PostgreSQL de la base SOURCE (lue seulement)")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--queries", required=True, help="JSON : [{qid, query, gold: [ids], note}]")
    ap.add_argument("--out", required=True, help="Dossier de sortie, HORS du dépôt")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    # Garde-fou open source : jamais de contenu privé sous l'arbre du dépôt.
    if os.path.commonpath([out, _ROOT]) == _ROOT:
        print(f"REFUS : {out} est dans le dépôt ({_ROOT}). Choisir un dossier externe.")
        return 2
    os.makedirs(out, exist_ok=True)

    # Lecture seule imposée CÔTÉ SERVEUR : toute écriture lèverait ReadOnlySqlTransaction.
    conn = psycopg2.connect(args.dsn, options="-c default_transaction_read_only=on")
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SHOW default_transaction_read_only")
            if cur.fetchone()["default_transaction_read_only"] != "on":
                raise RuntimeError("Connexion NON lecture seule : export annulé.")

            cur.execute("SELECT now() AS export_now")
            export_now = cur.fetchone()["export_now"]

            cur.execute(
                f"SELECT {_COLONNES_MEMOIRE} FROM memories "  # noqa: S608 (constante du module)
                "WHERE tenant_id = %s AND agent_id = %s ORDER BY created_at",
                (args.tenant, args.agent),
            )
            memoires = cur.fetchall()
            ids = [str(m["id"]) for m in memoires]

            # Seules les arêtes internes au corpus : une arête vers un autre agent n'a
            # rien à faire dans un banc mono-agent (et violerait la clé étrangère).
            cur.execute(
                "SELECT source_memory_id, target_memory_id, relation_type, weight, created_at "
                "FROM relationships "
                "WHERE source_memory_id = ANY(%s::uuid[]) AND target_memory_id = ANY(%s::uuid[])",
                (ids, ids),
            )
            aretes = cur.fetchall()

            cur.execute(
                "SELECT name, family, description, description_embedding::text AS "
                "description_embedding, entangle, packet_key, created_at "
                "FROM memory_collections WHERE tenant_id = %s AND agent_id = %s "
                "AND created_by = 'agent'",
                (args.tenant, args.agent),
            )
            collections = cur.fetchall()
    finally:
        conn.close()

    # Requêtes : on y ajoute le vecteur, calculé par le MÊME modèle que les souvenirs.
    with open(args.queries, encoding="utf-8") as f:
        requetes = json.load(f)
    connus = set(ids)
    for r in requetes:
        manquants = [g for g in r.get("gold", []) if g not in connus]
        if manquants:
            print(f"ATTENTION {r['qid']} : gold absent du corpus exporté : {manquants}")
    embedder = get_embedder()
    vecteurs = embedder.embed([r["query"] for r in requetes])
    for r, v in zip(requetes, vecteurs, strict=True):
        r["vector"] = v

    # Contrôle de cohérence du modèle : un souvenir ré-embarqué doit redonner son vecteur.
    # Même dimension mais modèle différent ne lève AUCUNE erreur ailleurs (incident §3ter).
    if memoires:
        echantillon = memoires[-1]
        stocke = json.loads(echantillon["embedding"])
        recalcule = embedder.embed_one(echantillon["content"])
        cos = _cosinus(stocke, recalcule)
        print(f"Cohérence du modèle d'embedding : cosinus = {cos:.4f}")
        if cos < 0.999:
            print("REFUS : le modèle courant n'est pas celui qui a écrit les vecteurs.")
            return 3

    fixture = {
        "format": 1,
        "source": {"tenant": args.tenant, "agent": args.agent},
        "export_now": export_now,
        "memories": memoires,
        "relationships": aretes,
        "collections": collections,
        "queries": requetes,
    }
    chemin = os.path.join(out, f"recall_{args.agent}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, default=_json_defaut)
    print(f"{len(memoires)} souvenirs, {len(aretes)} arêtes, {len(collections)} collections, "
          f"{len(requetes)} requêtes -> {chemin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
