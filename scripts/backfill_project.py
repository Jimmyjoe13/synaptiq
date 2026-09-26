"""Rattache les souvenirs EXISTANTS d'un agent à leur projet (lot B, 26/09).

Deux temps, jamais un seul :

  1. `--dry-run` (défaut) : lecture seule. Classe chaque souvenir sans projet par mots-clés
     et écrit un CSV `id, projet_propose, methode, confiance, extrait` à RELIRE. Rien n'est
     écrit en base (connexion `default_transaction_read_only`).
  2. `--apply fichier.csv` : applique le CSV (éventuellement corrigé à la main) en UNE
     transaction, avec une ligne `audit_log`. Seules les lignes dont `projet_propose` est
     renseigné sont appliquées ; une colonne vidée à la relecture laisse le souvenir global.

Règles de classement (volontairement prudentes : un souvenir global à tort reste visible
partout, un souvenir rattaché au mauvais projet disparaît du bon) :
  - collections « transverses » (`preference`, `jimmy_workflow`…) -> toujours globales ;
  - un seul projet cité -> ce projet (confiance haute) ;
  - plusieurs projets, l'un cité au moins deux fois plus que les autres -> celui-là
    (confiance moyenne) ;
  - sinon -> non résolu, laissé global et signalé dans le CSV.

Usage :
  python scripts/backfill_project.py --dsn ... --agent claude_code_orchestrator --out propositions.csv
  python scripts/backfill_project.py --dsn ... --agent claude_code_orchestrator --apply propositions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import psycopg2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "packages", "core"))

from synaptiq_core.project import normaliser_projet  # noqa: E402

# Dictionnaire par défaut (projets de l'instance de Jimmy). Surchargeable par --mots-cles
# (JSON {projet: [motifs regex]}). Les motifs sont cherchés en minuscules, bornés aux mots.
MOTS_CLES_DEFAUT: dict[str, list[str]] = {
    "synaptiq": [r"synaptiq", r"q-em", r"build_context", r"store_memory"],
    "emile": [r"[ée]mile"],
    "elu-scraper": [r"elu-scraper", r"elus?[_ -]scraper", r"carel", r"salesforce"],
    "jobxpress": [r"jobxpress"],
    "dashjoe": [r"dashjoe"],
    "muse": [r"\bmuse\b"],
    "hermes": [r"\bhermes\b", r"richard", r"juliette"],
}

# Collections qui décrivent l'utilisateur ou la façon de travailler : jamais d'un projet.
COLLECTIONS_GLOBALES = {"preference", "jimmy_workflow", "session_log"}


def classer(contenu: str, subtype: str | None,
            mots_cles: dict[str, list[str]]) -> tuple[str | None, str, str]:
    """(projet ou None, méthode, confiance) pour un souvenir."""
    if subtype in COLLECTIONS_GLOBALES:
        return None, "collection_globale", "haute"
    texte = contenu.lower()
    hits = {p: sum(len(re.findall(m, texte)) for m in motifs)
            for p, motifs in mots_cles.items()}
    hits = {p: n for p, n in hits.items() if n}
    if not hits:
        return None, "aucun_projet_cite", "haute"
    if len(hits) == 1:
        return next(iter(hits)), "mot_cle_unique", "haute"
    classes = sorted(hits.items(), key=lambda kv: -kv[1])
    if classes[0][1] >= 2 * classes[1][1]:
        return classes[0][0], "mot_cle_dominant", "moyenne"
    return None, "ambigu:" + ",".join(f"{p}={n}" for p, n in classes), "basse"


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill de memories.project (lot B)")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--out", help="CSV de propositions (mode --dry-run)")
    ap.add_argument("--apply", help="CSV relu à appliquer")
    ap.add_argument("--mots-cles", help="JSON {projet: [motifs regex]}")
    args = ap.parse_args()

    if args.apply:
        return appliquer(args)

    mots_cles = MOTS_CLES_DEFAUT
    if args.mots_cles:
        with open(args.mots_cles, encoding="utf-8") as f:
            mots_cles = json.load(f)

    # Lecture seule imposée côté serveur : le dry-run ne peut rien écrire.
    conn = psycopg2.connect(args.dsn, options="-c default_transaction_read_only=on")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, subtype, content FROM memories
            WHERE tenant_id = %s AND agent_id = %s AND status = 'active' AND project IS NULL
            ORDER BY created_at
        """, (args.tenant, args.agent))
        lignes = cur.fetchall()
    conn.close()

    compte: dict[str, int] = {}
    out = args.out or f"backfill_project_{args.agent}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "projet_propose", "methode", "confiance", "extrait"])
        for mem_id, subtype, contenu in lignes:
            projet, methode, confiance = classer(contenu, subtype, mots_cles)
            compte[projet or "(global)"] = compte.get(projet or "(global)", 0) + 1
            w.writerow([mem_id, projet or "", methode, confiance,
                        contenu[:140].replace("\n", " ")])
    print(f"{len(lignes)} souvenirs sans projet -> {out}")
    for projet, n in sorted(compte.items(), key=lambda kv: -kv[1]):
        print(f"  {projet:15} {n}")
    print("Relire le CSV (vider `projet_propose` pour garder global), puis --apply.")
    return 0


def appliquer(args) -> int:
    """Applique le CSV relu : une transaction, une ligne d'audit, jamais d'écrasement."""
    with open(args.apply, encoding="utf-8", newline="") as f:
        props = [(r["id"], normaliser_projet(r["projet_propose"]))
                 for r in csv.DictReader(f) if r["projet_propose"].strip()]
    if not props:
        print("Aucune proposition renseignée : rien à appliquer.")
        return 0
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            modifies = 0
            for mem_id, projet in props:
                # `project IS NULL` : ne jamais écraser un rattachement déjà posé (par
                # l'agent lui-même, ou par un passage précédent de ce script).
                cur.execute("""
                    UPDATE memories SET project = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND tenant_id = %s AND agent_id = %s AND project IS NULL
                """, (projet, mem_id, args.tenant, args.agent))
                modifies += cur.rowcount
            par_projet: dict[str, int] = {}
            for _, projet in props:
                par_projet[projet] = par_projet.get(projet, 0) + 1
            cur.execute(
                "INSERT INTO audit_log (tenant_id, agent_id, action, actor, details) "
                "VALUES (%s, %s, %s, %s, %s)",
                (args.tenant, args.agent, "backfill_project", "scripts/backfill_project.py",
                 json.dumps({"fichier": os.path.basename(args.apply), "proposes": len(props),
                             "modifies": modifies, "par_projet": par_projet})))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print(f"{modifies}/{len(props)} souvenirs rattachés à leur projet (audit_log écrit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
