"""Reclasse par similarité les souvenirs que la migration v0.4 n'a rangés qu'« au jugé ».

La migration `20261122_compartments` convertit les anciens noms de collection connus
(`preference`, `infra_access`, `code_error_resolution`…) avec une table exacte. Pour tous
les autres (`crm_ultimate_deployment`, `scraping_methodologies`, sous-type vide…), elle n'a
eu que la famille d'origine pour décider : un repli correct en moyenne, mais grossier.
Ce script propose un meilleur compartiment en comparant le vecteur DÉJÀ STOCKÉ de chaque
souvenir aux prototypes des compartiments (les mêmes que l'API et le worker).

Deux temps, jamais un seul (même protocole que `backfill_project.py`) :

  1. `--out propositions.csv` (défaut, lecture seule : connexion
     `default_transaction_read_only`) : écrit `id, agent, ancien_nom, actuel, propose,
     similarite, ecart, extrait` pour chaque souvenir dont le classement diffère du
     rangement actuel. À RELIRE : vider la colonne `propose` d'une ligne la laisse en place.
  2. `--apply propositions.csv` : applique le CSV en UNE transaction (compartiment + famille
     déduite), avec une ligne `audit_log`.

⚠️ Les prototypes sont calculés avec l'embedder du `.env` courant. Il DOIT être celui qui a
écrit les vecteurs en base (cf. CLAUDE.md, incident `EMBEDDING_MODEL` du 29/07 : mêmes
dimensions, aucune erreur, classement faux). Le script compare donc d'abord le vecteur
stocké d'un souvenir à son vecteur recalculé et refuse de continuer sous 0,99.

Usage :
  python scripts/reclassify_compartments.py --dsn ... --out propositions.csv
  python scripts/reclassify_compartments.py --dsn ... --apply propositions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import psycopg2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "packages", "core"))

from synaptiq_core.compartments import (  # noqa: E402
    ALIAS,
    ALIAS_PAR_FAMILLE,
    PAR_NOM,
    classer,
    cosinus,
    famille_de,
    prototypes,
)
from synaptiq_core.embeddings import get_embedder  # noqa: E402

# Noms convertis EXACTEMENT par la migration : ceux-là ne sont pas reproposés.
NOMS_EXACTS = set(ALIAS) | {nom for _, nom in ALIAS_PAR_FAMILLE} | set(PAR_NOM)


def _vecteur(texte: str) -> list[float]:
    return [float(x) for x in texte.strip("[]").split(",")] if texte else []


def _verifier_modele(cur, embedder) -> None:
    """Refuse de classer avec un autre modèle que celui des vecteurs stockés."""
    cur.execute("SELECT content, embedding::text FROM memories "
                "WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT 1")
    ligne = cur.fetchone()
    if ligne is None:
        return
    cos = cosinus(_vecteur(ligne[1]), embedder.embed_one(ligne[0]))
    if cos < 0.99:
        sys.exit(f"Modèle d'embedding incohérent avec la base (cosinus {cos:.3f} < 0.99) : "
                 f"vérifier EMBEDDING_MODEL / EMBEDDING_BASE_URL avant de reclasser.")


def proposer(dsn: str, tenant: str, agent: str | None, sortie: str, ecart_min: float) -> int:
    conn = psycopg2.connect(dsn, options="-c default_transaction_read_only=on")
    embedder = get_embedder()
    with conn, conn.cursor() as cur:
        _verifier_modele(cur, embedder)
        protos = prototypes(embedder)
        filtre_agent = "AND agent_id = %s" if agent else ""
        cur.execute(
            # `filtre_agent` : fragment fixe (deux formes), l agent passe en paramètre lié.
            f"SELECT id, agent_id, legacy_subtype, compartment, content, embedding::text "  # noqa: S608
            f"FROM memories WHERE tenant_id = %s {filtre_agent} AND status = 'active' "
            f"AND compartment <> 'croyances' ORDER BY agent_id, created_at",
            (tenant, agent) if agent else (tenant,))
        lignes = cur.fetchall()
    conn.close()

    propositions = []
    for mem_id, agent_id, ancien, actuel, contenu, vecteur in lignes:
        if ancien in NOMS_EXACTS:
            continue
        v = _vecteur(vecteur)
        propose, sim = classer(v, protos)
        # Écart avec le compartiment actuel : sous `ecart_min`, le gain ne vaut pas le
        # déplacement (les prototypes sont proches, le classement serait du bruit).
        ecart = sim - cosinus(v, protos.get(actuel, []))
        if propose == actuel or ecart < ecart_min:
            continue
        propositions.append({
            "id": str(mem_id), "agent": agent_id, "ancien_nom": ancien or "",
            "actuel": actuel, "propose": propose, "similarite": f"{sim:.3f}",
            "ecart": f"{ecart:.3f}", "extrait": (contenu or "")[:140].replace("\n", " "),
        })

    with open(sortie, "w", newline="", encoding="utf-8") as f:
        champs = ["id", "agent", "ancien_nom", "actuel", "propose", "similarite", "ecart",
                  "extrait"]
        ecrivain = csv.DictWriter(f, fieldnames=champs)
        ecrivain.writeheader()
        ecrivain.writerows(propositions)
    examines = sum(1 for li in lignes if li[2] not in NOMS_EXACTS)
    print(f"{len(lignes)} souvenirs lus, {examines} rangés au jugé, "
          f"{len(propositions)} propositions -> {sortie}")
    return 0


def appliquer(dsn: str, tenant: str, fichier: str) -> int:
    with open(fichier, encoding="utf-8") as f:
        lignes = [li for li in csv.DictReader(f) if (li.get("propose") or "").strip()]
    inconnus = {li["propose"] for li in lignes} - set(PAR_NOM) | (
        {"croyances"} & {li["propose"] for li in lignes})
    if inconnus:
        sys.exit(f"Compartiments refusés dans le CSV : {sorted(inconnus)}")
    conn = psycopg2.connect(dsn)
    with conn, conn.cursor() as cur:
        for li in lignes:
            cur.execute(
                "UPDATE memories SET compartment = %s, type = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = %s AND tenant_id = %s",
                (li["propose"], famille_de(li["propose"]), li["id"], tenant))
        cur.execute(
            "INSERT INTO audit_log (tenant_id, agent_id, action, actor, details) "
            "VALUES (%s, NULL, 'reclassify_compartments', 'script', %s)",
            (tenant, json.dumps({"fichier": os.path.basename(fichier),
                                 "souvenirs": len(lignes)})))
    conn.close()
    print(f"{len(lignes)} souvenirs reclassés.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Reclassement par similarité (v0.4)")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--agent", help="Limiter à un agent (défaut : tous)")
    ap.add_argument("--out", default="propositions_compartiments.csv")
    ap.add_argument("--apply", help="CSV relu à appliquer")
    ap.add_argument("--ecart-min", type=float, default=0.02,
                    help="Gain de similarité minimal pour proposer un déplacement")
    args = ap.parse_args()
    if args.apply:
        return appliquer(args.dsn, args.tenant, args.apply)
    return proposer(args.dsn, args.tenant, args.agent, args.out, args.ecart_min)


if __name__ == "__main__":
    raise SystemExit(main())
