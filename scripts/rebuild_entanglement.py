#!/usr/bin/env python3
"""Reconstruit le graphe d'intrication `entangled_with` d'un agent, ou de tous.

Usage :
    python scripts/rebuild_entanglement.py --dry-run
    python scripts/rebuild_entanglement.py --agent claude_code_orchestrator
    python scripts/rebuild_entanglement.py --agent mon_agent --threshold 0.62
    python scripts/rebuild_entanglement.py --all --threshold 0.62 --yes

`--all` et `--purge` agissent en masse : ils exigent `--yes`. La base visee vient de
`DATABASE_URL` (`.env` racine ou environnement) et NON d'un defaut code en dur.

## Pourquoi ce script est nécessaire, et pas seulement pratique

Deux situations qu'aucune écriture ne rattrape jamais d'elle-même :

1. **Les instances antérieures au 01/08.** Jusque-là, seul le worker tissait des arêtes. Tout
   agent qui écrivait par `store_memory` (donc `POST /v1/memories`) a une mémoire sans aucun
   voisin, et rien ne la remplira rétroactivement — l'intrication est un effet d'ÉCRITURE.
   Mesuré sur une instance réelle : 28 souvenirs, 0 arête, après des semaines d'usage.

2. **Tout changement de `QEM_ENTANGLE_THRESHOLD`.** Le seuil ne s'applique qu'aux écritures
   suivantes. Le baisser sans reconstruire laisse la mémoire existante aussi peu connectée
   qu'avant, avec l'illusion d'avoir corrigé le problème. Et ce changement est fréquent : le
   défaut `0.7` est calibré sur des corpus anglophones et ne se transpose pas (mesuré : 8
   arêtes à 0.70 contre 52 à 0.62 sur 55 souvenirs français en MiniLM multilingue).

## Deux garde-fous délibérés

**Aucune arête `supersedes_by` n'est produite.** Le chemin d'écriture en crée une entre
`coding_best_practices` et `code_error_resolution`, mais la phase d'interférence ANNULE la
cible d'un `supersedes_by` : rejouer cette règle en masse supprimerait silencieusement des
souvenirs d'erreurs résolues encore valides. Supprimer doit rester un acte d'écriture, jamais
un effet de bord d'un script de maintenance.

**Rien n'est supprimé.** Le script ajoute (`ON CONFLICT DO NOTHING`), il ne retire pas les
arêtes devenues sous le seuil. Relancer avec un seuil PLUS HAUT ne resserre donc pas le graphe
— utiliser `--purge` pour cela, explicitement, et en sachant que ça détruit aussi les arêtes
qu'une écriture avait légitimement posées.
"""
import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "packages", "core"))
from synaptiq_core.entanglement import VOISINS_EXAMINES, seuil_intrication

# `.env` RACINE, comme `main.py` / `worker.py` / `relay.py`. Un `load_dotenv()` nu cherche à
# partir du répertoire COURANT : lancé depuis `scripts/`, il ne trouvait rien, et le défaut
# ci-dessous décidait seul de la base visée. Pour un script qui sait faire `--all --purge`,
# c'est le pire endroit du dépôt où laisser un défaut implicite.
_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_RACINE, ".env"))
# Défaut sur `synaptiq_dev` : ce dépôt est celui de DÉVELOPPEMENT. `synaptiq_db` est la base
# de production servie par l'instance `C:\\Users\\jimmy\\synaptiq`.
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev"
)
TENANT = os.getenv("SYNAPTIQ_TENANT", "default")

# Une seule requête ensembliste plutôt qu'une boucle Python : `CROSS JOIN LATERAL` rejoue le
# `ORDER BY embedding <=> ... LIMIT n` du chemin d'écriture pour CHAQUE souvenir, en profitant
# de l'index HNSW. Les collections dont `entangle` est faux sont exclues par la jointure sur
# `memory_collections` — un rayon marqué non structurant ne doit pas se retrouver tissé ici.
SQL_RECONSTRUCTION = """
INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
SELECT s.id, n.id, 'entangled_with', n.sim
FROM memories s
LEFT JOIN memory_collections mc
       ON mc.tenant_id = s.tenant_id AND mc.agent_id = s.agent_id AND mc.name = s.subtype
CROSS JOIN LATERAL (
    SELECT m.id, (1 - (m.embedding <=> s.embedding)) AS sim
    FROM memories m
    WHERE m.tenant_id = s.tenant_id AND m.agent_id = s.agent_id
      AND m.id <> s.id AND m.status = 'active'
    ORDER BY m.embedding <=> s.embedding
    LIMIT %(voisins)s
) n
WHERE s.tenant_id = %(tenant)s
  AND (%(agent)s IS NULL OR s.agent_id = %(agent)s)
  AND s.status = 'active'
  AND n.sim > %(seuil)s
  -- Collection déclarée : son propre flag décide. Collection libre : défaut historique de la
  -- famille (`procedural` et `semantic` structurent, `episodic` et `working` non).
  AND COALESCE(mc.entangle, s.type IN ('procedural', 'semantic'))
ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING;
"""

SQL_ETAT = """
SELECT s.agent_id,
       count(DISTINCT s.id) AS souvenirs,
       count(r.source_memory_id) AS aretes,
       round(count(r.source_memory_id)::numeric / greatest(count(DISTINCT s.id), 1), 2) AS ratio
FROM memories s
LEFT JOIN relationships r ON r.source_memory_id = s.id AND r.relation_type = 'entangled_with'
WHERE s.tenant_id = %(tenant)s AND s.status = 'active'
  AND (%(agent)s IS NULL OR s.agent_id = %(agent)s)
GROUP BY 1 ORDER BY 2 DESC;
"""

SQL_DISTRIBUTION = """
SELECT round(n.sim::numeric, 2) AS similarite, count(*) AS paires
FROM memories s
CROSS JOIN LATERAL (
    SELECT (1 - (m.embedding <=> s.embedding)) AS sim
    FROM memories m
    WHERE m.tenant_id = s.tenant_id AND m.agent_id = s.agent_id
      AND m.id <> s.id AND m.status = 'active'
    ORDER BY m.embedding <=> s.embedding
    LIMIT %(voisins)s
) n
WHERE s.tenant_id = %(tenant)s AND s.status = 'active'
  AND (%(agent)s IS NULL OR s.agent_id = %(agent)s)
GROUP BY 1 ORDER BY 1 DESC;
"""

SQL_PURGE = """
DELETE FROM relationships r
USING memories s
WHERE r.source_memory_id = s.id AND r.relation_type = 'entangled_with'
  AND s.tenant_id = %(tenant)s
  AND (%(agent)s IS NULL OR s.agent_id = %(agent)s);
"""


def _masquer(url: str) -> str:
    """`postgres://user:pass@hote:port/base` -> `hote:port/base`.

    La base visee doit etre LISIBLE avant toute operation en masse (c'est tout l'enjeu de la
    separation prod/dev), mais le mot de passe n'a rien a faire dans une sortie console.
    """
    return url.rsplit("@", 1)[-1] if "@" in url else url


def _afficher_etat(cur, params, titre: str) -> None:
    cur.execute(SQL_ETAT, params)
    lignes = cur.fetchall()
    print(f"\n{titre}")
    if not lignes:
        print("  (aucun souvenir actif)")
        return
    print(f"  {'agent':<32} {'souvenirs':>10} {'aretes':>8} {'ratio':>7}")
    for agent, souvenirs, aretes, ratio in lignes:
        print(f"  {agent:<32} {souvenirs:>10} {aretes:>8} {ratio:>7}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cible = ap.add_mutually_exclusive_group(required=True)
    cible.add_argument("--agent", help="Reconstruire pour ce seul agent.")
    cible.add_argument("--all", action="store_true",
                       help="Tous les agents du tenant. A utiliser sciemment.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Seuil de similarite (defaut : QEM_ENTANGLE_THRESHOLD).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Affiche l'etat et la distribution des similarites, n'ecrit rien.")
    ap.add_argument("--purge", action="store_true",
                    help="Supprime les aretes 'entangled_with' existantes AVANT de "
                         "reconstruire. Necessaire pour RESSERRER un graphe, mais detruit "
                         "aussi les aretes posees legitimement a l'ecriture.")
    ap.add_argument("--yes", action="store_true",
                    help="Confirme une operation en masse (--all et/ou --purge).")
    args = ap.parse_args()

    # Deux operations irreversibles a l'echelle du tenant : --purge detruit des aretes que
    # personne ne reconstruira a l'identique, --all touche TOUS les agents de l'instance et
    # pas seulement celui sur lequel on travaille. Elles exigent donc un accord explicite,
    # jamais deduit du contexte. `--dry-run` en est dispense : il finit par un rollback.
    if not args.dry_run and (args.purge or args.all):
        operations = " + ".join(n for n, actif in (("--all", args.all),
                                                   ("--purge", args.purge)) if actif)
        if not args.yes:
            print(f"Refus : {operations} agit en masse sur le tenant '{TENANT}' de la base\n"
                  f"  {_masquer(DATABASE_URL)}\n"
                  "  Relancer avec --dry-run pour inspecter, ou avec --yes pour confirmer.",
                  file=sys.stderr)
            return 2
        print(f"Confirme (--yes) : {operations} sur le tenant '{TENANT}' de la base "
              f"{_masquer(DATABASE_URL)}")

    seuil = args.threshold if args.threshold is not None else seuil_intrication()
    params = {"tenant": TENANT, "agent": args.agent, "seuil": seuil,
              "voisins": VOISINS_EXAMINES}

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            print(f"Tenant '{TENANT}' | cible : {args.agent or 'TOUS les agents'} "
                  f"| seuil {seuil}")
            _afficher_etat(cur, params, "Avant :")

            if args.dry_run:
                cur.execute(SQL_DISTRIBUTION, params)
                print("\nDistribution des similarites des plus proches voisins :")
                print("  (viser un seuil qui laisse environ 1 arete par souvenir)")
                for similarite, paires in cur.fetchall():
                    barre = "#" * min(int(paires), 40)
                    marque = "  <- seuil" if abs(float(similarite) - seuil) < 0.005 else ""
                    print(f"  {similarite:>5} {paires:>5} {barre}{marque}")
                conn.rollback()
                print("\n--dry-run : rien n'a ete ecrit.")
                return 0

            if args.purge:
                cur.execute(SQL_PURGE, params)
                print(f"\nPurge : {cur.rowcount} arete(s) 'entangled_with' supprimee(s).")

            cur.execute(SQL_RECONSTRUCTION, params)
            print(f"Reconstruction : {cur.rowcount} arete(s) ajoutee(s).")
            conn.commit()
            _afficher_etat(cur, params, "Apres :")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
