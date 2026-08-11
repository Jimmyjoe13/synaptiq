#!/usr/bin/env python3
"""Crée une clé API SynaptiQ pour un tenant et affiche la clé en clair (une seule fois).

Usage :
    python scripts/create_api_key.py --name "agent-A" --agents agentA
    python scripts/create_api_key.py --name "lecture-seule" --scopes read --agents agentA
    python scripts/create_api_key.py --name "instance-mono-agent" --all-agents
    python scripts/create_api_key.py --name "admin-purge" --scopes read write admin --all-agents

Seul le hash SHA256 est stocké en base ; conserve la clé affichée, elle n'est pas récupérable.

Deux restrictions à connaître (ajoutées le 29/07) :
  - `--scopes` porte les permissions. Défaut : `read write`. **`admin` (purge RGPD) doit
    être demandé explicitement** — une clé d'agent ne doit pas pouvoir vider l'instance.
  - `--agents` restreint la clé à une liste d'agents. **Obligatoire depuis le 11/08**, sauf
    `--all-agents` explicite : le périmètre par défaut valait `NULL`, donc « tous les
    agents », et l'isolation entre agents — promesse centrale du produit — n'était active
    que si l'exploitant y pensait. Un défaut ne doit pas être le mode le plus permissif.
"""
import argparse
import hashlib
import os
import secrets

import psycopg2
from dotenv import load_dotenv

# `.env` RACINE, comme `main.py` / `worker.py` / `rebuild_entanglement.py`. Un
# `load_dotenv()` nu cherche à partir du répertoire COURANT : lancé depuis `scripts/`, il ne
# trouvait rien, et le défaut ci-dessous décidait seul de la base visée.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))
# Défaut sur `synaptiq_dev` : ce dépôt est celui de DÉVELOPPEMENT. `synaptiq_db` est la base
# de production servie par l'instance `C:\Users\jimmy\synaptiq`.
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev"
)


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Créer une clé API SynaptiQ.")
    parser.add_argument(
        "--tenant",
        default=os.getenv("SYNAPTIQ_TENANT", "default"),
        help="Identifiant du tenant (défaut : SYNAPTIQ_TENANT du .env, sinon 'default'). "
             "En instance auto-hébergée, laisser la valeur par défaut.",
    )
    parser.add_argument("--name", default=None, help="Libellé lisible de la clé")
    parser.add_argument(
        "--scopes", nargs="+", default=["read", "write"], choices=["read", "write", "admin"],
        help="Permissions de la clé (défaut : read write). 'admin' autorise la purge RGPD.",
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Restreindre la clé à ces agent_id. OBLIGATOIRE, sauf --all-agents.",
    )
    parser.add_argument(
        "--all-agents", action="store_true",
        help="Échappatoire explicite : clé valable pour TOUS les agents du tenant "
             "(cas légitime d'une instance mono-agent). Incompatible avec --agents.",
    )
    return parser


def resoudre_perimetre(args, parser: argparse.ArgumentParser) -> list[str] | None:
    """Périmètre d'agents de la clé : une liste, ou None pour « tous les agents ».

    Aucun défaut : le choix est EXPLICITE dans les deux sens. Auparavant l'absence de
    `--agents` produisait `agent_scope = NULL`, soit la clé la plus permissive possible —
    et `resolve_agent` la laissait alors agir au nom de n'importe quel agent. L'isolation
    annoncée par le produit n'existait donc que pour l'exploitant qui connaissait le
    drapeau.
    """
    if args.agents and args.all_agents:
        parser.error(
            "--agents et --all-agents s'excluent : soit la clé est bornée à des agents "
            "précis, soit elle vaut pour tous."
        )
    if args.agents:
        return list(args.agents)
    if args.all_agents:
        return None
    parser.error(
        "Périmètre d'agents non précisé. L'isolation entre agents d'une instance repose "
        "sur agent_id : une clé sans périmètre peut lire et écrire la mémoire de TOUS les "
        "agents.\n"
        "  --agents <agent_id> [...]  pour borner la clé (recommandé)\n"
        "  --all-agents               pour l'assumer explicitement (instance mono-agent)"
    )
    return None      # inatteignable : parser.error termine le processus (SystemExit 2)


def main(argv: list[str] | None = None) -> None:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    agents = resoudre_perimetre(args, parser)

    raw = "sk-synaptiq-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, tenant_id, name, scopes, agent_scope) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id;",
                (key_hash, args.tenant, args.name, args.scopes, agents),
            )
            key_id = cur.fetchone()[0]
            conn.commit()
    finally:
        conn.close()

    print(f"Clé API créée (id={key_id}) pour le tenant '{args.tenant}'.")
    print(f"  permissions : {' '.join(args.scopes)}")
    print(f"  agents      : {' '.join(agents) if agents else 'TOUS (--all-agents)'}")
    print("Clé en clair (à copier MAINTENANT, non stockée) :")
    print(f"  {raw}")
    print("\nUtilisation :  Authorization: Bearer " + raw)


if __name__ == "__main__":
    main()
