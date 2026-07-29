#!/usr/bin/env python3
"""Crée une clé API SynaptiQ pour un tenant et affiche la clé en clair (une seule fois).

Usage :
    python scripts/create_api_key.py --name "agent-ouroboros-prod"
    python scripts/create_api_key.py --name "lecture-seule" --scopes read
    python scripts/create_api_key.py --name "agent-A" --agents agentA
    python scripts/create_api_key.py --name "admin-purge" --scopes read write admin

Seul le hash SHA256 est stocké en base ; conserve la clé affichée, elle n'est pas récupérable.

Deux restrictions à connaître (ajoutées le 29/07) :
  - `--scopes` porte les permissions. Défaut : `read write`. **`admin` (purge RGPD) doit
    être demandé explicitement** — une clé d'agent ne doit pas pouvoir vider l'instance.
  - `--agents` restreint la clé à une liste d'agents. Sans ce drapeau, la clé accède à tous
    les agents de son tenant (comportement historique, cas normal d'une instance mono-agent).
"""
import argparse
import hashlib
import os
import secrets

import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db"
)


def main() -> None:
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
        help="Restreindre la clé à ces agent_id (défaut : tous les agents du tenant).",
    )
    args = parser.parse_args()

    raw = "sk-synaptiq-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, tenant_id, name, scopes, agent_scope) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id;",
                (key_hash, args.tenant, args.name, args.scopes, args.agents),
            )
            key_id = cur.fetchone()[0]
            conn.commit()
    finally:
        conn.close()

    print(f"Clé API créée (id={key_id}) pour le tenant '{args.tenant}'.")
    print(f"  permissions : {' '.join(args.scopes)}")
    print(f"  agents      : {' '.join(args.agents) if args.agents else 'tous'}")
    print("Clé en clair (à copier MAINTENANT, non stockée) :")
    print(f"  {raw}")
    print("\nUtilisation :  Authorization: Bearer " + raw)


if __name__ == "__main__":
    main()
