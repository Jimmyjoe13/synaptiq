#!/usr/bin/env python3
"""Crée une clé API SynaptiQ pour un tenant et affiche la clé en clair (une seule fois).

Usage :
    python scripts/create_api_key.py --tenant org_01 --name "agent-ouroboros-prod"

Seul le hash SHA256 est stocké en base ; conserve la clé affichée, elle n'est pas récupérable.
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
    args = parser.parse_args()

    raw = "sk-synaptiq-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, tenant_id, name) VALUES (%s, %s, %s) RETURNING id;",
                (key_hash, args.tenant, args.name),
            )
            key_id = cur.fetchone()[0]
            conn.commit()
    finally:
        conn.close()

    print(f"Clé API créée (id={key_id}) pour le tenant '{args.tenant}'.")
    print("Clé en clair (à copier MAINTENANT, non stockée) :")
    print(f"  {raw}")
    print("\nUtilisation :  Authorization: Bearer " + raw)


if __name__ == "__main__":
    main()
