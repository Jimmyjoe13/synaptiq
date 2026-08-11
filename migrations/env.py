import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

# Le `.env` racine est la source unique de configuration du dépôt (comme pour `main.py`,
# `worker.py` et `relay.py`). Sans ce chargement, `os.getenv("DATABASE_URL")` était vide dès
# qu'alembic était lancé à la main, et l'ancien défaut codé dans `alembic.ini` prenait le
# relais — c'est-à-dire la base de PRODUCTION. `load_dotenv` ne remplace pas une variable
# déjà positionnée : l'environnement (Compose, CI, Makefile) reste prioritaire.
_RACINE = Path(__file__).resolve().parents[1]
load_dotenv(_RACINE / ".env")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
if not DATABASE_URL:
    # Échec explicite, jamais de repli silencieux : une migration porte sur UNE base précise,
    # et se tromper de base est irréversible. Voir `alembic.ini`.
    print(
        "alembic : DATABASE_URL est absente.\n"
        f"  Renseigne-la dans {_RACINE / '.env'} ou dans l'environnement, par exemple :\n"
        "  DATABASE_URL=postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev\n"
        "  (dépôt de développement = base `synaptiq_dev` ; `synaptiq_db` est la production)",
        file=sys.stderr,
    )
    raise SystemExit(2)


def run_migrations_online() -> None:
    # L'URL est injectée dans la section lue par `engine_from_config`, et non via
    # `config.set_main_option` : ce dernier passe par l'interpolation de configparser, où un
    # `%` dans un mot de passe casserait la configuration.
    section = dict(config.get_section(config.config_ini_section) or {})
    section["sqlalchemy.url"] = DATABASE_URL
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
