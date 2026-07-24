import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url")))
if config.config_file_name:
    fileConfig(config.config_file_name)


def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
