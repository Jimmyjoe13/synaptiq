"""Fixtures et configuration pytest partagées.

- Rend importables la racine du repo et les packages (core, sdk-python).
- Marque automatiquement `integration` tout test hors de tests/unit/ (ceux-ci
  exigent Postgres + Redis actifs). Les tests unitaires tournent sans infra.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "packages", "core"), os.path.join(ROOT, "packages", "sdk-python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" not in path:
            item.add_marker(pytest.mark.integration)


def purge_tenants(conn, *tenants: str) -> None:
    """Vide les données des tenants indiqués, et EUX SEULS.

    Les tests d'intégration faisaient auparavant `TRUNCATE TABLE memories/events CASCADE`,
    qui efface TOUTE la base : lancer la suite sur une instance contenant des données
    réelles les détruisait sans avertissement (corpus de benchmark, données de démo…).
    Comme la base de développement est souvent aussi celle qui sert aux essais, le nettoyage
    est désormais borné au périmètre du test.

    `relationships` n'a pas de `tenant_id` : ses lignes partent en cascade avec les mémoires
    (`ON DELETE CASCADE`). Les mémoires sont supprimées avant les événements, faute de quoi
    `memories.source_event_id` serait simplement mis à NULL et les mémoires subsisteraient.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE tenant_id = ANY(%s)", (list(tenants),))
        cur.execute("DELETE FROM events WHERE tenant_id = ANY(%s)", (list(tenants),))
        cur.execute("DELETE FROM api_keys WHERE tenant_id = ANY(%s)", (list(tenants),))
        conn.commit()
