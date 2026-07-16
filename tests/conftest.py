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
