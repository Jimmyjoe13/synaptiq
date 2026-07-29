"""Tests unitaires du cache de résolution des clés API (audit F10).

Le cache remplace un SELECT + UPDATE + COMMIT par requête. Ce qu'il faut verrouiller :
l'expiration effective, l'invalidation explicite, et le fait que `AUTH_CACHE_TTL=0`
désactive complètement la mise en cache (révocation immédiate).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "core"))

import apps.api.main as main

VALEUR = ("tenant1", ["read", "write"], None)


def _reset():
    main.invalidate_auth_cache()


def test_put_puis_get(monkeypatch):
    monkeypatch.setattr(main, "AUTH_CACHE_TTL", 60.0)
    _reset()
    main._auth_cache_put("hash1", VALEUR)
    assert main._auth_cache_get("hash1") == VALEUR


def test_get_inconnu_renvoie_none(monkeypatch):
    monkeypatch.setattr(main, "AUTH_CACHE_TTL", 60.0)
    _reset()
    assert main._auth_cache_get("jamais-vu") is None


def test_entree_expiree_est_ignoree_et_purgee(monkeypatch):
    """Horloge monotone avancée à la main : pas de sleep dans les tests."""
    monkeypatch.setattr(main, "AUTH_CACHE_TTL", 60.0)
    _reset()
    instant = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: instant[0])

    main._auth_cache_put("hash1", VALEUR)
    assert main._auth_cache_get("hash1") == VALEUR

    instant[0] += 59.0          # encore dans la fenêtre
    assert main._auth_cache_get("hash1") == VALEUR

    instant[0] += 2.0           # au-delà des 60 s
    assert main._auth_cache_get("hash1") is None
    assert "hash1" not in main._auth_cache      # l'entrée morte est retirée


def test_ttl_nul_desactive_le_cache(monkeypatch):
    """AUTH_CACHE_TTL=0 : rien n'est mémorisé, la base est interrogée à chaque requête."""
    monkeypatch.setattr(main, "AUTH_CACHE_TTL", 0.0)
    _reset()
    main._auth_cache_put("hash1", VALEUR)
    assert main._auth_cache_get("hash1") is None
    assert main._auth_cache == {}


def test_invalidation_explicite(monkeypatch):
    monkeypatch.setattr(main, "AUTH_CACHE_TTL", 60.0)
    _reset()
    main._auth_cache_put("hash1", VALEUR)
    main._auth_cache_put("hash2", VALEUR)
    main.invalidate_auth_cache()
    assert main._auth_cache_get("hash1") is None
    assert main._auth_cache_get("hash2") is None


def test_plafond_empeche_une_croissance_non_bornee(monkeypatch):
    monkeypatch.setattr(main, "AUTH_CACHE_TTL", 60.0)
    monkeypatch.setattr(main, "AUTH_CACHE_MAX", 4)
    _reset()
    for i in range(10):
        main._auth_cache_put(f"hash{i}", VALEUR)
    assert len(main._auth_cache) <= 4
