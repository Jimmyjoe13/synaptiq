"""Empreinte de contenu — clé de déduplication PARTAGÉE par l'API et le worker.

Cette fonction vivait dans `apps/worker/worker.py`, donc elle ne s'appliquait qu'au chemin
d'extraction : `POST /v1/memories` n'écrivait aucun `content_hash` et n'avait donc aucune
protection contre le doublon. C'est exactement la mésaventure de la taxonomie (cf. le
commentaire de `_VALID_SUBTYPES` dans le worker) : deux chemins d'écriture, une seule règle
implémentée. Elle est remontée ici pour que les deux chemins produisent la MÊME empreinte
pour le même contenu — sans quoi un index unique sur cette colonne ne voudrait rien dire.
"""
import hashlib

__all__ = ["content_hash", "normalize_for_hash"]


def normalize_for_hash(content: str) -> str:
    """Normalise avant empreinte : blancs unifiés, bords rognés, casse abaissée.

    ⚠️ `str.split()` sans argument découpe sur **tout** blanc Unicode, U+00A0 (espace
    insécable) et U+202F (espace fine insécable) comprises. Ce n'est pas une subtilité
    théorique : les extractions du worker en contiennent (« 10:37 U+202F am », « MySQL
    U+202F 8.0 »), et cette normalisation-là est la seule qui fasse foi.

    Corollaire pratique, payé une fois : **on ne réplique PAS cette normalisation en SQL.**
    Le `\\s` de PostgreSQL est ASCII-only et ne reconnaît ni U+00A0 ni U+202F, donc un
    `encode(sha256(lower(regexp_replace(content, '\\s+', ' ', 'g'))), 'hex')` produit une
    empreinte DIFFÉRENTE sur ces contenus, sans lever la moindre erreur. Un backfill écrit
    en SQL diverge donc en silence de ce que calculera le code — et l'index unique cesse de
    protéger précisément les lignes qu'il croit couvrir. Tout backfill passe par cette
    fonction (cf. `migrations/versions/20260801_memory_idempotency.py`).
    """
    return " ".join(content.split()).strip().lower()


def content_hash(content: str) -> str:
    """SHA-256 du contenu normalisé.

    Deux usages, deux index :
      - `(source_event_id, content_hash)` sur le chemin worker : plusieurs faits par
        événement, mais un replay du même événement ne duplique rien.
      - `(tenant_id, agent_id, content_hash)` sur le chemin direct : une relance de
        `store_memory` après timeout devient un no-op au lieu d'une seconde ligne.
    """
    return hashlib.sha256(normalize_for_hash(content).encode("utf-8")).hexdigest()
