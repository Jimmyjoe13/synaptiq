"""L'empreinte de contenu, socle de la déduplication des DEUX chemins d'écriture.

Ce qui est verrouillé ici tient à une seule chose : cette fonction est la référence, et un
index unique est bâti sur son résultat. Toute divergence entre deux façons de la calculer
rend cet index silencieusement inopérant sur les lignes concernées.
"""
import hashlib

from synaptiq_core import content_hash, normalize_for_hash


def test_meme_contenu_meme_empreinte():
    assert content_hash("Jimmy préfère les e-mails courts.") == \
           content_hash("Jimmy préfère les e-mails courts.")


def test_contenus_differents_empreintes_differentes():
    assert content_hash("MySQL 8.0") != content_hash("PostgreSQL 16")


def test_la_casse_et_les_blancs_sont_neutralises():
    """Une reformulation cosmétique n'est pas un nouveau souvenir."""
    reference = content_hash("Le serveur ecoute sur 8443.")
    assert content_hash("  LE SERVEUR   ECOUTE\tSUR 8443.  ") == reference
    assert content_hash("Le serveur\necoute sur 8443.") == reference


def test_les_blancs_unicode_sont_neutralises_aussi():
    """U+00A0 et U+202F comptent comme des blancs — c'est TOUT l'enjeu du backfill.

    `str.split()` les traite comme des séparateurs, le `\\s` de PostgreSQL non. Un backfill
    écrit en SQL produirait donc une autre empreinte sur ces contenus, sans erreur, et
    l'index unique cesserait de couvrir ces lignes. Les extractions du worker en contiennent
    réellement (« 10:37 U+202F am », « MySQL U+202F 8.0 »).
    """
    assert content_hash("MySQL 8.0") == content_hash("MySQL 8.0")
    assert content_hash("10:37 am") == content_hash("10:37 am")


def test_l_empreinte_est_bien_le_sha256_du_normalise():
    """Verrou explicite sur l'algorithme : la migration en dépend pour son backfill."""
    normalise = normalize_for_hash("  Deux   Mots  ")
    assert normalise == "deux mots"
    assert content_hash("  Deux   Mots  ") == \
           hashlib.sha256(normalise.encode("utf-8")).hexdigest()


def test_le_worker_et_l_api_partagent_la_meme_fonction():
    """Régression : la fonction vivait dans le worker, donc l'API n'en avait aucune.

    C'était la cause racine du doublon silencieux sur `POST /v1/memories` : deux chemins
    d'écriture, une seule règle implémentée. Le test compare les objets, pas les résultats —
    deux copies qui coïncident aujourd'hui peuvent diverger demain.
    """
    from apps.api.main import content_hash as depuis_api
    from apps.worker.worker import content_hash as depuis_worker

    assert depuis_api is content_hash
    assert depuis_worker is content_hash
