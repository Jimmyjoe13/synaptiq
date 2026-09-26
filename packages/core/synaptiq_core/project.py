"""SynaptiQ — dimension PROJET des souvenirs (lot B, 26/09).

Un souvenir appartient soit à UN projet (`project = 'emile'`), soit à tous (`project IS
NULL`, souvenir global : préférences de l'utilisateur, conventions générales). Le projet
est orthogonal à la collection : `decisions` existe dans chaque projet, et un filtre
`project='synaptiq'` + `collections=['decisions']` vise exactement les décisions de
SynaptiQ.

Ce module porte la normalisation du nom (appelée par l'API, le worker et le MCP, pour
qu'un même projet ne s'écrive jamais de deux façons), la construction du fragment SQL de
filtrage partagée par tous les chemins de recherche, et l'activation du scan itératif HNSW
sans lequel un filtre sélectif amputerait les résultats.
"""
from __future__ import annotations

import os
import re

# Minuscules, chiffres, `_`, `.`, `-` ; commence par une lettre ou un chiffre ; 64 au plus
# (taille de la colonne). Assez large pour `elu-scraper`, `dash.joe`, `jobxpress_v2`.
_MOTIF = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def normaliser_projet(nom: str | None) -> str | None:
    """Nom de projet canonique, ou None (souvenir global / pas de filtre).

    La normalisation (espaces retirés, minuscules) est ce qui empêche `Emile`, `emile` et
    ` emile ` de devenir trois projets distincts — le même écueil que les collections
    `clients_paca` / `clients_region_paca`, en plus bête. Lève `ValueError` sur un nom
    invalide plutôt que de le « corriger » en silence : un projet mal orthographié qui
    passerait créerait une partition fantôme, vide à la relecture.
    """
    if nom is None:
        return None
    propre = nom.strip().lower()
    if not propre:
        return None
    if not _MOTIF.match(propre):
        raise ValueError(
            f"Nom de projet invalide : {nom!r}. Attendu : minuscules, chiffres, '_', '.', "
            "'-', 64 caractères au plus, commençant par une lettre ou un chiffre.")
    return propre


def filtre_projet(project: str | None, include_global: bool = True,
                  alias: str = "") -> tuple[str, list]:
    """Fragment SQL `AND …` et ses paramètres liés pour filtrer par projet.

    - `project` None           -> pas de filtre (tous projets confondus, historique) ;
    - `include_global` vrai    -> les souvenirs du projet ET les globaux (défaut : une
                                  préférence de l'utilisateur vaut dans tous ses projets) ;
    - `include_global` faux    -> strictement ceux du projet.

    Le fragment est CHOISI par le serveur parmi trois formes fixes ; seul le nom du projet
    varie, et il passe en paramètre lié. `alias` préfixe les colonnes (`"m."`) quand la
    requête aliase `memories`.

    ⚠️ À appeler dans CHAQUE chemin (vectoriel, plein texte), jamais factorisé dans une
    CTE commune : cf. `_fetch_candidates`, une CTE référencée plusieurs fois neutralise
    l'index HNSW.
    """
    if project is None:
        return "", []
    if include_global:
        return f"AND ({alias}project = %s OR {alias}project IS NULL)", [project]
    return f"AND {alias}project = %s", [project]


def activer_scan_iteratif(cur, mode: str = "strict_order") -> None:
    """Active le scan ITÉRATIF de l'index HNSW pour la transaction courante (pgvector ≥ 0.8).

    Sans lui, un filtre sélectif (un projet parmi dix) est appliqué APRÈS le parcours de
    l'index : HNSW ne rend que ses `ef_search` (40) plus proches voisins, le filtre en
    écarte la plupart, et la recherche renvoie moins de résultats que demandé — voire zéro
    pour un petit projet noyé dans un gros corpus. En mode itératif, l'index poursuit le
    parcours jusqu'à remplir le LIMIT.

    `SET LOCAL` : limité à la transaction, rien ne fuit vers l'appel suivant du pool.
    Sous SAVEPOINT : sur un pgvector antérieur à 0.8 le paramètre n'existe pas ; l'erreur
    est alors avalée (la recherche reste correcte, seulement moins complète) plutôt que
    d'avorter la transaction. `mode="off"` désactive.
    """
    if mode not in ("strict_order", "relaxed_order"):
        return
    cur.execute("SAVEPOINT synaptiq_hnsw")
    try:
        # `mode` vient d'une liste fermée vérifiée ci-dessus : aucune donnée d'appelant.
        cur.execute(f"SET LOCAL hnsw.iterative_scan = {mode}")
        cur.execute("RELEASE SAVEPOINT synaptiq_hnsw")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT synaptiq_hnsw")
        cur.execute("RELEASE SAVEPOINT synaptiq_hnsw")


def mode_scan_iteratif() -> str:
    """Mode du scan itératif HNSW (`RETRIEVAL_HNSW_ITERATIVE`), lu à chaque appel."""
    return os.getenv("RETRIEVAL_HNSW_ITERATIVE", "strict_order")
