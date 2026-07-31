"""SynaptiQ — les collections logiques, désormais des objets et non plus une fiction.

## Pourquoi ce module existe

Jusqu'ici, « collection » était un résultat de calcul : `route_memory()` regardait un couple
(type, sous-type) et renvoyait une des sept clés du `context_packet`, en dur, dans une
cascade de `if`. Rien n'existait qu'un agent puisse consulter, créer ou faire évoluer.

Conséquence directe, observée en production : un agent écrit `nana_intelligence_lead_webhook`
comme sous-type, la mémoire est bien stockée et bien retrouvée — mais elle est servie dans
`facts`, exactement comme un fait ordinaire. Le classement fin que l'agent croyait faire
n'a jamais eu lieu, et **rien ne le lui disait**.

## Le partage des rôles : famille au moteur, collection à l'agent

C'est l'invariant central de ce module, et il tient en une phrase :
**le `type` n'est pas une étiquette, c'est un comportement.**

Les quatre types (`semantic`, `episodic`, `procedural`, `working`) décident si un souvenir
est intriqué dans le graphe, comment il décroît, et vers quelle section il retombe par
défaut. Ce sont donc des FAMILLES cognitives, propriété du moteur — pas des catégories de
rangement. Elles restent fermées.

Le `subtype`, lui, est le nom de la collection : champ libre, propriété de l'agent. Il en
crée autant qu'il veut, il les nomme comme il veut. Ce découpage a une vertu décisive :
il ne demande AUCUNE migration de données. Les sous-types déjà écrits — y compris ceux
que l'agent avait inventés — deviennent rétroactivement de vraies collections.

## Aucune collection ne mène plus au vide

`route_memory` renvoyait `None` pour un type inconnu, et `collapse_by_utility` ignorait
alors silencieusement la mémoire — après l'avoir comptée dans les ids sélectionnés et avoir
consommé son budget de tokens. Une mémoire retrouvée, payée, et absente du contexte.

Un registre résout donc TOUJOURS vers une clé de paquet. Une collection inconnue retombe
sur celle de sa famille (le comportement historique du champ libre, à l'identique), et une
famille inconnue retombe sur `facts`. Mal rangé se voit et se corrige ; disparu en silence,
non.

Ce module ne fait aucune I/O : le registre se construit à partir d'une liste de
`Collection`, fournie par un `CollectionStore` implémenté côté API (même motif que
`MemoryStore` dans `context_builder`).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger("synaptiq-core.collections")

# Les quatre familles cognitives. Fermées, et c'est le point : chacune porte un
# comportement du moteur, pas un thème.
FAMILIES: tuple[str, ...] = ("semantic", "episodic", "procedural", "working")

# Section du context_packet servie par défaut pour une collection LIBRE de cette famille.
# Reproduit exactement l'ancien comportement de `route_memory` sur un sous-type inconnu :
# semantic -> facts, episodic -> episodes, procedural -> rules, working -> examples.
FAMILY_FALLBACK_KEY: dict[str, str] = {
    "semantic": "facts",
    "episodic": "episodes",
    "procedural": "rules",
    "working": "examples",
}

# Dernier recours, pour une famille elle-même inconnue (donnée corrompue, ou écrite par une
# version antérieure). `facts` est la section la plus neutre du paquet. On préfère un
# rangement imparfait mais VISIBLE à une disparition silencieuse.
ULTIMATE_FALLBACK_KEY = "facts"


@dataclass(frozen=True)
class Collection:
    """Une collection logique : le nom que l'agent donne à un rayon de sa mémoire.

    `name` correspond à `memories.subtype`, `family` à `memories.type`.
    """

    name: str
    family: str
    # Section du `context_packet` où les souvenirs de cette collection sont servis.
    packet_key: str
    description: str = ""
    # Cette collection tisse-t-elle des arêtes `entangled_with` ? C'est la propriété qui
    # remplacera la variable globale `QEM_ENTANGLE_TYPES` du worker (lot 2) : l'intrication
    # devient une décision par collection, prise par l'agent, et non un réglage d'instance.
    entangle: bool = True
    # `system` = livrée avec le moteur, valable pour tous les agents. `agent` = créée par
    # l'agent pour lui-même. Un agent ne peut jamais modifier une collection `system`.
    created_by: str = "agent"
    memory_count: int = 0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(
                f"Famille inconnue '{self.family}' pour la collection '{self.name}'. "
                f"Familles valides : {', '.join(FAMILIES)}."
            )


# ─── Collections livrées avec le moteur ──────────────────────────────────────
# Ce sont les sept sections historiques du context_packet, exprimées dans le nouveau
# modèle. Elles constituent la SOURCE UNIQUE de la taxonomie canonique : `taxonomy.py` en
# dérive désormais `VALID_SUBTYPES` et `DEFAULT_SUBTYPE`, qui étaient jusqu'ici une seconde
# liste à tenir à jour en parallèle.
#
# `entangle` reproduit le défaut historique `QEM_ENTANGLE_TYPES=procedural,semantic` :
# les épisodes bruts sont nombreux et peu discriminants, les intriquer densifie le graphe
# sans gain de pertinence. La différence est qu'il devient possible d'en décider
# collection par collection au lieu de subir le réglage global.
SYSTEM_COLLECTIONS: tuple[Collection, ...] = (
    Collection("fact", "semantic", "facts", created_by="system", entangle=True,
               description="Faits stables sur une personne, une entité ou le monde."),
    Collection("preference", "semantic", "preferences", created_by="system", entangle=True,
               description="Gouts, choix et preferences explicites de l'utilisateur."),
    Collection("interaction", "episodic", "episodes", created_by="system", entangle=False,
               description="Episodes bruts d'interaction, quand rien de durable n'est enonce."),
    Collection("rule", "procedural", "rules", created_by="system", entangle=True,
               description="Regles de conduite et procedures a appliquer."),
    Collection("coding_best_practices", "procedural", "best_practices", created_by="system",
               entangle=True,
               description="Regles d'architecture, conventions et bonnes pratiques de code."),
    Collection("code_error_resolution", "procedural", "errors", created_by="system",
               entangle=True,
               description="Erreurs rencontrees et leur resolution."),
    Collection("scratch", "working", "examples", created_by="system", entangle=False,
               description="Memoire de travail volatile, exemples ponctuels."),
)

# Ordre des sections canoniques dans le paquet. Le lot 2 y ajoutera les clés des
# collections déclarées par l'agent ; en attendant, ce tuple est le contrat public.
SYSTEM_PACKET_KEYS: tuple[str, ...] = (
    "facts", "preferences", "episodes", "rules", "best_practices", "errors", "examples")


class CollectionStore(Protocol):
    """Accès aux collections, borné à un (tenant, agent) fixés à la construction.

    Même motif que `MemoryStore` : aucune méthode ne prend de `tenant_id` ni d'`agent_id`,
    donc aucune implémentation ne peut « oublier » le filtre d'isolation. Une collection
    créée par un agent ne doit jamais devenir visible depuis la mémoire d'un autre.
    """

    def fetch_collections(self) -> list[Collection]:
        """Collections système + celles déclarées par cet agent."""


@dataclass
class CollectionRegistry:
    """Résout un couple (famille, collection) vers une section du context_packet.

    Construit à partir d'une liste plate de `Collection`. Les collections d'agent
    l'emportent sur les collections système de même nom : un agent peut spécialiser le
    routage d'un nom canonique pour lui-même, sans affecter les autres.
    """

    collections: tuple[Collection, ...] = field(default_factory=tuple)
    _par_cle: dict[tuple[str, str], Collection] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._par_cle = {}
        # Les `system` d'abord, les `agent` ensuite : la seconde écriture écrase la
        # première, donc l'agent gagne. L'ordre de la liste d'entrée n'a pas d'importance.
        for source in ("system", "agent"):
            for col in self.collections:
                if col.created_by == source:
                    self._par_cle[(col.family, col.name)] = col

    @classmethod
    def systeme(cls) -> CollectionRegistry:
        """Registre par défaut : les sept collections livrées, sans base de données.

        Sert de repli partout où le registre n'est pas encore injecté (worker, tests,
        appels historiques de `route_memory`), afin que ce lot n'introduise aucun
        changement de comportement observable.
        """
        return cls(SYSTEM_COLLECTIONS)

    @classmethod
    def depuis(cls, collections: Iterable[Collection]) -> CollectionRegistry:
        """Registre à partir des collections d'un magasin, complété par les système."""
        fournies = tuple(collections)
        noms_fournis = {(c.family, c.name) for c in fournies}
        manquantes = tuple(c for c in SYSTEM_COLLECTIONS
                           if (c.family, c.name) not in noms_fournis)
        return cls(fournies + manquantes)

    def get(self, family: str, name: str | None) -> Collection | None:
        """La collection déclarée pour ce couple, ou None si elle est libre."""
        if not name:
            return None
        return self._par_cle.get((family, name))

    def packet_key(self, family: str, name: str | None) -> str:
        """Section du context_packet. Ne renvoie JAMAIS None — voir l'en-tête du module."""
        declaree = self.get(family, name)
        if declaree is not None:
            return declaree.packet_key
        repli = FAMILY_FALLBACK_KEY.get(family)
        if repli is not None:
            # Cas normal du sous-type libre : comportement historique, à l'identique.
            return repli
        logger.warning(
            "Famille inconnue '%s' (collection '%s') : rangement de repli dans '%s'. "
            "Auparavant cette mémoire était retirée du paquet en silence, après avoir "
            "consommé son budget de tokens.", family, name, ULTIMATE_FALLBACK_KEY)
        return ULTIMATE_FALLBACK_KEY

    def entangle_pour(self, family: str, name: str | None) -> bool:
        """L'intrication est-elle souhaitée pour ce couple ?

        Non câblé au worker dans ce lot : la décision reste portée par
        `QEM_ENTANGLE_TYPES` jusqu'au lot 2. La propriété est déjà lisible ici pour que la
        migration et le registre soient complets, et pour que le basculement du worker ne
        soit qu'un changement d'appelant.
        """
        declaree = self.get(family, name)
        if declaree is not None:
            return declaree.entangle
        # Collection libre : hérite du défaut historique de sa famille.
        return family in ("procedural", "semantic")

    def packet_keys(self) -> tuple[str, ...]:
        """Sections du paquet : les sept canoniques d'abord, puis celles de l'agent.

        L'ordre compte : les canoniques restent en tête et toujours présentes, de sorte
        qu'un consommateur écrit avant les collections continue de lire les mêmes clés au
        même endroit. Les sections de l'agent s'ajoutent à la suite.
        """
        supplementaires = []
        for col in self.collections:
            if col.packet_key not in SYSTEM_PACKET_KEYS and col.packet_key not in supplementaires:
                supplementaires.append(col.packet_key)
        return SYSTEM_PACKET_KEYS + tuple(supplementaires)


# Registre par défaut du processus. Utilisé par `route_memory` quand aucun registre n'est
# fourni, ce qui garantit que tout le code existant conserve son comportement exact.
REGISTRE_SYSTEME = CollectionRegistry.systeme()


# ─── Chargement depuis la base ───────────────────────────────────────────────
# Prend un CURSEUR, comme `governance.handle_contradictions` : le paquet reste sans
# dépendance psycopg2, tout en évitant que l'API et le worker entretiennent chacun leur
# copie de la même requête. Les deux chemins d'écriture doivent voir le MÊME registre —
# c'est la leçon de la taxonomie, qui vivait dans le worker et laissait l'API sans règle.

_SQL_COLLECTIONS = """
    SELECT name, family, packet_key, description, entangle, created_by
    FROM memory_collections
    WHERE created_by = 'system' OR (tenant_id = %s AND agent_id = %s)
    ORDER BY created_by, name;
"""


def charger_registre(cur, tenant_id: str, agent_id: str) -> CollectionRegistry:
    """Registre des collections d'un (tenant, agent), complété par les collections système.

    Le repli sur le registre système est délibéré et vaut pour toute erreur : la taxonomie
    ne doit jamais être une dépendance dure du rappel ni de l'écriture. Refuser une mémoire
    parce que son ÉTIQUETTE est illisible serait disproportionné.

    Les deux causes de repli ne se valent pas et ne se journalisent donc pas pareil :
    une table absente est un état d'exploitation légitime (migration pas encore tirée),
    tout le reste est un défaut et doit crier — sinon le repli masque la panne exactement
    comme les silences que ce projet passe son temps à fermer.

    La lecture se fait par POSITION : les appelants n'ouvrent pas tous leur curseur avec
    `RealDictCursor` (`create_memory` utilise un curseur à tuples).
    """
    try:
        cur.execute(_SQL_COLLECTIONS, (tenant_id, agent_id))
        lignes = cur.fetchall()
    except Exception as exc:
        # `UndefinedTable` porte ce code SQLSTATE. Comparé sur le code plutôt que sur la
        # classe pour ne pas importer psycopg2 ici.
        if getattr(exc, "pgcode", None) == "42P01":
            logger.warning("Table `memory_collections` absente (migration "
                           "20260731_collections non appliquée) : collections système.")
        else:
            logger.error("Registre de collections illisible — repli sur les collections "
                         "système. Le rangement fin de cet agent est INACTIF.",
                         exc_info=True)
        return CollectionRegistry.systeme()

    collections = []
    for ligne in lignes:
        valeurs = list(ligne.values()) if isinstance(ligne, dict) else list(ligne)
        nom, famille, cle, description, entangle, origine = valeurs[:6]
        try:
            collections.append(Collection(
                name=nom, family=famille, packet_key=cle, description=description or "",
                entangle=bool(entangle), created_by=origine,
            ))
        except ValueError:
            # Famille hors des quatre : ligne écrite à la main ou par une version
            # antérieure. On l'écarte plutôt que de priver l'agent de TOUT son registre.
            logger.warning("Collection ignorée (famille invalide) : %s/%s", famille, nom)
    return CollectionRegistry.depuis(collections)
