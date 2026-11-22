"""SynaptiQ — les compartiments : une mémoire rangée comme un cerveau (v0.4.0).

## Pourquoi ce module remplace les collections

Jusqu'à la v0.3, chaque agent se créait ses propres collections. L'audit du 22/11/2026 a
mesuré le résultat en production : 80 collections d'agent, dont ~25 ne contenant qu'un
seul souvenir, des quasi-doublons (`decision` / `decisions` / `architecture-decision`,
`errors` / `error_resolution` / `resolution-erreur` / `bugfix`), et surtout une même
collection répartie sur 3 ou 4 familles (`project_state` de `master` : 810 souvenirs en
episodic/procedural/semantic). Or la famille décide de l'oubli et de l'intrication : une
collection n'avait donc même pas un comportement homogène.

Un cerveau ne fonctionne pas ainsi. Il a un nombre FIXE de régions, chacune avec sa
fonction, et une information est traitée là où sa nature l'envoie. SynaptiQ adopte ce
modèle : **9 compartiments fixes, propriété du moteur. L'agent ne crée rien, il juge
seulement dans quel compartiment ranger ce qu'il stocke.**

## Un compartiment est un COMPORTEMENT, pas une étiquette

Chaque compartiment fixe, pour tous les souvenirs qu'il contient :
  - sa vitesse d'oubli (`facteur_demi_vie`, multiplié par la demi-vie de l'instance) ;
  - s'il tisse des liens dans le graphe d'intrication (`intrique`) ;
  - son poids au rappel (`poids` : l'amygdale renforce les souvenirs marquants) ;
  - sa part maximale du budget de contexte (`part_max` : régulation, rôle de
    l'hypothalamus) ;
  - s'il est injecté à chaque contexte, quelle que soit la question (`toujours_injecte` :
    les réflexes du tronc cérébral ne se décident pas).

La famille historique (`memories.type`) n'est plus fournie par personne : elle est
DÉDUITE du compartiment, et ne sert plus qu'aux filtres internes (ex. la famille
`reflective` des croyances).

## Le rangement ne mène jamais au vide

Héritage direct de la v0.3 (`route_memory` ne renvoie jamais None) : un compartiment
absent ou inconnu n'est ni refusé ni perdu. Un ancien nom de collection est converti par
la table `ALIAS` ; tout le reste est CLASSÉ PAR LE MOTEUR, par similarité entre le souvenir
et les prototypes de chaque compartiment (`classer`). La réponse dit toujours qui a choisi
(`origine`) pour que l'agent voie le rangement et puisse le corriger.

Ce module ne fait aucune I/O. Les vecteurs prototypes sont calculés par l'appelant (API,
worker) avec SON embedder et passés à `classer` : aucun modèle n'est chargé à l'import.
"""
from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger("synaptiq-core.compartments")

# Familles internes (= valeurs possibles de `memories.type`). Elles ne sont plus exposées :
# chaque compartiment en porte une, et seul le moteur les manipule.
FAMILLE_REFLEXIVE = "reflective"
FAMILIES: tuple[str, ...] = ("semantic", "episodic", "procedural", "reflective")


@dataclass(frozen=True)
class Compartiment:
    """Une région de la mémoire, avec son comportement propre."""

    nom: str
    # Région du cerveau dont le compartiment s'inspire (affichée dans le contexte rendu).
    region: str
    titre: str
    famille: str
    # Montrée aux agents (outil `list_compartments`) ET utilisée pour le classement.
    description: str
    # Phrases types : elles forment, avec la description, le prototype vectoriel du
    # compartiment. Plusieurs exemples concrets classent mieux qu'une définition abstraite.
    exemples: tuple[str, ...] = ()
    # Multiplicateur de la demi-vie de l'instance (QEM_RECENCY_HALFLIFE_DAYS).
    # 0 = aucun oubli ; 1 = demi-vie de l'instance ; 4 = quatre fois plus lent.
    facteur_demi_vie: float = 1.0
    intrique: bool = True
    poids: float = 1.0
    # Part maximale du budget de tokens d'un paquet. 1.0 = pas de plafond propre.
    part_max: float = 1.0
    toujours_injecte: bool = False
    # Le moteur peut-il choisir ce compartiment tout seul ? Non pour les croyances : une
    # croyance est formulée délibérément (`note_belief`), avec confiance et indices.
    classable: bool = True
    # Accessible par l'écriture ordinaire (`store_memory`) ? Non pour les croyances, qui
    # passent obligatoirement par le garde-fou de `note_belief`.
    ecriture_directe: bool = True

    def __post_init__(self) -> None:
        if self.famille not in FAMILIES:
            raise ValueError(f"Famille inconnue '{self.famille}' pour le compartiment "
                             f"'{self.nom}'.")
        if not 0.0 < self.part_max <= 1.0:
            raise ValueError(f"part_max hors de ]0, 1] pour '{self.nom}'.")


# ─── Les 9 compartiments ─────────────────────────────────────────────────────
# L'ORDRE est celui des sections du context_packet (et de leur rendu) : les réflexes
# d'abord, comme le tronc cérébral passe avant toute délibération.
COMPARTIMENTS: tuple[Compartiment, ...] = (
    Compartiment(
        "reflexes", "Tronc cérébral", "Réflexes", "procedural",
        "Règles absolues et préférences fermes de l'utilisateur, garde-fous à appliquer "
        "à chaque fois sans réfléchir.",
        exemples=(
            "Toujours répondre en français et tutoyer l'utilisateur.",
            "Ne jamais pousser sur git sans confirmation explicite.",
            "L'utilisateur préfère les réponses courtes, sans titres inutiles.",
        ),
        facteur_demi_vie=0.0, intrique=False, part_max=0.20, toujours_injecte=True),
    Compartiment(
        "decisions", "Lobe frontal", "Décisions", "semantic",
        "Décisions prises et leur raison, objectifs, plans, priorités, arbitrages "
        "techniques ou métier.",
        exemples=(
            "Décision : on garde PostgreSQL plutôt que MongoDB, pour les jointures.",
            "Objectif du trimestre : livrer la v1 du CRM avant fin mars.",
            "Choix validé : déployer sur OCI derrière Tailscale, sans port public.",
        ),
        facteur_demi_vie=3.0),
    Compartiment(
        "savoir_faire", "Cervelet", "Savoir-faire", "procedural",
        "Procédures, playbooks, gestes techniques précis, bonnes pratiques et conventions "
        "de code : comment faire.",
        exemples=(
            "Pour déployer : git pull, docker compose build sans liste, puis up -d.",
            "Toujours utiliser TestClient comme context manager pour lancer le lifespan.",
            "Procédure de restauration d'une base Postgres avec pg_restore --clean.",
        ),
        facteur_demi_vie=4.0),
    Compartiment(
        "alertes", "Amygdale", "Alertes", "procedural",
        "Erreurs rencontrées et leur résolution, incidents, pièges à ne plus jamais "
        "refaire.",
        exemples=(
            "Erreur : TypeError on_startup ; cause : starlette trop récent ; corrigé en "
            "épinglant fastapi.",
            "Incident : la suite de tests écrivait en production, défaut corrigé.",
            "Piège : docker cp écrase les fichiers modifiés avec des versions obsolètes.",
        ),
        facteur_demi_vie=4.0, poids=1.15),
    Compartiment(
        "environnement", "Lobe pariétal", "Environnement", "semantic",
        "Carte du terrain : serveurs, accès, chemins, ports, topologie, outils et services "
        "disponibles, où se trouvent les choses.",
        exemples=(
            "Le serveur OCI est joignable en SSH via Tailscale sur 100.88.158.55.",
            "Le vault Obsidian est dans C:\\Obsidian\\Jimmy.",
            "L'API écoute sur le port 8000, Redis sur 6399, Postgres sur 5435.",
        ),
        facteur_demi_vie=3.0),
    Compartiment(
        "savoirs", "Lobe temporal", "Savoirs", "semantic",
        "Faits et connaissances durables : état d'un projet, définitions, informations "
        "sur une personne, une entreprise ou le monde.",
        exemples=(
            "Le projet Carel est un CRM pour une agence immobilière.",
            "SynaptiQ stocke ses vecteurs en 384 dimensions avec pgvector.",
            "Le client principal de la mission est une PME du secteur médical.",
        ),
        facteur_demi_vie=2.0),
    Compartiment(
        "episodes", "Hippocampe", "Épisodes", "episodic",
        "Journal de ce qui s'est passé : sessions de travail, interactions, jalons, "
        "déroulé d'une tâche.",
        exemples=(
            "Session du 12/11 : migration de la base terminée, tests verts.",
            "L'utilisateur a demandé un audit complet, rapport livré dans Obsidian.",
            "Jalon atteint : première mise en production de l'agent vocal.",
        ),
        facteur_demi_vie=1.0, intrique=False),
    Compartiment(
        "observations", "Lobe occipital", "Observations", "episodic",
        "Constats bruts et mesures : résultats d'audit, chiffres, logs, sorties d'outils, "
        "benchmarks, état observé à un instant donné.",
        exemples=(
            "Mesure : la recherche hybride répond en 6 à 9 ms après correction de l'index.",
            "Constat : 39 % des souvenirs n'ont aucun lien dans le graphe.",
            "Le serveur affiche 1 Go de swap utilisé sur 2.",
        ),
        facteur_demi_vie=0.5, intrique=False),
    Compartiment(
        "croyances", "Cortex préfrontal", "Croyances", FAMILLE_REFLEXIVE,
        "Hypothèses de l'agent sur l'utilisateur et sur les humains qu'il côtoie, avec un "
        "niveau de confiance. Écriture réservée à note_belief.",
        facteur_demi_vie=0.0, intrique=False, part_max=0.15,
        classable=False, ecriture_directe=False),
)

PAR_NOM: dict[str, Compartiment] = {c.nom: c for c in COMPARTIMENTS}
NOMS: tuple[str, ...] = tuple(c.nom for c in COMPARTIMENTS)
# Compartiments que le moteur peut choisir seul (classement par similarité).
CLASSABLES: tuple[str, ...] = tuple(c.nom for c in COMPARTIMENTS if c.classable)
COMPARTIMENT_CROYANCES = "croyances"
COMPARTIMENT_REFLEXES = "reflexes"
# Repli ultime quand même le classement est impossible (aucun vecteur) : le compartiment le
# plus neutre. Un rangement imparfait mais VISIBLE vaut mieux qu'une disparition.
COMPARTIMENT_PAR_DEFAUT = "savoirs"

# Sujets de croyance (`about` de note_belief), conservés dans `memories.provenance->>'about'`
# (ils vivaient dans `subtype` : `user_model` / `human_insights`). Ce ne sont PAS des
# compartiments : une seule région, deux sujets.
SUJETS_CROYANCE: tuple[str, ...] = ("user", "humans")


def compartiment(nom: str) -> Compartiment:
    """Le compartiment `nom`. KeyError explicite s'il n'existe pas."""
    try:
        return PAR_NOM[nom]
    except KeyError:
        raise KeyError(f"Compartiment inconnu '{nom}'. Valides : {', '.join(NOMS)}.") from None


def famille_de(nom: str) -> str:
    """Famille interne déduite du compartiment (repli : `semantic`)."""
    c = PAR_NOM.get(nom)
    return c.famille if c else "semantic"


# ─── Anciens noms de collection → compartiment ───────────────────────────────
# Sert à TROIS endroits, d'où sa place dans le cœur :
#   1. la migration `20261122_compartments` (copie figée de cette table) ;
#   2. l'écriture : un client v0.3 qui envoie encore `subtype='preference'` n'est pas
#      cassé le jour du déploiement (conversion journalisée, origine `alias`) ;
#   3. le worker, dont l'extracteur LLM peut encore produire un ancien nom.
# Seuls les noms GÉNÉRIQUES figurent ici. Les collections propres à un projet
# (`crm_ultimate_*`…) relèvent du classement par similarité, pas d'une liste.
ALIAS: dict[str, str] = {
    # Tronc cérébral
    "preference": "reflexes", "preferences": "reflexes", "rule": "reflexes",
    "rules": "reflexes", "jimmy_workflow": "reflexes",
    # Lobe frontal
    "decisions": "decisions", "decision": "decisions",
    "architecture-decision": "decisions", "architecture_decision": "decisions",
    "config-decision": "decisions",
    # Cervelet
    "coding_best_practices": "savoir_faire", "best_practices": "savoir_faire",
    "deploy_playbooks": "savoir_faire", "deploy_playbook": "savoir_faire",
    "harness_procedures": "savoir_faire",
    # Amygdale
    "code_error_resolution": "alertes", "errors": "alertes", "error_resolution": "alertes",
    "resolution-erreur": "alertes", "bugfix": "alertes", "harness_errors": "alertes",
    # Lobe pariétal
    "infra_access": "environnement",
    # Lobe temporal
    "fact": "savoirs", "facts": "savoirs", "project_info": "savoirs",
    "harness_facts": "savoirs",
    # Décision de Jimmy (22/11) : des souvenirs ordinaires, sans confiance ni indices,
    # ne deviennent pas des croyances en contournant le garde-fou.
    "reflexions_humaines": "savoirs",
    # Hippocampe
    "interaction": "episodes", "session_log": "episodes", "milestone": "episodes",
    "harness_episodes": "episodes",
    # Lobe occipital
    "scratch": "observations", "examples": "observations", "audit_reports": "observations",
    # Cortex préfrontal (croyances)
    "user_model": "croyances", "human_insights": "croyances",
}

# Noms dont le compartiment dépend de la famille d'origine. `project_state` est le cas
# d'école de l'audit : écrit en épisodique, c'est un journal ; ailleurs, un état de projet.
ALIAS_PAR_FAMILLE: dict[tuple[str, str], str] = {
    ("episodic", "project_state"): "episodes",
    ("semantic", "project_state"): "savoirs",
    ("procedural", "project_state"): "savoirs",
    ("working", "project_state"): "observations",
}

# Repli par famille d'origine pour un nom inconnu (migration uniquement : à l'écriture,
# un nom inconnu est classé par similarité, ce qui est plus fin).
REPLI_FAMILLE: dict[str, str] = {
    "semantic": "savoirs",
    "episodic": "episodes",
    "procedural": "savoir_faire",
    "working": "observations",
    "reflective": "croyances",
}


def depuis_alias(nom: str | None, famille: str | None = None) -> str | None:
    """Compartiment correspondant à un ancien nom de collection, ou None s'il est inconnu."""
    if not nom:
        return None
    if nom in PAR_NOM:
        return nom
    if famille and (famille, nom) in ALIAS_PAR_FAMILLE:
        return ALIAS_PAR_FAMILLE[(famille, nom)]
    return ALIAS.get(nom)


def depuis_ancien(famille: str | None, nom: str | None) -> tuple[str, bool]:
    """Migration v0.3 → v0.4 : (compartiment, exact).

    `exact` vaut False quand seul le repli par famille a joué : ces souvenirs sont ceux
    que `scripts/reclassify_compartments.py` propose de reclasser par similarité.
    """
    trouve = depuis_alias(nom, famille)
    if trouve is not None:
        return trouve, True
    return REPLI_FAMILLE.get(famille or "", COMPARTIMENT_PAR_DEFAUT), False


# ─── Classement par similarité ───────────────────────────────────────────────

def textes_prototypes() -> dict[str, list[str]]:
    """Textes à vectoriser pour chaque compartiment classable (description + exemples)."""
    return {c.nom: [c.description, *c.exemples] for c in COMPARTIMENTS if c.classable}


# Prototypes calculés UNE fois par processus et par modèle d'embedding. Le modèle fait
# partie de la clé : comparer un souvenir à des prototypes d'un autre modèle donnerait un
# classement faux sans la moindre erreur (cf. l'incident `EMBEDDING_MODEL` du 29/07).
_PROTOTYPES: dict[str, dict[str, list[float]]] = {}
_PROTOTYPES_VERROU = threading.Lock()


def prototypes(embedder) -> dict[str, list[float]]:
    """Vecteur prototype de chaque compartiment classable, pour CET embedder.

    Centroïde de la description et des exemples, vectorisés en un seul lot. Partagé par
    l'API et le worker : les deux chemins d'écriture classent selon les MÊMES prototypes.
    Seule fonction du module qui fasse un appel externe — et uniquement via l'embedder
    fourni par l'appelant.
    """
    cle = f"{type(embedder).__name__}:{getattr(embedder, 'model', '')}"
    with _PROTOTYPES_VERROU:
        if cle not in _PROTOTYPES:
            textes = textes_prototypes()
            a_plat = [(nom, t) for nom, liste in textes.items() for t in liste]
            vecteurs = embedder.embed([t for _, t in a_plat])
            par_nom: dict[str, list] = {}
            for (nom, _), v in zip(a_plat, vecteurs, strict=True):
                par_nom.setdefault(nom, []).append(list(v))
            _PROTOTYPES[cle] = {nom: centroide(vs) for nom, vs in par_nom.items()}
        return _PROTOTYPES[cle]


def _norme(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def centroide(vecteurs: Iterable[Sequence[float]]) -> list[float]:
    """Moyenne normalisée d'un ensemble de vecteurs (prototype d'un compartiment)."""
    liste = [list(v) for v in vecteurs]
    if not liste:
        return []
    somme = [sum(col) for col in zip(*liste, strict=True)]
    n = _norme(somme)
    return [x / n for x in somme] if n else somme


def cosinus(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norme(a), _norme(b)
    if not na or not nb or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def classer(vecteur: Sequence[float] | None,
            prototypes: Mapping[str, Sequence[float]]) -> tuple[str, float]:
    """Compartiment classable le plus proche du souvenir : (nom, similarité).

    Ne renvoie jamais rien d'autre qu'un compartiment valide : sans vecteur ni prototype,
    repli sur `COMPARTIMENT_PAR_DEFAUT` avec une similarité de 0 (et l'appelant le dit).
    """
    meilleur, score = COMPARTIMENT_PAR_DEFAUT, 0.0
    if not vecteur:
        return meilleur, score
    trouve = False
    for nom, proto in prototypes.items():
        if nom not in CLASSABLES or not proto:
            continue
        s = cosinus(vecteur, proto)
        if not trouve or s > score:
            meilleur, score, trouve = nom, s, True
    return meilleur, score


@dataclass(frozen=True)
class Rangement:
    """Résultat de la résolution d'un compartiment à l'écriture."""

    compartiment: str
    # `agent` : nom valide fourni ; `alias` : ancien nom converti ; `moteur` : classé par
    # similarité (compartiment absent ou inconnu).
    origine: str
    similarite: float | None = None
    demande: str | None = None


def resoudre(demande: str | None, vecteur: Sequence[float] | None,
             prototypes: Mapping[str, Sequence[float]],
             famille: str | None = None) -> Rangement:
    """Décide du compartiment d'un souvenir à l'écriture.

    `croyances` n'est JAMAIS rendu ici par classement ; une demande explicite de
    `croyances` est rendue telle quelle et c'est à l'appelant de la refuser s'il n'est pas
    le chemin `note_belief` (voir `Compartiment.ecriture_directe`).
    """
    if demande in PAR_NOM:
        return Rangement(demande, "agent", demande=demande)
    converti = depuis_alias(demande, famille)
    if converti is not None:
        logger.info("Ancien nom de collection '%s' converti en compartiment '%s'.",
                    demande, converti)
        return Rangement(converti, "alias", demande=demande)
    nom, sim = classer(vecteur, prototypes)
    if demande:
        logger.info("Compartiment inconnu '%s' : classé par le moteur dans '%s' (%.2f).",
                    demande, nom, sim)
    return Rangement(nom, "moteur", similarite=round(sim, 3), demande=demande)


def normaliser_extraction(demande: str | None, vecteur: Sequence[float] | None,
                          prototypes: Mapping[str, Sequence[float]],
                          famille: str | None = None) -> Rangement:
    """Comme `resoudre`, pour une sortie d'extracteur LLM (worker).

    Un extracteur ne produit JAMAIS de croyance : une croyance est formulée délibérément par
    l'agent et passe par le garde-fou. Un `croyances` (ou `user_model`…) halluciné est donc
    reclassé par similarité parmi les compartiments ouverts.
    """
    rangement = resoudre(demande, vecteur, prototypes, famille)
    if PAR_NOM[rangement.compartiment].ecriture_directe:
        return rangement
    nom, sim = classer(vecteur, prototypes)
    return Rangement(nom, "moteur", similarite=round(sim, 3), demande=demande)


def demi_vie(nom: str, demi_vie_instance: float) -> float:
    """Demi-vie effective (jours) d'un compartiment. <= 0 = pas d'oubli."""
    c = PAR_NOM.get(nom)
    facteur = c.facteur_demi_vie if c else 1.0
    if demi_vie_instance <= 0 or facteur <= 0:
        return 0.0
    return demi_vie_instance * facteur


def parts_max(surcharges: Mapping[str, float] | None = None) -> dict[str, float]:
    """Part max du budget par compartiment plafonné, surcharges d'instance comprises."""
    parts = {c.nom: c.part_max for c in COMPARTIMENTS if c.part_max < 1.0}
    for nom, part in (surcharges or {}).items():
        if nom in PAR_NOM and 0.0 < part <= 1.0:
            parts[nom] = part
    return parts


def packet_vide() -> dict[str, list[str]]:
    """Paquet de contexte vide : les 9 sections, toujours présentes, dans l'ordre."""
    return {nom: [] for nom in NOMS}


def catalogue() -> list[dict]:
    """Description publique des compartiments (API `GET /v1/compartments`, MCP)."""
    return [{
        "name": c.nom,
        "region": c.region,
        "title": c.titre,
        "description": c.description,
        "examples": list(c.exemples),
        "forgetting": ("aucun" if c.facteur_demi_vie <= 0 else
                       f"x{c.facteur_demi_vie:g} la demi-vie de l'instance"),
        "entangle": c.intrique,
        "always_injected": c.toujours_injecte,
        "max_share": c.part_max,
        "direct_write": c.ecriture_directe,
    } for c in COMPARTIMENTS]
