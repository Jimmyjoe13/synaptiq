"""SynaptiQ — garde-fou des CROYANCES (famille `reflective`, 26/09).

Une croyance est ce que l'agent pense de l'utilisateur ou des humains. Contrairement à un
fait, elle est INFÉRÉE — et c'est ce qui la rend délicate : un agent qui « déduit » l'état
de santé, l'orientation ou les opinions politiques de son utilisateur à partir d'indices
fabrique exactement le type de donnée que le RGPD (art. 9) interdit de traiter sans
consentement explicite, et que personne n'a demandé à l'agent de produire.

Règles, toutes déterministes (aucun appel réseau, testables) :

  1. **Catégories sensibles refusées** : santé, opinions politiques, convictions
     religieuses, orientation sexuelle, origine ethnique, appartenance syndicale. Le refus
     porte sur le TEXTE de la croyance, par lexique FR/EN : faux positifs possibles (une
     croyance « Jimmy est allergique aux réunions » serait refusée pour « allergique »),
     assumés — un refus se reformule, une inférence sensible stockée ne se rattrape pas.
  2. **Confiance plafonnée** à `CONFIANCE_MAX` (0,9) : une croyance n'est jamais une
     certitude. Au-delà, c'est un fait, qui a sa famille (`semantic`).
  3. **Pas de confiance sans indices** : au-dessus de `SEUIL_PREUVE` (0,5), au moins un
     identifiant de souvenir à l'appui est exigé. Une hypothèse forte sans trace ne se
     contrôle pas, et c'est précisément ce que l'utilisateur doit pouvoir faire.

Les tiers (clients, collègues) sont AUTORISÉS dans `human_insights` (décision de Jimmy du
26/09) ; les catégories sensibles restent interdites quel que soit le sujet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CONFIANCE_MAX = 0.9
SEUIL_PREUVE = 0.5

# Lexique des catégories sensibles (art. 9 RGPD), FR et EN, en racines pour couvrir les
# flexions. Bornées au début de mot : « santé » ne doit pas matcher dans « assainir ».
_CATEGORIES: dict[str, tuple[str, ...]] = {
    "sante": (
        r"sant[ée]", r"malad", r"d[ée]press", r"anxi[ée]t", r"burn.?out", r"cancer",
        r"diab[èe]t", r"handicap", r"autis", r"tdah", r"adhd", r"bipolai", r"th[ée]rap",
        r"m[ée]dicament", r"traitement m[ée]dical", r"allergi", r"enceinte", r"grossesse",
        r"health", r"illness", r"disease", r"depress", r"medication", r"pregnan",
        r"psychiatr", r"trouble mental", r"mental health",
    ),
    "politique": (
        r"politique", r"[ée]lecteur de", r"vote pour", r"de gauche", r"de droite",
        r"extr[êe]me.(gauche|droite)", r"militant", r"political", r"votes? for",
        r"left.wing", r"right.wing",
    ),
    "religion": (
        r"religi", r"croyant", r"ath[ée]e", r"musulman", r"chr[ée]tien", r"catholique",
        r"juif", r"juive", r"bouddhis", r"prati(que|quant) (sa|la) foi", r"believer",
        r"muslim", r"christian", r"jewish", r"atheist",
    ),
    "orientation": (
        r"homosexu", r"h[ée]t[ée]rosexu", r"bisexu", r"\bgay\b", r"lesbien", r"orientation sexuelle",
        r"\blgbt", r"transgenre", r"transgender", r"sexual orientation", r"sexualit",
    ),
    "origine": (
        r"origine ethnique", r"\brace\b", r"racial", r"ethni", r"couleur de peau",
        r"immigr[ée]", r"skin colou?r",
    ),
    "syndicat": (
        r"syndiqu", r"syndicat", r"trade union", r"union member",
    ),
}
_MOTIFS = {cat: re.compile(r"\b(?:" + "|".join(racines) + r")", re.IGNORECASE)
           for cat, racines in _CATEGORIES.items()}


class CroyanceRefusee(ValueError):
    """La croyance viole une règle du garde-fou ; le message dit laquelle et comment faire."""


@dataclass(frozen=True)
class CroyanceValidee:
    content: str
    confidence: float          # plafonnée à CONFIANCE_MAX
    evidence: tuple[str, ...]  # identifiants de souvenirs à l'appui


def categorie_sensible(texte: str) -> str | None:
    """Première catégorie sensible détectée dans le texte, ou None."""
    for categorie, motif in _MOTIFS.items():
        if motif.search(texte):
            return categorie
    return None


def valider_croyance(content: str, confidence: float,
                     evidence: list[str] | tuple[str, ...] | None = None) -> CroyanceValidee:
    """Applique les trois règles ; lève `CroyanceRefusee` avec un message actionnable."""
    categorie = categorie_sensible(content)
    if categorie is not None:
        raise CroyanceRefusee(
            f"Croyance refusée : elle touche une catégorie sensible ({categorie}). Un agent "
            "ne doit pas inférer la santé, les opinions politiques, la religion, "
            "l'orientation sexuelle, l'origine ou l'appartenance syndicale d'une personne. "
            "Reformuler en décrivant un comportement observable, sans cette inférence.")
    preuves = tuple(e for e in (evidence or ()) if e)
    if confidence > SEUIL_PREUVE and not preuves:
        raise CroyanceRefusee(
            f"Croyance refusée : une confiance supérieure à {SEUIL_PREUVE} exige au moins "
            "un souvenir à l'appui (evidence_ids). Sans indice, noter l'hypothèse avec "
            f"une confiance de {SEUIL_PREUVE} au plus.")
    return CroyanceValidee(content=content, confidence=min(confidence, CONFIANCE_MAX),
                           evidence=preuves)
