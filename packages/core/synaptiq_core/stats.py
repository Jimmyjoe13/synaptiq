"""SynaptiQ — statistiques élémentaires pour les mesures de qualité.

## Pourquoi ce module existe

Le README annonçait **+3,29 points** d'exactitude sur **152 questions**. À cette taille
d'échantillon, l'intervalle de confiance à 95 % d'une proportion voisine de 50 % vaut
environ **±8 points** : le gain annoncé n'est pas distinguable de zéro. Un lecteur
technique fait ce calcul en dix secondes et referme l'onglet — ce qui est dommage, car les
gains par catégorie racontent une histoire cohérente avec ce que fait l'algorithme.

Publier une différence sans son incertitude n'est pas une exagération, c'est une mesure
inexploitable : elle ne permet à personne de décider quoi que ce soit. Ce module rend
l'incertitude obligatoire, parce qu'elle est calculée en même temps que le résultat.

Aucune dépendance : `math` de la bibliothèque standard suffit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Quantile normal bilatéral à 95 % (1,959964). Table figée plutôt qu'une dépendance à scipy.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Proportion:
    """Une proportion mesurée, avec son incertitude.

    `low`/`high` bornent l'intervalle de Wilson à 95 %, préféré à l'approximation normale
    (« Wald ») : celle-ci produit des bornes hors [0, 1] et devient franchement fausse pour
    les petits échantillons ou les proportions extrêmes — exactement les cas d'un benchmark
    par catégorie, où une catégorie ne compte parfois que 16 questions.
    """
    successes: int
    total: int

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def low(self) -> float:
        return wilson_interval(self.successes, self.total)[0]

    @property
    def high(self) -> float:
        return wilson_interval(self.successes, self.total)[1]

    @property
    def margin(self) -> float:
        """Demi-largeur de l'intervalle, en points de proportion (0..1)."""
        return (self.high - self.low) / 2.0

    def as_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "successes": self.successes,
            "total": self.total,
            "ci95_low": round(self.low, 4),
            "ci95_high": round(self.high, 4),
            "ci95_margin_points": round(self.margin * 100, 2),
        }

    def __str__(self) -> str:
        return (f"{self.value * 100:.2f}% "
                f"[{self.low * 100:.2f}–{self.high * 100:.2f}] (n={self.total})")


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion binomiale.

    Retourne (borne_basse, borne_haute), toujours dans [0, 1]. Un total nul rend (0, 0) :
    aucune mesure, donc aucune borne à annoncer.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominateur = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominateur
    demi_largeur = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
                    / denominateur)
    return (max(0.0, centre - demi_largeur), min(1.0, centre + demi_largeur))


@dataclass(frozen=True)
class Difference:
    """Écart entre deux proportions mesurées sur des échantillons indépendants.

    `significant` est le seul champ qui compte pour une conclusion : quand l'intervalle de
    l'écart contient zéro, la mesure ne permet PAS d'affirmer qu'un bras est meilleur.
    C'est précisément le cas du « +3,29 pts » annoncé sur 152 questions.
    """
    a: Proportion
    b: Proportion

    @property
    def delta(self) -> float:
        return self.a.value - self.b.value

    @property
    def interval(self) -> tuple[float, float]:
        """IC 95 % de la différence (approximation normale sur l'erreur type combinée)."""
        if not self.a.total or not self.b.total:
            return (0.0, 0.0)
        pa, pb = self.a.value, self.b.value
        erreur_type = math.sqrt(pa * (1 - pa) / self.a.total + pb * (1 - pb) / self.b.total)
        marge = Z_95 * erreur_type
        return (self.delta - marge, self.delta + marge)

    @property
    def significant(self) -> bool:
        bas, haut = self.interval
        return bas > 0.0 or haut < 0.0

    def as_dict(self) -> dict:
        bas, haut = self.interval
        return {
            "delta_points": round(self.delta * 100, 2),
            "ci95_low_points": round(bas * 100, 2),
            "ci95_high_points": round(haut * 100, 2),
            "significant_at_95": self.significant,
            # Rendu explicite : sans cette phrase, un lecteur pressé retient le delta seul.
            "verdict": (
                "écart statistiquement significatif à 95 %"
                if self.significant else
                "écart NON significatif : l'intervalle contient zéro, "
                "l'échantillon ne permet pas de conclure"
            ),
        }


def required_sample_size(marge_points: float, p: float = 0.5) -> int:
    """Taille d'échantillon nécessaire pour atteindre une marge donnée (en points).

    Sert à dimensionner un run : pour ±2 points autour de 50 %, il faut ~2 400 questions —
    d'où l'intérêt de passer aux 10 conversations LOCOMO (~1 990 questions) plutôt qu'une
    seule (152).
    """
    if marge_points <= 0:
        raise ValueError("La marge doit être strictement positive.")
    marge = marge_points / 100.0
    return math.ceil((Z_95 ** 2) * p * (1 - p) / (marge ** 2))
