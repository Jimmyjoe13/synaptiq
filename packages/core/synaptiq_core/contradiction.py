"""SynaptiQ — verdict de contradiction entre deux souvenirs.

## Pourquoi ce module existe

Jusqu'au 28/07, `governance.handle_contradictions` archivait toute préférence active dont
le cosinus avec la nouvelle dépassait 0,8. Autrement dit : **« sémantiquement proche »
était traité comme « contradictoire »**. Or ce sont deux choses différentes.

    « Jimmy préfère les mails courts »        \\  cosinus ~0,85
    « Jimmy préfère les mails en français »   /   → l'ancienne était archivée

Les deux préférences sont compatibles ; l'une disparaissait silencieusement. Le bug était
actif à chaque préférence extraite par le worker, et invisible (aucune erreur, aucun log
d'alerte, la donnée passait simplement en `archived`).

La proximité sémantique reste le bon **pré-filtre** (elle évite d'interroger un juge sur
toutes les paires possibles), mais elle ne peut plus être le verdict. D'où ce module : la
décision d'archiver exige désormais un verdict EXPLICITE.

## Contrat

Un juge est un `Callable[[str, str], bool]` — `judge(existant, nouveau)` répond « le
nouveau contredit-il l'existant ? ». Deux implémentations :

- `no_judge` : répond toujours False. Rien n'est archivé. C'est le comportement retenu
  quand aucun LLM n'est configuré : **ne rien faire est préférable à détruire à tort**.
- `LLMContradictionJudge` : une question fermée à un LLM, `temperature=0`.
  **Fail-closed** : toute erreur (réseau, parse, réponse ambiguë) renvoie False, donc
  n'archive pas. Un juge en panne dégrade la déduplication, il ne perd pas de données.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import lru_cache

from synaptiq_core.llm import ClientLLM, LLMIndisponible, get_client_llm

logger = logging.getLogger("synaptiq-core.contradiction")

ContradictionJudge = Callable[[str, str], bool]


def no_judge(existing: str, new: str) -> bool:
    """Juge nul : aucune contradiction n'est jamais constatée, donc rien n'est archivé."""
    return False


class LLMContradictionJudge:
    """Juge binaire délégué à un LLM, via le client à rotation (`synaptiq_core.llm`).

    Une seule question fermée par paire candidate. Le prompt est en anglais pour la même
    raison que celui de l'extraction (cf. `worker.call_llm_extractor`) : un prompt français
    fait dériver les modèles vers de la reformulation au lieu du classement.

    Fail-closed : si aucun modèle ne répond (tous en échec), les deux souvenirs sont
    conservés. Un juge muet ne doit jamais détruire de donnée.
    """

    def __init__(self, client: ClientLLM) -> None:
        self.client = client

    def __call__(self, existing: str, new: str) -> bool:
        prompt = (
            "Two statements were recorded about the same user, at different times.\n\n"
            f"OLD: {existing}\n"
            f"NEW: {new}\n\n"
            "Does NEW *contradict* OLD — meaning they cannot both be true at once, so OLD "
            "is now obsolete?\n"
            "Answer NO if they are merely related, on the same topic, or simply add detail: "
            "two compatible preferences must both survive.\n"
            "Answer with exactly one word: YES or NO."
        )
        try:
            # 64 tokens et non 4 : certains modèles gratuits raisonnent avant de répondre et
            # rendraient une réponse vide sous un plafond trop serré.
            reponse = self.client.chat(
                [{"role": "system", "content": "You classify contradictions. Reply YES or NO."},
                 {"role": "user", "content": prompt}],
                max_tokens=64,
                valider=_oui_ou_non,
            )
        except LLMIndisponible as e:
            logger.warning("Juge de contradiction indisponible (%s) : aucun archivage.", e)
            return False
        verdict = reponse.texte.strip().upper().lstrip("*\"' ")
        contradicts = verdict.startswith("YES")
        logger.info("Verdict de contradiction : %s (modèle %s, réponse brute : %r).",
                    "contradiction" if contradicts else "compatibles", reponse.modele,
                    verdict[:20])
        return contradicts


def _oui_ou_non(texte: str) -> None:
    """Valide une réponse du juge : YES ou NO, sinon le modèle suivant est essayé."""
    if not texte.strip().upper().lstrip("*\"' ").startswith(("YES", "NO")):
        raise ValueError(f"Réponse hors format : {texte[:40]!r}")


@lru_cache(maxsize=1)
def get_contradiction_judge() -> ContradictionJudge:
    """Instancie le juge configuré (mis en cache pour tout le process).

    `CONTRADICTION_JUDGE` : `auto` (défaut) | `llm` | `off`
      - auto / llm : juge LLM si un LLM est configuré (`synaptiq_core.llm.depuis_env`),
        sinon aucun archivage ;
      - off  : jamais d'archivage automatique, quelle que soit la configuration LLM.
    """
    mode = os.getenv("CONTRADICTION_JUDGE", "auto").lower()
    client = None if mode == "off" else get_client_llm()
    if client is None:
        logger.info("Juge de contradiction désactivé (mode=%s) : aucun archivage automatique. "
                    "Les préférences proches coexisteront.", mode)
        return no_judge
    logger.info("Juge de contradiction = LLM (%s, rotation=%s).", client.base_url, client.rotation)
    return LLMContradictionJudge(client)
