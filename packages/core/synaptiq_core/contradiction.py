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

import requests

logger = logging.getLogger("synaptiq-core.contradiction")

ContradictionJudge = Callable[[str, str], bool]


def no_judge(existing: str, new: str) -> bool:
    """Juge nul : aucune contradiction n'est jamais constatée, donc rien n'est archivé."""
    return False


class LLMContradictionJudge:
    """Juge binaire délégué à un LLM (endpoint OpenAI-compatible).

    Une seule question fermée par paire candidate. Le prompt est en anglais pour la même
    raison que celui de l'extraction (cf. `worker.call_llm_extractor`) : un prompt français
    fait dériver les modèles vers de la reformulation au lieu du classement.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

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
        headers = {"Content-Type": "application/json"}
        if self.api_key and "your_api_key" not in self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You classify contradictions. Reply YES or NO."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 4,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            verdict = resp.json()["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:
            # Fail-closed : en cas de doute, on conserve les deux souvenirs.
            logger.warning("Juge de contradiction indisponible (%s) : aucun archivage.", e)
            return False

        contradicts = verdict.startswith("YES")
        logger.info("Verdict de contradiction : %s (réponse brute : %r).",
                    "contradiction" if contradicts else "compatibles", verdict[:20])
        return contradicts


def _llm_available() -> bool:
    """Un LLM est-il réellement joignable pour juger ?

    Même règle que l'extracteur du worker : un endpoint LOCAL (LM Studio, Ollama) n'exige
    aucune clé, un endpoint distant en exige une valide.
    """
    if os.getenv("LLM_PROVIDER", "mock") == "mock":
        return False
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    local = any(h in base_url for h in ("localhost", "127.0.0.1", "host.docker.internal"))
    api_key = os.getenv("LLM_API_KEY", "")
    return local or bool(api_key and "your_api_key" not in api_key)


@lru_cache(maxsize=1)
def get_contradiction_judge() -> ContradictionJudge:
    """Instancie le juge configuré (mis en cache pour tout le process).

    `CONTRADICTION_JUDGE` : `auto` (défaut) | `llm` | `off`
      - auto : juge LLM si un LLM est configuré, sinon aucun archivage.
      - llm  : force le juge LLM (échoue en fail-closed s'il est injoignable).
      - off  : jamais d'archivage automatique, quelle que soit la configuration LLM.
    """
    mode = os.getenv("CONTRADICTION_JUDGE", "auto").lower()
    if mode == "off" or (mode == "auto" and not _llm_available()):
        logger.info("Juge de contradiction désactivé (mode=%s) : aucun archivage automatique. "
                    "Les préférences proches coexisteront.", mode)
        return no_judge
    judge = LLMContradictionJudge(
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )
    logger.info("Juge de contradiction = LLM (%s, %s).", judge.base_url, judge.model)
    return judge
