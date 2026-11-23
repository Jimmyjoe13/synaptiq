"""SynaptiQ — client LLM avec ROTATION des modèles gratuits (lot 2, 23/11/2026).

## Pourquoi

Le moteur a besoin d'un LLM à trois endroits : l'extraction du worker, le juge de
contradiction et le sommeil (consolidation des épisodes). L'audit du 22/11 a montré qu'en
production AUCUN n'était joignable (`LLM_BASE_URL` pointait sur un serveur d'embeddings) :
l'extraction échouait à chaque événement et le juge n'archivait jamais rien, sans erreur
visible.

Décision de Jimmy (23/11) : utiliser les modèles GRATUITS de la bibliothèque OpenCode Go.
Leur liste change régulièrement — un modèle disparaît, un autre arrive, un troisième sature
son quota. Un identifiant figé dans la configuration serait donc une panne programmée. Ce
client découvre lui-même les modèles disponibles et passe au suivant dès qu'un modèle
échoue.

## Comment

- **Découverte** : `GET {base}/models`, filtrée par `LLM_FREE_MODEL_PATTERN` (défaut :
  suffixe `-free`), rafraîchie toutes les `LLM_ROTATION_REFRESH_S` secondes. Si la
  découverte échoue, la dernière liste connue (ou `LLM_MODEL_FALLBACKS`) sert de repli.
- **Ordre** : `LLM_MODEL_PREFERENCE` d'abord (si encore proposés), puis l'ordre du
  catalogue.
- **Rotation** : sur un 429, une 5xx, un délai dépassé, une réponse VIDE (modèle qui épuise
  ses tokens à raisonner) ou un JSON invalide quand on en attend un, le modèle est mis en
  pénalité (`Retry-After` respecté, sinon `LLM_ROTATION_COOLDOWN_S`) et le suivant est
  essayé. Tous en échec : `LLMIndisponible`, et chaque appelant applique son repli
  (heuristique pour l'extraction, fail-closed pour le juge, consolidation reportée).
- **OpenCode Go** exige un en-tête `x-opencode-session` (constaté le 23/11 : 400
  `MissingSessionID` sans lui) : un identifiant par processus.

Aucune lecture d'`os.getenv` hors de `depuis_env()` : le client se teste avec un faux
transport HTTP, sans réseau.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import requests

logger = logging.getLogger("synaptiq-core.llm")

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
MOTIF_GRATUIT_DEFAUT = r"-free$"
# Codes qui justifient de passer au modèle suivant (et non de renoncer).
CODES_ROTATION = {404, 408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMIndisponible(RuntimeError):
    """Aucun modèle n'a pu répondre (tous en échec ou en pénalité)."""


@dataclass
class Reponse:
    texte: str
    modele: str


def extraire_json(texte: str) -> Any:
    """Premier objet (ou liste) JSON d'une réponse, balises markdown tolérées."""
    brut = texte.strip()
    if "```" in brut:
        brut = re.sub(r"```(?:json)?", "", brut).strip()
    for ouvrant, fermant in (("{", "}"), ("[", "]")):
        debut, fin = brut.find(ouvrant), brut.rfind(fermant)
        if debut != -1 and fin > debut:
            try:
                return json.loads(brut[debut:fin + 1])
            except ValueError:
                continue
    return json.loads(brut)


@dataclass
class ClientLLM:
    """Client OpenAI-compatible (`/chat/completions`), avec ou sans rotation."""

    base_url: str
    api_key: str = ""
    # Rotation : liste découverte. Sans rotation : `modeles_fixes[0]` seul.
    rotation: bool = True
    modeles_fixes: Sequence[str] = ()
    preference: Sequence[str] = ()
    motif_gratuit: str = MOTIF_GRATUIT_DEFAUT
    rafraichir_s: float = 3600.0
    penalite_s: float = 600.0
    timeout: float = 60.0
    # Transport injectable (tests) : mêmes signatures que `requests.get` / `requests.post`.
    http_get: Callable[..., Any] = field(default=requests.get, repr=False)
    http_post: Callable[..., Any] = field(default=requests.post, repr=False)
    horloge: Callable[[], float] = field(default=time.time, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._catalogue: list[str] = []
        self._catalogue_a = 0.0
        self._penalites: dict[str, float] = {}
        self._verrou = threading.Lock()
        self._session = str(uuid.uuid4())
        self.dernier_modele: str | None = None

    # ── Catalogue ────────────────────────────────────────────────────────────

    def _entetes(self) -> dict[str, str]:
        entetes = {"Content-Type": "application/json"}
        if self.api_key and "your_api_key" not in self.api_key:
            entetes["Authorization"] = f"Bearer {self.api_key}"
        if "opencode.ai" in self.base_url:
            entetes["x-opencode-session"] = self._session
        return entetes

    def catalogue(self, forcer: bool = False) -> list[str]:
        """Modèles gratuits proposés par le fournisseur (cache `rafraichir_s`)."""
        if not self.rotation:
            return list(self.modeles_fixes)
        maintenant = self.horloge()
        with self._verrou:
            if not forcer and self._catalogue and maintenant - self._catalogue_a < self.rafraichir_s:
                return list(self._catalogue)
        try:
            r = self.http_get(f"{self.base_url}/models", headers=self._entetes(),
                              timeout=min(self.timeout, 20))
            r.raise_for_status()
            ids = [str(m.get("id")) for m in r.json().get("data", []) if m.get("id")]
            gratuits = [i for i in ids if re.search(self.motif_gratuit, i)]
            if gratuits:
                with self._verrou:
                    nouveaux = set(gratuits) - set(self._catalogue)
                    if self._catalogue and nouveaux:
                        logger.info("Rotation LLM : nouveaux modèles gratuits %s.", sorted(nouveaux))
                    self._catalogue, self._catalogue_a = gratuits, maintenant
            else:
                logger.warning("Catalogue LLM sans modèle gratuit (%d modèles listés).", len(ids))
        except Exception as e:  # le catalogue n'est jamais une dépendance dure
            logger.warning("Découverte des modèles impossible (%s) : dernière liste conservée.", e)
        with self._verrou:
            return list(self._catalogue or self.modeles_fixes)

    def candidats(self) -> list[str]:
        """Ordre d'essai : préférences encore proposées, puis le catalogue ; hors pénalité."""
        catalogue = self.catalogue()
        ordre = [m for m in self.preference if m in catalogue or not self.rotation]
        ordre += [m for m in catalogue if m not in ordre]
        maintenant = self.horloge()
        with self._verrou:
            disponibles = [m for m in ordre if self._penalites.get(m, 0.0) <= maintenant]
        return disponibles

    def penaliser(self, modele: str, secondes: float | None = None) -> None:
        with self._verrou:
            self._penalites[modele] = self.horloge() + (secondes or self.penalite_s)

    # ── Appels ───────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], max_tokens: int = 800, temperature: float = 0.0,
             valider: Callable[[str], Any] | None = None) -> Reponse:
        """Premier modèle qui rend une réponse non vide (et valide, si `valider` est fourni)."""
        candidats = self.candidats()
        if not candidats:
            raise LLMIndisponible("Aucun modèle LLM disponible (catalogue vide ou tous en pénalité).")
        derniere_erreur = "aucun essai"
        for modele in candidats:
            try:
                r = self.http_post(
                    f"{self.base_url}/chat/completions", headers=self._entetes(),
                    json={"model": modele, "messages": messages, "max_tokens": max_tokens,
                          "temperature": temperature},
                    timeout=self.timeout)
            except Exception as e:      # délai, connexion coupée : modèle suivant
                derniere_erreur = f"{modele}: {type(e).__name__}"
                self.penaliser(modele, 120)
                continue
            if r.status_code in CODES_ROTATION or r.status_code >= 500:
                retry = (getattr(r, "headers", None) or {}).get("Retry-After", "")
                self.penaliser(modele, float(retry) if str(retry).isdigit() else None)
                derniere_erreur = f"{modele}: HTTP {r.status_code}"
                logger.info("Rotation LLM : %s indisponible (HTTP %s), modèle suivant.",
                            modele, r.status_code)
                continue
            if r.status_code >= 400:
                # Erreur de requête (400, 401, 403) : la changer de modèle ne la corrigera
                # généralement pas, mais un modèle retiré du catalogue répond aussi 400.
                self.penaliser(modele)
                derniere_erreur = f"{modele}: HTTP {r.status_code} {str(getattr(r, 'text', ''))[:120]}"
                continue
            try:
                texte = (r.json()["choices"][0]["message"].get("content") or "").strip()
            except Exception:
                texte = ""
            if not texte:
                # Cas réel (23/11) : `hy4-flash-free` répond 200 avec un contenu vide.
                self.penaliser(modele, 300)
                derniere_erreur = f"{modele}: réponse vide"
                continue
            if valider is not None:
                try:
                    valider(texte)
                except Exception:
                    self.penaliser(modele, 120)
                    derniere_erreur = f"{modele}: réponse invalide"
                    continue
            self.dernier_modele = modele
            return Reponse(texte, modele)
        raise LLMIndisponible(f"Tous les modèles ont échoué (dernier : {derniere_erreur}).")

    def chat_json(self, messages: list[dict], max_tokens: int = 1200) -> tuple[Any, str]:
        """Comme `chat`, en exigeant une réponse JSON exploitable : (objet, modèle)."""
        reponse = self.chat(messages, max_tokens=max_tokens, valider=extraire_json)
        return extraire_json(reponse.texte), reponse.modele


def depuis_env(env: dict | None = None) -> ClientLLM | None:
    """Client configuré par l'environnement, ou None si aucun LLM n'est utilisable.

    - `LLM_PROVIDER=mock` (défaut) : None.
    - `LLM_PROVIDER=opencode-go` : base `https://opencode.ai/zen/go/v1`, ROTATION des modèles
      gratuits (sauf `LLM_MODEL` explicite et différent de `auto`).
    - autre fournisseur : `LLM_BASE_URL` + `LLM_MODEL` fixe (rotation si `LLM_MODEL=auto`).
    Un endpoint distant exige une clé ; un endpoint local (LM Studio, Ollama) non.
    """
    e = os.environ if env is None else env
    fournisseur = e.get("LLM_PROVIDER", "mock").lower()
    if fournisseur == "mock":
        return None
    base = e.get("LLM_BASE_URL") or (OPENCODE_GO_BASE_URL if fournisseur == "opencode-go"
                                     else "https://openrouter.ai/api/v1")
    local = any(h in base for h in ("localhost", "127.0.0.1", "host.docker.internal"))
    if fournisseur == "opencode-go" and local:
        # Cas réel de la prod (23/11) : le Compose impose `LLM_BASE_URL` vers le serveur
        # d'embeddings local. Une URL locale ne peut pas être OpenCode Go : on revient à
        # l'endpoint officiel plutôt que d'échouer à chaque appel.
        logger.warning("LLM_PROVIDER=opencode-go avec une URL locale (%s) : endpoint "
                       "officiel %s utilisé.", base, OPENCODE_GO_BASE_URL)
        base, local = OPENCODE_GO_BASE_URL, False
    cle = e.get("LLM_API_KEY", "")
    if not local and not (cle and "your_api_key" not in cle):
        return None
    modele = e.get("LLM_MODEL", "auto" if fournisseur == "opencode-go" else "")
    rotation = e.get("LLM_ROTATION", "").lower() in ("1", "true", "yes") or modele in ("", "auto")

    def _liste(cle_env: str) -> list[str]:
        return [m.strip() for m in e.get(cle_env, "").split(",") if m.strip()]

    return ClientLLM(
        base_url=base, api_key=cle, rotation=rotation,
        modeles_fixes=_liste("LLM_MODEL_FALLBACKS") if rotation else [modele],
        preference=_liste("LLM_MODEL_PREFERENCE"),
        motif_gratuit=e.get("LLM_FREE_MODEL_PATTERN", MOTIF_GRATUIT_DEFAUT),
        rafraichir_s=float(e.get("LLM_ROTATION_REFRESH_S", "3600")),
        penalite_s=float(e.get("LLM_ROTATION_COOLDOWN_S", "600")),
        timeout=float(e.get("LLM_TIMEOUT_S", "60")),
    )


@lru_cache(maxsize=1)
def get_client_llm() -> ClientLLM | None:
    """Client partagé par le processus (l'état de rotation vit avec lui)."""
    client = depuis_env()
    if client is None:
        logger.info("Aucun LLM configuré (LLM_PROVIDER=%s).", os.getenv("LLM_PROVIDER", "mock"))
    else:
        logger.info("Client LLM : %s (rotation=%s).", client.base_url, client.rotation)
    return client
