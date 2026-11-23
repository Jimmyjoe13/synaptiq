from typing import Any

import requests


class SynaptiqClient:
    """
    Mini SDK Python pour faciliter l'intégration de SynaptiQ
    dans n'importe quel pipeline d'agent IA.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str | None = None):
        self.base_url = base_url.rstrip('/')
        # En-tête d'auth propagé à chaque appel (Phase 3, multi-tenant)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def health(self) -> dict[str, Any]:
        """
        Vérifie l'état des services de SynaptiQ.
        """
        try:
            response = requests.get(f"{self.base_url}/v1/health", headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def capture(self, agent_id: str, session_id: str, content: str, metadata: dict[str, Any] | None = None,
                idempotency_key: str | None = None, project: str | None = None) -> dict[str, Any]:
        """
        Enregistre un événement ou une interaction brute dans SynaptiQ.
        Cet événement sera classifié et extrait en arrière-plan de manière asynchrone.
        """
        url = f"{self.base_url}/v1/events"
        payload = {
            "agent_id": agent_id,
            "session_id": session_id,
            "content": content,
            "metadata": metadata or {},
            "idempotency_key": idempotency_key,
        }
        # Projet (lot B) : appliqué à tous les faits que le worker extraira de l'événement.
        if project is not None:
            payload["project"] = project
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de l'enregistrement de l'événement : {e}") from e

    def build_context(self, agent_id: str, session_id: str, task: str, query: str, max_tokens: int = 1200,
                      compartments: list[str] | None = None, explain: bool = False,
                      project: str | None = None,
                      include_global: bool = True) -> dict[str, Any]:
        """
        Récupère un paquet de contexte structuré et minimaliste pour alimenter le prompt du LLM.

        `context_packet` a une forme FIXE (v0.4) : une section par compartiment (`reflexes`,
        `decisions`, `savoir_faire`, `alertes`, `environnement`, `savoirs`, `episodes`,
        `observations`, `croyances`), toujours présente, même vide. `compartments` restreint
        le rappel à certains d'entre eux (None = tous).
        """
        url = f"{self.base_url}/v1/context/build"
        contraintes: dict[str, Any] = {"max_tokens": max_tokens}
        # Omis quand None : le serveur prend alors les 9 compartiments, alors qu'une liste
        # vide serait refusée.
        if compartments is not None:
            contraintes["compartments"] = compartments
        # Projet (lot B) : ce projet + les souvenirs globaux (sauf include_global=False).
        if project is not None:
            contraintes["project"] = project
            contraintes["include_global"] = include_global
        payload = {
            "agent_id": agent_id,
            "session_id": session_id,
            "task": task,
            "query": query,
            "constraints": contraintes,
            "explain": explain,
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la récupération du contexte mémoire : {e}") from e

    def store_memory(self, agent_id: str, content: str, compartment: str | None = None,
                     confidence: float = 1.0, importance: float = 0.5,
                     project: str | None = None) -> dict[str, Any]:
        """
        Enregistre un souvenir dans le compartiment choisi par l'agent.

        `compartment` est facultatif : absent ou inconnu, le serveur classe le souvenir par
        similarité. La réponse dit toujours où il a été rangé (`compartment`) et par qui
        (`compartment_origin` : `agent`, `alias`, `moteur`, `existant`).
        """
        url = f"{self.base_url}/v1/memories"
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "content": content,
            "confidence": confidence,
            "importance": importance
        }
        if compartment is not None:
            payload["compartment"] = compartment
        # None = souvenir GLOBAL (valable dans tous les projets).
        if project is not None:
            payload["project"] = project
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de l'enregistrement de la mémoire par l'agent : {e}") from e

    def retrieve(self, agent_id: str, query: str, limit: int = 5,
                 compartments: list[str] | None = None, project: str | None = None,
                 include_global: bool = True) -> dict[str, Any]:
        """
        Permet à l'agent IA de rechercher sémantiquement dans ses souvenirs.

        `compartments` restreint la recherche à certains compartiments (None = tous).
        """
        url = f"{self.base_url}/v1/retrieve"
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "query": query,
            "limit": limit,
        }
        # Omis quand None : une liste vide serait un filtre qui ne ramène rien, alors que
        # l'absence de filtre doit tout balayer.
        if compartments is not None:
            payload["compartments"] = compartments
        if project is not None:
            payload["project"] = project
            payload["include_global"] = include_global
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la récupération des souvenirs : {e}") from e

    # ─── Croyances : ce que l'agent pense (compartiment `croyances`) ──────────
    # Hypothèses sur l'utilisateur (`about="user"`) ou les humains (`about="humans"`) :
    # validées par le garde-fou côté serveur (422 sinon), servies comme « hypothèse,
    # confiance x », consultables et contestables par l'utilisateur.

    def note_belief(self, agent_id: str, content: str, about: str = "user",
                    confidence: float = 0.5, evidence: list[str] | None = None,
                    replaces: str | None = None, project: str | None = None) -> dict[str, Any]:
        """Note une croyance. Confiance > 0.5 : `evidence` (ids de souvenirs) obligatoire."""
        payload: dict[str, Any] = {
            "agent_id": agent_id, "compartment": "croyances", "about": about,
            "content": content, "confidence": confidence,
        }
        if evidence:
            payload["evidence"] = evidence
        if replaces:
            payload["replaces"] = replaces
        if project is not None:
            payload["project"] = project
        try:
            response = requests.post(f"{self.base_url}/v1/memories", json=payload,
                                     headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de l'enregistrement de la croyance : {e}") from e

    def list_beliefs(self, agent_id: str, about: str | None = None) -> dict[str, Any]:
        """Croyances actives de l'agent, la plus assurée d'abord."""
        params = {"agent_id": agent_id, **({"about": about} if about else {})}
        try:
            response = requests.get(f"{self.base_url}/v1/beliefs", params=params,
                                    headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la lecture des croyances : {e}") from e

    def contest_belief(self, agent_id: str, belief_id: str,
                       reason: str | None = None) -> dict[str, Any]:
        """Conteste une croyance : retirée du contexte, non réinscriptible à l'identique."""
        try:
            response = requests.post(f"{self.base_url}/v1/beliefs/{belief_id}/contest",
                                     json={"agent_id": agent_id, "reason": reason},
                                     headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la contestation : {e}") from e

    # ─── Récompense, plasticité, archives (lot 2) ─────────────────────────────

    def reinforce(self, agent_id: str, memory_ids: list[str],
                  useful: bool = True) -> dict[str, Any]:
        """Récompense (±1) des souvenirs ; s'ils ont servi ensemble, renforce leurs liens."""
        try:
            response = requests.post(f"{self.base_url}/v1/memories/reinforce",
                                     json={"agent_id": agent_id, "memory_ids": memory_ids,
                                           "useful": useful},
                                     headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec du renforcement : {e}") from e

    def restore_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Sort un souvenir des archives, vers son compartiment d'origine."""
        try:
            response = requests.post(f"{self.base_url}/v1/memories/{memory_id}/restore",
                                     json={"agent_id": agent_id}, headers=self.headers,
                                     timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la restauration : {e}") from e

    # ─── Compartiments : les régions fixes de la mémoire ─────────────────────
    # Propriété du moteur : l'agent ne crée rien, il choisit où ranger.

    def list_compartments(self, agent_id: str) -> dict[str, Any]:
        """Les compartiments, leur fonction et le nombre de souvenirs de l'agent dans chacun."""
        url = f"{self.base_url}/v1/compartments"
        try:
            response = requests.get(url, params={"agent_id": agent_id},
                                    headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la lecture des compartiments : {e}") from e
