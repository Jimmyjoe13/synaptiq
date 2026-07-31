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
                idempotency_key: str | None = None) -> dict[str, Any]:
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
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de l'enregistrement de l'événement : {e}") from e

    def build_context(self, agent_id: str, session_id: str, task: str, query: str, max_tokens: int = 1200,
                      memory_types: list[str] | None = None, explain: bool = False,
                      collections: list[str] | None = None) -> dict[str, Any]:
        """
        Récupère un paquet de contexte structuré et minimaliste pour alimenter le prompt du LLM.

        `memory_types` filtre par FAMILLE cognitive (les 4, fermées) ; `collections` filtre
        finement par rayon déclaré par l'agent (`memories.subtype`). Viser une collection
        précise réduit le nombre de candidats en entrée de Q-EM, donc le bruit à budget égal.

        ⚠️ `context_packet` n'a plus un nombre de clés fixe. Il porte toujours les sept
        sections canoniques (`facts`, `preferences`, `episodes`, `rules`, `best_practices`,
        `errors`, `examples`), et EN PLUS une section par collection déclarée par l'agent —
        même vide. Itérer sur `.items()` plutôt que de lire sept clés en dur.
        """
        url = f"{self.base_url}/v1/context/build"
        contraintes: dict[str, Any] = {
            "max_tokens": max_tokens,
            "memory_types": memory_types or ["semantic", "episodic", "procedural", "working"],
        }
        # Omis quand None : le serveur distingue « toutes les collections » (absent) de
        # « cette liste », et une liste vide y serait un filtre qui ne ramène rien.
        if collections is not None:
            contraintes["collections"] = collections
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

    def store_memory(self, agent_id: str, memory_type: str, content: str, subtype: str | None = None, confidence: float = 1.0, importance: float = 0.5) -> dict[str, Any]:
        """
        Permet à l'agent IA d'enregistrer de lui-même une information sémantique,
        procédurale ou épisodique dans sa mémoire à long terme.
        """
        url = f"{self.base_url}/v1/memories"
        payload = {
            "agent_id": agent_id,
            "type": memory_type,
            "subtype": subtype,
            "content": content,
            "confidence": confidence,
            "importance": importance
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de l'enregistrement de la mémoire par l'agent : {e}") from e

    def retrieve(self, agent_id: str, query: str, limit: int = 5, memory_type: str | None = None) -> dict[str, Any]:
        """
        Permet à l'agent IA de rechercher sémantiquement dans ses souvenirs.
        """
        url = f"{self.base_url}/v1/retrieve"
        payload = {
            "agent_id": agent_id,
            "query": query,
            "limit": limit,
            "memory_type": memory_type
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la récupération des souvenirs : {e}") from e
