import requests
from typing import Dict, Any, List, Optional

class SynaptiqClient:
    """
    Mini SDK Python pour faciliter l'intégration de SynaptiQ
    dans n'importe quel pipeline d'agent IA.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        # En-tête d'auth propagé à chaque appel (Phase 3, multi-tenant)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
    def health(self) -> Dict[str, Any]:
        """
        Vérifie l'état des services de SynaptiQ.
        """
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def capture(self, tenant_id: str, agent_id: str, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enregistre un événement ou une interaction brute dans SynaptiQ.
        Cet événement sera classifié et extrait en arrière-plan de manière asynchrone.
        """
        url = f"{self.base_url}/events"
        payload = {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "content": content,
            "metadata": metadata or {}
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de l'enregistrement de l'événement : {e}")

    def build_context(self, tenant_id: str, agent_id: str, session_id: str, task: str, query: str, max_tokens: int = 1200, memory_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Récupère un paquet de contexte structuré et minimaliste pour alimenter le prompt du LLM.
        """
        url = f"{self.base_url}/context/build"
        payload = {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "task": task,
            "query": query,
            "constraints": {
                "max_tokens": max_tokens,
                "memory_types": memory_types or ["semantic", "episodic", "procedural", "working"]
            }
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Échec de la récupération du contexte mémoire : {e}")

    def store_memory(self, tenant_id: str, agent_id: str, memory_type: str, content: str, subtype: Optional[str] = None, confidence: float = 1.0, importance: float = 0.5) -> Dict[str, Any]:
        """
        Permet à l'agent IA d'enregistrer de lui-même une information sémantique, 
        procédurale ou épisodique dans sa mémoire à long terme.
        """
        url = f"{self.base_url}/memories"
        payload = {
            "tenant_id": tenant_id,
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
            raise RuntimeError(f"Échec de l'enregistrement de la mémoire par l'agent : {e}")

    def retrieve(self, tenant_id: str, agent_id: str, query: str, limit: int = 5, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Permet à l'agent IA de rechercher sémantiquement dans ses souvenirs.
        """
        url = f"{self.base_url}/retrieve"
        payload = {
            "tenant_id": tenant_id,
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
            raise RuntimeError(f"Échec de la récupération des souvenirs : {e}")

