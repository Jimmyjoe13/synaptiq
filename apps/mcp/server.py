import os
import sys
import logging
from typing import Optional
from fastmcp import FastMCP
import requests
from dotenv import load_dotenv

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synaptiq-mcp")

# Charger les variables d'environnement
load_dotenv()
SYNAPTIQ_API_URL = os.getenv("SYNAPTIQ_API_URL", "http://127.0.0.1:8000")

# Initialiser FastMCP
mcp = FastMCP("SynaptiQ Memory Engine")

@mcp.tool()
def store_memory(content: str, memory_type: str, subtype: Optional[str] = None, tenant_id: str = "default_tenant", agent_id: str = "qwen_code_agent") -> str:
    """
    Enregistre de maniere autonome un fait, une preference, une regle ou un episode dans la memoire SynaptiQ.
    
    Args:
        content: Le souvenir ou fait a retenir (ex: 'L'utilisateur prefere les rapports courts').
        memory_type: Le type de memoire ('semantic' pour les faits/preferences, 'procedural' pour les regles/playbooks, 'episodic' pour les actions/resultats).
        subtype: Precision optionnelle (ex: 'preference', 'rule', 'error_resolution').
        tenant_id: Identifiant du locataire (default: 'default_tenant').
        agent_id: Identifiant de l'agent qui ecrit (default: 'qwen_code_agent').
    """
    url = f"{SYNAPTIQ_API_URL}/memories"
    payload = {
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "type": memory_type,
        "subtype": subtype,
        "content": content,
        "confidence": 1.0,
        "importance": 0.5
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        res_data = response.json()
        return f"[SUCCESS] Memoire enregistree avec succes. ID: {res_data.get('memory_id')}"
    except Exception as e:
        return f"[ERROR] Echec de l'enregistrement de la memoire : {e}"

@mcp.tool()
def recall_memories(query: str, limit: int = 5, memory_type: Optional[str] = None, tenant_id: str = "default_tenant", agent_id: str = "qwen_code_agent") -> str:
    """
    Recherche sementiquement des souvenirs ou regles dans la memoire SynaptiQ pour adapter les reponses ou actions de l'agent.
    
    Args:
        query: Le sujet ou mot-cle a rechercher (ex: 'preferences style ecriture').
        limit: Nombre maximum de souvenirs a ramener (default: 5).
        memory_type: Filtrer par type de memoire ('semantic', 'procedural', 'episodic').
        tenant_id: Identifiant du locataire.
        agent_id: Identifiant de l'agent.
    """
    url = f"{SYNAPTIQ_API_URL}/retrieve"
    payload = {
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "query": query,
        "limit": limit,
        "memory_type": memory_type
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        memories = response.json().get("memories", [])
        
        if not memories:
            return "Aucun souvenir correspondant trouve dans la base."
            
        output = ["Souvenirs retrouves dans SynaptiQ :"]
        for mem in memories:
            output.append(f"- [{mem['type'].upper()} / {mem['subtype'] or 'general'}] {mem['content']} (Confidence: {mem['confidence']})")
        return "\n".join(output)
    except Exception as e:
        return f"[ERROR] Echec de la recherche de souvenirs : {e}"

if __name__ == "__main__":
    # Lancement du serveur MCP
    mcp.run()
