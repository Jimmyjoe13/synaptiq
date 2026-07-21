import os
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
SYNAPTIQ_API_KEY = os.getenv("SYNAPTIQ_API_KEY", "")

# En-tête d'auth propagé à l'API si une clé est configurée (Phase 3, multi-tenant)
HEADERS = {"Authorization": f"Bearer {SYNAPTIQ_API_KEY}"} if SYNAPTIQ_API_KEY else {}

# Initialiser FastMCP
mcp = FastMCP("SynaptiQ Memory Engine")

@mcp.tool()
def store_memory(content: str, memory_type: str, subtype: Optional[str] = None, agent_id: str = "qwen_code_agent") -> str:
    """
    Enregistre de maniere autonome un fait, une preference, une regle ou un episode dans la memoire SynaptiQ.

    Args:
        content: Le souvenir ou fait a retenir (ex: 'L'utilisateur prefere les rapports courts').
        memory_type: Le type de memoire ('semantic' pour les faits/preferences, 'procedural' pour les regles/playbooks, 'episodic' pour les actions/resultats).
        subtype: Precision optionnelle (ex: 'preference', 'rule', 'error_resolution').
        agent_id: Identifiant de l'agent qui ecrit (default: 'qwen_code_agent').
    """
    url = f"{SYNAPTIQ_API_URL}/memories"
    payload = {
        "agent_id": agent_id,
        "type": memory_type,
        "subtype": subtype,
        "content": content,
        "confidence": 1.0,
        "importance": 0.5
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=5)
        response.raise_for_status()
        res_data = response.json()
        return f"[SUCCESS] Memoire enregistree avec succes. ID: {res_data.get('memory_id')}"
    except Exception as e:
        return f"[ERROR] Echec de l'enregistrement de la memoire : {e}"

@mcp.tool()
def recall_memories(query: str, limit: int = 5, memory_type: Optional[str] = None, agent_id: str = "qwen_code_agent") -> str:
    """
    Recherche sementiquement des souvenirs ou regles dans la memoire SynaptiQ pour adapter les reponses ou actions de l'agent.

    Args:
        query: Le sujet ou mot-cle a rechercher (ex: 'preferences style ecriture').
        limit: Nombre maximum de souvenirs a ramener (default: 5).
        memory_type: Filtrer par type de memoire ('semantic', 'procedural', 'episodic').
        agent_id: Identifiant de l'agent.
    """
    url = f"{SYNAPTIQ_API_URL}/retrieve"
    payload = {
        "agent_id": agent_id,
        "query": query,
        "limit": limit,
        "memory_type": memory_type
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=5)
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


@mcp.tool()
def build_context(task: str, query: str, max_tokens: int = 1200,
                  agent_id: str = "qwen_code_agent") -> str:
    """
    Assemble un paquet de contexte compact (Q-EM) pret a injecter dans le prompt systeme
    de l'agent : faits, preferences, episodes, regles, bonnes pratiques, erreurs.

    Args:
        task: La tache en cours (ex: 'Rediger un email de suivi B2B').
        query: La requete de rappel semantique (ex: 'style d'ecriture, preferences client').
        max_tokens: Budget de tokens du contexte (default: 1200).
        agent_id: Identifiant de l'agent.
    """
    url = f"{SYNAPTIQ_API_URL}/context/build"
    payload = {
        "agent_id": agent_id,
        "session_id": "mcp-session",
        "task": task,
        "query": query,
        "constraints": {"max_tokens": max_tokens,
                        "memory_types": ["semantic", "episodic", "procedural", "working"]},
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        packet = data.get("context_packet", {})
        lines = [f"Contexte SynaptiQ (~{data.get('token_estimate', 0)} tokens) :"]
        labels = {
            "facts": "FAITS", "preferences": "PREFERENCES", "episodes": "EPISODES",
            "rules": "REGLES", "best_practices": "BONNES PRATIQUES",
            "errors": "ERREURS", "examples": "EXEMPLES",
        }
        for key, label in labels.items():
            for item in packet.get(key, []):
                lines.append(f"- [{label}] {item}")
        return "\n".join(lines) if len(lines) > 1 else "Aucun contexte pertinent trouve."
    except Exception as e:
        return f"[ERROR] Echec de la construction du contexte : {e}"


if __name__ == "__main__":
    # Transport configurable : stdio par défaut (Claude Desktop / Cursor),
    # sse / streamable-http pour un déploiement conteneurisé joignable en réseau.
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run()
    else:
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8765"))
        logger.info(f"Démarrage du serveur MCP en transport '{transport}' sur {host}:{port}")
        mcp.run(transport=transport, host=host, port=port)
