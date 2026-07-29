import logging
import os

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synaptiq-mcp")

# Charger les variables d'environnement
load_dotenv()
SYNAPTIQ_API_URL = os.getenv("SYNAPTIQ_API_URL", "http://127.0.0.1:8000")
SYNAPTIQ_API_KEY = os.getenv("SYNAPTIQ_API_KEY", "")

# Identite de l'agent, fixee par la CONFIGURATION du serveur MCP et non par l'appelant.
# Jusqu'au 29/07, `agent_id` etait un parametre des outils : c'etait donc le LLM qui
# choisissait sous quelle identite lire et ecrire, et il suffisait d'une autre chaine pour
# atteindre la memoire d'un autre agent de l'instance. Une valeur d'environnement n'est
# atteignable par aucun prompt.
SYNAPTIQ_AGENT_ID = os.getenv("SYNAPTIQ_AGENT_ID", "qwen_code_agent")

# En-tête d'auth propagé à l'API si une clé est configurée (Phase 3, multi-tenant)
HEADERS = {"Authorization": f"Bearer {SYNAPTIQ_API_KEY}"} if SYNAPTIQ_API_KEY else {}

# Initialiser FastMCP
mcp = FastMCP("SynaptiQ Memory Engine")

@mcp.tool()
def store_memory(content: str, memory_type: str, subtype: str | None = None) -> str:
    """
    Enregistre de maniere autonome un fait, une preference, une regle ou un episode dans la memoire SynaptiQ.

    Args:
        content: Le souvenir ou fait a retenir (ex: 'L'utilisateur prefere les rapports courts').
        memory_type: Le type de memoire ('semantic' pour les faits/preferences, 'procedural' pour les regles/playbooks, 'episodic' pour les actions/resultats).
        subtype: Precision optionnelle (ex: 'preference', 'rule', 'error_resolution').
    """
    url = f"{SYNAPTIQ_API_URL}/v1/memories"
    payload = {
        "agent_id": SYNAPTIQ_AGENT_ID,
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
def recall_memories(query: str, limit: int = 5, memory_type: str | None = None) -> str:
    """
    Recherche sementiquement des souvenirs ou regles dans la memoire SynaptiQ pour adapter les reponses ou actions de l'agent.

    Args:
        query: Le sujet ou mot-cle a rechercher (ex: 'preferences style ecriture').
        limit: Nombre maximum de souvenirs a ramener (default: 5).
        memory_type: Filtrer par type de memoire ('semantic', 'procedural', 'episodic').
    """
    url = f"{SYNAPTIQ_API_URL}/v1/retrieve"
    payload = {
        "agent_id": SYNAPTIQ_AGENT_ID,
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
def build_context(task: str, query: str, max_tokens: int = 1200) -> str:
    """
    Assemble un paquet de contexte compact (Q-EM) pret a injecter dans le prompt systeme
    de l'agent : faits, preferences, episodes, regles, bonnes pratiques, erreurs.

    Args:
        task: La tache en cours (ex: 'Rediger un email de suivi B2B').
        query: La requete de rappel semantique (ex: 'style d'ecriture, preferences client').
        max_tokens: Budget de tokens du contexte (default: 1200).
    """
    url = f"{SYNAPTIQ_API_URL}/v1/context/build"
    payload = {
        "agent_id": SYNAPTIQ_AGENT_ID,
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
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport != "stdio" and not SYNAPTIQ_API_KEY:
        raise RuntimeError("SYNAPTIQ_API_KEY est obligatoire pour exposer MCP en réseau.")
    if transport == "stdio":
        mcp.run()
    else:
        # 0.0.0.0 est intentionnel : en conteneur, se lier a 127.0.0.1 rendrait le serveur
        # injoignable depuis l'hote. Le port n'est publie que sur 127.0.0.1 par Compose,
        # et ce transport exige SYNAPTIQ_API_KEY (verifie ci-dessus).
        host = os.getenv("MCP_HOST", "0.0.0.0")  # noqa: S104
        port = int(os.getenv("MCP_PORT", "8765"))
        logger.info(f"Démarrage du serveur MCP en transport '{transport}' sur {host}:{port}")
        mcp.run(transport=transport, host=host, port=port)
