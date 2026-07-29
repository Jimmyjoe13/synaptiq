import logging
import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

# Rendre packages/core importable (meme hack sys.path que l'API et le worker).
_racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_racine, os.path.join(_racine, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synaptiq_core.observability import configure_logging

# Journalisation sur STDERR, jamais stdout : en transport `stdio`, stdout porte le JSON-RPC
# du protocole MCP et la moindre ligne de log y corromprait la session. Ne pas « harmoniser »
# cet appel avec les autres services (qui, eux, écrivent bien sur stdout).
configure_logging("synaptiq-mcp", stream=sys.stderr)
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
#
# AUCUN DEFAUT, volontairement. Il valait `qwen_code_agent`, ce qui a produit une panne en
# production le 29/07 : le serveur lisait une partition vide et repondait « aucun souvenir
# trouve » — sans erreur, sans avertissement. Pour un moteur de memoire, ce symptome est
# indiscernable d'une memoire reellement vide, donc indebuggable de l'exterieur.
# Un serveur qui refuse de demarrer vaut mieux qu'un serveur qui repond a cote en silence.
SYNAPTIQ_AGENT_ID = os.getenv("SYNAPTIQ_AGENT_ID", "").strip()

# En-tête d'auth propagé à l'API si une clé est configurée (Phase 3, multi-tenant)
HEADERS = {"Authorization": f"Bearer {SYNAPTIQ_API_KEY}"} if SYNAPTIQ_API_KEY else {}

_AIDE_AGENT_ID = (
    "SYNAPTIQ_AGENT_ID n'est pas defini. C'est l'identite memoire sous laquelle ce serveur "
    "lit et ecrit ; sans elle, toute lecture porterait sur une partition arbitraire et "
    "renverrait un resultat vide sans erreur.\n"
    "Le declarer dans la configuration MCP, par exemple :\n"
    '  "env": { "SYNAPTIQ_AGENT_ID": "mon_agent" }\n'
    "Utiliser le meme identifiant qu'a l'ecriture des souvenirs existants "
    "(SELECT DISTINCT agent_id FROM memories pour les retrouver)."
)


def require_agent_id() -> str:
    """Retourne l'identite configuree, ou echoue avec un message actionnable."""
    if not SYNAPTIQ_AGENT_ID:
        raise RuntimeError(_AIDE_AGENT_ID)
    return SYNAPTIQ_AGENT_ID


# Initialiser FastMCP
mcp = FastMCP("SynaptiQ Memory Engine")


def ensure_api_running(timeout_s: float = 10.0) -> bool:
    """Demarre l'API HTTP si elle ne repond pas. Retourne True si elle est joignable.

    ⚠️ N'est PAS appelee a l'import. Un import qui demarre un serveur est un effet de bord
    inacceptable : importer ce module depuis la suite de tests lancait un uvicorn et
    attendait plusieurs secondes.

    Deux precautions sur le sous-processus :
      - stdout/stderr rediriges vers un fichier. En transport `stdio`, le protocole MCP
        utilise stdout pour son JSON-RPC : laisser uvicorn ecrire dedans CORROMPT le flux.
      - processus detache, pour qu'il survive a l'arret du client MCP sans en heriter la
        console.
    """
    def _joignable() -> bool:
        try:
            return requests.get(f"{SYNAPTIQ_API_URL}/health", timeout=2).status_code == 200
        except Exception:
            return False

    if _joignable():
        logger.info("API SynaptiQ deja joignable sur %s.", SYNAPTIQ_API_URL)
        return True

    racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    journal = os.path.join(racine, "api.log")
    logger.info("API injoignable : demarrage d'uvicorn (journal : %s).", journal)

    creation = 0
    if os.name == "nt":
        # DETACHED_PROCESS : pas de console heritee, survit a la fin du client MCP.
        creation = getattr(subprocess, "DETACHED_PROCESS", 0)

    try:
        with open(journal, "ab") as sortie:
            # Arguments entierement litteraux, aucune entree exterieure : `sys.executable`
            # est l'interpreteur courant et le reste est fige. Pas de shell.
            subprocess.Popen(  # noqa: S603
                [sys.executable, "-m", "uvicorn", "apps.api.main:app",
                 "--host", "127.0.0.1", "--port", "8000"],
                cwd=racine, stdout=sortie, stderr=sortie,
                stdin=subprocess.DEVNULL, creationflags=creation,
            )
    except Exception:
        logger.error("Impossible de demarrer l'API automatiquement.", exc_info=True)
        return False

    echeance = time.monotonic() + timeout_s
    while time.monotonic() < echeance:
        time.sleep(0.5)
        if _joignable():
            logger.info("API SynaptiQ demarree.")
            return True
    logger.warning("API toujours injoignable apres %.0f s : voir %s.", timeout_s, journal)
    return False

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
    try:
        # `require_agent_id()` DANS le try : une identite manquante devient un message
        # exploitable par l'agent, pas une exception qui traverse le protocole MCP.
        payload = {
            "agent_id": require_agent_id(),
            "type": memory_type,
            "subtype": subtype,
            "content": content,
            "confidence": 1.0,
            "importance": 0.5,
        }
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
    try:
        payload = {
            "agent_id": require_agent_id(),
            "query": query,
            "limit": limit,
            "memory_type": memory_type,
        }
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
    try:
        payload = {
            "agent_id": require_agent_id(),
            "session_id": "mcp-session",
            "task": task,
            "query": query,
            "constraints": {"max_tokens": max_tokens,
                            "memory_types": ["semantic", "episodic", "procedural", "working"]},
        }
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

    # Echec au demarrage, et non a la premiere requete : le probleme se voit dans les logs
    # MCP du client au lieu de se manifester comme une memoire mysterieusement vide.
    if not SYNAPTIQ_AGENT_ID:
        raise RuntimeError(_AIDE_AGENT_ID)

    if transport != "stdio" and not SYNAPTIQ_API_KEY:
        raise RuntimeError("SYNAPTIQ_API_KEY est obligatoire pour exposer MCP en réseau.")

    # Demarrage de l'API a la demande. Ici et pas a l'import : un import ne doit jamais
    # lancer de serveur. Desactivable, car en conteneur l'API est un service a part.
    if os.getenv("SYNAPTIQ_AUTOSTART_API", "true").lower() in ("1", "true", "yes"):
        ensure_api_running()

    logger.info("Serveur MCP SynaptiQ : agent_id=%s, API=%s.",
                SYNAPTIQ_AGENT_ID, SYNAPTIQ_API_URL)

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
