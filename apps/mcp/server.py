import logging
import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

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
    """Retourne l'identite configuree, ou echoue avec un message actionnable.

    Leve ICI, au moment de l'appel d'outil, et non au demarrage : c'est le seul endroit ou
    le message atteint reellement l'utilisateur (cf. `verifier_configuration`).
    """
    if not SYNAPTIQ_AGENT_ID:
        raise RuntimeError(_AIDE_AGENT_ID)
    return SYNAPTIQ_AGENT_ID


def verifier_configuration() -> bool:
    """Journalise l'etat de la configuration au demarrage. Retourne True si elle est complete.

    ⚠️ NE QUITTE PAS quand `SYNAPTIQ_AGENT_ID` manque, et c'est deliberé.

    La version precedente levait une RuntimeError pour « echouer vite ». En contexte MCP,
    c'etait exactement le mauvais choix : le client ne montre que
    `failed to stop mcp instance: synaptiq: exit status 1`, jette le contenu de stderr, et
    le serveur DISPARAIT de la liste des serveurs. Le message d'aide, si soigneusement
    redige soit-il, n'atteignait donc jamais personne — et l'utilisateur se retrouvait avec
    un code d'erreur opaque au lieu d'un diagnostic.

    Echouer vite n'a de valeur que si quelqu'un LIT l'echec. Ici, le seul canal que
    l'utilisateur lit vraiment est la reponse d'un outil. Le serveur demarre donc toujours,
    expose ses outils, et chaque outil renvoie l'explication complete via `require_agent_id`.
    """
    if not SYNAPTIQ_AGENT_ID:
        logger.error("DEMARRAGE DEGRADE — aucune identite memoire configuree. Le serveur "
                     "demarre et expose ses outils, mais chaque appel echouera avec ce "
                     "message : %s", _AIDE_AGENT_ID)
        return False
    logger.info("Serveur MCP SynaptiQ : agent_id=%s, API=%s.",
                SYNAPTIQ_AGENT_ID, SYNAPTIQ_API_URL)
    return True


# Initialiser FastMCP
mcp = FastMCP("SynaptiQ Memory Engine")


def ensure_api_running(timeout_s: float | None = None) -> bool:
    """Demarre l'API HTTP si elle ne repond pas. Retourne True si elle est joignable.

    ⚠️ NE BLOQUE PAS par defaut (`SYNAPTIQ_AUTOSTART_WAIT_S=0`).

    Version precedente : elle attendait jusqu'a 10 s que l'API reponde, AVANT `mcp.run()`.
    Mesure : 14 s avant le handshake MCP quand l'API etait injoignable, contre 1,9 s
    sinon. Or le client MCP a son propre delai d'initialisation, bien plus court : il tuait
    le process, et sur Windows un TerminateProcess se lit `exit status 1` — ce qui faisait
    echouer le rechargement de TOUS les serveurs du client.

    Le handshake d'un protocole ne doit jamais attendre une tache annexe. On lance donc
    uvicorn et on rend la main immediatement ; si le premier appel d'outil arrive avant que
    l'API ecoute, `_poster()` reessaie une fois (cf. RETRY_DELAI_S).

    Deux precautions sur le sous-processus :
      - stdout/stderr rediriges vers un fichier. En transport `stdio`, le protocole MCP
        utilise stdout pour son JSON-RPC : laisser uvicorn ecrire dedans CORROMPT le flux.
      - processus detache, pour qu'il survive a l'arret du client MCP sans en heriter la
        console.
    """
    if timeout_s is None:
        timeout_s = float(os.getenv("SYNAPTIQ_AUTOSTART_WAIT_S", "0"))
    def _joignable() -> bool:
        # Timeout court : c'est un appel local. Un port TENU par un process fige (plutot que
        # ferme) ferait sinon patienter le handshake pour rien.
        try:
            return requests.get(f"{SYNAPTIQ_API_URL}/health", timeout=1).status_code == 200
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

    if timeout_s <= 0:
        # Cas par defaut : on n'attend pas. Le handshake MCP passe immediatement.
        logger.info("uvicorn lance en arriere-plan (pas d'attente, journal : %s).", journal)
        return False

    echeance = time.monotonic() + timeout_s
    while time.monotonic() < echeance:
        time.sleep(0.25)
        if _joignable():
            logger.info("API SynaptiQ demarree.")
            return True
    logger.warning("API toujours injoignable apres %.1f s : voir %s.", timeout_s, journal)
    return False

# Delai avant l'unique nouvelle tentative quand l'API n'ecoute pas encore. Le demarrage
# automatique d'uvicorn ne bloque plus le handshake : un premier appel d'outil peut donc
# arriver pendant que l'API finit de demarrer.
RETRY_DELAI_S = float(os.getenv("SYNAPTIQ_RETRY_DELAI_S", "2.0"))

# Le premier appel d'une session paie le chargement du modele d'embedding (LM Studio, en
# local) : mesure a plus de 5 s a froid, ~2,6 s une fois chaud. L'ancien defaut de 5 s en dur
# faisait donc echouer le premier `recall_memories` de chaque session -- precisement celui qui
# construit le contexte initial de l'agent, quand la memoire lui est la plus utile.
TIMEOUT_S = float(os.getenv("SYNAPTIQ_TIMEOUT_S", "30"))


def _poster(url: str, payload: dict, timeout: float | None = None):
    """POST vers l'API, avec UNE nouvelle tentative si la connexion a ete REFUSEE.

    ⚠️ La relance ne couvre que le cas ou la requete n'a PAS pu partir. `ConnectionError` de
    `requests` est plus large que ca : elle englobe aussi une connexion coupee APRES l'envoi
    (`ConnectionResetError`, `ProtocolError`), cas ou le serveur a tres bien pu committer
    avant la coupure. Relancer aveuglement sur toute `ConnectionError` faisait donc du
    serveur MCP lui-meme un producteur de doublons, sans aucune relance du client.

    On distingue les deux : seul un refus de connexion franc (`ConnectionRefusedError`, l'API
    pas encore levee -- le scenario que cette relance existe pour couvrir) est rejoue. Toute
    autre coupure remonte a l'appelant, qui saura que l'ecriture est d'issue INCONNUE.

    L'ecriture directe est desormais idempotente sur le contenu cote API, donc une relance
    legitime est un no-op ; mais une relance dont on ne sait pas si elle est legitime n'a rien
    a faire ici -- le filet cote serveur ne dispense pas d'etre correct cote client.
    """
    if timeout is None:
        timeout = TIMEOUT_S
    try:
        return requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
    except requests.ConnectionError as err:
        if not _connexion_refusee(err):
            raise
        logger.info("API pas encore joignable, nouvelle tentative dans %.1f s.", RETRY_DELAI_S)
        time.sleep(RETRY_DELAI_S)
        return requests.post(url, json=payload, headers=HEADERS, timeout=timeout)


def _connexion_refusee(err: BaseException) -> bool:
    """La requete n'a-t-elle jamais atteint le serveur ?

    `requests` empile ses causes (`ConnectionError` -> `urllib3.NewConnectionError` ->
    `OSError`), donc on descend la chaine `__cause__`/`__context__` a la recherche d'un
    `ConnectionRefusedError`. Repondre False par defaut est le choix sur : on ne rejoue que
    ce qu'on a formellement identifie comme n'etant jamais parti.
    """
    vu = set()
    courant: BaseException | None = err
    while courant is not None and id(courant) not in vu:
        vu.add(id(courant))
        if isinstance(courant, ConnectionRefusedError):
            return True
        # urllib3 emballe l'OSError sans la chainer : son texte reste le seul indice.
        if "refused" in str(courant).lower():
            return True
        courant = courant.__cause__ or courant.__context__
    return False


def _lire(url: str, params: dict, timeout: float | None = None):
    """GET vers l'API, avec UNE nouvelle tentative si la connexion est refusee.

    NE PAS « harmoniser » avec `_poster` : la restriction que celui-ci s'impose n'a aucune
    raison d'etre ici. Un GET ne cree rien, donc le rejouer ne peut pas dupliquer de
    souvenir, quelle que soit l'etape ou la connexion a lache.
    """
    if timeout is None:
        timeout = TIMEOUT_S
    try:
        return requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    except requests.ConnectionError:
        logger.info("API pas encore joignable, nouvelle tentative dans %.1f s.", RETRY_DELAI_S)
        time.sleep(RETRY_DELAI_S)
        return requests.get(url, params=params, headers=HEADERS, timeout=timeout)


def _echec(operation: str, err: Exception) -> str:
    """Traduit une exception d'outil en reponse MCP, selon qu'elle est corrigeable ou non.

    Une identite manquante (`RuntimeError` de `require_agent_id`) reste un message texte :
    c'est un defaut de CONFIGURATION, dont le message porte la marche a suivre, et le
    commentaire de `require_agent_id` explique pourquoi il ne doit pas traverser le protocole.

    Tout le reste -- timeout, API injoignable, 5xx -- est une PANNE, et doit sortir en
    `isError: true`. Rendu en texte, un echec est indiscernable d'un resultat : le timeout du
    30/07 est ainsi remonte jusqu'a l'agent, qui a affiche « [ERROR] Read timed out » comme
    s'il s'agissait d'un souvenir. Une memoire qui echoue en silence est pire qu'une memoire
    absente.
    """
    if isinstance(err, RuntimeError):
        return f"[ERROR] {operation} : {err}"
    logger.error("%s : %s", operation, err, exc_info=True)
    raise ToolError(f"{operation} : {err}") from err


@mcp.tool()
def store_memory(content: str, memory_type: str, subtype: str | None = None) -> str:
    """
    Enregistre de maniere autonome un fait, une preference, une regle ou un episode dans la memoire SynaptiQ.

    Le couple (memory_type, subtype) designe une COLLECTION. `subtype` est le nom du rayon :
    utiliser `list_collections` pour voir ceux qui existent, `create_collection` pour en
    ouvrir un nouveau. Un nom NON declare est accepte, mais le souvenir est alors range dans
    la section par defaut de sa famille -- la reponse le dit explicitement.

    Args:
        content: Le souvenir ou fait a retenir (ex: 'L'utilisateur prefere les rapports courts').
        memory_type: La famille de memoire ('semantic' pour les faits/preferences, 'procedural' pour les regles/playbooks, 'episodic' pour les actions/resultats, 'working' pour le volatil).
        subtype: Nom de la collection (ex: 'preference', 'rule', 'clients_paca').
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
        response = _poster(url, payload)
        response.raise_for_status()
        res_data = response.json()

        # ── Le verdict de rangement est RENDU a l'agent ──
        # L'API renvoie `collection` et `canonical_subtype` depuis toujours ; cet outil les
        # jetait et ne retournait que l'identifiant. L'agent croyait donc ranger finement
        # alors que son libelle metier retombait dans la section par defaut de sa famille,
        # et rien ne le detrompait. Un moteur de memoire ne doit pas laisser croire a un
        # classement qui n'a pas eu lieu.
        collection = res_data.get("collection")
        # `duplicate` : ce contenu etait deja en base sous cette identite, l'API a neutralise
        # l'ecriture et rend la ligne existante. C'est dit a l'agent plutot que presente comme
        # une creation -- sinon il croit avoir ajoute un souvenir alors qu'il a reecrit le
        # meme, et peut boucler en reformulant pour « corriger » un echec qui n'existe pas.
        if res_data.get("status") == "duplicate":
            message = (f"[DEJA PRESENT] Ce contenu est deja en memoire (ID: "
                       f"{res_data.get('memory_id')}, section '{collection}'). Rien n'a ete "
                       f"ajoute : l'ecriture directe est idempotente sur le contenu. Pour "
                       f"corriger un souvenir, en ecrire un NOUVEAU qui enonce la version a "
                       f"jour -- une reformulation a l'identique ne cree rien.")
            return message
        message = (f"[SUCCESS] Memoire enregistree. ID: {res_data.get('memory_id')} | "
                   f"servie dans la section '{collection}'")
        if subtype and not res_data.get("canonical_subtype") and collection != subtype:
            message += (f".\n[INFO] '{subtype}' n'est pas une collection declaree : le "
                        f"souvenir est range dans '{collection}', la section par defaut de "
                        f"la famille '{memory_type}'. Pour qu'il ait sa propre section, "
                        f"appeler create_collection(name='{subtype}', "
                        f"family='{memory_type}', description=...).")
        return message
    except Exception as e:
        return _echec("Echec de l'enregistrement de la memoire", e)


@mcp.tool()
def list_collections() -> str:
    """
    Liste les collections de la memoire de cet agent : les rayons ou ranger un souvenir.

    A APPELER AVANT de creer une collection, et avant d'inventer un `subtype` dans
    store_memory. Une collection existante qui convient doit toujours etre reutilisee :
    une taxonomie qui se demultiplie a chaque nuance devient illisible, pour l'agent comme
    pour le modele qui lit le contexte.

    Deux origines :
      - `systeme` : livrees avec le moteur, communes a tous les agents.
      - `agent`   : creees par cet agent, pour lui seul.
    """
    url = f"{SYNAPTIQ_API_URL}/v1/collections"
    try:
        response = _lire(url, {"agent_id": require_agent_id()})
        response.raise_for_status()
        data = response.json()
        collections = data.get("collections", [])
        if not collections:
            return "Aucune collection (registre vide)."

        lignes = ["Collections de la memoire SynaptiQ :"]
        dormantes = []
        for col in collections:
            origine = "systeme" if col["created_by"] == "system" else "agent"
            graphe = "intriquee" if col["entangle"] else "hors graphe"
            marque = " | DORMANTE" if col.get("stale") else ""
            if col.get("stale"):
                dormantes.append(col["name"])
            lignes.append(
                f"- {col['name']} [{origine} | famille={col['family']} | {graphe} | "
                f"{col['memory_count']} souvenir(s){marque}] -> section '{col['packet_key']}'"
                + (f"\n    {col['description']}" if col.get("description") else "")
            )

        limites = data.get("limits") or {}
        if limites:
            lignes.append(f"\nQuota : {limites.get('used')} / "
                          f"{limites.get('max_collections')} collections creees.")
        if dormantes:
            # Une collection declaree puis jamais remplie est le premier symptome d'une
            # taxonomie qui se disperse. La signaler est ce qui permet de la corriger.
            lignes.append(
                f"[ATTENTION] Collections creees mais restees vides : {', '.join(dormantes)}. "
                f"Les reutiliser, ou les verser dans un autre rayon via merge_collections.")
        return "\n".join(lignes)
    except Exception as e:
        return _echec("Echec de la lecture des collections", e)


@mcp.tool()
def merge_collections(source: str, target: str) -> str:
    """
    Verse tous les souvenirs de la collection `source` dans `target`, puis supprime `source`.

    C'est l'outil d'entretien de la taxonomie. A utiliser des que deux rayons se recouvrent
    (`clients_paca` et `clients_region_paca`, par exemple) : une memoire eparpillee entre
    des rayons quasi identiques est une memoire ou plus rien ne se retrouve.

    Aucun souvenir n'est detruit -- seule leur etiquette change.

    Contraintes : `source` doit etre une collection creee par cet agent (les collections
    systeme servent tous les agents), et les deux doivent appartenir a la MEME famille --
    la famille decide de l'intrication et de la decroissance, la changer modifierait le
    traitement des souvenirs et pas seulement leur rangement.

    Args:
        source: Collection a vider puis supprimer.
        target: Collection qui recoit les souvenirs.
    """
    url = f"{SYNAPTIQ_API_URL}/v1/collections/merge"
    try:
        response = _poster(url, {"agent_id": require_agent_id(),
                                 "source": source, "target": target})
        if response.status_code in (404, 409, 422):
            return f"[REFUSE] {response.json().get('detail', response.text)}"
        response.raise_for_status()
        data = response.json()
        return (f"[SUCCESS] '{data.get('source')}' versee dans '{data.get('target')}' : "
                f"{data.get('moved_memories')} souvenir(s) deplace(s).")
    except Exception as e:
        return _echec("Echec de la fusion", e)


@mcp.tool()
def create_collection(name: str, family: str, description: str,
                      entangle: bool = True) -> str:
    """
    Cree un nouveau rayon dans la memoire de cet agent, pour y classer une categorie de
    souvenirs qui n'a pas encore sa place.

    A n'utiliser QUE si `list_collections` ne montre rien de convenable : reutiliser vaut
    toujours mieux que multiplier. Une collection est un engagement durable, pas une
    etiquette jetable.

    La FAMILLE n'est pas une categorie de rangement, c'est un COMPORTEMENT du moteur, et
    elle ne peut pas etre inventee. Choisir parmi :
      - 'semantic'   : savoirs stables (faits, preferences, profils, referentiels).
      - 'procedural' : regles, procedures, resolutions d'erreur, conventions.
      - 'episodic'   : evenements dates (comptes rendus, incidents, historique).
      - 'working'    : volatil, jetable apres usage.

    Args:
        name: Nom du rayon, en minuscules_avec_underscores (ex: 'clients_paca').
        family: 'semantic' | 'procedural' | 'episodic' | 'working'.
        description: A quoi sert ce rayon et ce qu'on y range. Sert a decider plus tard s'il faut le reutiliser -- etre concret.
        entangle: True pour un savoir structurant (le souvenir tisse des liens semantiques exploites par le rappel multi-saut). False pour du volumineux peu discriminant (brouillons, journaux bruts), afin de ne pas encombrer le graphe.
    """
    url = f"{SYNAPTIQ_API_URL}/v1/collections"
    try:
        payload = {
            "agent_id": require_agent_id(),
            "name": name,
            "family": family,
            "description": description,
            "entangle": entangle,
        }
        response = _poster(url, payload)
        if response.status_code in (409, 422):
            # Refus METIER (doublon, nom canonique, plafond) : le detail de l'API est
            # actionnable, il doit atteindre l'agent tel quel plutot que sous la forme
            # d'un « HTTP 409 » qu'il ne saura pas interpreter.
            detail = response.json().get("detail", response.text)
            return f"[REFUSE] {detail}"
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return (f"[SUCCESS] Collection '{data.get('name')}' creee "
                f"(famille {data.get('family')}, section '{data.get('packet_key')}'). "
                f"Pour y ecrire : store_memory(content=..., "
                f"memory_type='{usage.get('type')}', subtype='{usage.get('subtype')}').")
    except Exception as e:
        return _echec("Echec de la creation de la collection", e)

@mcp.tool()
def recall_memories(query: str, limit: int = 5, memory_type: str | None = None,
                    collections: list[str] | None = None) -> str:
    """
    Recherche sementiquement des souvenirs ou regles dans la memoire SynaptiQ pour adapter les reponses ou actions de l'agent.

    Args:
        query: Le sujet ou mot-cle a rechercher (ex: 'preferences style ecriture').
        limit: Nombre maximum de souvenirs a ramener (default: 5).
        memory_type: Filtrer par famille de memoire ('semantic', 'procedural', 'episodic', 'working').
        collections: Restreindre a ces collections (cf. list_collections). Cible un rayon precis plutot qu'une famille entiere : moins de candidats, donc moins de bruit.
    """
    url = f"{SYNAPTIQ_API_URL}/v1/retrieve"
    try:
        payload = {
            "agent_id": require_agent_id(),
            "query": query,
            "limit": limit,
            "memory_type": memory_type,
        }
        # Omis quand absent : une liste vide serait un filtre qui ne ramene rien, alors que
        # l'absence de filtre doit tout balayer.
        if collections:
            payload["collections"] = collections
        response = _poster(url, payload)
        response.raise_for_status()
        memories = response.json().get("memories", [])

        if not memories:
            return "Aucun souvenir correspondant trouve dans la base."

        output = ["Souvenirs retrouves dans SynaptiQ :"]
        for mem in memories:
            output.append(f"- [{mem['type'].upper()} / {mem['subtype'] or 'general'}] {mem['content']} (Confidence: {mem['confidence']})")
        return "\n".join(output)
    except Exception as e:
        return _echec("Echec de la recherche de souvenirs", e)


# Libelles lisibles des sept sections livrees avec le moteur. Les sections creees par
# l'agent n'y figurent pas : elles portent deja le nom qu'il leur a donne.
_LIBELLES_CANONIQUES = {
    "facts": "FAITS", "preferences": "PREFERENCES", "episodes": "EPISODES",
    "rules": "REGLES", "best_practices": "BONNES PRATIQUES",
    "errors": "ERREURS", "examples": "EXEMPLES",
}


@mcp.tool()
def build_context(task: str, query: str, max_tokens: int = 1200,
                  collections: list[str] | None = None) -> str:
    """
    Assemble un paquet de contexte compact (Q-EM) pret a injecter dans le prompt systeme
    de l'agent : faits, preferences, episodes, regles, bonnes pratiques, erreurs, plus une
    section par collection que cet agent s'est creee.

    Args:
        task: La tache en cours (ex: 'Rediger un email de suivi B2B').
        query: La requete de rappel semantique (ex: 'style d'ecriture, preferences client').
        max_tokens: Budget de tokens du contexte (default: 1200).
        collections: Restreindre le rappel a ces collections (cf. list_collections). Cible un rayon precis au lieu de ratisser toute la memoire : moins de bruit a budget de tokens egal. Omettre pour tout balayer.
    """
    url = f"{SYNAPTIQ_API_URL}/v1/context/build"
    try:
        contraintes: dict = {
            "max_tokens": max_tokens,
            "memory_types": ["semantic", "episodic", "procedural", "working"],
        }
        if collections:
            contraintes["collections"] = collections
        payload = {
            "agent_id": require_agent_id(),
            "session_id": "mcp-session",
            "task": task,
            "query": query,
            "constraints": contraintes,
        }
        response = _poster(url, payload)
        response.raise_for_status()
        data = response.json()
        packet = data.get("context_packet", {})
        lines = [f"Contexte SynaptiQ (~{data.get('token_estimate', 0)} tokens) :"]

        # Parcours du paquet REELLEMENT renvoye, et non d'une liste figee de sept cles.
        # Le paquet porte desormais une section par collection de l'agent : iterer sur une
        # table de libelles codee en dur les aurait toutes ecartees en silence -- le
        # rangement qu'il s'est donne aurait ete invisible dans son propre contexte, et
        # rien ne l'aurait signale. Les canoniques restent en tete, dans leur ordre.
        ordre = [c for c in _LIBELLES_CANONIQUES if c in packet]
        ordre += [c for c in packet if c not in _LIBELLES_CANONIQUES]
        for cle in ordre:
            libelle = _LIBELLES_CANONIQUES.get(cle, cle.upper())
            for item in packet.get(cle, []):
                lines.append(f"- [{libelle}] {item}")
        return "\n".join(lines) if len(lines) > 1 else "Aucun contexte pertinent trouve."
    except Exception as e:
        return _echec("Echec de la construction du contexte", e)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    # Journalise l'etat de la configuration SANS quitter : un serveur MCP qui refuse de
    # demarrer disparait de la liste du client, qui n'affiche qu'un « exit status 1 ».
    verifier_configuration()

    # Ici, en revanche, refuser de demarrer est le bon choix : exposer MCP sur le RESEAU
    # sans clé serait une ouverture, pas une gene de diagnostic.
    if transport != "stdio" and not SYNAPTIQ_API_KEY:
        raise RuntimeError("SYNAPTIQ_API_KEY est obligatoire pour exposer MCP en réseau.")

    # Demarrage de l'API a la demande. Ici et pas a l'import : un import ne doit jamais
    # lancer de serveur. Desactivable, car en conteneur l'API est un service a part.
    if os.getenv("SYNAPTIQ_AUTOSTART_API", "true").lower() in ("1", "true", "yes"):
        ensure_api_running()

    if transport == "stdio":
        # ⚠️ Limite CONNUE du transport stdio avec ce client.
        #
        # Apres fermeture de stdin, `mcp.run()` met 141 a 250 ms (mediane 157) a se
        # denouer — le temps est passe dans la boucle anyio de fastmcp, pas dans l'arret de
        # l'interpreteur : un `os._exit()` en sortie de `mcp.run()` n'y change donc RIEN
        # (verifie). Or antigravity CLI n'accorde qu'une fenetre de grace de l'ordre de
        # 100 ms avant d'appeler Kill() ; sur Windows, TerminateProcess(handle, 1) se lit
        # « exit status 1 », et son gestionnaire abandonne alors le rechargement de TOUS ses
        # serveurs. Les serveurs Node passent sous cette limite, Python non.
        #
        # Contournement recommande avec ce client : exposer ce serveur en HTTP/SSE et le
        # declarer avec `serverUrl` cote client. Il n'y a alors plus de processus enfant a
        # arreter, donc plus de fenetre de grace a respecter (cf. README, section MCP).
        mcp.run()
    else:
        # 0.0.0.0 est intentionnel : en conteneur, se lier a 127.0.0.1 rendrait le serveur
        # injoignable depuis l'hote. Le port n'est publie que sur 127.0.0.1 par Compose,
        # et ce transport exige SYNAPTIQ_API_KEY (verifie ci-dessus).
        host = os.getenv("MCP_HOST", "0.0.0.0")  # noqa: S104
        port = int(os.getenv("MCP_PORT", "8765"))
        logger.info(f"Démarrage du serveur MCP en transport '{transport}' sur {host}:{port}")
        mcp.run(transport=transport, host=host, port=port)
