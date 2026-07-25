"""Passerelle OpenAI-compatible devant Antigravity CLI (`agy`).

Expose `POST /v1/chat/completions` en local et relaie chaque requête à `agy --print`,
qui s'authentifie avec le compte Google de la machine. SynaptiQ (worker, API, harness
LOCOMO) parle donc à Antigravity SANS UNE SEULE LIGNE DE CODE PRODUIT MODIFIÉE : il
suffit de pointer `LLM_BASE_URL` sur ce serveur.

Pourquoi une passerelle plutôt qu'un provider `agy` dans le worker :
  - le worker reste un pur client OpenAI-compatible (un seul chemin de code à tester) ;
  - `agy` est un binaire interactif, pas une API : l'isoler ici évite de faire entrer
    du lancement de sous-processus dans le chemin de consolidation ;
  - le benchmark reste rejouable avec n'importe quel endpoint OpenAI-compatible.

Usage :
    python benchmarks/agy_openai_shim.py --port 8899 --model gemini-3.6-flash-low
    # puis, dans .env :  LLM_BASE_URL=http://127.0.0.1:8899/v1

Limites assumées : pas de streaming, pas d'auth (écoute sur la boucle locale), et le
`model` du corps de requête est ignoré au profit de `--model`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agy-shim")

AGY = shutil.which("agy") or os.path.expandvars(r"%LOCALAPPDATA%\agy\bin\agy.exe")

# `agy` démarre un agent dans le répertoire courant : lancé depuis le dépôt, il en
# explorerait les fichiers (lenteur, et contexte parasite dans les réponses). On
# l'exécute donc dans un dossier vide.
_SANDBOX = tempfile.mkdtemp(prefix="agy-shim-")
# HOME dédié (voir build_isolated_home) ; vide = HOME réel de l'utilisateur.
_AGY_HOME: str = ""

_stats = {"calls": 0, "errors": 0, "total_seconds": 0.0}
_stats_lock = threading.Lock()


def _discover_models() -> set[str]:
    """Modèles exposés par `agy models` (permet de surcharger par requête)."""
    try:
        env = dict(os.environ)
        if _AGY_HOME:
            env["HOME"] = _AGY_HOME
            env["USERPROFILE"] = _AGY_HOME
        out = subprocess.run([AGY, "models"], capture_output=True, text=True,
                             timeout=90, cwd=_SANDBOX, env=env)
        return {line.strip() for line in (out.stdout or "").splitlines() if line.strip()}
    except Exception:
        return set()


KNOWN_MODELS: set[str] = set()


def _flatten(messages: list[dict]) -> str:
    """Aplati les messages en un prompt unique (`agy --print` ne prend pas de rôles)."""
    parts = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if isinstance(content, list):  # contenu multimodal : on ne garde que le texte
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(content if role == "user" else f"[{role.upper()}] {content}")
    return "\n\n".join(p for p in parts if p)


def _json_instruction(response_format: dict | None) -> str:
    """Traduit `response_format` en consigne texte : `agy` n'a pas de mode structuré natif."""
    if not response_format:
        return ""
    kind = response_format.get("type")
    if kind == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema", {})
        return ("\n\nReply with ONLY a single valid JSON object matching this schema, "
                "with no markdown fence and no commentary:\n" + json.dumps(schema))
    if kind == "json_object":
        return ("\n\nReply with ONLY a single valid JSON object, "
                "with no markdown fence and no commentary.")
    return ""


def _extract_json(text: str) -> str:
    """Isole l'objet JSON quand le modèle l'entoure de texte ou de balises markdown."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start:end + 1] if 0 <= start < end else stripped


def build_isolated_home(dest: str) -> str:
    """Prépare un HOME dédié à `agy` : authentification seule, sans contexte utilisateur.

    Deux raisons de ne pas réutiliser le HOME réel :
      - `~/.gemini/GEMINI.md` (instructions personnelles) est injecté dans CHAQUE session.
        S'il impose une langue ou un rôle, il contamine toutes les extractions et rend le
        benchmark non reproductible sur une autre machine.
      - `settings.json` déclare des serveurs MCP qu'`agy` démarre à chaque appel : ~2x de
        latence pour des outils dont l'extraction n'a aucun usage.
    On copie donc uniquement les fichiers d'authentification.
    """
    src = os.path.join(os.path.expanduser("~"), ".gemini")
    dst = os.path.join(dest, ".gemini")
    os.makedirs(dst, exist_ok=True)
    for name in ("oauth_creds.json", "google_accounts.json", "installation_id",
                 "projects.json", "state.json", "trustedFolders.json"):
        candidate = os.path.join(src, name)
        if os.path.exists(candidate):
            shutil.copy2(candidate, os.path.join(dst, name))
    # settings.json réduit à l'authentification : ni MCP, ni préférences d'affichage.
    auth = {}
    try:
        with open(os.path.join(src, "settings.json"), encoding="utf-8") as fh:
            auth = {"security": json.load(fh).get("security", {})}
    except Exception:
        auth = {"security": {"auth": {"selectedType": "oauth-personal"}}}
    with open(os.path.join(dst, "settings.json"), "w", encoding="utf-8") as fh:
        json.dump(auth, fh, indent=2)
    return dest


def call_agy(prompt: str, model: str, timeout: int) -> str:
    cmd = [AGY, "--dangerously-skip-permissions", "--model", model, "-p", prompt]
    env = dict(os.environ)
    if _AGY_HOME:
        env["HOME"] = _AGY_HOME
        env["USERPROFILE"] = _AGY_HOME
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, cwd=_SANDBOX, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"agy exit {proc.returncode}: {(proc.stderr or '')[:300]}")
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(f"agy n'a rien renvoyé. stderr: {(proc.stderr or '')[:300]}")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence le log HTTP par requête
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": self.server.model,
                                                         "object": "model"}]})
        elif self.path.rstrip("/").endswith("/stats"):
            with _stats_lock:
                done = _stats["calls"] or 1
                self._send(200, {**_stats, "avg_seconds": round(_stats["total_seconds"] / done, 2)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            req = json.loads(raw or b"{}")
        except Exception as exc:
            self._send(400, {"error": {"message": f"corps illisible: {exc}"}})
            return

        prompt = _flatten(req.get("messages", [])) + _json_instruction(req.get("response_format"))
        wants_json = bool(req.get("response_format"))
        # Le `model` du corps prime s'il correspond à un modèle `agy` connu : cela permet
        # de comparer plusieurs modèles sans redémarrer la passerelle. Sinon on garde
        # celui de --model (le client envoie souvent un identifiant d'un autre provider).
        requested = req.get("model") or ""
        model = requested if requested in KNOWN_MODELS else self.server.model

        started = time.perf_counter()
        try:
            text = call_agy(prompt, model, self.server.call_timeout)
            if wants_json:
                text = _extract_json(text)
        except Exception as exc:
            with _stats_lock:
                _stats["errors"] += 1
            log.warning("Échec agy : %s", exc)
            # 503 (et non 400) : le worker SynaptiQ ne réessaie que sur 429/5xx.
            self._send(503, {"error": {"message": str(exc), "type": "agy_error"}})
            return

        elapsed = time.perf_counter() - started
        with _stats_lock:
            _stats["calls"] += 1
            _stats["total_seconds"] += elapsed
            n, avg = _stats["calls"], _stats["total_seconds"] / _stats["calls"]
        if n % 25 == 0:
            log.info("%d appels relayés (moyenne %.1fs, %d erreurs)", n, avg, _stats["errors"])

        self._send(200, {
            "id": f"chatcmpl-agy-{n}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            # `agy` ne rapporte pas sa consommation : approximation ~4 caractères/token,
            # suffisante pour que les clients qui lisent `usage` ne cassent pas.
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": len(text) // 4,
                      "total_tokens": (len(prompt) + len(text)) // 4},
        })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--model", default="gemini-3.6-flash-low")
    ap.add_argument("--call-timeout", type=int, default=300)
    ap.add_argument("--use-real-home", action="store_true",
                    help="Utiliser le HOME réel (charge GEMINI.md et les serveurs MCP : "
                         "plus lent et non reproductible). Par défaut, HOME isolé.")
    args = ap.parse_args()

    if not os.path.exists(AGY):
        raise SystemExit(f"binaire `agy` introuvable ({AGY}). Installer Antigravity CLI.")

    if not args.use_real_home:
        _AGY_HOME = build_isolated_home(tempfile.mkdtemp(prefix="agy-home-"))
        log.info("HOME isolé pour agy : %s (sans GEMINI.md ni MCP)", _AGY_HOME)

    KNOWN_MODELS.update(_discover_models())
    log.info("Modèles agy détectés : %s", ", ".join(sorted(KNOWN_MODELS)) or "(aucun)")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.model = args.model
    server.call_timeout = args.call_timeout
    log.info("Passerelle prête : http://127.0.0.1:%d/v1  (modèle %s, sandbox %s)",
             args.port, args.model, _SANDBOX)
    log.info("À mettre dans .env :  LLM_BASE_URL=http://127.0.0.1:%d/v1", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Arrêt. %d appels, %d erreurs.", _stats["calls"], _stats["errors"])
