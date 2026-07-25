"""Harness benchmark LOCOMO pour SynaptiQ (mémoire long-terme conversationnelle).

Protocole (inspiré du harness Mem0, adapté au pipeline SynaptiQ) :
  1. INGESTION  : chaque tour de dialogue d'une conversation multi-sessions est capturé
     comme un `event` puis consolidé par le VRAI pipeline worker (extraction LLM →
     mémoire typée + intrication auto). Tenant/agent dédiés (aucune pollution).
  2. QA         : pour chaque question, on reconstruit un contexte via `/context/build`
     (moteur Q-EM), on génère une réponse avec un LLM, puis un LLM-juge décide si elle
     correspond à la réponse attendue (« J-score », protocole dominant sur LOCOMO).
  3. SCORING    : exactitude globale et PAR CATÉGORIE (1 multi-hop, 2 temporal,
     3 open-domain, 4 single-hop ; 5 adversarial exclue par convention).

Tout passe par OpenRouter (LLM) + LM Studio (embeddings, local) via le `.env` racine.
Pacing + retry pour rester sous la limite du tier gratuit (~20 req/min, 1000/jour).

Usage :
  python benchmarks/locomo_runner.py DATASET.json --conv 0 --limit-qa 120 \
      --pace 4.0 --out benchmarks/results_locomo_conv0.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import requests

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_ROOT, ".env"))

from apps.worker import worker  # noqa: E402
from apps.api import main as api  # noqa: E402
from apps.api.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
# Même estimateur de tokens que le collapse Q-EM : indispensable pour que les deux bras
# soient comparés à budget de contexte réellement identique.
from synaptiq_core.qem import estimate_tokens  # noqa: E402

logging.getLogger("synaptiq-worker").setLevel(logging.WARNING)
logging.getLogger("synaptiq-core.embeddings").setLevel(logging.WARNING)
log = logging.getLogger("locomo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Un modèle par rôle. Le RÉPONDEUR doit être identique sur les deux bras (sinon on mesure
# le modèle, pas le moteur de mémoire) ; le JUGE est délibérément différent du répondeur
# pour écarter l'auto-préférence.
MODEL_QA = os.getenv("LOCOMO_MODEL_QA", LLM_MODEL)
MODEL_JUDGE = os.getenv("LOCOMO_MODEL_JUDGE", LLM_MODEL)

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def _sessions(conv: dict) -> list[tuple[str, str, list]]:
    """Retourne [(clé_session, date, tours), ...] triées par index de session."""
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    out = []
    for k in keys:
        date = conv.get(f"{k}_date_time", "")
        out.append((k, date, conv[k]))
    return out


def _llm_chat(messages: list[dict], pace: float, model: str | None = None,
              max_retries: int = 5) -> str:
    """Appel chat OpenAI-compatible avec pacing + retry sur 429/5xx."""
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY and "your_api_key" not in LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {"model": model or LLM_MODEL, "messages": messages, "temperature": 0}
    for attempt in range(max_retries):
        time.sleep(pace)
        resp = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            ra = resp.headers.get("Retry-After")
            delay = float(ra) if (ra or "").isdigit() else 3.0 * (2 ** attempt)
            log.warning("LLM %s (essai %d/%d) → attente %.1fs", resp.status_code, attempt + 1, max_retries, delay)
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    resp.raise_for_status()
    return ""


class _FallbackCounter:
    """Compte les extractions retombées sur les heuristiques regex.

    Le worker rattrape toute erreur LLM (429, timeout…) en repliant sur les regex : la
    consolidation continue, mais la classification se dégrade en `episodic/interaction`,
    donc hors du graphe d'intrication. Un run qui subit ça mesure un Q-EM handicapé SANS
    QUE RIEN NE LE SIGNALE dans le rapport. On instrumente donc le repli pour pouvoir
    invalider le run.
    """

    def __init__(self):
        self.count = 0
        self._original = worker._heuristic_extract

    def __enter__(self):
        def instrumente(event_content: str) -> dict:
            self.count += 1
            return self._original(event_content)
        worker._heuristic_extract = instrumente
        return self

    def __exit__(self, *exc):
        worker._heuristic_extract = self._original
        return False


def ingest(conv: dict, tenant: str, agent: str, pace: float, limit_turns: int | None,
           skip: int = 0) -> int:
    """Capture chaque tour comme event puis consolide via le worker (extraction LLM).

    `skip` saute les N premiers tours déjà ingérés (reprise après interruption). L'ordre
    de `_sessions` étant déterministe, reprendre au rang N redonne exactement la même
    séquence — le graphe d'intrication se construit donc à l'identique.
    """
    db = psycopg2.connect(DATABASE_URL)
    count = 0
    try:
        for skey, date, turns in _sessions(conv):
            for turn in turns:
                speaker = turn.get("speaker", "?")
                text = turn.get("text", "")
                if not text:
                    continue
                count += 1
                if count <= skip:
                    continue          # déjà ingéré lors d'un run précédent
                content = f"[{date}] {speaker}: {text}"
                # L'event DOIT exister avant la mémoire (FK memories.source_event_id).
                with db.cursor() as cur:
                    cur.execute(
                        "INSERT INTO events (tenant_id, agent_id, session_id, content) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (tenant, agent, skey, content),
                    )
                    event_id = str(cur.fetchone()[0])
                    db.commit()
                time.sleep(pace)  # respecter le rate-limit LLM (extraction dans process_event)
                worker.process_event({
                    "id": event_id, "tenant_id": tenant, "agent_id": agent,
                    "session_id": skey, "content": content,
                    # Date de la session LOCOMO : référence pour résoudre « yesterday »,
                    # « last week »… en dates absolues. Sans elle, aucune mémoire n'est
                    # datée et les questions temporelles restent insolubles.
                    "created_at": date,
                })
                if count % 25 == 0:
                    log.info("Ingéré %d/%d tours…", count, skip + limit_turns if limit_turns else count)
                if limit_turns and count >= limit_turns:
                    return count
    finally:
        db.close()
    return count


def context_qem(client: TestClient, agent: str, question: str, max_tokens: int) -> tuple[str, int]:
    """BRAS Q-EM — moteur complet : intrication multi-hop, interférence, collapse.

    Retourne (contexte aplati, tokens estimés) — l'API rend elle-même son décompte.
    """
    r = client.post("/context/build", json={
        "agent_id": agent, "session_id": "bench",
        "task": question, "query": question,
        "constraints": {"max_tokens": max_tokens,
                        "memory_types": ["semantic", "episodic", "procedural", "working"]},
    })
    if r.status_code != 200:
        return "", 0
    body = r.json()
    lines = [f"- {it}" for items in body.get("context_packet", {}).values() for it in items]
    return "\n".join(lines), body.get("token_estimate", 0)


def context_vector(client: TestClient, agent: str, question: str, max_tokens: int,
                   top_k: int) -> tuple[str, int]:
    """BRAS BASELINE — top-k vectoriel pur (le RAG de référence), à budget de tokens ÉGAL.

    Même embedder, même base, même question. Seule différence : aucune intrication, aucun
    filtrage d'interférence, aucun collapse par densité d'utilité — on prend les k plus
    proches par cosinus et on remplit le budget dans cet ordre.

    Le budget est tronqué avec `estimate_tokens`, l'estimateur EXACT qu'utilise Q-EM :
    sans cela, un bras disposerait de plus de contexte que l'autre et la comparaison
    d'exactitude ne voudrait rien dire.
    """
    r = client.post("/retrieve", json={"agent_id": agent, "query": question, "limit": top_k})
    if r.status_code != 200:
        return "", 0
    lines, total = [], 0
    for mem in r.json().get("memories", []):
        content = mem.get("content", "")
        cost = estimate_tokens(content)
        if total + cost > max_tokens:
            continue  # même règle que collapse_by_utility : on saute, on ne s'arrête pas
        lines.append(f"- {content}")
        total += cost
    return "\n".join(lines), total


def answer_question(context: str, question: str, pace: float) -> str:
    return _llm_chat([
        {"role": "system", "content": "Tu réponds à des questions sur une conversation à partir "
         "de la MÉMOIRE fournie. Réponse TRÈS concise (quelques mots). Si l'information n'est pas "
         "présente, réponds exactement 'Non mentionné'."},
        {"role": "user", "content": f"MÉMOIRE :\n{context}\n\nQUESTION : {question}\nRÉPONSE :"},
    ], pace, model=MODEL_QA)


def judge(question: str, gold: str, hyp: str, pace: float) -> bool:
    verdict = _llm_chat([
        {"role": "system", "content": "Tu es un évaluateur strict mais tolérant aux reformulations. "
         "Réponds uniquement 'oui' ou 'non'."},
        {"role": "user", "content": f"Question : {question}\nRéponse attendue : {gold}\n"
         f"Réponse du système : {hyp}\n\nLa réponse du système est-elle correcte "
         "(même sens que l'attendue, reformulation tolérée) ?"},
    ], pace, model=MODEL_JUDGE)
    # Les modèles de raisonnement préfixent parfois leur verdict : on retient le premier
    # marqueur rencontré DANS LE TEXTE, en mot entier (`\b` évite que « Louis » compte
    # comme « oui », ou « nous » comme « no »). Sans verdict lisible -> incorrect.
    match = re.search(r"\b(oui|yes|non|no)\b", verdict.strip().lower())
    return bool(match) and match.group(1) in ("oui", "yes")


def run(args) -> dict:
    data = json.load(open(args.dataset, encoding="utf-8"))
    sample = data[args.conv]
    conv = sample["conversation"]
    tenant = args.tenant
    agent = f"conv{args.conv}"

    # SynaptiQ décide du tenant CÔTÉ SERVEUR (jamais depuis le body) : l'API le relit dans
    # SYNAPTIQ_TENANT à chaque requête. Sans cette ligne, l'ingestion écrit sous
    # `bench_locomo` pendant que /context/build et /retrieve interrogent `default`,
    # et les deux bras ramènent un contexte vide.
    os.environ["SYNAPTIQ_TENANT"] = tenant

    # Reset du périmètre benchmark (jamais les données d'autres tenants), sauf reprise.
    db = psycopg2.connect(DATABASE_URL)
    already = 0
    with db.cursor() as cur:
        if args.resume:
            cur.execute("SELECT count(*) FROM events WHERE tenant_id = %s", (tenant,))
            already = cur.fetchone()[0]
            log.info("Reprise : %d tours déjà ingérés, on continue à partir de là.", already)
        else:
            cur.execute("DELETE FROM memories WHERE tenant_id = %s", (tenant,))
            cur.execute("DELETE FROM events WHERE tenant_id = %s", (tenant,))
            db.commit()
    db.close()

    t0 = time.time()
    with _FallbackCounter() as fallbacks:
        n_turns = ingest(conv, tenant, agent, args.pace, args.limit_turns, skip=already)
    degraded = fallbacks.count
    traites = max(0, n_turns - already)   # tours ingérés par CE run
    degraded_ratio = round(degraded / traites, 4) if traites else 0.0
    log.info("Ingestion terminée : %d tours (%d par ce run) en %.0fs — %d dégradée(s), %.1f%%",
             n_turns, traites, time.time() - t0, degraded, 100 * degraded_ratio)
    if degraded_ratio > args.max_degraded:
        raise SystemExit(
            f"Run ABANDONNÉ : {degraded} extractions sur {traites} ({100 * degraded_ratio:.1f}%) "
            f"sont retombées sur les heuristiques regex, au-delà du seuil de "
            f"{100 * args.max_degraded:.0f}%. Les mémoires seraient majoritairement classées "
            "'episodic' et exclues du graphe d'intrication : le score mesurerait un Q-EM "
            "handicapé, pas le moteur. Cause probable : saturation du rate limit LLM "
            "(augmenter --pace, LLM_MAX_RETRIES, ou changer de modèle)."
        )

    qas = [q for q in sample["qa"] if q.get("category") != 5 and q.get("answer") is not None]
    if args.limit_qa:
        qas = qas[:args.limit_qa]

    arms = ["qem", "vector"] if args.arm == "both" else [args.arm]
    results: list[dict] = []
    done = 0
    done_lock = threading.Lock()

    def evaluate(qa: dict, arm: str) -> dict:
        """Évalue UNE question sur UN bras. Indépendant des autres : parallélisable.

        L'ingestion, elle, reste séquentielle : le worker tisse le graphe d'intrication au
        fil des insertions, donc l'ordre d'arrivée influe sur les arêtes créées.
        """
        q, gold, cat = qa["question"], str(qa["answer"]), qa.get("category")
        if arm == "qem":
            ctx, ctx_tokens = context_qem(client, agent, q, args.max_tokens)
        else:
            ctx, ctx_tokens = context_vector(client, agent, q, args.max_tokens, args.top_k)
        hyp = answer_question(ctx, q, args.pace)
        ok = judge(q, gold, hyp, args.pace)
        nonlocal done
        with done_lock:
            done += 1
            if done % 10 == 0:
                log.info("QA %d/%d évaluées", done, len(qas) * len(arms))
        return {"arm": arm, "category": cat, "question": q, "gold": gold,
                "hyp": hyp, "correct": ok, "context_tokens": ctx_tokens}

    with TestClient(app) as client:
        jobs = [(qa, arm) for qa in qas for arm in arms]
        if args.qa_workers > 1:
            with ThreadPoolExecutor(max_workers=args.qa_workers) as pool:
                futures = [pool.submit(evaluate, qa, arm) for qa, arm in jobs]
                for fut in as_completed(futures):
                    results.append(fut.result())
        else:
            results = [evaluate(qa, arm) for qa, arm in jobs]

    for arm in arms:
        log.info("Bras %s : exactitude %.1f%%",
                 arm, 100 * _accuracy([r for r in results if r["arm"] == arm]))

    report = {
        "dataset": os.path.basename(args.dataset),
        "conv_index": args.conv,
        "sample_id": sample.get("sample_id"),
        "model_extraction": LLM_MODEL,
        "model_qa": MODEL_QA,
        "model_judge": MODEL_JUDGE,
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        # Seuils EFFECTIFS lus dans les modules (pas os.getenv, qui rendrait "" quand la
        # valeur vient du défaut codé) : un score n'est reproductible qu'avec eux.
        "qem_settings": {
            "entangle_threshold": worker.QEM_ENTANGLE_THRESHOLD,
            "entangle_types": sorted(worker.QEM_ENTANGLE_TYPES),
            "entangle_damping": api.QEM_ENTANGLE_DAMPING,
            "entangle_max_hops": api.QEM_ENTANGLE_MAX_HOPS,
            "redundancy_threshold": api.QEM_REDUNDANCY_THRESHOLD,
            "recency_halflife_days": api.QEM_RECENCY_HALFLIFE_DAYS,
        },
        "max_tokens": args.max_tokens,
        "vector_top_k": args.top_k,
        "turns_ingested": n_turns,
        # Part de l'ingestion classée par regex au lieu du LLM : au-delà de quelques
        # pourcents, le corpus n'est plus représentatif et le score n'est pas publiable.
        "degraded_extractions": degraded,
        "degraded_ratio": degraded_ratio,
        "memories_consolidated": _count(tenant, "memories"),
        "relationships_created": _count_relationships(tenant),
        "qa_evaluated": len(qas),
        "arms": {arm: _arm_report([r for r in results if r["arm"] == arm]) for arm in arms},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if len(arms) == 2:
        qem_acc = report["arms"]["qem"]["accuracy_overall"]
        vec_acc = report["arms"]["vector"]["accuracy_overall"]
        report["delta_qem_minus_vector"] = round(qem_acc - vec_acc, 4)
    return {"report": report, "results": results}


def _accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def _arm_report(rows: list[dict]) -> dict:
    by_cat: dict = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    return {
        "accuracy_overall": round(_accuracy(rows), 4),
        "accuracy_by_category": {
            CATEGORY_NAMES.get(c, str(c)): round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())
        },
        # À exactitude comparable, le contexte le plus court gagne : c'est l'argument coût.
        "avg_context_tokens": round(sum(r["context_tokens"] for r in rows) / len(rows), 1) if rows else 0.0,
    }


def _count(tenant: str, table: str) -> int:
    db = psycopg2.connect(DATABASE_URL)
    try:
        with db.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table} WHERE tenant_id = %s", (tenant,))  # noqa: S608
            return cur.fetchone()[0]
    finally:
        db.close()


def _count_relationships(tenant: str) -> int:
    """Nombre d'arêtes d'intrication du périmètre : à 0, Q-EM ne peut PAS battre le top-k."""
    db = psycopg2.connect(DATABASE_URL)
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM relationships r JOIN memories m "
                "ON m.id = r.source_memory_id WHERE m.tenant_id = %s", (tenant,))
            return cur.fetchone()[0]
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Chemin vers locomo10.json")
    ap.add_argument("--conv", type=int, default=0, help="Index de la conversation (0-9)")
    ap.add_argument("--limit-turns", type=int, default=None, help="Cap de tours ingérés (smoke test)")
    ap.add_argument("--limit-qa", type=int, default=None, help="Cap de questions évaluées")
    ap.add_argument("--arm", choices=["qem", "vector", "both"], default="both",
                    help="Bras évalué : moteur Q-EM, baseline top-k vectorielle, ou les deux")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Candidats ramenés par la baseline vectorielle avant troncature au budget")
    # Groq plafonne à 8000 tokens/minute sur les modèles d'extraction : ~1 appel toutes
    # les 4-5s pour rester sous la limite sur un run soutenu.
    ap.add_argument("--pace", type=float, default=1.0, help="Délai (s) avant chaque appel LLM (rate-limit)")
    ap.add_argument("--resume", action="store_true",
                    help="Reprendre une ingestion interrompue au lieu de repartir de zéro")
    ap.add_argument("--qa-workers", type=int, default=4,
                    help="Questions évaluées en parallèle (l'ingestion reste séquentielle)")
    ap.add_argument("--max-degraded", type=float, default=0.05,
                    help="Part maximale d'extractions repliées sur les regex avant abandon du run")
    ap.add_argument("--max-tokens", type=int, default=1500, help="Budget tokens du context_packet")
    ap.add_argument("--tenant", default="bench_locomo")
    ap.add_argument("--out", default=None, help="Fichier JSON de sortie")
    args = ap.parse_args()

    payload = run(args)
    print(json.dumps(payload["report"], ensure_ascii=False, indent=2))
    if args.out:
        json.dump(payload, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log.info("Rapport écrit dans %s", args.out)
