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

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from fastapi.testclient import TestClient

from apps.api import main as api
from apps.api.main import app
from apps.worker import worker

# Règle de budget commune à TOUS les bras (elle s'appuie sur `estimate_tokens`, l'estimateur
# du collapse Q-EM) : sans elle, un écart d'exactitude pourrait n'être qu'un écart de budget.
from benchmarks.budget import fit_to_budget

# Intervalles de confiance : une exactitude sans incertitude n'est pas exploitable.
from synaptiq_core.entanglement import seuil_intrication
from synaptiq_core.stats import Difference, Proportion, required_sample_size

logging.getLogger("synaptiq-worker").setLevel(logging.WARNING)
logging.getLogger("synaptiq-core.embeddings").setLevel(logging.WARNING)
log = logging.getLogger("locomo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Defaut sur la base de DEVELOPPEMENT : ce harness ingere des milliers de memoires (1011 des
# 1025 memoires de la base de dev viennent d'ici). `synaptiq_db` est la base de production.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
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
           skip: int = 0, *, synaptiq: bool = True, mem0_arm=None) -> int:
    """Capture chaque tour de dialogue dans les moteurs demandés, dans le même ordre.

    Les deux moteurs reçoivent la MÊME chaîne `[date] locuteur: texte`, construite une seule
    fois : c'est ce qui rend la comparaison honnête. Une boucle par moteur aurait fini par
    diverger sur un détail de formatage, et l'écart se serait lu comme une différence de
    qualité de mémoire.

    `skip` saute les N premiers tours déjà ingérés (reprise après interruption). L'ordre
    de `_sessions` étant déterministe, reprendre au rang N redonne exactement la même
    séquence — le graphe d'intrication se construit donc à l'identique.
    """
    db = psycopg2.connect(DATABASE_URL) if synaptiq else None
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
                if synaptiq:
                    # L'event DOIT exister avant la mémoire (FK memories.source_event_id).
                    with db.cursor() as cur:
                        cur.execute(
                            "INSERT INTO events (tenant_id, agent_id, session_id, content) "
                            "VALUES (%s, %s, %s, %s) RETURNING id",
                            (tenant, agent, skey, content),
                        )
                        event_id = str(cur.fetchone()[0])
                        db.commit()
                    time.sleep(pace)  # rate-limit LLM (extraction dans process_event)
                    worker.process_event({
                        "id": event_id, "tenant_id": tenant, "agent_id": agent,
                        "session_id": skey, "content": content,
                        # Date de la session LOCOMO : référence pour résoudre « yesterday »,
                        # « last week »… en dates absolues. Sans elle, aucune mémoire n'est
                        # datée et les questions temporelles restent insolubles.
                        "created_at": date,
                    })
                if mem0_arm is not None:
                    # mem0 fait lui aussi un appel LLM par ajout : il lui faut son propre
                    # pacing, sinon les deux ingestions se cumulent sur la même fenêtre de
                    # rate limit et c'est l'extraction SynaptiQ qui se dégrade en premier.
                    time.sleep(pace)
                    mem0_arm.ingest_turn(content, session_id=skey, date=date)
                if count % 25 == 0:
                    log.info("Ingéré %d/%d tours…", count, skip + limit_turns if limit_turns else count)
                if limit_turns and count >= limit_turns:
                    return count
    finally:
        if db is not None:
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

    La troncature passe par `fit_to_budget`, partagée avec le bras mem0 et fondée sur
    l'estimateur EXACT qu'utilise Q-EM : sans cela, un bras disposerait de plus de contexte
    que l'autre et la comparaison d'exactitude ne voudrait rien dire.
    """
    r = client.post("/retrieve", json={"agent_id": agent, "query": question, "limit": top_k})
    if r.status_code != 200:
        return "", 0
    return fit_to_budget((m.get("content", "") for m in r.json().get("memories", [])), max_tokens)


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


ARMS_SYNAPTIQ = ("qem", "vector")


def _resoudre_bras(choix: str) -> list[str]:
    """Traduit `--arm` en liste de bras. `both` reste l'ancien couple (compat des scripts)."""
    if choix == "both":
        return ["qem", "vector"]
    if choix == "all":
        return ["qem", "vector", "mem0"]
    return [choix]


def run(args) -> dict:
    data = json.load(open(args.dataset, encoding="utf-8"))
    sample = data[args.conv]
    conv = sample["conversation"]
    tenant = args.tenant
    agent = f"conv{args.conv}"
    arms = _resoudre_bras(args.arm)
    besoin_synaptiq = any(a in ARMS_SYNAPTIQ for a in arms)
    besoin_mem0 = "mem0" in arms

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

    # Le bras mem0 est monté APRÈS le reset : ses tables sont supprimées puis recréées par
    # le SDK à l'instanciation. Import tardif — mem0 est une dépendance optionnelle, et un
    # run `--arm both` ne doit pas exiger son installation.
    mem0_arm = None
    if besoin_mem0:
        from benchmarks.mem0_arm import Mem0Arm, reset_collection
        if not args.resume:
            # Purge AVANT instanciation : le SDK ouvre ses tables au démarrage, les détruire
            # sous lui ensuite le laisserait avec un pool pointant sur des tables mortes.
            reset_collection(DATABASE_URL, args.mem0_collection)
        mem0_arm = Mem0Arm.from_env(user_id=agent, dsn=DATABASE_URL,
                                    collection_name=args.mem0_collection)
        etat = mem0_arm.stats()
        log.info("Bras mem0 prêt : %s", etat)
        if not etat["nlp"]["full_capacity"]:
            # Avertissement bruyant plutôt qu'abandon : mesurer un mem0 purement sémantique
            # reste légitime si c'est ASSUMÉ. Ce qui ne l'est pas, c'est de le publier
            # comme le score de mem0. Le rapport porte l'information dans tous les cas.
            log.warning(
                "mem0 tourne SANS spaCy/en_core_web_sm : BM25 et liaison d'entités sont "
                "hors service, deux des trois signaux de rappel de la v3. Le score obtenu "
                "sous-estimera mem0. Corriger avec : python -m spacy download en_core_web_sm"
            )

    t0 = time.time()
    with _FallbackCounter() as fallbacks:
        n_turns = ingest(conv, tenant, agent, args.pace, args.limit_turns, skip=already,
                         synaptiq=besoin_synaptiq, mem0_arm=mem0_arm)
    degraded = fallbacks.count
    traites = max(0, n_turns - already)   # tours ingérés par CE run
    degraded_ratio = round(degraded / traites, 4) if traites else 0.0
    log.info("Ingestion terminée : %d tours (%d par ce run) en %.0fs — %d dégradée(s), %.1f%%",
             n_turns, traites, time.time() - t0, degraded, 100 * degraded_ratio)
    if besoin_mem0:
        # Même exigence que pour les extractions dégradées de SynaptiQ : un corpus troué
        # d'un côté produit un score bas qui ne mesure pas le moteur mais la panne.
        mem0_ratio = round(mem0_arm.add_failures / traites, 4) if traites else 0.0
        if mem0_ratio > args.max_degraded:
            raise SystemExit(
                f"Run ABANDONNÉ : {mem0_arm.add_failures} ajouts mem0 sur {traites} "
                f"({100 * mem0_ratio:.1f}%) ont échoué, au-delà du seuil de "
                f"{100 * args.max_degraded:.0f}%. Le corpus mem0 serait incomplet et son "
                "score mesurerait les échecs d'ingestion, pas la qualité du rappel."
            )
    if besoin_synaptiq and degraded_ratio > args.max_degraded:
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
        elif arm == "mem0":
            ctx, ctx_tokens = mem0_arm.context(q, args.max_tokens, args.top_k)
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
            # Lu par la fonction du cœur : le seuil n'est plus une constante figée à
            # l'import du worker, il est partagé avec l'API qui intrique aussi désormais.
            "entangle_threshold": seuil_intrication(),
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
    if besoin_mem0:
        # Version, extras NLP et volume consolidé : sans eux, personne ne peut savoir si le
        # bras mem0 tournait à pleine capacité ni reproduire le run.
        report["mem0"] = mem0_arm.stats()
    report.update(_deltas(results, arms))
    return {"report": report, "results": results}


def _deltas(results: list[dict], arms: list[str]) -> dict:
    """Écarts entre bras, chacun avec son intervalle de confiance et son verdict.

    Le delta seul est trompeur : sur une conversation (~152 questions), la marge à 95 %
    vaut ~±8 points et le verdict est « non significatif ». C'est une information, pas un
    échec — elle dit combien de questions il faudrait pour conclure.

    ⚠️ `Difference` traite les deux bras comme des échantillons INDÉPENDANTS, alors qu'ils
    répondent aux mêmes questions. C'est conservateur (l'intervalle réel d'un test apparié
    est plus étroit), donc jamais à l'avantage d'un bras — mais un écart déclaré non
    significatif ici pourrait l'être avec le test apparié qui convient.

    Les clés historiques `delta_qem_minus_vector*` sont conservées : des scripts et le
    README les lisent.
    """
    sortie: dict = {}
    if len(arms) < 2:
        return sortie
    sortie["sample_size_needed_for_2pts_margin"] = required_sample_size(2.0)
    paires = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    comparaisons = {}
    for a, b in paires:
        difference = Difference(a=_proportion([r for r in results if r["arm"] == a]),
                                b=_proportion([r for r in results if r["arm"] == b]))
        comparaisons[f"{a}_minus_{b}"] = difference.as_dict()
        if (a, b) == ("qem", "vector"):
            sortie["delta_qem_minus_vector"] = round(difference.delta, 4)
            sortie["delta_qem_minus_vector_ci"] = difference.as_dict()
    sortie["comparisons"] = comparaisons
    return sortie


def _accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def _proportion(rows: list[dict]) -> Proportion:
    return Proportion(successes=sum(1 for r in rows if r["correct"]), total=len(rows))


def _arm_report(rows: list[dict]) -> dict:
    by_cat: dict = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    return {
        "accuracy_overall": round(_accuracy(rows), 4),
        # L'exactitude est TOUJOURS accompagnée de son intervalle de confiance : une
        # proportion sans incertitude ne permet à personne de décider quoi que ce soit,
        # et sur 152 questions la marge à 95 % vaut ~±8 points.
        "accuracy_overall_ci": _proportion(rows).as_dict(),
        "accuracy_by_category": {
            CATEGORY_NAMES.get(c, str(c)): round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())
        },
        "accuracy_by_category_ci": {
            CATEGORY_NAMES.get(c, str(c)): Proportion(sum(v), len(v)).as_dict()
            for c, v in sorted(by_cat.items())
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
    ap.add_argument("--arm", choices=["qem", "vector", "mem0", "both", "all"], default="both",
                    help="Bras évalué : Q-EM, baseline top-k vectorielle, mem0 (SDK open "
                         "source), 'both' = qem+vector (défaut historique), 'all' = les trois")
    ap.add_argument("--mem0-collection", default="mem0_bench_locomo",
                    help="Nom de la collection pgvector du bras mem0. DOIT commencer par "
                         "'mem0' : le reset supprime toutes les tables de ce préfixe.")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Candidats ramenés par les bras vectoriel ET mem0 avant troncature au budget")
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
