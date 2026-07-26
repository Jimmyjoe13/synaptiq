"""Benchmark comparatif : SynaptiQ (Q-EM) vs Obsidian MCP (Markdown / Recherche vectorielle classique).

Ce script compare l'efficacité de SynaptiQ face à une approche Obsidian MCP classique :
  - Bras Obsidian MCP : Recherche vectorielle brute sur des notes Markdown statiques.
  - Bras SynaptiQ (Q-EM) : Ingestion avec classification, intrication auto, filtrage des
    contradictions et collapse glouton sous budget de tokens (/v1/context/build).

Évaluation par un LLM-juge sur l'exactitude des réponses produites.
"""
import os
import sys
import json
import time
import requests
import hashlib
import psycopg2
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from apps.worker.worker import process_event
from synaptiq_core import get_embedder

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8899/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-120b-medium")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db")

def _call_llm(prompt: str, system: str = "Tu es un assistant IA précis.") -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    r = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _judge_answer(question: str, gold: str, hyp: str) -> bool:
    prompt = (
        f"Question : {question}\n"
        f"Réponse exacte attendue (Gold) : {gold}\n"
        f"Réponse fournie par le modèle (Hypothèse) : {hyp}\n\n"
        "La réponse fournie est-elle FACTUELLEMENT CORRECTE et conforme à la réponse attendue ?\n"
        "Réponds uniquement par 'OUI' ou 'NON'."
    )
    res = _call_llm(prompt, system="Tu es un juge strict d'exactitude factuelle.")
    return "OUI" in res.upper()


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(y * y for y in v2) ** 0.5
    return dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0


def run_benchmark(dataset_path: str) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    embedder = get_embedder()
    tenant = "bench_obsidian_vs_synaptiq"
    agent_id = "agent_bench"

    # Purger les données précédentes du tenant & insérer une clé API de benchmark valide
    raw_key = "test-key-obsidian-vs-synaptiq"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE tenant_id = %s", (tenant,))
        cur.execute("DELETE FROM events WHERE tenant_id = %s", (tenant,))
        cur.execute("DELETE FROM api_keys WHERE tenant_id = %s", (tenant,))
        cur.execute(
            "INSERT INTO api_keys (tenant_id, name, key_hash, active) VALUES (%s, %s, %s, true)",
            (tenant, "bench_key", key_hash)
        )
        conn.commit()

    print("[BENCHMARK START] SynaptiQ Q-EM vs Obsidian MCP")
    
    obsidian_results = []
    synaptiq_results = []

    for scenario in ds["scenarios"]:
        print(f"\nScenario: {scenario['title']} ({scenario['category']})")
        events = scenario["events"]
        question = scenario["question"]
        gold = scenario["gold_answer"]

        # --- Ingestion pour Obsidian MCP (Notes Markdown brutes + Vector Search) ---
        markdown_notes = []
        for ev in events:
            emb = embedder.embed_one(ev["content"])
            markdown_notes.append({
                "id": ev["id"],
                "content": ev["content"],
                "embedding": emb,
                "timestamp": ev["timestamp"]
            })

        # Simuler Obsidian MCP Vector Search (Top-2 cosinus brut)
        q_emb = embedder.embed_one(question)
        markdown_notes.sort(key=lambda n: _cosine_similarity(q_emb, n["embedding"]), reverse=True)
        top_obsidian_notes = markdown_notes[:2]
        obsidian_context = "\n".join([f"- Note Obsidian : {n['content']}" for n in top_obsidian_notes])
        
        obsidian_prompt = (
            f"Contexte récupéré depuis le vault Obsidian :\n{obsidian_context}\n\n"
            f"Question : {question}\n\n"
            "Réponds de manière concise et factuelle d'après le contexte ci-dessus."
        )
        obsidian_answer = _call_llm(obsidian_prompt)
        obsidian_correct = _judge_answer(question, gold, obsidian_answer)
        obsidian_results.append({
            "scenario_id": scenario["id"],
            "category": scenario["category"],
            "correct": obsidian_correct,
            "answer": obsidian_answer
        })

        # --- Ingestion dans SynaptiQ Q-EM ---
        with conn.cursor() as cur:
            for ev in events:
                cur.execute(
                    "INSERT INTO events (id, tenant_id, agent_id, session_id, content) VALUES (%s, %s, %s, %s, %s)",
                    (ev["id"], tenant, agent_id, "sess_1", ev["content"])
                )
            conn.commit()

        for ev in events:
            event_payload = {
                "id": ev["id"],
                "tenant_id": tenant,
                "agent_id": agent_id,
                "session_id": "sess_1",
                "content": ev["content"],
                "metadata": "{}",
                "created_at": ev["timestamp"]
            }
            process_event(event_payload)

        # Appel de SynaptiQ build_context (API FastAPI locale avec clé API valide)
        syn_resp = requests.post(
            "http://127.0.0.1:8000/v1/context/build",
            json={
                "agent_id": agent_id,
                "session_id": "sess_1",
                "task": "Benchmark QA",
                "query": question,
                "constraints": {"max_tokens": 800}
            },
            headers={"Authorization": f"Bearer {raw_key}"},
            timeout=10
        )
        syn_resp.raise_for_status()
        ctx_data = syn_resp.json()["context_packet"]
        
        synaptiq_context_items = []
        for k, items in ctx_data.items():
            for item in items:
                synaptiq_context_items.append(f"- [{k}] {item}")
        synaptiq_context = "\n".join(synaptiq_context_items)

        synaptiq_prompt = (
            f"Contexte structuré SynaptiQ Q-EM :\n{synaptiq_context}\n\n"
            f"Question : {question}\n\n"
            "Réponds de manière concise et factuelle d'après le contexte ci-dessus."
        )
        synaptiq_answer = _call_llm(synaptiq_prompt)
        synaptiq_correct = _judge_answer(question, gold, synaptiq_answer)
        synaptiq_results.append({
            "scenario_id": scenario["id"],
            "category": scenario["category"],
            "correct": synaptiq_correct,
            "answer": synaptiq_answer
        })

        print(f"  Obsidian MCP: {'[OK]' if obsidian_correct else '[ECHEC]'} -> {obsidian_answer}")
        print(f"  SynaptiQ Q-EM: {'[OK]' if synaptiq_correct else '[ECHEC]'} -> {synaptiq_answer}")

    conn.close()

    # Calcul des scores finaux
    obsidian_acc = sum(r["correct"] for r in obsidian_results) / len(obsidian_results)
    synaptiq_acc = sum(r["correct"] for r in synaptiq_results) / len(synaptiq_results)

    report = {
        "obsidian_mcp_accuracy": round(obsidian_acc * 100, 2),
        "synaptiq_qem_accuracy": round(synaptiq_acc * 100, 2),
        "delta_points": round((synaptiq_acc - obsidian_acc) * 100, 2),
        "obsidian_details": obsidian_results,
        "synaptiq_details": synaptiq_results
    }
    return report

if __name__ == "__main__":
    dataset = os.path.join(_ROOT, "benchmarks", "obsidian_vs_synaptiq_dataset.json")
    res = run_benchmark(dataset)
    print("\n" + "="*50)
    print("RESULTATS DU BENCHMARK SYNAPTIQ Q-EM vs OBSIDIAN MCP")
    print(f"Obsidian MCP Exactitude : {res['obsidian_mcp_accuracy']} %")
    print(f"SynaptiQ Q-EM Exactitude : {res['synaptiq_qem_accuracy']} %")
    print(f"Avantage SynaptiQ       : +{res['delta_points']} points")
    print("="*50)
    
    out_file = os.path.join(_ROOT, "benchmarks", "results_obsidian_vs_synaptiq.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"Rapport enregistre dans {out_file}")
