"""Le sommeil de SynaptiQ : consolidation nocturne, agent par agent (lot 2, 23/11/2026).

Lancé chaque nuit par cron, APRÈS la sauvegarde :
    docker compose run --rm --no-deps worker python apps/worker/sommeil.py --tous

Phases (cf. `synaptiq_core.sommeil`) : doublons, consolidation, oubli, graphe, plasticité.
Règle absolue (décision de Jimmy du 23/11) : **AUCUNE suppression**. Ce qui s'efface est
rangé dans le compartiment `archives`, avec sa provenance (`archived_from`,
`archived_reason`, `archived_at`) pour pouvoir le restaurer.

Chaque phase s'exécute dans sa propre transaction : une phase en échec n'annule pas les
précédentes, et le rapport (`audit_log`, action `sommeil`) dit ce qui a été fait.
`--dry-run` compte sans rien écrire ni appeler de LLM.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime

_ici = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_ici, "..", ".."))
for _p in (_root, os.path.join(_root, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import psycopg2  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

load_dotenv(os.path.join(_root, ".env"))

from synaptiq_core import content_hash, entangle, get_embedder, to_pgvector  # noqa: E402
from synaptiq_core.compartments import (  # noqa: E402
    COMPARTIMENT_ARCHIVES,
    COMPARTIMENT_CROYANCES,
    COMPARTIMENTS,
    FAMILLE_DORMANTE,
    PAR_NOM,
)
from synaptiq_core.llm import LLMIndisponible, get_client_llm  # noqa: E402
from synaptiq_core.observability import configure_logging  # noqa: E402
from synaptiq_core.plasticite import POIDS_LIEN_NOUVEAU  # noqa: E402
from synaptiq_core.sommeil import (  # noqa: E402
    JAMAIS_OUBLIES,
    ReglagesSommeil,
    doit_oublier,
    grouper,
    paire_doublon,
    prompt_consolidation,
    valider_consolidation,
)

configure_logging("synaptiq-sommeil")
logger = logging.getLogger("synaptiq-sommeil")

DATABASE_URL = os.getenv("DATABASE_URL",
                         "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_dev")
PHASES = ("doublons", "consolidation", "oubli", "graphe", "plasticite")


def reglages_depuis_env() -> ReglagesSommeil:
    e = os.getenv
    return ReglagesSommeil(
        doublon_seuil=float(e("SOMMEIL_DOUBLON_SEUIL", "0.95")),
        oubli_seuil=float(e("SOMMEIL_OUBLI_SEUIL", "0.05")),
        oubli_age_min_jours=float(e("SOMMEIL_OUBLI_AGE_MIN_JOURS", "30")),
        consolidation_age_min_jours=float(e("SOMMEIL_CONSOLIDATION_AGE_MIN_JOURS", "3")),
        consolidation_groupe_seuil=float(e("SOMMEIL_CONSOLIDATION_SEUIL", "0.75")),
        consolidation_groupes_max=int(e("SOMMEIL_CONSOLIDATION_GROUPES_MAX", "20")),
        graphe_max=int(e("SOMMEIL_GRAPHE_MAX", "500")),
        plasticite_facteur=float(e("SOMMEIL_PLASTICITE_FACTEUR", "0.05")),
        plasticite_repos_jours=float(e("SOMMEIL_PLASTICITE_REPOS_JOURS", "7")),
    )


def _vecteur(texte):
    if texte is None:
        return []
    return [float(x) for x in str(texte).strip("[]").split(",") if x]


def _archiver(cur, mem_id: str, origine: str, motif: str, **extra) -> None:
    """Range un souvenir dans les archives. Jamais de DELETE : c'est tout l'objet."""
    trace = {"archived_from": origine, "archived_reason": motif,
             "archived_at": datetime.now(UTC).isoformat(), **extra}
    cur.execute(
        "UPDATE memories SET compartment = %s, type = %s, updated_at = CURRENT_TIMESTAMP, "
        "provenance = COALESCE(provenance, '{}'::jsonb) || %s::jsonb WHERE id = %s",
        (COMPARTIMENT_ARCHIVES, FAMILLE_DORMANTE, json.dumps(trace), mem_id))


# ─── Phases ──────────────────────────────────────────────────────────────────

def phase_doublons(cur, tenant: str, agent: str, r: ReglagesSommeil, dry: bool) -> int:
    """Quasi-identiques d'un même compartiment et d'un même projet : le moins bon s'archive."""
    cur.execute("""
        SELECT m.id::text AS a, n.id::text AS b
        FROM memories m
        CROSS JOIN LATERAL (
            SELECT x.id, 1 - (x.embedding <=> m.embedding) AS sim
            FROM memories x
            WHERE x.tenant_id = m.tenant_id AND x.agent_id = m.agent_id AND x.id <> m.id
              AND x.status = 'active' AND x.compartment = m.compartment
              AND x.project IS NOT DISTINCT FROM m.project
            ORDER BY x.embedding <=> m.embedding
            LIMIT 1
        ) n
        WHERE m.tenant_id = %s AND m.agent_id = %s AND m.status = 'active'
          AND m.compartment <> ALL(%s) AND n.sim >= %s
    """, (tenant, agent, [COMPARTIMENT_CROYANCES, COMPARTIMENT_ARCHIVES], r.doublon_seuil))
    paires = {tuple(sorted((li["a"], li["b"]))) for li in cur.fetchall()}
    if not paires:
        return 0
    ids = sorted({i for p in paires for i in p})
    cur.execute("SELECT id::text AS id, compartment, reward, access_count, importance, "
                "created_at FROM memories WHERE id = ANY(%s::uuid[])", (ids,))
    fiches = {li["id"]: li for li in cur.fetchall()}
    archives: set[str] = set()
    for a, b in sorted(paires):
        if a in archives or b in archives:
            continue      # déjà rangé cette nuit : on ne fusionne pas en chaîne
        survivant, perdant = paire_doublon(fiches[a], fiches[b])
        archives.add(perdant)
        if not dry:
            _archiver(cur, perdant, fiches[perdant]["compartment"], "doublon",
                      merged_into=survivant)
            cur.execute("INSERT INTO relationships (source_memory_id, target_memory_id, "
                        "relation_type, weight) VALUES (%s, %s, 'supersedes_by', 1.0) "
                        "ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING",
                        (survivant, perdant))
    return len(archives)


def phase_consolidation(cur, tenant: str, agent: str, r: ReglagesSommeil,
                        dry: bool) -> dict:
    """Épisodes rejoués par le LLM, transformés en souvenirs durables."""
    cur.execute("""
        SELECT id::text AS id, content, embedding::text AS embedding, project, created_at
        FROM memories
        WHERE tenant_id = %s AND agent_id = %s AND status = 'active'
          AND compartment = 'episodes' AND provenance->>'consolidated' IS NULL
          AND created_at < CURRENT_TIMESTAMP - make_interval(secs => %s)
        ORDER BY project NULLS FIRST, created_at
        LIMIT 400
    """, (tenant, agent, r.consolidation_age_min_jours * 86400))
    episodes = [dict(li, embedding=_vecteur(li["embedding"])) for li in cur.fetchall()]
    rapport = {"episodes_candidats": len(episodes), "groupes": 0, "souvenirs_crees": 0,
               "modele": None}
    if not episodes:
        return rapport
    par_projet: dict = {}
    for e in episodes:
        par_projet.setdefault(e["project"], []).append(e)
    groupes = [g for liste in par_projet.values()
               for g in grouper(liste, r.consolidation_groupe_seuil,
                                r.consolidation_groupe_taille_max)]
    # Les sujets les plus rejoués d'abord : c'est là que la consolidation rapporte le plus.
    groupes.sort(key=len, reverse=True)
    groupes = groupes[:r.consolidation_groupes_max]
    rapport["groupes"] = len(groupes)
    client = get_client_llm()
    if dry or client is None:
        rapport["note"] = "dry-run" if dry else "aucun LLM configuré : consolidation reportée"
        return rapport
    embedder = get_embedder()
    for groupe in groupes:
        try:
            donnees, modele = client.chat_json(prompt_consolidation(groupe), max_tokens=1500)
        except LLMIndisponible as e:
            logger.warning("Consolidation interrompue : %s", e)
            rapport["note"] = "LLM indisponible : consolidation partielle"
            break
        rapport["modele"] = modele
        souvenirs = valider_consolidation(donnees)
        vecteurs = embedder.embed([c for _, c in souvenirs]) if souvenirs else []
        crees = []
        for (compartiment, contenu), vecteur in zip(souvenirs, vecteurs, strict=True):
            fiche = PAR_NOM[compartiment]
            cur.execute("""
                INSERT INTO memories (tenant_id, agent_id, type, compartment, content, embedding,
                                      confidence, importance, status, content_hash, project,
                                      provenance)
                VALUES (%s, %s, %s, %s, %s, %s, 0.8, 0.6, 'active', %s, %s, %s)
                ON CONFLICT DO NOTHING RETURNING id
            """, (tenant, agent, fiche.famille, compartiment, contenu, to_pgvector(vecteur),
                  content_hash(contenu), groupe[0]["project"],
                  json.dumps({"source": "sommeil", "model": modele,
                              "consolidated_from": [e["id"] for e in groupe]})))
            ligne = cur.fetchone()
            if ligne is None:
                continue          # déjà su : la consolidation ne crée pas de doublon
            crees.append(str(ligne["id"]))
            if fiche.intrique:
                # Curseur ORDINAIRE : `entangle` lit ses lignes par position.
                with cur.connection.cursor() as brut:
                    entangle(brut, tenant, agent, ligne["id"], compartiment, vecteur,
                             project=groupe[0]["project"])
        # Les épisodes source restent où ils sont (l'oubli les archivera plus tard) : on
        # note seulement qu'ils ont été rejoués, pour ne pas les rejouer chaque nuit.
        cur.execute(
            "UPDATE memories SET provenance = COALESCE(provenance, '{}'::jsonb) || %s::jsonb "
            "WHERE id = ANY(%s::uuid[])",
            (json.dumps({"consolidated": True, "consolidated_into": crees}),
             [e["id"] for e in groupe]))
        rapport["souvenirs_crees"] += len(crees)
    return rapport


def phase_oubli(cur, tenant: str, agent: str, r: ReglagesSommeil, dry: bool,
                demi_vie_instance: float) -> int:
    """Jamais relus et tombés sous le seuil de récence : rangés dans les archives."""
    cur.execute("""
        SELECT id::text AS id, compartment, access_count,
               EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - COALESCE(last_accessed_at, created_at)))
                 AS age_s
        FROM memories
        WHERE tenant_id = %s AND agent_id = %s AND status = 'active'
          AND compartment <> ALL(%s) AND access_count = 0
    """, (tenant, agent, sorted(JAMAIS_OUBLIES)))
    oublies = [li for li in cur.fetchall()
               if doit_oublier(li["compartment"], float(li["age_s"] or 0), li["access_count"],
                               demi_vie_instance, r)]
    if not dry:
        for li in oublies:
            _archiver(cur, li["id"], li["compartment"], "oubli")
    return len(oublies)


def phase_graphe(cur, tenant: str, agent: str, r: ReglagesSommeil, dry: bool) -> int:
    """Souvenirs isolés des compartiments qui intriquent : reliés à leurs voisins."""
    intriquants = [c.nom for c in COMPARTIMENTS if c.intrique]
    cur.execute("""
        SELECT m.id, m.compartment, m.project, m.embedding::text AS embedding
        FROM memories m
        WHERE m.tenant_id = %s AND m.agent_id = %s AND m.status = 'active'
          AND m.compartment = ANY(%s)
          AND NOT EXISTS (SELECT 1 FROM relationships x WHERE x.source_memory_id = m.id)
          AND NOT EXISTS (SELECT 1 FROM relationships x WHERE x.target_memory_id = m.id)
        LIMIT %s
    """, (tenant, agent, intriquants, r.graphe_max))
    isoles = cur.fetchall()
    if dry:
        return len(isoles)
    relies = 0
    # Curseur ORDINAIRE : `entangle` lit ses lignes par position.
    with cur.connection.cursor() as brut:
        for li in isoles:
            if entangle(brut, tenant, agent, li["id"], li["compartment"],
                        _vecteur(li["embedding"]), project=li["project"]):
                relies += 1
    return relies


def phase_plasticite(cur, tenant: str, agent: str, r: ReglagesSommeil, dry: bool) -> int:
    """Liens renforcés puis délaissés : affaiblis jusqu'au poids d'un lien neuf (jamais supprimés)."""
    filtre = """
        FROM memories m
        WHERE r.source_memory_id = m.id AND m.tenant_id = %s AND m.agent_id = %s
          AND r.relation_type = 'entangled_with' AND r.last_reinforced_at IS NOT NULL
          AND r.last_reinforced_at < CURRENT_TIMESTAMP - make_interval(secs => %s)
          AND r.weight > %s
    """
    params = (tenant, agent, r.plasticite_repos_jours * 86400, POIDS_LIEN_NOUVEAU)
    if dry:
        # Fragments SQL constants de cette fonction : aucune donnée d'appelant interpolée.
        cur.execute("SELECT count(*) AS n FROM relationships r, "  # noqa: S608
                    + filtre.split("FROM ", 1)[1], params)
        return int(cur.fetchone()["n"])
    cur.execute("UPDATE relationships r SET weight = GREATEST(%s, r.weight * (1 - %s)) "  # noqa: S608
                + filtre, (POIDS_LIEN_NOUVEAU, r.plasticite_facteur, *params))
    return cur.rowcount


# ─── Orchestration ───────────────────────────────────────────────────────────

def dormir(conn, tenant: str, agent: str, phases: tuple[str, ...], r: ReglagesSommeil,
           dry: bool = False) -> dict:
    """Une nuit de sommeil pour un agent. Une transaction par phase."""
    demi_vie_instance = float(os.getenv("QEM_RECENCY_HALFLIFE_DAYS", "14"))
    rapport: dict = {"agent_id": agent, "dry_run": dry}
    for phase in PHASES:
        if phase not in phases:
            continue
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if phase == "doublons":
                    rapport[phase] = phase_doublons(cur, tenant, agent, r, dry)
                elif phase == "consolidation":
                    rapport[phase] = phase_consolidation(cur, tenant, agent, r, dry)
                elif phase == "oubli":
                    rapport[phase] = phase_oubli(cur, tenant, agent, r, dry, demi_vie_instance)
                elif phase == "graphe":
                    rapport[phase] = phase_graphe(cur, tenant, agent, r, dry)
                elif phase == "plasticite":
                    rapport[phase] = phase_plasticite(cur, tenant, agent, r, dry)
            if dry:
                conn.rollback()
            else:
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Phase %s en échec pour %s.", phase, agent, exc_info=True)
            rapport[phase] = {"erreur": type(e).__name__}
    if not dry:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO audit_log (tenant_id, agent_id, action, actor, details) "
                        "VALUES (%s, %s, 'sommeil', 'sommeil', %s)",
                        (tenant, agent, json.dumps(rapport, default=str)))
        conn.commit()
    logger.info("Sommeil de %s : %s", agent, json.dumps(rapport, default=str))
    return rapport


def agents_du_tenant(conn, tenant: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT agent_id FROM memories WHERE tenant_id = %s "
                    "AND status = 'active' AND compartment <> %s ORDER BY 1",
                    (tenant, COMPARTIMENT_ARCHIVES))
        return [li[0] for li in cur.fetchall()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sommeil de SynaptiQ (consolidation nocturne)")
    ap.add_argument("--tenant", default=os.getenv("SYNAPTIQ_TENANT", "default"))
    cible = ap.add_mutually_exclusive_group(required=True)
    cible.add_argument("--agent")
    cible.add_argument("--tous", action="store_true", help="Tous les agents du tenant")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    phases = tuple(p.strip() for p in args.phases.split(",") if p.strip())
    inconnues = set(phases) - set(PHASES)
    if inconnues:
        ap.error(f"Phases inconnues : {sorted(inconnues)}")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        agents = agents_du_tenant(conn, args.tenant) if args.tous else [args.agent]
        rapports = [dormir(conn, args.tenant, a, phases, reglages_depuis_env(), args.dry_run)
                    for a in agents]
    finally:
        conn.close()
    print(json.dumps(rapports, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
