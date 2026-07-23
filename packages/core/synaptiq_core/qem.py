"""SynaptiQ — cœur algorithmique Q-EM (Quantum Entanglement Memory).

Fonctions PURES, sans aucun accès base de données ni dépendance FastAPI/psycopg2.
Extraites de `build_context` (apps/api/main.py) pour être testables en isolation.

Le moteur Q-EM se déroule en 4 phases (voir CLAUDE.md) :
  1. Superposition  — recherche sémantique -> candidats scorés (similarité x récence).
  2. Intrication    — propagation d'activation amortie le long des liens 'entangled_with'.
  3. Interférence   — filtrage destructif (contradictions + redondances sémantiques).
  4. Mesure         — collapse glouton par densité d'utilité/token sous budget de tokens.

Toutes les fonctions opèrent sur :
  - `candidates` : dict[str, dict] indexé par id de mémoire. Chaque valeur porte au moins :
        id, type, subtype, content, confidence, importance, created_at,
        last_accessed_at, embedding (list[float]), similarity, score.
    (`recency_factor` est présent pour les candidats issus de la recherche directe.)
  - `relationships` : list[dict] avec les clés source_memory_id, target_memory_id,
        relation_type, weight.

Les SEUILS (damping, threshold, halflife) sont passés en PARAMÈTRES : aucune lecture
d'os.getenv ici, afin de tester chaque phase de manière déterministe.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("synaptiq-core.qem")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Superposition : scoring initial (similarité pondérée par la récence)
# ─────────────────────────────────────────────────────────────────────────────

def compute_recency_factor(age_seconds: float, halflife_days: float) -> float:
    """Facteur de décroissance temporelle (demi-vie exponentielle).

    - age_seconds : âge de la mémoire depuis son dernier accès (en secondes).
    - halflife_days : demi-vie en jours. <= 0 => décroissance désactivée (1.0).

    age=0 -> 1.0 ; age=halflife -> 0.5 ; age=2*halflife -> 0.25.
    """
    if halflife_days <= 0:
        return 1.0
    age_days = float(age_seconds or 0.0) / 86400.0
    return 0.5 ** (age_days / halflife_days)


def initial_score(similarity: float, recency_factor: float) -> float:
    """Score de départ d'un candidat : similarité cosinus pondérée par la récence."""
    return similarity * recency_factor


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Intrication : propagation d'activation amortie ('entangled_with')
# ─────────────────────────────────────────────────────────────────────────────

def propagate_entanglement(
    candidates: Dict[str, dict],
    relationships: List[dict],
    damping: float,
) -> None:
    """Propage l'activation le long des liens 'entangled_with' (mute les scores).

    Bidirectionnel et amorti : chaque extrémité reçoit la SIMILARITÉ directe de
    l'autre extrémité * poids du lien * damping. On lit bien `similarity` (activation
    « source ») et non `score` — réplique exacte de l'original. N'agit que si les deux
    extrémités sont présentes dans `candidates`.
    """
    for rel in relationships:
        if rel['relation_type'] == 'entangled_with':
            src = str(rel['source_memory_id'])
            tgt = str(rel['target_memory_id'])
            weight = float(rel['weight'] or 1.0)

            if src in candidates and tgt in candidates:
                candidates[tgt]['score'] += candidates[src]['similarity'] * weight * damping
                candidates[src]['score'] += candidates[tgt]['similarity'] * weight * damping


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Interférence destructive : contradictions puis redondances
# ─────────────────────────────────────────────────────────────────────────────

def apply_contradictions(
    candidates: Dict[str, dict],
    relationships: List[dict],
) -> None:
    """Annule (score=0) la mémoire la plus ANCIENNE d'un couple en contradiction.

    Concerne les relations 'contradicts' et 'supersedes_by'. La comparaison porte
    sur `created_at` : la plus ancienne des deux voit son score remis à zéro.
    """
    for rel in relationships:
        if rel['relation_type'] in ('contradicts', 'supersedes_by'):
            src = str(rel['source_memory_id'])
            tgt = str(rel['target_memory_id'])
            if src in candidates and tgt in candidates:
                c_src = candidates[src]
                c_tgt = candidates[tgt]
                if c_src['created_at'] < c_tgt['created_at']:
                    c_src['score'] = 0.0
                    logger.info(f"Q-EM: Interférence destructive (contradiction) : {src} annulé par {tgt}")
                else:
                    c_tgt['score'] = 0.0
                    logger.info(f"Q-EM: Interférence destructive (contradiction) : {tgt} annulé par {src}")


def filter_redundancy(
    candidates: Dict[str, dict],
    threshold: float,
) -> None:
    """Filtre les redondances sémantiques par similarité cosinus des embeddings.

    Parmi les candidats encore actifs (score > 0), triés par (importance, created_at)
    décroissants, deux mémoires dont le cosinus dépasse `threshold` sont jugées
    redondantes : la moins prioritaire (celle qui vient après dans le tri) est annulée.
    Les embeddings sont supposés normalisés (cosinus = produit scalaire).
    """
    active_ids = [cid for cid, c in candidates.items() if c['score'] > 0.0]
    # Conserver en priorité les plus importants / récents (annulera les suivants).
    active_ids.sort(key=lambda cid: (candidates[cid]['importance'], candidates[cid]['created_at']), reverse=True)

    for i in range(len(active_ids)):
        id_i = active_ids[i]
        if candidates[id_i]['score'] == 0.0:
            continue
        emb_i = candidates[id_i]['embedding']

        for j in range(i + 1, len(active_ids)):
            id_j = active_ids[j]
            if candidates[id_j]['score'] == 0.0:
                continue
            emb_j = candidates[id_j]['embedding']

            if emb_i and emb_j:
                cosine_sim = sum(x * y for x, y in zip(emb_i, emb_j))
                if cosine_sim > threshold:
                    candidates[id_j]['score'] = 0.0
                    logger.info(f"Q-EM: Interférence destructive (redondance sim={cosine_sim:.2f}) : {id_j} annulé au profit de {id_i}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Mesure : routage + collapse glouton sous budget de tokens
# ─────────────────────────────────────────────────────────────────────────────

# Les 7 clés du context_packet (contrat stable côté consommateur).
_PACKET_KEYS = ("facts", "preferences", "episodes", "rules", "best_practices", "errors", "examples")


def estimate_tokens(content: str) -> int:
    """Estimation grossière du coût en tokens d'un contenu (>=1)."""
    return max(1, int(len(content.split()) * 1.3))


def route_memory(m_type: str, m_subtype) -> str:
    """Détermine la clé du context_packet cible pour une mémoire (type/subtype).

    Routage « correct » (intention du code d'origine) :
      - semantic/preference           -> preferences
      - semantic/<autre>              -> facts
      - episodic/<*>                  -> episodes
      - procedural/coding_best_practices -> best_practices
      - procedural/code_error_resolution -> errors
      - procedural/<autre>            -> rules
      - working/<*>                   -> examples
      - type inconnu                  -> None (non routé, comme l'original)

    ⚠️ Voir `collapse_by_utility` : dans le chemin d'origine préservé, le sous-type
    n'est PAS transmis (bug historique) et tout ce qui est semantic tombe donc dans
    `facts` et tout ce qui est procedural dans `rules`.
    """
    if m_type == 'semantic':
        return 'preferences' if m_subtype == 'preference' else 'facts'
    if m_type == 'episodic':
        return 'episodes'
    if m_type == 'procedural':
        if m_subtype == 'coding_best_practices':
            return 'best_practices'
        if m_subtype == 'code_error_resolution':
            return 'errors'
        return 'rules'
    if m_type == 'working':
        return 'examples'
    return None


def collapse_by_utility(
    candidates: Dict[str, dict],
    max_tokens: int,
) -> Tuple[Dict[str, list], List[str], int]:
    """Collapse glouton : maximise l'utilité/token sous contrainte `max_tokens`.

    Retourne `(context_packet, selected_ids, token_count)` où `context_packet`
    porte toujours les 7 clés (listes, éventuellement vides).

    Routage COMPLET par type ET sous-type vers les 7 collections logiques
    (via `route_memory`) : `semantic/preference` -> preferences, `semantic/*` -> facts,
    `episodic/*` -> episodes, `procedural/coding_best_practices` -> best_practices,
    `procedural/code_error_resolution` -> errors, `procedural/*` -> rules,
    `working/*` -> examples. Le sous-type est propagé dans l'entrée collapsée.
    """
    # Sélection des candidats survivants + densité d'utilité par token.
    collapsed_candidates = []
    for mem_id, c in candidates.items():
        if c['score'] > 0.0:
            tokens = estimate_tokens(c['content'])
            utility_density = c['score'] / tokens
            collapsed_candidates.append({
                "id": mem_id,
                "type": c['type'],
                "subtype": c.get('subtype'),
                "content": c['content'],
                "tokens": tokens,
                "utility_density": utility_density,
            })

    # Tri par densité d'utilité par token décroissante (stable sur l'ordre d'insertion).
    collapsed_candidates.sort(key=lambda x: x['utility_density'], reverse=True)

    packet = {k: [] for k in _PACKET_KEYS}
    selected_ids: List[str] = []
    token_count = 0

    for c in collapsed_candidates:
        if token_count + c['tokens'] <= max_tokens:
            selected_ids.append(c['id'])
            token_count += c['tokens']

            # Routage complet type + sous-type vers la bonne collection logique.
            key = route_memory(c['type'], c.get('subtype'))
            if key is not None:
                packet[key].append(c['content'])
        else:
            logger.debug(f"Q-EM: Hors budget pour {c['id']} (tokens={c['tokens']}, restant={max_tokens - token_count})")

    return packet, selected_ids, token_count
