// `reflective` : croyances de l'agent (ce qu'il pense de l'utilisateur et des humains).
export type MemoryType = "semantic" | "episodic" | "procedural" | "working" | "reflective";

export interface Belief {
  id: string;
  about: "user" | "humans";
  content: string;
  confidence: number;
  evidence: string[];
  project: string | null;
  created_at: string;
}

export interface SynaptiqClientOptions {
  baseUrl?: string;
  apiKey?: string;
  fetch?: typeof globalThis.fetch;
}

export interface Collection {
  name: string;
  family: MemoryType;
  packet_key: string;
  description: string;
  entangle: boolean;
  created_by: "system" | "agent";
  memory_count: number;
  /** Declaree mais restee vide au-dela de COLLECTION_STALE_DAYS : candidate a la fusion. */
  stale: boolean;
}

export interface CollectionsResult {
  agent_id: string;
  collections: Collection[];
  /** Sections du context_packet : les 7 canoniques, puis celles de l'agent. */
  packet_keys: string[];
  limits: { max_collections: number; used: number };
}

export interface CreatedCollection {
  status: string;
  name: string;
  family: MemoryType;
  packet_key: string;
  entangle: boolean;
  /** Ce qu'il faut passer a storeMemory pour ecrire dedans. */
  usage: { type: MemoryType; subtype: string };
}

export interface ContextResult {
  /**
   * /!\ Nombre de cles VARIABLE : les 7 sections canoniques (`facts`, `preferences`,
   * `episodes`, `rules`, `best_practices`, `errors`, `examples`) sont toujours presentes,
   * plus une par collection declaree par l'agent -- meme vide. Iterer sur les entrees
   * plutot que de lire sept cles en dur.
   */
  context_packet: Record<string, string[]>;
  token_estimate: number;
  selected_memory_ids: string[];
  trace_id: string;
  retrieval_trace?: Array<{ memory_id: string; similarity: number; recency_factor: number; score: number; selection_reason: string }>;
}

export class SynaptiqClient {
  private readonly baseUrl: string;
  private readonly request: typeof globalThis.fetch;
  private readonly headers: HeadersInit;

  constructor(options: SynaptiqClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8000").replace(/\/$/, "");
    this.request = options.fetch ?? globalThis.fetch;
    this.headers = { "content-type": "application/json", ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}) };
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await this.request(`${this.baseUrl}${path}`, { method: "POST", headers: this.headers, body: JSON.stringify(body) });
    if (!response.ok) throw new Error(`SynaptiQ ${response.status}: ${await response.text()}`);
    return response.json() as Promise<T>;
  }

  private async get<T>(path: string, params: Record<string, string>): Promise<T> {
    const url = `${this.baseUrl}${path}?${new URLSearchParams(params).toString()}`;
    const response = await this.request(url, { method: "GET", headers: this.headers });
    if (!response.ok) throw new Error(`SynaptiQ ${response.status}: ${await response.text()}`);
    return response.json() as Promise<T>;
  }

  /** `project` (lot B) : applique a tous les faits que le worker extraira de l'evenement. */
  capture(agent_id: string, session_id: string, content: string, metadata: Record<string, unknown> = {}, idempotency_key?: string, project?: string) {
    const body: Record<string, unknown> = { agent_id, session_id, content, metadata, idempotency_key };
    if (project !== undefined) body.project = project;
    return this.post<{ status: string; event_id: string }>("/v1/events", body);
  }

  /** `project` absent = souvenir GLOBAL (valable dans tous les projets). */
  storeMemory(agent_id: string, type: MemoryType, content: string, subtype?: string, confidence = 1, importance = 0.5, project?: string) {
    const body: Record<string, unknown> = { agent_id, type, content, subtype, confidence, importance };
    if (project !== undefined) body.project = project;
    return this.post<{ status: string; memory_id: string; project: string | null }>("/v1/memories", body);
  }

  /**
   * `memory_type` filtre par famille cognitive, `collections` par rayon precis, `project`
   * par projet (les souvenirs globaux restent inclus sauf `includeGlobal = false`).
   */
  retrieve(agent_id: string, query: string, limit = 5, memory_type?: MemoryType, collections?: string[], project?: string, includeGlobal = true) {
    const body: Record<string, unknown> = { agent_id, query, limit, memory_type };
    // Omis quand absent : une liste vide serait un filtre qui ne ramene rien.
    if (collections !== undefined) body.collections = collections;
    if (project !== undefined) {
      body.project = project;
      body.include_global = includeGlobal;
    }
    return this.post<{ memories: unknown[] }>("/v1/retrieve", body);
  }

  // ─── Collections : la taxonomie que l'agent se donne ──────────────────────
  // La FAMILLE appartient au moteur et porte un comportement (intrication, decroissance,
  // section de repli). La COLLECTION appartient a l'agent : il la nomme, la decrit, et
  // elle obtient sa propre section dans le context_packet.

  // ─── Croyances (famille `reflective`) ─────────────────────────────────────
  // Validees par le garde-fou serveur (422 sinon), servies comme « hypothese, confiance x »,
  // consultables et contestables par l'utilisateur.

  /** Confiance > 0.5 : `evidence` (ids de souvenirs) obligatoire. */
  noteBelief(agent_id: string, content: string, options: { about?: "user" | "humans"; confidence?: number; evidence?: string[]; replaces?: string; project?: string } = {}) {
    const body: Record<string, unknown> = {
      agent_id, type: "reflective", content, confidence: options.confidence ?? 0.5,
      subtype: (options.about ?? "user") === "user" ? "user_model" : "human_insights",
    };
    if (options.evidence?.length) body.evidence = options.evidence;
    if (options.replaces) body.replaces = options.replaces;
    if (options.project !== undefined) body.project = options.project;
    return this.post<{ status: string; memory_id: string }>("/v1/memories", body);
  }

  listBeliefs(agent_id: string, about?: "user" | "humans") {
    const params: Record<string, string> = { agent_id };
    if (about) params.about = about;
    return this.get<{ beliefs: Belief[] }>("/v1/beliefs", params);
  }

  contestBelief(agent_id: string, belief_id: string, reason?: string) {
    return this.post<{ status: string; belief_id: string }>(`/v1/beliefs/${belief_id}/contest`, { agent_id, reason });
  }

  listCollections(agent_id: string) {
    return this.get<CollectionsResult>("/v1/collections", { agent_id });
  }

  /**
   * Declare une collection. La `description` est obligatoire et vectorisee : une
   * collection trop proche d'une existante est refusee (409) en nommant le doublon.
   */
  createCollection(agent_id: string, name: string, family: MemoryType, description: string,
                   options: { entangle?: boolean; packetKey?: string } = {}) {
    const body: Record<string, unknown> = {
      agent_id, name, family, description, entangle: options.entangle ?? true,
    };
    if (options.packetKey !== undefined) body.packet_key = options.packetKey;
    return this.post<CreatedCollection>("/v1/collections", body);
  }

  /** Verse `source` dans `target` puis supprime `source`. Aucun souvenir n'est detruit. */
  mergeCollections(agent_id: string, source: string, target: string) {
    return this.post<{ status: string; source: string; target: string; moved_memories: number }>(
      "/v1/collections/merge", { agent_id, source, target });
  }

  /**
   * `memoryTypes` filtre par FAMILLE cognitive (les 4, fermees) ; `collections` filtre
   * finement par rayon declare par l'agent (`memories.subtype`).
   *
   * /!\ `context_packet` n'a plus un nombre de cles fixe : les sept sections canoniques
   * sont toujours presentes, plus une section par collection declaree par l'agent, meme
   * vide. Iterer sur les entrees plutot que de lire sept cles en dur.
   */
  buildContext(agent_id: string, session_id: string, task: string, query: string, options: { maxTokens?: number; memoryTypes?: MemoryType[]; collections?: string[]; explain?: boolean; project?: string; includeGlobal?: boolean } = {}) {
    const constraints: Record<string, unknown> = {
      max_tokens: options.maxTokens ?? 1200,
      memory_types: options.memoryTypes ?? ["semantic", "episodic", "procedural", "working", "reflective"],
    };
    // Omis quand absent : le serveur distingue « toutes les collections » d'une liste
    // explicite, et une liste vide serait un filtre qui ne ramene rien.
    if (options.collections !== undefined) constraints.collections = options.collections;
    // Projet (lot B) : ce projet + les souvenirs globaux, sauf includeGlobal = false.
    if (options.project !== undefined) {
      constraints.project = options.project;
      constraints.include_global = options.includeGlobal ?? true;
    }
    return this.post<ContextResult>("/v1/context/build", {
      agent_id, session_id, task, query, explain: options.explain ?? false, constraints,
    });
  }
}
