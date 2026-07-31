export type MemoryType = "semantic" | "episodic" | "procedural" | "working";

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

  capture(agent_id: string, session_id: string, content: string, metadata: Record<string, unknown> = {}, idempotency_key?: string) {
    return this.post<{ status: string; event_id: string }>("/v1/events", { agent_id, session_id, content, metadata, idempotency_key });
  }

  storeMemory(agent_id: string, type: MemoryType, content: string, subtype?: string, confidence = 1, importance = 0.5) {
    return this.post<{ status: string; memory_id: string }>("/v1/memories", { agent_id, type, content, subtype, confidence, importance });
  }

  /** `memory_type` filtre par famille cognitive, `collections` par rayon precis. */
  retrieve(agent_id: string, query: string, limit = 5, memory_type?: MemoryType, collections?: string[]) {
    const body: Record<string, unknown> = { agent_id, query, limit, memory_type };
    // Omis quand absent : une liste vide serait un filtre qui ne ramene rien.
    if (collections !== undefined) body.collections = collections;
    return this.post<{ memories: unknown[] }>("/v1/retrieve", body);
  }

  // ─── Collections : la taxonomie que l'agent se donne ──────────────────────
  // La FAMILLE appartient au moteur et porte un comportement (intrication, decroissance,
  // section de repli). La COLLECTION appartient a l'agent : il la nomme, la decrit, et
  // elle obtient sa propre section dans le context_packet.

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
  buildContext(agent_id: string, session_id: string, task: string, query: string, options: { maxTokens?: number; memoryTypes?: MemoryType[]; collections?: string[]; explain?: boolean } = {}) {
    const constraints: Record<string, unknown> = {
      max_tokens: options.maxTokens ?? 1200,
      memory_types: options.memoryTypes ?? ["semantic", "episodic", "procedural", "working"],
    };
    // Omis quand absent : le serveur distingue « toutes les collections » d'une liste
    // explicite, et une liste vide serait un filtre qui ne ramene rien.
    if (options.collections !== undefined) constraints.collections = options.collections;
    return this.post<ContextResult>("/v1/context/build", {
      agent_id, session_id, task, query, explain: options.explain ?? false, constraints,
    });
  }
}
