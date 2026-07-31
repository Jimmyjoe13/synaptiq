export type MemoryType = "semantic" | "episodic" | "procedural" | "working";

export interface SynaptiqClientOptions {
  baseUrl?: string;
  apiKey?: string;
  fetch?: typeof globalThis.fetch;
}

export interface ContextResult {
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

  capture(agent_id: string, session_id: string, content: string, metadata: Record<string, unknown> = {}, idempotency_key?: string) {
    return this.post<{ status: string; event_id: string }>("/v1/events", { agent_id, session_id, content, metadata, idempotency_key });
  }

  storeMemory(agent_id: string, type: MemoryType, content: string, subtype?: string, confidence = 1, importance = 0.5) {
    return this.post<{ status: string; memory_id: string }>("/v1/memories", { agent_id, type, content, subtype, confidence, importance });
  }

  retrieve(agent_id: string, query: string, limit = 5, memory_type?: MemoryType) {
    return this.post<{ memories: unknown[] }>("/v1/retrieve", { agent_id, query, limit, memory_type });
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
