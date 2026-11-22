// Les 9 compartiments fixes de la memoire (v0.4), dans l'ordre du context_packet.
// Proprietes du moteur : l'agent ne cree rien, il choisit ou ranger.
export const COMPARTMENTS = [
  "reflexes", "decisions", "savoir_faire", "alertes", "environnement",
  "savoirs", "episodes", "observations", "croyances",
] as const;
export type Compartment = (typeof COMPARTMENTS)[number];

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

export interface CompartmentInfo {
  name: Compartment;
  /** Region du cerveau dont le compartiment s'inspire. */
  region: string;
  title: string;
  description: string;
  examples: string[];
  forgetting: string;
  entangle: boolean;
  /** Injecte dans chaque contexte, quelle que soit la question (reflexes). */
  always_injected: boolean;
  max_share: number;
  /** false = ecriture reservee a noteBelief (croyances). */
  direct_write: boolean;
  memory_count: number;
}

export interface CompartmentsResult {
  agent_id: string;
  compartments: CompartmentInfo[];
  /** Anciens noms de collection encore acceptes a l'ecriture (transition v0.4 -> v0.5). */
  legacy_aliases: Record<string, Compartment>;
}

export interface StoredMemory {
  status: "created" | "duplicate";
  memory_id: string;
  compartment: Compartment;
  region: string;
  /** `agent` : choisi ; `alias` : ancien nom converti ; `moteur` : classe par similarite ; `existant` : doublon. */
  compartment_origin: "agent" | "alias" | "moteur" | "existant";
  classification_similarity: number | null;
  requested: string | null;
  project: string | null;
}

export interface ContextResult {
  /** Forme FIXE : une section par compartiment, toujours presente (meme vide), dans l'ordre de COMPARTMENTS. */
  context_packet: Record<Compartment, string[]>;
  token_estimate: number;
  selected_memory_ids: string[];
  trace_id: string;
  retrieval_trace?: Array<{ memory_id: string; similarity: number; recency_factor: number; score: number; compartment: Compartment; selection_reason: string }>;
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

  /**
   * Range un souvenir dans un compartiment. `compartment` absent (ou inconnu) : le serveur
   * classe par similarite et le dit dans `compartment_origin`.
   * `project` absent = souvenir GLOBAL (valable dans tous les projets).
   */
  storeMemory(agent_id: string, content: string, options: { compartment?: Compartment; confidence?: number; importance?: number; project?: string } = {}) {
    const body: Record<string, unknown> = {
      agent_id, content, confidence: options.confidence ?? 1, importance: options.importance ?? 0.5,
    };
    if (options.compartment !== undefined) body.compartment = options.compartment;
    if (options.project !== undefined) body.project = options.project;
    return this.post<StoredMemory>("/v1/memories", body);
  }

  /**
   * `compartments` restreint la recherche, `project` filtre par projet (les souvenirs
   * globaux restent inclus sauf `includeGlobal = false`).
   */
  retrieve(agent_id: string, query: string, options: { limit?: number; compartments?: Compartment[]; project?: string; includeGlobal?: boolean } = {}) {
    const body: Record<string, unknown> = { agent_id, query, limit: options.limit ?? 5 };
    // Omis quand absent : une liste vide serait refusee, l'absence balaie tout.
    if (options.compartments !== undefined) body.compartments = options.compartments;
    if (options.project !== undefined) {
      body.project = options.project;
      body.include_global = options.includeGlobal ?? true;
    }
    return this.post<{ memories: unknown[] }>("/v1/retrieve", body);
  }

  /** Les 9 compartiments, leur fonction et le nombre de souvenirs de l'agent dans chacun. */
  listCompartments(agent_id: string) {
    return this.get<CompartmentsResult>("/v1/compartments", { agent_id });
  }

  // ─── Croyances (compartiment `croyances`) ─────────────────────────────────
  // Validees par le garde-fou serveur (422 sinon), servies comme « hypothese, confiance x »,
  // consultables et contestables par l'utilisateur.

  /** Confiance > 0.5 : `evidence` (ids de souvenirs) obligatoire. */
  noteBelief(agent_id: string, content: string, options: { about?: "user" | "humans"; confidence?: number; evidence?: string[]; replaces?: string; project?: string } = {}) {
    const body: Record<string, unknown> = {
      agent_id, compartment: "croyances", about: options.about ?? "user", content,
      confidence: options.confidence ?? 0.5,
    };
    if (options.evidence?.length) body.evidence = options.evidence;
    if (options.replaces) body.replaces = options.replaces;
    if (options.project !== undefined) body.project = options.project;
    return this.post<StoredMemory>("/v1/memories", body);
  }

  listBeliefs(agent_id: string, about?: "user" | "humans") {
    const params: Record<string, string> = { agent_id };
    if (about) params.about = about;
    return this.get<{ beliefs: Belief[] }>("/v1/beliefs", params);
  }

  contestBelief(agent_id: string, belief_id: string, reason?: string) {
    return this.post<{ status: string; belief_id: string }>(`/v1/beliefs/${belief_id}/contest`, { agent_id, reason });
  }

  /** Paquet de contexte : une section par compartiment, les reflexes injectes d'office. */
  buildContext(agent_id: string, session_id: string, task: string, query: string, options: { maxTokens?: number; compartments?: Compartment[]; explain?: boolean; project?: string; includeGlobal?: boolean } = {}) {
    const constraints: Record<string, unknown> = { max_tokens: options.maxTokens ?? 1200 };
    // Omis quand absent : le serveur prend alors les 9 compartiments.
    if (options.compartments !== undefined) constraints.compartments = options.compartments;
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
