-- Activation de l'extension pgvector pour la recherche sémantique
CREATE EXTENSION IF NOT EXISTS vector;

-- Table des événements bruts
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    session_id VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    idempotency_key VARCHAR(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_tenant_idempotency
    ON events(tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Index pour accélérer les recherches par session et agent
CREATE INDEX IF NOT EXISTS idx_events_tenant_agent_session ON events(tenant_id, agent_id, session_id);

-- Table des mémoires consolidées
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    type VARCHAR(20) NOT NULL, -- working, episodic, semantic, procedural
    subtype VARCHAR(50),        -- ex: preference, rule, facts, error_resolution
    content TEXT NOT NULL,
    summary TEXT,
    embedding VECTOR(384) NOT NULL, -- Dimension 384 pour modèles type all-MiniLM-L6-v2
    confidence DOUBLE PRECISION DEFAULT 1.0,
    importance DOUBLE PRECISION DEFAULT 0.5,
    recency_score DOUBLE PRECISION DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active', -- active, archived, disputed
    version INTEGER DEFAULT 1,
    provenance JSONB DEFAULT '{}'::jsonb,
    source_event_id UUID,
    -- Date à laquelle le FAIT s'est produit, résolue à l'extraction (« yesterday » ->
    -- date absolue). À distinguer de created_at, qui date l'écriture de la ligne.
    occurred_at TIMESTAMP WITH TIME ZONE,
    -- SHA256 du contenu : discrimine les faits multiples issus d'un même événement.
    content_hash VARCHAR(64)
);
-- Un événement peut produire PLUSIEURS faits ; le replay du même événement reste
-- dédupliqué car les mêmes faits produisent les mêmes hachages (idempotence).
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_event_fact
    ON memories(source_event_id, content_hash) WHERE source_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_occurred
    ON memories(tenant_id, agent_id, occurred_at) WHERE occurred_at IS NOT NULL;
ALTER TABLE memories ADD CONSTRAINT fk_memories_source_event
    FOREIGN KEY (source_event_id) REFERENCES events(id) ON DELETE SET NULL;

-- Index vectoriel HNSW pour la recherche sémantique (opérateur <=> = distance cosinus).
-- vector_cosine_ops car les embeddings sont L2-normalisés et toutes les requêtes trient par <=>.
-- HNSW se construit à vide et se met à jour à l'insertion (pas de reindex différé nécessaire).
-- Paramètres par défaut (m=16, ef_construction=64) adaptés à la volumétrie visée ; ajuster si besoin.
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops);

-- Index B-tree pour le filtrage (tenant/agent/type/status) appliqué avant/avec la recherche vectorielle.
CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(tenant_id, agent_id, type, status);

-- Recherche hybride : le plein texte rattrape les correspondances LITTÉRALES (noms propres,
-- dates, identifiants) que la similarité vectorielle manque. Config 'simple' car le corpus
-- d'une instance est souvent multilingue et l'on cherche ici l'exactitude, pas la
-- généralisation (déjà couverte par le vecteur).
ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;
CREATE INDEX IF NOT EXISTS idx_memories_content_tsv ON memories USING GIN (content_tsv);

-- Table des clés API (auth + scoping multi-tenant, Phase 3)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) NOT NULL UNIQUE,   -- SHA256 hex de la clé en clair (jamais stockée en clair)
    tenant_id VARCHAR(50) NOT NULL,          -- tenant auquel la clé donne accès
    name VARCHAR(100),                       -- libellé lisible (ex: 'agent-ouroboros-prod')
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash) WHERE active;

-- Transactional outbox: the relay publishes committed events to Redis Streams.
CREATE TABLE IF NOT EXISTS event_outbox (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL UNIQUE REFERENCES events(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP WITH TIME ZONE,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_event_outbox_pending ON event_outbox(created_at) WHERE published_at IS NULL;

-- Table des relations (Intrication Quantique)
CREATE TABLE IF NOT EXISTS relationships (
    source_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL, -- e.g., 'entangled_with', 'supersedes_by', 'contradicts'
    weight DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_memory_id, target_memory_id)
);
