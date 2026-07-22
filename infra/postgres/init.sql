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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

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
    provenance JSONB DEFAULT '{}'::jsonb
);

-- Index vectoriel HNSW pour la recherche sémantique (opérateur <=> = distance cosinus).
-- vector_cosine_ops car les embeddings sont L2-normalisés et toutes les requêtes trient par <=>.
-- HNSW se construit à vide et se met à jour à l'insertion (pas de reindex différé nécessaire).
-- Paramètres par défaut (m=16, ef_construction=64) adaptés à la volumétrie visée ; ajuster si besoin.
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops);

-- Index B-tree pour le filtrage (tenant/agent/type/status) appliqué avant/avec la recherche vectorielle.
CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(tenant_id, agent_id, type, status);

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

-- Table des relations (Intrication Quantique)
CREATE TABLE IF NOT EXISTS relationships (
    source_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL, -- e.g., 'entangled_with', 'supersedes_by', 'contradicts'
    weight DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_memory_id, target_memory_id)
);
