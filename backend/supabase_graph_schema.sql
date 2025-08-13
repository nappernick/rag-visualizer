-- Additional tables for graph storage (entities and relationships)

-- Create entities table
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    document_ids TEXT[],
    chunk_ids TEXT[],
    frequency INTEGER DEFAULT 1,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create relationships table
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id TEXT REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    document_ids TEXT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for entities
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_document_ids ON entities USING GIN(document_ids);
CREATE INDEX idx_entities_created_at ON entities(created_at DESC);

-- Create indexes for relationships
CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON relationships(relationship_type);
CREATE INDEX idx_relationships_document_ids ON relationships USING GIN(document_ids);
CREATE INDEX idx_relationships_created_at ON relationships(created_at DESC);

-- Create triggers for updated_at
CREATE TRIGGER update_entities_updated_at BEFORE UPDATE
    ON entities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_relationships_updated_at BEFORE UPDATE
    ON relationships FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();