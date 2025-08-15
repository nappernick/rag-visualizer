-- Add weight field to documents table
ALTER TABLE rag_documents 
ADD COLUMN IF NOT EXISTS weight FLOAT DEFAULT 1.0 CHECK (weight >= 0.1 AND weight <= 10.0);

-- Create weight rules table
CREATE TABLE IF NOT EXISTS weight_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('document_type', 'title_pattern', 'temporal', 'content', 'manual')),
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    conditions JSONB NOT NULL,
    weight_modifier FLOAT NOT NULL CHECK (weight_modifier >= 0.1 AND weight_modifier <= 10.0),
    affected_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create weight calculations cache table
CREATE TABLE IF NOT EXISTS weight_calculations (
    id SERIAL PRIMARY KEY,
    document_id TEXT REFERENCES rag_documents(id) ON DELETE CASCADE,
    base_weight FLOAT DEFAULT 1.0,
    applied_rules JSONB,
    final_weight FLOAT NOT NULL,
    calculation_path TEXT,
    calculated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(document_id)
);

-- Create indexes for performance
CREATE INDEX idx_weight_rules_enabled ON weight_rules(enabled);
CREATE INDEX idx_weight_rules_priority ON weight_rules(priority DESC);
CREATE INDEX idx_weight_rules_type ON weight_rules(rule_type);
CREATE INDEX idx_weight_calculations_document ON weight_calculations(document_id);
CREATE INDEX idx_weight_calculations_final ON weight_calculations(final_weight DESC);
CREATE INDEX idx_rag_documents_weight ON rag_documents(weight DESC);

-- Create trigger for weight_rules updated_at
CREATE TRIGGER update_weight_rules_updated_at BEFORE UPDATE
    ON weight_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default weight rules
INSERT INTO weight_rules (id, name, rule_type, priority, conditions, weight_modifier) VALUES
    ('default_doc_type', 'Document Type Weights', 'document_type', 100, 
     '{"pdf": 1.5, "markdown": 1.2, "text": 1.0, "code": 0.8, "image": 0.5}'::jsonb, 1.0),
    ('default_temporal', 'Recency Boost', 'temporal', 90,
     '{"ranges": [{"within": "7d", "weight": 2.0}, {"within": "30d", "weight": 1.5}, {"within": "90d", "weight": 1.2}, {"older_than": "365d", "weight": 0.7}]}'::jsonb, 1.0),
    ('default_title_important', 'Important Documents', 'title_pattern', 80,
     '{"patterns": [{"match": "contains", "value": "important", "weight": 2.0}, {"match": "contains", "value": "critical", "weight": 2.5}, {"match": "contains", "value": "policy", "weight": 1.8}]}'::jsonb, 1.0)
ON CONFLICT (id) DO NOTHING;

-- Function to recalculate document weights
CREATE OR REPLACE FUNCTION recalculate_document_weight(doc_id TEXT)
RETURNS FLOAT AS $$
DECLARE
    final_weight FLOAT := 1.0;
    doc_record RECORD;
    rule RECORD;
    rule_weight FLOAT;
BEGIN
    -- Get document details
    SELECT * INTO doc_record FROM rag_documents WHERE id = doc_id;
    IF NOT FOUND THEN
        RETURN 1.0;
    END IF;
    
    -- Start with base weight
    final_weight := COALESCE(doc_record.weight, 1.0);
    
    -- Apply enabled rules in priority order
    FOR rule IN 
        SELECT * FROM weight_rules 
        WHERE enabled = true 
        ORDER BY priority DESC
    LOOP
        -- Apply rule based on type (simplified for now)
        -- In production, this would have more complex logic
        rule_weight := 1.0;
        
        -- Document type rule
        IF rule.rule_type = 'document_type' THEN
            rule_weight := COALESCE((rule.conditions->>(doc_record.doc_type))::FLOAT, 1.0);
        END IF;
        
        -- Apply the weight modifier
        final_weight := final_weight * rule_weight;
    END LOOP;
    
    -- Clamp final weight to valid range
    final_weight := GREATEST(0.1, LEAST(10.0, final_weight));
    
    -- Cache the calculation
    INSERT INTO weight_calculations (document_id, base_weight, final_weight, calculated_at)
    VALUES (doc_id, doc_record.weight, final_weight, NOW())
    ON CONFLICT (document_id) 
    DO UPDATE SET 
        base_weight = doc_record.weight,
        final_weight = final_weight,
        calculated_at = NOW();
    
    RETURN final_weight;
END;
$$ LANGUAGE plpgsql;