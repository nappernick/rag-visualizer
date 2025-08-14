-- Create entities table for storing extracted entities
CREATE TABLE IF NOT EXISTS public.entities (
    id text NOT NULL,
    name text NOT NULL,
    entity_type text NOT NULL,
    document_ids text[],
    chunk_ids text[],
    frequency integer DEFAULT 1,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT entities_pkey PRIMARY KEY (id)
);

-- Create relationships table for storing entity relationships
CREATE TABLE IF NOT EXISTS public.relationships (
    id text NOT NULL,
    source_entity_id text NOT NULL,
    target_entity_id text NOT NULL,
    relationship_type text NOT NULL,
    weight float DEFAULT 1.0,
    document_ids text[],
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT relationships_pkey PRIMARY KEY (id),
    CONSTRAINT relationships_source_entity_id_fkey FOREIGN KEY (source_entity_id) REFERENCES public.entities(id) ON DELETE CASCADE,
    CONSTRAINT relationships_target_entity_id_fkey FOREIGN KEY (target_entity_id) REFERENCES public.entities(id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_entities_name ON public.entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON public.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_document_ids ON public.entities USING GIN(document_ids);
CREATE INDEX IF NOT EXISTS idx_entities_created_at ON public.entities(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON public.relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON public.relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON public.relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_document_ids ON public.relationships USING GIN(document_ids);
CREATE INDEX IF NOT EXISTS idx_relationships_created_at ON public.relationships(created_at DESC);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_entities_updated_at 
    BEFORE UPDATE ON public.entities 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_relationships_updated_at 
    BEFORE UPDATE ON public.relationships 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust based on your RLS needs)
GRANT ALL ON public.entities TO anon;
GRANT ALL ON public.relationships TO anon;
GRANT ALL ON public.entities TO authenticated;
GRANT ALL ON public.relationships TO authenticated;