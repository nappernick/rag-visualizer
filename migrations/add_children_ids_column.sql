-- Add children_ids column to rag_chunks table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'rag_chunks' 
        AND column_name = 'children_ids'
    ) THEN
        ALTER TABLE rag_chunks ADD COLUMN children_ids JSONB DEFAULT '[]';
    END IF;
END $$;