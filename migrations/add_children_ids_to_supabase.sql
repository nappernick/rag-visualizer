-- Add children_ids column to rag_chunks table in Supabase
-- Run this in Supabase SQL Editor

ALTER TABLE public.rag_chunks 
ADD COLUMN IF NOT EXISTS children_ids JSONB DEFAULT '[]'::jsonb;

-- Create index for children_ids if needed
CREATE INDEX IF NOT EXISTS idx_rag_chunks_children_ids 
ON public.rag_chunks USING GIN(children_ids);

-- Verify the column was added
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'rag_chunks' 
AND column_name = 'children_ids';