// Type definitions for RAG Visualizer

export interface Document {
  id: string;
  title: string;
  content: string;
  doc_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Chunk {
  id: string;
  content: string;
  document_id: string;
  chunk_index: number;
  chunk_type: 'standard' | 'hierarchical' | 'summary' | 'code' | 'table' | 'section';
  tokens: number;
  embedding?: number[];
  metadata: Record<string, any>;
  parent_id?: string;
  children_ids: string[];
  created_at: string;
}

export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  document_ids: string[];
  chunk_ids: string[];
  frequency: number;
  metadata: Record<string, any>;
}

export interface Relationship {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  weight: number;
  document_ids: string[];
  metadata: Record<string, any>;
}

export interface RetrievalResult {
  chunk_id: string;
  content: string;
  score: number;
  source: 'vector' | 'graph' | 'hybrid';
  metadata: Record<string, any>;
  highlights: string[];
}

export interface ChunkingRequest {
  document_id: string;
  content: string;
  strategy: 'standard' | 'hierarchical' | 'semantic';
  max_chunk_size: number;
  chunk_overlap: number;
}

export interface ChunkingResponse {
  document_id: string;
  chunks: Chunk[];
  total_chunks: number;
  strategy_used: string;
  processing_time_ms: number;
  hierarchy_depth?: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  frequency?: number;
  level?: number;
  tokens?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight?: number;
}

export interface VisualizationData {
  chunk_hierarchy?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  knowledge_graph?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
    communities?: string[][];
  };
  embedding_space?: {
    points: Array<{
      id: string;
      x: number;
      y: number;
      z?: number;
      label: string;
      cluster?: number;
    }>;
    clusters?: Array<{
      id: number;
      label: string;
      center: { x: number; y: number; z?: number };
    }>;
  };
  retrieval_flow?: Record<string, any>;
}