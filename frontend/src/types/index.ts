// Type definitions for RAG Visualizer

export interface Document {
  id: string;
  title: string;
  content: string;
  doc_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  weight: number; // Document priority weight (0.1-10.0)
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
  performance?: {
    file_read?: number;
    content_processing?: number;
    chunking?: number;
    chunk_count?: number;
    entity_extraction?: number;
    entity_count?: number;
    relationship_count?: number;
    entity_storage?: number;
    document_storage?: number;
    total_time?: number;
    content_length?: number;
  };
}

export interface Chunk {
  id: string;
  content: string;
  document_id: string;
  chunk_index: number;
  chunk_type: 'standard' | 'hierarchical' | 'summary' | 'code' | 'table' | 'section' | 'semantic';
  tokens: number;
  embedding?: number[];
  metadata: Record<string, any>;
  parent_id?: string | null;
  children_ids?: string[];
  level?: number;  // Hierarchy level: 0 = root, 1 = section, 2 = subsection, etc.
  created_at?: string;
  position?: number;
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

// Weight Rules Types
export type RuleType = 'document_type' | 'title_pattern' | 'temporal' | 'content' | 'manual';

export type PatternMatchType = 'contains' | 'startsWith' | 'endsWith' | 'regex' | 'exact';

export interface PatternMatch {
  match: PatternMatchType;
  value: string;
  weight: number;
  case_sensitive?: boolean;
}

export interface TemporalRange {
  within?: string; // e.g., "7d", "30d", "1y"
  older_than?: string;
  newer_than?: string;
  weight: number;
}

export interface WeightRuleConditions {
  // For document_type rules
  type_weights?: Record<string, number>;
  
  // For title_pattern rules
  patterns?: PatternMatch[];
  
  // For temporal rules
  ranges?: TemporalRange[];
  
  // For content rules
  content_patterns?: PatternMatch[];
  min_length?: number;
  max_length?: number;
  
  // For manual rules
  document_ids?: string[];
  document_patterns?: string[];
}

export interface WeightRule {
  id: string;
  name: string;
  rule_type: RuleType;
  enabled: boolean;
  priority: number;
  conditions: WeightRuleConditions;
  weight_modifier: number;
  affected_count: number;
  created_at: string;
  updated_at: string;
}

export interface AppliedRule {
  rule_id: string;
  rule_name: string;
  rule_type: string;
  weight_applied: number;
  reason: string;
}

export interface WeightCalculation {
  document_id: string;
  document_title: string;
  base_weight: number;
  applied_rules: AppliedRule[];
  final_weight: number;
  calculation_path: string;
  calculated_at: string;
}

export interface WeightDistribution {
  range: string;
  count: number;
  percentage: number;
}

export interface WeightSimulation {
  total_documents: number;
  calculations: WeightCalculation[];
  weight_distribution: Record<string, number>;
  average_weight: number;
  median_weight: number;
}