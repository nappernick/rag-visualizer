// Demo API service for RAG Visualizer Demo Tab
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE || import.meta.env.VITE_API_URL || 'http://localhost:8642';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface DemoSearchOptions {
  query: string;
  mode?: 'smart' | 'vector' | 'graph' | 'hybrid';
  includeExplanations?: boolean;
  includeDecomposition?: boolean;
  maxResults?: number;
}

export interface DemoSearchResult {
  id: string;
  content: string;
  score: number;
  source: string;
  document_id?: string;
  document_title?: string;
  chunk_id?: string;
  metadata?: any;
  explanation?: string;
  highlights?: string[];
}

export interface DemoSearchResponse {
  results: DemoSearchResult[];
  decomposition?: any;
  total_results: number;
  retrieval_strategy: string;
  processing_time_ms: number;
  average_confidence: number;
  metadata?: any;
}

export const demoApi = {
  // Enhanced search with explanations and decomposition
  search: async (options: DemoSearchOptions): Promise<DemoSearchResponse> => {
    const response = await api.post('/api/demo/search', {
      query: options.query,
      mode: options.mode || 'smart',
      includeExplanations: options.includeExplanations !== false,
      includeDecomposition: options.includeDecomposition || false,
      maxResults: options.maxResults || 10,
    });
    return response.data;
  },

  // Get query suggestions
  getSuggestions: async (query: string): Promise<{ suggestions: string[] }> => {
    const response = await api.post('/api/demo/suggest', {
      query,
      max_suggestions: 5,
    });
    return response.data;
  },

  // Decompose complex query
  decomposeQuery: async (query: string): Promise<any> => {
    const response = await api.post('/api/demo/decompose', { query });
    return response.data;
  },

  // Summarize document
  summarizeDocument: async (
    documentId: string,
    style: 'brief' | 'detailed' | 'technical' = 'brief'
  ): Promise<any> => {
    const response = await api.post('/api/demo/summarize', {
      document_id: documentId,
      style,
    });
    return response.data;
  },

  // Explore graph from entities
  exploreGraph: async (
    entityIds: string[],
    maxHops: number = 2,
    limit: number = 10
  ): Promise<any> => {
    const response = await api.post('/api/demo/explore', {
      entity_ids: entityIds,
      max_hops: maxHops,
      limit,
    });
    return response.data;
  },

  // Find path between entities
  findPath: async (
    startEntity: string,
    endEntity: string,
    maxHops: number = 3
  ): Promise<any> => {
    const response = await api.post('/api/demo/find-path', {
      start_entity: startEntity,
      end_entity: endEntity,
      max_hops: maxHops,
    });
    return response.data;
  },

  // Analyze documents
  analyzeDocuments: async (
    documentIds: string[],
    analysisType: 'comprehensive' | 'summary' | 'entities' | 'relationships' = 'comprehensive'
  ): Promise<any> => {
    const response = await api.post('/api/demo/analyze', {
      document_ids: documentIds,
      analysis_type: analysisType,
    });
    return response.data;
  },
};

export default demoApi;