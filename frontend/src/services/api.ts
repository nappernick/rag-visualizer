// API service for RAG Visualizer

import axios from 'axios';
import type {
  Document,
  Chunk,
  ChunkingRequest,
  ChunkingResponse,
  Entity,
  Relationship,
  VisualizationData,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8734';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Document APIs
export const documentApi = {
  create: async (title: string, content: string): Promise<Document> => {
    const response = await api.post('/api/documents', { title, content });
    return response.data;
  },

  upload: async (file: File): Promise<Document> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  list: async (): Promise<Document[]> => {
    const response = await api.get('/api/documents');
    return response.data;
  },

  get: async (id: string): Promise<Document> => {
    const response = await api.get(`/api/documents/${id}`);
    return response.data;
  },
};

// Chunking APIs
export const chunkingApi = {
  chunkDocument: async (request: ChunkingRequest): Promise<ChunkingResponse> => {
    const response = await api.post('/api/chunking', request);
    return response.data;
  },

  getChunks: async (documentId: string): Promise<Chunk[]> => {
    const response = await api.get(`/api/documents/${documentId}/chunks`);
    return response.data;
  },
};

// Graph APIs
export const graphApi = {
  extractGraph: async (documentId: string, chunks: Chunk[]): Promise<{
    entities: Entity[];
    relationships: Relationship[];
  }> => {
    const response = await api.post('/api/graph/extract', {
      document_id: documentId,
      chunks,
      extract_entities: true,
      extract_relationships: true,
      use_spacy: true,
    });
    return response.data;
  },

  getEntities: async (documentId: string): Promise<Entity[]> => {
    const response = await api.get(`/api/graph/${documentId}/entities`);
    return response.data;
  },

  getRelationships: async (documentId: string): Promise<Relationship[]> => {
    const response = await api.get(`/api/graph/${documentId}/relationships`);
    return response.data;
  },
};

// Visualization APIs
export const visualizationApi = {
  getVisualizationData: async (documentId: string): Promise<VisualizationData> => {
    const response = await api.get(`/api/visualization/${documentId}`);
    return response.data;
  },
};

// Query APIs
export const queryApi = {
  query: async (
    query: string,
    options: {
      max_results?: number;
      retrieval_strategy?: 'vector' | 'graph' | 'hybrid';
      fusion_config?: any;
      preset?: string;
    } = {}
  ) => {
    const response = await api.post('/api/query', {
      query,
      max_results: options.max_results || 10,
      retrieval_strategy: options.retrieval_strategy || 'hybrid',
      include_metadata: true,
      rerank: true,
      fusion_config: options.fusion_config,
      preset: options.preset,
    });
    return response.data;
  },

  // Update fusion configuration
  updateFusionConfig: async (config: any) => {
    const response = await api.post('/api/fusion/tune', config);
    return response.data;
  },

  // Evaluate fusion configurations
  evaluateFusion: async (query: string, groundTruth: string[]) => {
    const response = await api.post('/api/fusion/evaluate', {
      query,
      ground_truth: groundTruth,
    });
    return response.data;
  },

  // Get current fusion configuration
  getFusionConfig: async () => {
    const response = await api.get('/api/fusion/config');
    return response.data;
  },
};

export default api;