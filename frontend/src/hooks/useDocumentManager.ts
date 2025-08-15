import { useState, useCallback } from 'react';
import { documentApi, chunkingApi, graphApi } from '../services/api';
import type { Document, Chunk, Entity, Relationship, ChunkingRequest } from '../types';

export const useDocumentManager = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const docs = await documentApi.list();
      setDocuments(docs);
    } catch (err) {
      setError('Failed to load documents');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const selectDocument = useCallback(async (doc: Document) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedDocument(doc);
      
      // Clear previous data first to avoid showing stale data
      setChunks([]);
      setEntities([]);
      setRelationships([]);
      
      // Load chunks with proper error handling
      const docChunks = await chunkingApi.getChunks(doc.id).catch((err) => {
        console.error('Error loading chunks:', err);
        return [];
      });
      setChunks(Array.isArray(docChunks) ? docChunks : []);
      
      // Load entities and relationships with individual error handling
      const [docEntities, docRelationships] = await Promise.all([
        graphApi.getEntities(doc.id).catch((err) => {
          console.error('Error loading entities:', err);
          return [];
        }),
        graphApi.getRelationships(doc.id).catch((err) => {
          console.error('Error loading relationships:', err);
          return [];
        })
      ]);
      
      setEntities(Array.isArray(docEntities) ? docEntities : []);
      setRelationships(Array.isArray(docRelationships) ? docRelationships : []);
      
      // Log what was loaded for debugging
      console.log(`Loaded for document ${doc.id}:`, {
        chunks: docChunks?.length || 0,
        entities: docEntities?.length || 0,
        relationships: docRelationships?.length || 0
      });
      
    } catch (err) {
      console.error('Error loading document data:', err);
      setError('Failed to load document data');
      // Still set empty arrays to avoid undefined errors
      setChunks([]);
      setEntities([]);
      setRelationships([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const createDocument = useCallback(async (title: string, content: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const document = await documentApi.create(title, content);
      setDocuments(prev => [...prev, document]);
      setSelectedDocument(document);
      
      // Process the document
      const chunkingRequest: ChunkingRequest = {
        document_id: document.id,
        content: document.content,
        strategy: 'hierarchical',
        max_chunk_size: 800,
        chunk_overlap: 100
      };
      
      const chunkingResponse = await chunkingApi.chunkDocument(chunkingRequest);
      setChunks(Array.isArray(chunkingResponse.chunks) ? chunkingResponse.chunks : []);
      
      return document;
    } catch (err) {
      setError('Failed to create document');
      console.error(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateDocument = useCallback((documentId: string, updates: Partial<Document>) => {
    setDocuments(prev => prev.map(d => d.id === documentId ? { ...d, ...updates } : d));
    if (selectedDocument?.id === documentId) {
      setSelectedDocument(prev => prev ? { ...prev, ...updates } : null);
    }
  }, [selectedDocument]);

  return {
    // State
    documents,
    selectedDocument,
    chunks,
    entities,
    relationships,
    loading,
    error,
    
    // Actions
    loadDocuments,
    selectDocument,
    createDocument,
    updateDocument,
    setDocuments,
    setSelectedDocument,
    setChunks,
    setEntities,
    setRelationships,
    setError
  };
};