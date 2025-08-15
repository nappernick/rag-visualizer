import { useState, useCallback } from 'react';
import { chunkingApi, graphApi } from '../services/api';
import type { Document, Chunk, Entity, Relationship } from '../types';

export const useDataLoader = () => {
  const [allChunks, setAllChunks] = useState<{[docId: string]: Chunk[]}>({});
  const [allEntities, setAllEntities] = useState<{[docId: string]: Entity[]}>({});
  const [allRelationships, setAllRelationships] = useState<{[docId: string]: Relationship[]}>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAllData = useCallback(async (documents: Document[]) => {
    try {
      setLoading(true);
      setError(null);
      
      const newAllChunks: {[key: string]: Chunk[]} = {};
      const newAllEntities: {[key: string]: Entity[]} = {};
      const newAllRelationships: {[key: string]: Relationship[]} = {};
      
      for (const doc of documents) {
        try {
          const docChunks = await chunkingApi.getChunks(doc.id);
          newAllChunks[doc.id] = Array.isArray(docChunks) ? docChunks : [];
          
          const [docEntities, docRelationships] = await Promise.all([
            graphApi.getEntities(doc.id).catch(() => []),
            graphApi.getRelationships(doc.id).catch(() => [])
          ]);
          newAllEntities[doc.id] = docEntities;
          newAllRelationships[doc.id] = docRelationships;
        } catch (err) {
          console.error(`Failed to load data for document ${doc.id}:`, err);
        }
      }
      
      setAllChunks(newAllChunks);
      setAllEntities(newAllEntities);
      setAllRelationships(newAllRelationships);
      
      return {
        chunks: newAllChunks,
        entities: newAllEntities,
        relationships: newAllRelationships
      };
    } catch (err) {
      setError('Failed to load all data');
      console.error(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearAllData = useCallback(() => {
    setAllChunks({});
    setAllEntities({});
    setAllRelationships({});
    setError(null);
  }, []);

  return {
    // State
    allChunks,
    allEntities,
    allRelationships,
    loading,
    error,
    
    // Actions
    loadAllData,
    clearAllData,
    setAllChunks,
    setAllEntities,
    setAllRelationships
  };
};