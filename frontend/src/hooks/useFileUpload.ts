import { useState, useCallback } from 'react';
import { documentApi, chunkingApi, graphApi } from '../services/api';
import type { Document } from '../types';

interface UploadProgress {
  status: string;
  metrics: any;
}

export const useFileUpload = (
  onDocumentCreated?: (doc: Document) => void,
  onDocumentsUpdate?: (docs: Document[]) => void
) => {
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null);
  const [uploading, setUploading] = useState(false);

  const uploadFiles = useCallback(async (files: FileList) => {
    if (!files || files.length === 0) return [];

    try {
      setUploading(true);
      setPerformanceMetrics(null);
      
      const totalFiles = files.length;
      const newDocuments: Document[] = [];
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileNum = `${i + 1}/${totalFiles}`;
        
        // Stage 1: Reading file
        setProcessingStatus(`📂 Reading file ${fileNum}: ${file.name}`);
        
        // Stage 2: Processing (includes upload, chunking, extraction)
        setProcessingStatus(`⚙️ Processing ${fileNum}: ${file.name}`);
        const uploadResponse = await documentApi.upload(file);
        
        // Extract performance metrics if available
        if (uploadResponse.performance) {
          setPerformanceMetrics(uploadResponse.performance);
          
          // Update status based on current stage from server
          if (uploadResponse.performance.chunking) {
            setProcessingStatus(`✂️ Chunking ${fileNum}: ${file.name} (${uploadResponse.performance.chunk_count} chunks)`);
          }
          if (uploadResponse.performance.entity_extraction) {
            setProcessingStatus(`🔍 Extracting entities ${fileNum}: ${file.name} (${uploadResponse.performance.entity_count} entities)`);
          }
        }
        
        newDocuments.push(uploadResponse);
        
        // Notify about document creation
        if (onDocumentCreated) {
          onDocumentCreated(uploadResponse);
        }
        
        // Select the first uploaded document and load its data
        if (i === 0) {
          try {
            const [docChunks, docEntities, docRelationships] = await Promise.all([
              chunkingApi.getChunks(uploadResponse.id).catch(() => []),
              graphApi.getEntities(uploadResponse.id).catch(() => []),
              graphApi.getRelationships(uploadResponse.id).catch(() => [])
            ]);
            
            // Update document status to completed
            const updatedDoc = { ...uploadResponse, status: 'completed' as const };
            if (onDocumentsUpdate) {
              onDocumentsUpdate([updatedDoc]);
            }
          } catch (e) {
            console.error('Error loading document data:', e);
          }
        }
      }
      
      // Show performance summary if available
      if (performanceMetrics) {
        const perf = performanceMetrics;
        setProcessingStatus(
          `✅ Processed in ${perf.total_time}s | ` +
          `Chunks: ${perf.chunk_count || 0} | ` +
          `Entities: ${perf.entity_count || 0} | ` +
          `Relationships: ${perf.relationship_count || 0}`
        );
      } else {
        setProcessingStatus(`✅ Successfully processed ${totalFiles} document${totalFiles > 1 ? 's' : ''}!`);
      }
      
      // Clear status after delay
      setTimeout(() => {
        setProcessingStatus('');
        setPerformanceMetrics(null);
      }, 5000);
      
      return newDocuments;
      
    } catch (err: any) {
      const errorMessage = err?.response?.data?.detail || err?.message || 'Failed to process document';
      setProcessingStatus(`❌ Error: ${errorMessage}`);
      console.error('Document processing error:', err);
      throw err;
    } finally {
      setUploading(false);
    }
  }, [onDocumentCreated, onDocumentsUpdate]);

  const clearStatus = useCallback(() => {
    setProcessingStatus('');
    setPerformanceMetrics(null);
  }, []);

  return {
    uploadFiles,
    processingStatus,
    performanceMetrics,
    uploading,
    clearStatus
  };
};