import React, { useState, useEffect } from 'react';
import type { Document, Chunk } from '../../../types';
import { chunkingApi } from '../../../services/api';

interface DocumentPreviewProps {
  document: Document;
  highlightedChunk?: string;
  onOpen: () => void;
}

export const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  document,
  highlightedChunk,
  onOpen
}) => {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedView, setExpandedView] = useState(false);

  useEffect(() => {
    const loadChunks = async () => {
      if (!document) return;
      
      setLoading(true);
      try {
        const docChunks = await chunkingApi.getChunks(document.id);
        setChunks(Array.isArray(docChunks) ? docChunks : []);
      } catch (error) {
        console.error('Error loading chunks:', error);
        setChunks([]);
      } finally {
        setLoading(false);
      }
    };

    loadChunks();
  }, [document]);

  const getHighlightedChunkIndex = () => {
    if (!highlightedChunk) return -1;
    return chunks.findIndex(c => c.id === highlightedChunk);
  };

  const formatContent = (content: string) => {
    // Truncate for preview unless expanded
    const maxLength = expandedView ? 10000 : 500;
    const truncated = content.length > maxLength 
      ? content.substring(0, maxLength) + '...' 
      : content;
    
    // Convert line breaks to HTML
    return truncated.split('\n').map((line, i) => (
      <span key={i}>
        {line}
        {i < truncated.split('\n').length - 1 && <br />}
      </span>
    ));
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-4 border-b border-gray-200">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 truncate">
              {document.title}
            </h3>
            <div className="mt-1 flex items-center space-x-3 text-xs text-gray-500">
              <span>ID: {document.id.substring(0, 8)}...</span>
              <span>•</span>
              <span>{chunks.length} chunks</span>
              <span>•</span>
              <span>{document.doc_type || 'text'}</span>
            </div>
          </div>
          <button
            onClick={onOpen}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
          >
            Open Full
          </button>
        </div>
      </div>

      {/* Document Stats */}
      <div className="p-4 bg-gray-50 border-b border-gray-200">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xl font-bold text-gray-700">
              {document.content?.length || 0}
            </div>
            <div className="text-xs text-gray-500">Characters</div>
          </div>
          <div>
            <div className="text-xl font-bold text-blue-600">
              {chunks.length}
            </div>
            <div className="text-xs text-gray-500">Chunks</div>
          </div>
          <div>
            <div className="text-xl font-bold text-green-600">
              {document.status === 'completed' ? '✓' : '○'}
            </div>
            <div className="text-xs text-gray-500">Status</div>
          </div>
        </div>
      </div>

      {/* Content Preview */}
      <div className="p-4 max-h-96 overflow-y-auto">
        {loading ? (
          <div className="animate-pulse space-y-2">
            <div className="h-3 bg-gray-200 rounded w-full"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6"></div>
            <div className="h-3 bg-gray-200 rounded w-4/6"></div>
          </div>
        ) : (
          <div>
            {highlightedChunk && getHighlightedChunkIndex() >= 0 && (
              <div className="mb-4 p-3 bg-yellow-50 border-l-4 border-yellow-400 rounded">
                <div className="text-xs font-medium text-yellow-800 mb-1">
                  Highlighted Chunk #{getHighlightedChunkIndex() + 1}
                </div>
                <p className="text-sm text-gray-700">
                  {chunks[getHighlightedChunkIndex()]?.content}
                </p>
              </div>
            )}
            
            <div className="prose prose-sm max-w-none">
              <div className="text-xs font-medium text-gray-500 mb-2">
                Document Preview {!expandedView && '(First 500 chars)'}
              </div>
              <div className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
                {formatContent(document.content || 'No content available')}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-4 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setExpandedView(!expandedView)}
            className="text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            {expandedView ? 'Show Less' : 'Show More'}
          </button>
          <div className="flex items-center space-x-2">
            <button className="p-2 text-gray-500 hover:text-gray-700">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
            <button className="p-2 text-gray-500 hover:text-gray-700">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
            <button className="p-2 text-gray-500 hover:text-gray-700">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m9.032 4.026A9.001 9.001 0 1112 3c4.243 0 7.771 2.936 8.716 6.893M19.5 12.5l-2.5 1.5V9" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};