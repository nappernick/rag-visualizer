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
  const [showFullscreen, setShowFullscreen] = useState(false);

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
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-gray-900 truncate pr-2">
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
            onClick={() => setShowFullscreen(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium whitespace-nowrap flex-shrink-0"
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
            <button 
              onClick={(e) => {
                navigator.clipboard.writeText(document.content || '');
                // Show feedback
                const btn = e.currentTarget;
                const originalTitle = btn.title;
                btn.title = 'Copied!';
                btn.classList.add('text-green-600');
                setTimeout(() => {
                  btn.title = originalTitle;
                  btn.classList.remove('text-green-600');
                }, 1500);
              }}
              className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
              title="Copy content"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
            <button 
              onClick={(e) => {
                e.preventDefault();
                const content = document.content || '';
                const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `${document.title || 'document'}.txt`;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                setTimeout(() => {
                  document.body.removeChild(link);
                  window.URL.revokeObjectURL(url);
                }, 100);
              }}
              className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
              title="Download document"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Fullscreen Modal */}
      {showFullscreen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowFullscreen(false);
            }
          }}
        >
          <div className="fixed inset-0 bg-black bg-opacity-50" onClick={() => setShowFullscreen(false)} />
          <div className="relative bg-white rounded-lg shadow-2xl" style={{ width: '95vw', height: '95vh', maxWidth: '1400px' }}>
            <div className="absolute top-4 right-4 z-10">
              <button
                onClick={() => setShowFullscreen(false)}
                className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-full transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="p-8 h-full overflow-y-auto">
              <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-6 pb-4 border-b border-gray-200">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">{document.title}</h2>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span>📄 Document ID: {document.id}</span>
                    <span>•</span>
                    <span>Type: {document.doc_type || 'text'}</span>
                    <span>•</span>
                    <span>{chunks.length} chunks</span>
                    <span>•</span>
                    <span>{document.content?.length || 0} characters</span>
                  </div>
                </div>
                
                {/* Content */}
                <div className="prose prose-lg max-w-none mb-6">
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <pre className="whitespace-pre-wrap font-mono text-sm">{document.content || 'No content available'}</pre>
                  </div>
                </div>
                
                {/* Chunks Section */}
                {chunks.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Chunks</h3>
                    <div className="space-y-3">
                      {chunks.map((chunk, idx) => (
                        <div key={chunk.id} className="bg-white border border-gray-200 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-gray-700">Chunk #{idx + 1}</span>
                            <span className="text-xs text-gray-500">ID: {chunk.id.substring(0, 8)}...</span>
                          </div>
                          <p className="text-sm text-gray-600 whitespace-pre-wrap">{chunk.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Actions */}
                <div className="mt-8 pt-4 border-t border-gray-200 flex items-center gap-3">
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(document.content || '');
                      const btn = event?.currentTarget as HTMLElement;
                      if (btn) {
                        const originalText = btn.textContent;
                        btn.textContent = '✅ Copied!';
                        setTimeout(() => { btn.textContent = originalText || ''; }, 1500);
                      }
                    }}
                    className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md transition-colors"
                  >
                    📋 Copy Full Content
                  </button>
                  
                  <button
                    onClick={() => {
                      const content = document.content || '';
                      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
                      const url = window.URL.createObjectURL(blob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = `${document.title || 'document'}-full.txt`;
                      link.style.display = 'none';
                      document.body.appendChild(link);
                      link.click();
                      setTimeout(() => {
                        document.body.removeChild(link);
                        window.URL.revokeObjectURL(url);
                      }, 100);
                    }}
                    className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-md transition-colors"
                  >
                    💾 Download Document
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};