import React from 'react';
import { ChunkVisualizer } from '../ChunkVisualizer/ChunkVisualizer';
import type { Chunk, Document } from '../../types';

interface ChunksTabProps {
  chunks: Chunk[];
  allChunks: {[docId: string]: Chunk[]};
  viewMode: 'single' | 'all';
  setViewMode: (mode: 'single' | 'all') => void;
  selectedChunkId?: string;
  onChunkSelect: (chunkId: string) => void;
  onRefresh: () => void;
  documents: Document[];
}

export const ChunksTab: React.FC<ChunksTabProps> = ({
  chunks,
  allChunks,
  viewMode,
  setViewMode,
  selectedChunkId,
  onChunkSelect,
  onRefresh,
  documents
}) => {
  // Calculate total chunks
  const totalChunks = viewMode === 'single' 
    ? chunks.length 
    : Object.values(allChunks).reduce((sum, c) => sum + c.length, 0);

  // Find selected chunk details
  const getSelectedChunkIndex = () => {
    if (!selectedChunkId) return 'None';
    
    const allChunksList = viewMode === 'single' 
      ? chunks 
      : Object.values(allChunks).flat();
    const idx = allChunksList.findIndex(c => c.id === selectedChunkId);
    return idx >= 0 ? `#${idx + 1}` : 'None';
  };

  // Get selected chunk content
  const getSelectedChunkContent = () => {
    if (!selectedChunkId) return null;
    
    // In single document mode, search in chunks
    if (viewMode === 'single') {
      return chunks.find(c => c.id === selectedChunkId)?.content;
    }
    
    // In all documents mode, search across all chunks
    for (const docChunks of Object.values(allChunks)) {
      const chunk = docChunks.find(c => c.id === selectedChunkId);
      if (chunk) return chunk.content;
    }
    return 'Chunk not found';
  };

  const selectedContent = getSelectedChunkContent();

  return (
    <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center">
          <span className="text-3xl mr-3">✂️</span>
          Chunk Visualization
        </h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setViewMode('single')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'single' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Current Document
          </button>
          <button
            onClick={() => setViewMode('all')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'all' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            All Documents
          </button>
          {viewMode === 'all' && (
            <button
              onClick={onRefresh}
              className="px-3 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
              title="Refresh all documents data"
            >
              🔄 Refresh
            </button>
          )}
        </div>
      </div>

      {/* Empty State */}
      {viewMode === 'single' && chunks.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <span className="text-6xl block mb-4">📄</span>
          <p className="text-lg">No chunks to display</p>
          <p className="text-sm mt-2">Please select a document first</p>
        </div>
      ) : viewMode === 'all' && (Object.keys(allChunks).length === 0 || Object.values(allChunks).every(chunks => chunks.length === 0)) ? (
        <div className="text-center py-12 text-gray-500">
          <span className="text-6xl block mb-4">📚</span>
          <p className="text-lg">No chunks in the corpus</p>
          <p className="text-sm mt-2">Upload and process documents to see chunks</p>
        </div>
      ) : (
        <div>
          {/* Statistics Bar */}
          <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6">
                <div>
                  <span className="text-sm text-gray-600">Total Chunks</span>
                  <p className="text-2xl font-bold text-green-600">{totalChunks}</p>
                </div>
                {viewMode === 'all' && (
                  <>
                    <div className="h-8 w-px bg-gray-300"></div>
                    <div>
                      <span className="text-sm text-gray-600">Documents</span>
                      <p className="text-lg font-semibold text-gray-800">{Object.keys(allChunks).length}</p>
                    </div>
                  </>
                )}
                <div className="h-8 w-px bg-gray-300"></div>
                <div>
                  <span className="text-sm text-gray-600">Selected</span>
                  <p className="text-lg font-semibold text-gray-800">{getSelectedChunkIndex()}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Chunk Visualizers */}
          {viewMode === 'single' ? (
            <ChunkVisualizer
              chunks={chunks}
              selectedChunkId={selectedChunkId}
              onChunkSelect={onChunkSelect}
            />
          ) : (
            <div className="space-y-6">
              {Object.entries(allChunks).map(([docId, docChunks]) => {
                const doc = documents.find(d => d.id === docId);
                return (
                  <div key={docId} className="border rounded-lg p-4">
                    <h3 className="font-semibold text-lg mb-3">
                      {doc?.title || 'Unknown Document'}
                      <span className="text-sm text-gray-500 ml-2">({docChunks.length} chunks)</span>
                    </h3>
                    <ChunkVisualizer
                      chunks={docChunks}
                      selectedChunkId={selectedChunkId}
                      onChunkSelect={onChunkSelect}
                    />
                  </div>
                );
              })}
            </div>
          )}

          {/* Selected Chunk Content */}
          {selectedChunkId && selectedContent && (
            <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
              <h3 className="font-semibold text-lg mb-3 flex items-center">
                <span className="text-2xl mr-2">📝</span>
                Selected Chunk Content
              </h3>
              <pre className="text-sm whitespace-pre-wrap bg-white p-4 rounded-lg border border-gray-200 max-h-64 overflow-y-auto">
                {selectedContent}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};