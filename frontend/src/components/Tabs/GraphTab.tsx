import React from 'react';
import { GraphViewer } from '../GraphViewer/GraphViewer';
import type { Entity, Relationship } from '../../types';

interface GraphTabProps {
  entities: Entity[];
  relationships: Relationship[];
  allEntities: {[docId: string]: Entity[]};
  allRelationships: {[docId: string]: Relationship[]};
  viewMode: 'single' | 'all';
  setViewMode: (mode: 'single' | 'all') => void;
  selectedEntityId?: string;
  onEntitySelect: (entityId: string) => void;
  onRefresh: () => void;
  loading: boolean;
}

export const GraphTab: React.FC<GraphTabProps> = ({
  entities,
  relationships,
  allEntities,
  allRelationships,
  viewMode,
  setViewMode,
  selectedEntityId,
  onEntitySelect,
  onRefresh,
  loading
}) => {
  // Calculate totals
  const totalEntities = viewMode === 'single' 
    ? entities.length 
    : Object.values(allEntities).reduce((sum, e) => sum + e.length, 0);
  
  const totalRelationships = viewMode === 'single' 
    ? relationships.length 
    : Object.values(allRelationships).reduce((sum, r) => sum + r.length, 0);

  // Handle linking documents
  const handleLinkDocuments = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/graph/link-documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const result = await response.json();
      if (response.ok) {
        alert(`Successfully linked documents!\n${result.cross_relationships} cross-document relationships created\n${result.entity_matches} entity matches found`);
        await onRefresh();
      } else {
        alert('Failed to link documents: ' + result.detail);
      }
    } catch (error) {
      console.error('Error linking documents:', error);
      alert('Error linking documents');
    }
  };

  // Get selected entity
  const getSelectedEntity = () => {
    if (!selectedEntityId) return null;
    
    if (viewMode === 'single') {
      return entities.find(e => e.id === selectedEntityId);
    }
    
    // Search across all entities
    for (const docEntities of Object.values(allEntities)) {
      const entity = docEntities.find(e => e.id === selectedEntityId);
      if (entity) return entity;
    }
    return null;
  };

  const selectedEntity = getSelectedEntity();

  return (
    <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center">
          <span className="text-3xl mr-3">🔮</span>
          Knowledge Graph
        </h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setViewMode('single')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'single' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Current Document
          </button>
          <button
            onClick={() => setViewMode('all')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'all' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            All Documents
          </button>
          {viewMode === 'all' && (
            <>
              <button
                onClick={onRefresh}
                className="px-3 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
                title="Refresh all documents data"
              >
                🔄 Refresh
              </button>
              <button
                onClick={handleLinkDocuments}
                className="px-3 py-2 rounded-lg bg-purple-600 text-white hover:bg-purple-700 transition-colors"
                title="Link entities across all documents"
                disabled={loading}
              >
                🔗 Link Graphs
              </button>
            </>
          )}
        </div>
      </div>

      {/* Empty State */}
      {viewMode === 'single' && entities.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <span className="text-6xl block mb-4">🕸️</span>
          <p className="text-lg">No graph data available</p>
          <p className="text-sm mt-2">Process a document first to see the knowledge graph</p>
        </div>
      ) : viewMode === 'all' && (Object.keys(allEntities).length === 0 || Object.values(allEntities).every(entities => entities.length === 0)) ? (
        <div className="text-center py-12 text-gray-500">
          <span className="text-6xl block mb-4">🌐</span>
          <p className="text-lg">No graph data in the corpus</p>
          <p className="text-sm mt-2">Process documents to see knowledge graphs</p>
        </div>
      ) : (
        <div>
          {/* Statistics Bar */}
          <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6">
                <div>
                  <span className="text-sm text-gray-600">Entities</span>
                  <p className="text-2xl font-bold text-purple-600">{totalEntities}</p>
                </div>
                <div className="h-8 w-px bg-gray-300"></div>
                <div>
                  <span className="text-sm text-gray-600">Relationships</span>
                  <p className="text-2xl font-bold text-pink-600">{totalRelationships}</p>
                </div>
                {viewMode === 'all' && (
                  <>
                    <div className="h-8 w-px bg-gray-300"></div>
                    <div>
                      <span className="text-sm text-gray-600">Documents</span>
                      <p className="text-lg font-semibold text-gray-800">{Object.keys(allEntities).length}</p>
                    </div>
                  </>
                )}
                <div className="h-8 w-px bg-gray-300"></div>
                <div>
                  <span className="text-sm text-gray-600">Selected</span>
                  <p className="text-lg font-semibold text-gray-800">{selectedEntityId || 'None'}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Graph Viewer */}
          {viewMode === 'single' ? (
            <div style={{ height: '600px' }}>
              <GraphViewer
                entities={entities}
                relationships={relationships}
                selectedEntityId={selectedEntityId}
                onNodeSelect={onEntitySelect}
              />
            </div>
          ) : (
            <div style={{ height: '800px' }} className="border rounded-lg p-4">
              <h3 className="font-semibold text-lg mb-3">Combined Knowledge Graph</h3>
              <GraphViewer
                entities={Object.values(allEntities).flat()}
                relationships={Object.values(allRelationships).flat()}
                selectedEntityId={selectedEntityId}
                onNodeSelect={onEntitySelect}
              />
            </div>
          )}

          {/* Selected Entity Details */}
          {selectedEntityId && selectedEntity && (
            <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
              <h3 className="font-semibold text-lg mb-3 flex items-center">
                <span className="text-2xl mr-2">🎯</span>
                Selected Entity
              </h3>
              <pre className="text-sm bg-white p-4 rounded-lg border border-gray-200 max-h-64 overflow-y-auto">
                {JSON.stringify(selectedEntity, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};