import React, { useState } from 'react';
import type { Entity, Relationship, Chunk, Document } from '../../../types';
import { demoApi } from '../../../services/demoApi';

interface KnowledgeNavigatorProps {
  entities: Entity[];
  relationships: Relationship[];
  chunks: Chunk[];
  documents: Document[];
  persistentState?: any;
  onStateChange?: (state: any) => void;
}

export const KnowledgeNavigator: React.FC<KnowledgeNavigatorProps> = ({
  entities,
  relationships,
  chunks,
  documents,
  persistentState,
  onStateChange
}) => {
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(persistentState?.selectedEntity || null);
  const [explorationResults, setExplorationResults] = useState<any>(persistentState?.explorationResults || null);
  const [pathFindingMode, setPathFindingMode] = useState(persistentState?.pathFindingMode || false);
  const [startEntity, setStartEntity] = useState<Entity | null>(persistentState?.startEntity || null);
  const [endEntity, setEndEntity] = useState<Entity | null>(persistentState?.endEntity || null);
  const [pathResults, setPathResults] = useState<any>(persistentState?.pathResults || null);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState(persistentState?.searchTerm || '');

  const filteredEntities = entities.filter(e =>
    e.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    e.entity_type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleEntitySelect = async (entity: Entity) => {
    if (pathFindingMode) {
      // In path finding mode, allow selecting start and end entities
      if (!startEntity) {
        setStartEntity(entity);
        setSelectedEntity(entity);
      } else if (!endEntity && entity.id !== startEntity.id) {
        setEndEntity(entity);
      } else {
        // Reset and start over
        setStartEntity(entity);
        setEndEntity(null);
        setPathResults(null);
        setSelectedEntity(entity);
      }
    } else {
      // Normal mode - explore from this entity
      setSelectedEntity(entity);
      setLoading(true);
      
      try {
        const results = await demoApi.exploreGraph([entity.id], 2, 10);
        setExplorationResults(results);
      } catch (error) {
        console.error('Error exploring graph:', error);
        setExplorationResults(null);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleFindPath = async () => {
    if (!startEntity || !endEntity) return;
    
    setLoading(true);
    try {
      const results = await demoApi.findPath(startEntity.id, endEntity.id, 3);
      setPathResults(results);
    } catch (error) {
      console.error('Error finding path:', error);
      setPathResults(null);
    } finally {
      setLoading(false);
    }
  };
  
  const resetPathFinding = () => {
    setStartEntity(null);
    setEndEntity(null);
    setPathResults(null);
    setPathFindingMode(false);
  };

  const getEntityTypeColor = (type: string) => {
    const colors: { [key: string]: string } = {
      'person': 'blue',
      'organization': 'green',
      'location': 'purple',
      'concept': 'orange',
      'technology': 'indigo',
      'default': 'gray'
    };
    return colors[type.toLowerCase()] || colors.default;
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl p-6 border border-purple-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Knowledge Graph Explorer</h3>
          <button
            onClick={() => setPathFindingMode(!pathFindingMode)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              pathFindingMode
                ? 'bg-purple-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            {pathFindingMode ? 'Exit Path Finding' : 'Find Path'}
          </button>
        </div>

        {pathFindingMode ? (
          <div className="space-y-3">
            <div className="text-sm text-gray-600 mb-2">
              Click entities below to select start and end points for path finding
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-white border-2 border-purple-300 rounded-lg">
                <div className="text-xs font-medium text-purple-600 mb-1">Start Entity</div>
                {startEntity ? (
                  <div className="font-medium text-gray-900">{startEntity.name}</div>
                ) : (
                  <div className="text-gray-400">Click to select...</div>
                )}
              </div>
              <div className="p-3 bg-white border-2 border-purple-300 rounded-lg">
                <div className="text-xs font-medium text-purple-600 mb-1">End Entity</div>
                {endEntity ? (
                  <div className="font-medium text-gray-900">{endEntity.name}</div>
                ) : (
                  <div className="text-gray-400">Click to select...</div>
                )}
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={handleFindPath}
                disabled={!startEntity || !endEntity || loading}
                className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 transition-colors"
              >
                {loading ? 'Finding Path...' : 'Find Connection'}
              </button>
              <button
                onClick={resetPathFinding}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
              >
                Reset
              </button>
            </div>
          </div>
        ) : (
          <div className="relative">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search entities..."
              className="w-full px-4 py-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <span className="absolute left-3 top-2.5 text-gray-400">🔍</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Entity List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="text-md font-semibold text-gray-900 mb-3">
              Entities ({filteredEntities.length})
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredEntities.slice(0, 50).map((entity) => {
                const color = getEntityTypeColor(entity.entity_type);
                return (
                  <div
                    key={entity.id}
                    onClick={() => handleEntitySelect(entity)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                      selectedEntity?.id === entity.id
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{entity.name}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full bg-${color}-100 text-${color}-700`}>
                            {entity.entity_type}
                          </span>
                          <span className="ml-2">Freq: {entity.frequency}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Graph Visualization / Results */}
        <div className="lg:col-span-2">
          {pathFindingMode && pathResults ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Path Results</h3>
              <div className="space-y-4">
                {pathResults.paths && pathResults.paths.map((path: any, idx: number) => (
                  <div key={idx} className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      {path.nodes.map((node: string, nodeIdx: number) => (
                        <React.Fragment key={nodeIdx}>
                          <div className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                            {node}
                          </div>
                          {nodeIdx < path.nodes.length - 1 && (
                            <span className="text-gray-400">→</span>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                    <div className="mt-2 text-xs text-gray-500">
                      Path length: {path.length} • Relationships: {path.relationships?.join(', ')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : selectedEntity && explorationResults ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Exploration from: {selectedEntity.name}
              </h3>
              
              {loading ? (
                <div className="animate-pulse space-y-3">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-4 bg-gray-200 rounded w-full"></div>
                  <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Related Entities */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">
                      Related Entities ({explorationResults.entities?.length || 0})
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {explorationResults.entities?.slice(0, 10).map((entity: any, idx: number) => {
                        const color = getEntityTypeColor(entity.entity_type || entity.type);
                        return (
                          <span
                            key={idx}
                            className={`px-3 py-1 bg-${color}-100 text-${color}-700 rounded-full text-sm`}
                          >
                            {entity.name}
                          </span>
                        );
                      })}
                    </div>
                  </div>

                  {/* Analysis */}
                  {explorationResults.analysis && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Insights</h4>
                      <p className="text-sm text-gray-600">{explorationResults.analysis.insights}</p>
                      
                      {explorationResults.analysis.patterns?.length > 0 && (
                        <div className="mt-3">
                          <h5 className="text-xs font-medium text-gray-500 mb-1">Patterns Found:</h5>
                          <ul className="space-y-1">
                            {explorationResults.analysis.patterns.map((pattern: string, idx: number) => (
                              <li key={idx} className="text-xs text-gray-600 flex items-start">
                                <span className="text-purple-500 mr-2">•</span>
                                {pattern}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {explorationResults.analysis.next_steps?.length > 0 && (
                        <div className="mt-3">
                          <h5 className="text-xs font-medium text-gray-500 mb-1">Suggested Next Steps:</h5>
                          <ul className="space-y-1">
                            {explorationResults.analysis.next_steps.map((step: string, idx: number) => (
                              <li key={idx} className="text-xs text-gray-600 flex items-start">
                                <span className="text-green-500 mr-2">→</span>
                                {step}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="bg-gray-50 rounded-xl p-12 text-center border-2 border-dashed border-gray-300">
              <div className="text-gray-400 text-6xl mb-4">🧭</div>
              <p className="text-gray-600">
                {pathFindingMode
                  ? 'Enter two entities to find the connection between them'
                  : 'Select an entity to explore its relationships'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};