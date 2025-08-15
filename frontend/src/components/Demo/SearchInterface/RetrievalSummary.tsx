import React from 'react';

interface RetrievalSummaryProps {
  results: any[];
  query: string;
  processingTime?: number;
}

export const RetrievalSummary: React.FC<RetrievalSummaryProps> = ({
  results,
  query,
  processingTime
}) => {
  // Calculate statistics
  const vectorResults = results.filter(r => r.source === 'vector');
  const graphResults = results.filter(r => r.source === 'graph');
  const hybridResults = results.filter(r => r.source === 'hybrid');
  
  const avgScore = results.length > 0
    ? results.reduce((sum, r) => sum + r.score, 0) / results.length
    : 0;
  
  const uniqueDocuments = new Set(results.map(r => r.document_id).filter(Boolean)).size;
  const uniqueEntities = new Set(
    results
      .filter(r => r.metadata?.entity_id)
      .map(r => r.metadata.entity_id)
  ).size;

  return (
    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-4 border border-indigo-200 mb-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900 flex items-center">
          <span className="mr-2">📊</span>
          Retrieval Summary
        </h3>
        {processingTime && (
          <span className="text-xs text-gray-500">
            {processingTime.toFixed(0)}ms
          </span>
        )}
      </div>
      
      {/* Visual Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
        {/* Vector Results */}
        {vectorResults.length > 0 && (
          <div className="bg-white rounded-lg p-2 border border-blue-200">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Chunks</span>
              <span className="text-lg font-bold text-blue-600">{vectorResults.length}</span>
            </div>
            <div className="mt-1">
              <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${(vectorResults.length / results.length) * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}
        
        {/* Graph Results */}
        {graphResults.length > 0 && (
          <div className="bg-white rounded-lg p-2 border border-purple-200">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Entities</span>
              <span className="text-lg font-bold text-purple-600">{graphResults.length}</span>
            </div>
            <div className="mt-1">
              <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-500 transition-all duration-500"
                  style={{ width: `${(graphResults.length / results.length) * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}
        
        {/* Average Score */}
        <div className="bg-white rounded-lg p-2 border border-green-200">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Avg Match</span>
            <span className="text-lg font-bold text-green-600">
              {(avgScore * 100).toFixed(0)}%
            </span>
          </div>
          <div className="mt-1">
            <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-green-500 transition-all duration-500"
                style={{ width: `${avgScore * 100}%` }}
              />
            </div>
          </div>
        </div>
        
        {/* Coverage */}
        <div className="bg-white rounded-lg p-2 border border-orange-200">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Sources</span>
            <span className="text-lg font-bold text-orange-600">
              {uniqueDocuments || uniqueEntities}
            </span>
          </div>
          <div className="mt-1 flex space-x-1">
            {uniqueDocuments > 0 && (
              <span className="text-xs bg-orange-100 text-orange-700 px-1 rounded">
                {uniqueDocuments} docs
              </span>
            )}
            {uniqueEntities > 0 && (
              <span className="text-xs bg-purple-100 text-purple-700 px-1 rounded">
                {uniqueEntities} entities
              </span>
            )}
          </div>
        </div>
      </div>
      
      {/* Retrieval Strategy Breakdown */}
      <div className="flex items-center space-x-2 text-xs">
        <span className="text-gray-500">Strategy:</span>
        <div className="flex items-center space-x-1">
          {vectorResults.length > 0 && (
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
              Vector: {((vectorResults.length / results.length) * 100).toFixed(0)}%
            </span>
          )}
          {graphResults.length > 0 && (
            <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full">
              Graph: {((graphResults.length / results.length) * 100).toFixed(0)}%
            </span>
          )}
          {hybridResults.length > 0 && (
            <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full">
              Hybrid: {((hybridResults.length / results.length) * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
};