import React, { useState } from 'react';

interface GraphResultPreviewProps {
  result: any;
  query: string;
  onOpen?: () => void;
}

export const GraphResultPreview: React.FC<GraphResultPreviewProps> = ({
  result,
  query,
  onOpen
}) => {
  const [showFullscreen, setShowFullscreen] = useState(false);
  
  // Extract entity information from the result
  const entityName = result.metadata?.entity_name || result.metadata?.entity_type || 'Entity';
  const entityType = result.metadata?.entity_type || 'Unknown';
  const relationships = result.metadata?.relationships || [];
  const confidence = result.metadata?.confidence || result.score;
  
  // Helper to highlight query terms
  const highlightText = (text: string) => {
    if (!query) return text;
    const words = query.toLowerCase().split(' ').filter(w => w.length > 2);
    let highlightedText = text;
    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
    });
    return highlightedText;
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 border-b border-purple-200">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">🕸️</span>
              <h3 className="font-semibold text-gray-900 truncate">
                Graph Result: {entityName}
              </h3>
            </div>
            <div className="flex items-center gap-3 text-xs text-gray-600">
              <span className="px-2 py-1 bg-purple-200 text-purple-800 rounded-full">
                {entityType}
              </span>
              <span>Score: {(confidence * 100).toFixed(1)}%</span>
              <span>•</span>
              <span>{relationships.length} relationships</span>
            </div>
          </div>
          <button
            onClick={() => setShowFullscreen(true)}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium whitespace-nowrap flex-shrink-0"
          >
            Explore Graph
          </button>
        </div>
      </div>

      {/* Query Relevance Section */}
      <div className="p-4 bg-blue-50 border-b border-blue-200">
        <h4 className="text-sm font-semibold text-blue-900 mb-2 flex items-center">
          <span className="mr-2">🎯</span>
          Why This Matches Your Query
        </h4>
        <div className="text-sm text-blue-800">
          <p dangerouslySetInnerHTML={{ __html: highlightText(result.explanation || `This ${entityType} entity matches your search for "${query}" based on its relationships and context within the knowledge graph.`) }} />
        </div>
      </div>

      {/* Main Content - Entity Context */}
      <div className="p-4 border-b border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
          <span className="mr-2">📝</span>
          Entity Context
        </h4>
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-sm text-gray-700" dangerouslySetInnerHTML={{ 
            __html: highlightText(result.content || 'No content available') 
          }} />
        </div>
      </div>

      {/* Relationships Section */}
      {relationships.length > 0 && (
        <div className="p-4 bg-purple-50 border-b border-purple-200">
          <h4 className="text-sm font-semibold text-purple-900 mb-3 flex items-center">
            <span className="mr-2">🔗</span>
            Connected Entities ({relationships.length})
          </h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {relationships.slice(0, 10).map((rel: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between bg-white p-2 rounded-lg border border-purple-200">
                <div className="flex items-center space-x-2 flex-1">
                  <span className="font-medium text-purple-700 text-sm">
                    {rel.source || entityName}
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className="px-2 py-0.5 bg-purple-100 text-purple-600 rounded text-xs font-medium">
                    {rel.type || rel.relationship_type || 'relates to'}
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className="font-medium text-purple-700 text-sm">
                    {rel.target || rel.target_entity}
                  </span>
                </div>
                {rel.weight && (
                  <span className="text-xs text-gray-500 ml-2">
                    {(rel.weight * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
            {relationships.length > 10 && (
              <p className="text-xs text-purple-600 font-medium text-center pt-2">
                +{relationships.length - 10} more relationships
              </p>
            )}
          </div>
        </div>
      )}

      {/* Source Document Section */}
      {result.document_title && (
        <div className="p-4 bg-gray-50 border-b border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
            <span className="mr-2">📄</span>
            Source Document
          </h4>
          <p className="text-sm text-gray-600">{result.document_title}</p>
          {result.chunk_id && (
            <p className="text-xs text-gray-500 mt-1">Chunk ID: {result.chunk_id}</p>
          )}
        </div>
      )}

      {/* Graph Visualization Placeholder */}
      <div className="p-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
          <span className="mr-2">🗺️</span>
          Graph Neighborhood
        </h4>
        <div className="bg-gray-100 rounded-lg p-8 text-center">
          <div className="text-4xl mb-2">🕸️</div>
          <p className="text-sm text-gray-600">
            {relationships.length > 0 
              ? `Connected to ${relationships.length} other entities`
              : 'No direct connections found'}
          </p>
          <button
            onClick={() => setShowFullscreen(true)}
            className="mt-3 text-sm text-purple-600 hover:text-purple-700 font-medium"
          >
            View Interactive Graph →
          </button>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="p-4 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-500">
            Result ID: {result.id?.substring(0, 8)}...
          </div>
          <div className="flex items-center space-x-2">
            <button 
              onClick={(e) => {
                const textContent = `Entity: ${entityName}\nType: ${entityType}\n\nContext:\n${result.content}\n\nRelationships:\n${relationships.map((r: any) => `${r.source} --[${r.type}]--> ${r.target}`).join('\n')}`;
                navigator.clipboard.writeText(textContent);
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
              title="Copy graph data"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
            <button 
              onClick={() => {
                const textContent = `Entity: ${entityName}\nType: ${entityType}\n\nContext:\n${result.content}\n\nRelationships:\n${relationships.map((r: any) => `${r.source} --[${r.type}]--> ${r.target}`).join('\n')}`;
                const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `graph-result-${result.id || 'unknown'}.txt`;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                setTimeout(() => {
                  document.body.removeChild(link);
                  window.URL.revokeObjectURL(url);
                }, 100);
              }}
              className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
              title="Download graph data"
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
              <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="mb-6 pb-4 border-b border-gray-200">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">🕸️ Graph Entity: {entityName}</h2>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full font-medium">
                      {entityType}
                    </span>
                    <span>Score: {(confidence * 100).toFixed(1)}%</span>
                    <span>•</span>
                    <span>{relationships.length} relationships</span>
                  </div>
                </div>
                
                {/* Full Relationships Graph */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Complete Relationship Network</h3>
                  <div className="bg-purple-50 p-6 rounded-lg">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {relationships.map((rel: any, idx: number) => (
                        <div key={idx} className="flex items-center justify-between bg-white p-3 rounded-lg border border-purple-200">
                          <div className="flex items-center space-x-2 flex-1">
                            <span className="font-medium text-purple-700">
                              {rel.source || entityName}
                            </span>
                            <span className="text-gray-400">→</span>
                            <span className="px-2 py-1 bg-purple-100 text-purple-600 rounded text-xs font-medium">
                              {rel.type || 'relates to'}
                            </span>
                            <span className="text-gray-400">→</span>
                            <span className="font-medium text-purple-700">
                              {rel.target}
                            </span>
                          </div>
                          {rel.weight && (
                            <span className="text-sm text-gray-500 ml-2">
                              {(rel.weight * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* Full Context */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Full Entity Context</h3>
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <pre className="whitespace-pre-wrap font-mono text-sm" dangerouslySetInnerHTML={{
                      __html: highlightText(result.content || 'No content available')
                    }} />
                  </div>
                </div>
                
                {/* Query Analysis */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Query Match Analysis</h3>
                  <div className="bg-blue-50 p-6 rounded-lg">
                    <p className="text-blue-800">
                      {result.explanation || `This entity matches your search for "${query}" through its relationships and contextual relevance in the knowledge graph.`}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};