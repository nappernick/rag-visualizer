import React, { useState } from 'react';

interface SearchResult {
  id: string;
  content: string;
  score: number;
  source: string;
  document_id?: string;
  document_title?: string;
  chunk_id?: string;
  metadata?: any;
  explanation?: string;
  highlights?: string[];
}

interface ResultsPanelProps {
  results: SearchResult[];
  loading: boolean;
  onResultSelect: (result: SearchResult) => void;
  selectedResult: SearchResult | null;
  query: string;
}

interface CollapsibleResultProps {
  result: SearchResult;
  index: number;
  isSelected: boolean;
  query: string;
  onSelect: (result: SearchResult) => void;
}

const CollapsibleResult: React.FC<CollapsibleResultProps> = ({
  result,
  index,
  isSelected,
  query,
  onSelect
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showRelationships, setShowRelationships] = useState(false);

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'vector': return '📊';
      case 'graph': return '🕸️';
      case 'hybrid': return '🔀';
      default: return '📄';
    }
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'vector': return 'blue';
      case 'graph': return 'purple';
      case 'hybrid': return 'emerald';
      default: return 'gray';
    }
  };

  const getSourceBgClass = (source: string) => {
    switch (source) {
      case 'vector': return 'bg-blue-500';
      case 'graph': return 'bg-purple-500';
      case 'hybrid': return 'bg-emerald-500';
      default: return 'bg-gray-500';
    }
  };

  const highlightText = (text: string, query: string) => {
    if (!query) return text;
    
    const words = query.toLowerCase().split(' ').filter(w => w.length > 2);
    let highlightedText = text;
    
    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
    });
    
    return highlightedText;
  };

  const truncateContent = (content: string, maxLines: number = 3) => {
    const words = content.split(' ');
    const wordsPerLine = 12; // Approximate words per line
    const maxWords = maxLines * wordsPerLine;
    
    if (words.length <= maxWords) return content;
    return words.slice(0, maxWords).join(' ') + '...';
  };

  const renderRelationships = () => {
    if (!result.metadata?.relationships || result.metadata.relationships.length === 0) {
      return null;
    }

    return (
      <div className="mt-3 p-3 bg-purple-50 rounded-lg border border-purple-200 transition-all duration-300">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-medium text-purple-900 flex items-center">
            <span className="mr-2">🔗</span>
            Entity Relationships ({result.metadata.relationships.length})
          </h4>
          <button
            onClick={() => setShowRelationships(!showRelationships)}
            className="text-purple-600 hover:text-purple-800 text-sm font-medium transition-colors"
          >
            {showRelationships ? 'Hide' : 'Show'}
          </button>
        </div>
        
        <div className={`overflow-hidden transition-all duration-300 ease-in-out ${
          showRelationships ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}>
          <div className="space-y-2 pt-2">
            {result.metadata.relationships.slice(0, 5).map((rel: any, idx: number) => (
              <div key={idx} className="flex items-center space-x-2 text-sm">
                <div className="flex items-center space-x-1 bg-white px-2 py-1 rounded border">
                  <span className="font-medium text-purple-700">{rel.source}</span>
                  <span className="text-gray-400">→</span>
                  <span className="text-xs px-1 py-0.5 bg-purple-100 text-purple-600 rounded">
                    {rel.type}
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className="font-medium text-purple-700">{rel.target}</span>
                  {rel.weight && (
                    <span className="text-xs text-gray-500 ml-1">
                      ({(rel.weight * 100).toFixed(0)}%)
                    </span>
                  )}
                </div>
              </div>
            ))}
            {result.metadata.relationships.length > 5 && (
              <p className="text-xs text-purple-600 font-medium">
                +{result.metadata.relationships.length - 5} more relationships
              </p>
            )}
          </div>
        </div>
      </div>
    );
  };

  const contentToShow = isExpanded ? result.content : truncateContent(result.content);
  const color = getSourceColor(result.source);
  const bgClass = getSourceBgClass(result.source);
  const needsExpansion = result.content.length > truncateContent(result.content).length;

  return (
    <div
      className={`p-4 cursor-pointer transition-all duration-300 hover:bg-gray-50 border-l-4 ${
        isSelected 
          ? `bg-${color}-50 border-${color}-400` 
          : 'border-transparent hover:border-gray-300'
      }`}
      onClick={() => onSelect(result)}
    >
      {/* Result Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className={`w-8 h-8 rounded-full ${bgClass} flex items-center justify-center text-white text-sm font-bold`}>
              #{index + 1}
            </div>
            <div className={`px-3 py-1.5 rounded-full bg-${color}-100 text-${color}-800 text-xs font-semibold flex items-center space-x-1 border border-${color}-200`}>
              <span className="text-base">{getSourceIcon(result.source)}</span>
              <span>{result.source.toUpperCase()}</span>
            </div>
          </div>
          <div className="px-3 py-1.5 rounded-full bg-green-100 text-green-800 text-xs font-semibold border border-green-200">
            <span className="mr-1">⚡</span>
            {(result.score * 100).toFixed(1)}% Match
          </div>
        </div>
        
        {/* Document Info */}
        {result.document_title && (
          <div className="text-xs text-gray-500 flex items-center">
            <span className="mr-1">📄</span>
            {result.document_title}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="mt-3">
        <div className="relative">
          <p 
            className={`text-gray-700 text-sm leading-relaxed transition-all duration-300 ${
              result.source === 'graph' ? 'bg-purple-50 p-3 rounded-lg border border-purple-100' : ''
            }`}
            dangerouslySetInnerHTML={{ 
              __html: highlightText(contentToShow, query) 
            }}
          />
          
          {needsExpansion && (
            <div className="mt-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsExpanded(!isExpanded);
                }}
                className={`text-${color}-600 hover:text-${color}-800 text-sm font-medium flex items-center space-x-1 transition-all duration-200 hover:bg-${color}-50 px-2 py-1 rounded`}
              >
                <span>{isExpanded ? '📖 Show less' : '📚 Show more'}</span>
                <span className={`transform transition-transform duration-200 ${
                  isExpanded ? 'rotate-180' : ''
                }`}>
                  ▼
                </span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Metadata */}
      <div className="mt-3 flex flex-wrap gap-2">
        {result.metadata?.entity_type && (
          <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full border border-purple-200 font-medium">
            <span className="mr-1">🏷️</span>
            {result.metadata.entity_type}
          </span>
        )}
        {result.metadata?.chunk_type && (
          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full border border-blue-200 font-medium">
            <span className="mr-1">📦</span>
            {result.metadata.chunk_type}
          </span>
        )}
        {result.metadata?.position !== undefined && (
          <span className="px-2 py-1 bg-orange-100 text-orange-800 text-xs rounded-full border border-orange-200 font-medium">
            <span className="mr-1">📍</span>
            Position {result.metadata.position}
          </span>
        )}
        {result.metadata?.confidence && (
          <span className="px-2 py-1 bg-teal-100 text-teal-800 text-xs rounded-full border border-teal-200 font-medium">
            <span className="mr-1">🎯</span>
            {(result.metadata.confidence * 100).toFixed(0)}% Confidence
          </span>
        )}
      </div>

      {/* Graph-specific: Relationships */}
      {result.source === 'graph' && renderRelationships()}

      {/* Explanation */}
      {result.explanation && isSelected && (
        <div className="mt-3 p-3 bg-gray-50 rounded-lg border border-gray-200 transition-all duration-300">
          <div className="text-xs font-semibold text-gray-700 mb-2 flex items-center">
            <span className="mr-1">🤔</span>
            Why this matches:
          </div>
          <p className="text-xs text-gray-600 leading-relaxed">{result.explanation}</p>
        </div>
      )}

      {/* Highlights */}
      {result.highlights && result.highlights.length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-gray-700 mb-2 flex items-center">
            <span className="mr-1">💡</span>
            Key phrases:
          </div>
          <div className="flex flex-wrap gap-1">
            {result.highlights.slice(0, 5).map((highlight, idx) => (
              <span key={idx} className="px-2 py-1 bg-yellow-100 text-yellow-900 text-xs rounded border border-yellow-200 font-medium">
                {highlight}
              </span>
            ))}
            {result.highlights.length > 5 && (
              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded border border-gray-200">
                +{result.highlights.length - 5} more
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  results,
  loading,
  onResultSelect,
  selectedResult,
  query
}) => {

  // Group results by source type for better organization
  const groupedResults = results.reduce((acc, result) => {
    if (!acc[result.source]) {
      acc[result.source] = [];
    }
    acc[result.source].push(result);
    return acc;
  }, {} as Record<string, SearchResult[]>);

  const getSourceStats = () => {
    const stats = {
      vector: groupedResults.vector?.length || 0,
      graph: groupedResults.graph?.length || 0,
      hybrid: groupedResults.hybrid?.length || 0
    };
    return stats;
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl p-8 border border-gray-200">
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-5/6"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="bg-white rounded-xl p-8 border border-gray-200">
        <div className="text-center">
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No results yet</h3>
          <p className="text-gray-600">
            Enter a query above to search your documents
          </p>
        </div>
      </div>
    );
  }

  const stats = getSourceStats();

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
      {/* Enhanced Header with Stats */}
      <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <span className="mr-2">🔍</span>
            Search Results ({results.length})
          </h3>
          
          {results.length > 0 && (
            <div className="flex items-center space-x-3">
              {stats.vector > 0 && (
                <div className="flex items-center space-x-1 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                  <span>📊</span>
                  <span>{stats.vector} Vector</span>
                </div>
              )}
              {stats.graph > 0 && (
                <div className="flex items-center space-x-1 px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                  <span>🕸️</span>
                  <span>{stats.graph} Graph</span>
                </div>
              )}
              {stats.hybrid > 0 && (
                <div className="flex items-center space-x-1 px-2 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs font-medium">
                  <span>🔀</span>
                  <span>{stats.hybrid} Hybrid</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Results List */}
      <div className="divide-y divide-gray-200 max-h-[700px] overflow-y-auto">
        {results.map((result, index) => {
          const isSelected = selectedResult?.id === result.id;
          
          return (
            <CollapsibleResult
              key={result.id}
              result={result}
              index={index}
              isSelected={isSelected}
              query={query}
              onSelect={onResultSelect}
            />
          );
        })}
      </div>
    </div>
  );
};