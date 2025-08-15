import React from 'react';

interface SearchResult {
  id: string;
  content: string;
  score: number;
  source: string;
  document_id?: string;
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

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  results,
  loading,
  onResultSelect,
  selectedResult,
  query
}) => {
  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'vector': return '🔢';
      case 'graph': return '🕸️';
      case 'hybrid': return '🔀';
      default: return '📄';
    }
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'vector': return 'blue';
      case 'graph': return 'purple';
      case 'hybrid': return 'green';
      default: return 'gray';
    }
  };

  const highlightText = (text: string, query: string) => {
    if (!query) return text;
    
    const words = query.toLowerCase().split(' ').filter(w => w.length > 2);
    let highlightedText = text;
    
    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200">$1</mark>');
    });
    
    return highlightedText;
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

  return (
    <div className="bg-white rounded-xl border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">
          Search Results ({results.length})
        </h3>
      </div>
      
      <div className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
        {results.map((result, index) => {
          const color = getSourceColor(result.source);
          const isSelected = selectedResult?.id === result.id;
          
          return (
            <div
              key={result.id}
              onClick={() => onResultSelect(result)}
              className={`p-4 cursor-pointer transition-all duration-200 hover:bg-gray-50 ${
                isSelected ? 'bg-blue-50 border-l-4 border-blue-500' : ''
              }`}
            >
              {/* Result Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <span className="text-lg font-bold text-gray-700">
                    #{index + 1}
                  </span>
                  <div className={`px-3 py-1 rounded-full bg-${color}-100 text-${color}-700 text-xs font-medium flex items-center space-x-1`}>
                    <span>{getSourceIcon(result.source)}</span>
                    <span>{result.source.toUpperCase()}</span>
                  </div>
                  <div className="px-3 py-1 rounded-full bg-green-100 text-green-700 text-xs font-medium">
                    {(result.score * 100).toFixed(0)}% Match
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="mt-3">
                <p 
                  className="text-gray-700 text-sm leading-relaxed line-clamp-3"
                  dangerouslySetInnerHTML={{ 
                    __html: highlightText(result.content, query) 
                  }}
                />
              </div>

              {/* Metadata */}
              {result.metadata && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {result.metadata.entity_type && (
                    <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                      Entity: {result.metadata.entity_type}
                    </span>
                  )}
                  {result.metadata.chunk_type && (
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                      Chunk: {result.metadata.chunk_type}
                    </span>
                  )}
                  {result.metadata.confidence && (
                    <span className="px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full">
                      Confidence: {(result.metadata.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              )}

              {/* Explanation */}
              {result.explanation && isSelected && (
                <div className="mt-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="text-xs font-medium text-gray-500 mb-1">Why this matches:</div>
                  <p className="text-xs text-gray-600">{result.explanation}</p>
                </div>
              )}

              {/* Highlights */}
              {result.highlights && result.highlights.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs font-medium text-gray-500 mb-1">Key phrases:</div>
                  <div className="flex flex-wrap gap-1">
                    {result.highlights.slice(0, 5).map((highlight, idx) => (
                      <span key={idx} className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded">
                        {highlight}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};