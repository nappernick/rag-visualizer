import React, { useState } from 'react';

interface SubQuery {
  question: string;
  type: string;
  dependencies: number[];
  priority: string;
  keywords: string[];
}

interface QueryDecompositionProps {
  decomposition: {
    original: string;
    sub_queries: SubQuery[];
    reasoning_path?: string;
    complexity_score: number;
    query_type: string;
  };
}

export const QueryDecomposition: React.FC<QueryDecompositionProps> = ({ decomposition }) => {
  const [expandedQueries, setExpandedQueries] = useState<number[]>([]);

  const toggleExpand = (index: number) => {
    setExpandedQueries(prev => 
      prev.includes(index) 
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'red';
      case 'medium': return 'yellow';
      case 'low': return 'green';
      default: return 'gray';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'factual': return '📊';
      case 'analytical': return '🧪';
      case 'comparative': return '⚖️';
      case 'exploratory': return '🔍';
      case 'navigational': return '🧭';
      case 'multi_hop': return '🔗';
      default: return '❓';
    }
  };

  return (
    <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl p-6 border border-purple-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <span className="mr-2">🧩</span>
          Query Decomposition
        </h3>
        <div className="flex items-center space-x-3">
          <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
            {decomposition.query_type}
          </span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-medium">
            Complexity: {(decomposition.complexity_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Original Query */}
      <div className="mb-4 p-3 bg-white rounded-lg border border-purple-200">
        <div className="text-xs font-medium text-gray-500 mb-1">Original Query:</div>
        <p className="text-sm text-gray-900 font-medium">{decomposition.original}</p>
      </div>

      {/* Sub-Queries */}
      <div className="space-y-3">
        <div className="text-sm font-medium text-gray-700 mb-2">
          Sub-Queries ({decomposition.sub_queries.length}):
        </div>
        
        {decomposition.sub_queries.map((subQuery, index) => {
          const isExpanded = expandedQueries.includes(index);
          const priorityColor = getPriorityColor(subQuery.priority);
          
          return (
            <div
              key={index}
              className="bg-white rounded-lg border border-purple-200 overflow-hidden"
            >
              <div
                onClick={() => toggleExpand(index)}
                className="p-3 cursor-pointer hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <span className="text-lg font-bold text-purple-600">
                      {index + 1}
                    </span>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span>{getTypeIcon(subQuery.type)}</span>
                        <span className="text-sm font-medium text-gray-900">
                          {subQuery.question}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-0.5 bg-${priorityColor}-100 text-${priorityColor}-700 rounded text-xs font-medium`}>
                          {subQuery.priority}
                        </span>
                        <span className="px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-xs">
                          {subQuery.type}
                        </span>
                        {subQuery.dependencies.length > 0 && (
                          <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                            Depends on: {subQuery.dependencies.join(', ')}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <button className="text-gray-400 hover:text-gray-600">
                    <svg
                      className={`w-5 h-5 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                </div>
              </div>
              
              {isExpanded && (
                <div className="p-3 bg-gray-50 border-t border-purple-100">
                  <div className="space-y-2">
                    {subQuery.keywords.length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-gray-500 mb-1">Keywords:</div>
                        <div className="flex flex-wrap gap-1">
                          {subQuery.keywords.map((keyword, kidx) => (
                            <span
                              key={kidx}
                              className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs"
                            >
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {subQuery.dependencies.length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-gray-500">
                          This query depends on results from query {subQuery.dependencies.join(' and ')}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Reasoning Path */}
      {decomposition.reasoning_path && (
        <div className="mt-4 p-3 bg-white rounded-lg border border-purple-200">
          <div className="text-xs font-medium text-gray-500 mb-1">Reasoning Path:</div>
          <p className="text-xs text-gray-700">{decomposition.reasoning_path}</p>
        </div>
      )}
    </div>
  );
};