import React from 'react';
import { FusionControls } from '../FusionControls/FusionControls';

interface QueryTabProps {
  queryText: string;
  queryResults: any[];
  loading: boolean;
  onQueryTextChange: (text: string) => void;
  onQuery: () => void;
  onFusionConfigChange: (config: any) => void;
  onPresetSelect: (preset: string) => void;
}

export const QueryTab: React.FC<QueryTabProps> = ({
  queryText,
  queryResults,
  loading,
  onQueryTextChange,
  onQuery,
  onFusionConfigChange,
  onPresetSelect
}) => {
  return (
    <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="text-3xl mr-3">🔍</span>
        Query Interface
      </h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Query Input and Results - Takes up 2 columns on large screens */}
        <div className="lg:col-span-2 p-6 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-200">
          <label className="block text-lg font-semibold text-gray-700 mb-4">
            What would you like to know?
          </label>
          <div className="flex gap-3 mb-6">
            <input
              type="text"
              value={queryText}
              onChange={(e) => onQueryTextChange(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && onQuery()}
              placeholder="Ask a question about your documents..."
              className="flex-1 px-5 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent text-lg"
            />
            <button
              onClick={onQuery}
              disabled={loading || !queryText.trim()}
              className="px-8 py-3 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-lg shadow-md hover:from-orange-600 hover:to-amber-600 disabled:opacity-50 transition-all duration-200 transform hover:scale-105 font-semibold"
            >
              <span className="text-xl mr-2">🔍</span>
              Search
            </button>
          </div>

          {/* Query Results Area */}
          <div className="h-[70vh]">
            <h3 className="text-lg font-semibold text-gray-700 mb-3 flex items-center">
              Results {queryResults.length > 0 && `(${queryResults.length})`}
            </h3>
            
            {queryResults.length > 0 ? (
              <div className="space-y-3 h-[calc(70vh-2rem)] overflow-y-auto">
                {queryResults.map((result, idx) => (
                  <div key={idx} className="p-4 bg-white rounded-lg border border-orange-200 hover:shadow-md transition-shadow duration-200">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center space-x-3">
                        <span className="text-lg font-bold text-orange-600">{idx + 1}.</span>
                        <div className="px-2 py-1 bg-gradient-to-r from-green-100 to-emerald-100 rounded-full">
                          <span className="text-xs font-semibold text-green-800">
                            {result.score.toFixed(1)}% Match
                          </span>
                        </div>
                      </div>
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                        {result.source}
                      </span>
                    </div>
                    <p className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap">{result.content}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[calc(70vh-2rem)] text-gray-500 bg-white bg-opacity-50 rounded-lg border-2 border-dashed border-orange-300">
                <div className="text-4xl mb-3">🔍</div>
                <h4 className="text-md font-medium mb-2">Ready to Search</h4>
                <p className="text-sm text-center max-w-md text-gray-600">
                  Enter your question above and click Search to find relevant information from your documents.
                </p>
                {loading && (
                  <div className="mt-3 flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-orange-500 border-t-transparent"></div>
                    <span className="text-sm text-orange-600">Searching...</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Fusion Controls - Takes up 1 column on large screens */}
        <div className="lg:col-span-1">
          <FusionControls
            onConfigChange={onFusionConfigChange}
            onPresetSelect={onPresetSelect}
          />
        </div>
      </div>
    </div>
  );
};