import React, { useState, useCallback, useRef, useEffect } from 'react';
import { QueryInput } from './QueryInput';
import { ResultsPanel } from './ResultsPanel';
import { DocumentPreview } from './DocumentPreview';
import { GraphResultPreview } from './GraphResultPreview';
import { QueryDecomposition } from './QueryDecomposition';
import { RetrievalSummary } from './RetrievalSummary';
import { demoApi } from '../../../services/demoApi';
import type { Document, Chunk, Entity } from '../../../types';

interface SearchInterfaceProps {
  documents: Document[];
  chunks: Chunk[];
  entities: Entity[];
  onSearchComplete: (searchData: any) => void;
  onDocumentSelect: (doc: Document) => void;
  persistentState?: any;
  onStateChange?: (state: any) => void;
}

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  documents,
  chunks,
  entities,
  onSearchComplete,
  onDocumentSelect,
  persistentState,
  onStateChange
}) => {
  // Use persistent state if provided, otherwise use local state
  const [query, setQuery] = useState(persistentState?.query || '');
  const [searchResults, setSearchResults] = useState<any[]>(persistentState?.results || []);
  const [decomposition, setDecomposition] = useState<any>(persistentState?.decomposition || null);
  const [selectedResult, setSelectedResult] = useState<any>(persistentState?.selectedResult || null);
  const [previewDocument, setPreviewDocument] = useState<Document | null>(persistentState?.previewDocument || null);
  const [loading, setLoading] = useState(false);
  const [searchMetrics, setSearchMetrics] = useState<any>(persistentState?.metrics || null);
  const [searchMode, setSearchMode] = useState<'smart' | 'vector' | 'graph' | 'hybrid'>(persistentState?.searchMode || 'smart');
  const [showDecomposition, setShowDecomposition] = useState(persistentState?.showDecomposition || false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  
  // Update persistent state when local state changes
  useEffect(() => {
    if (onStateChange) {
      onStateChange({
        query,
        results: searchResults,
        selectedResult,
        decomposition,
        metrics: searchMetrics,
        previewDocument,
        searchMode,
        showDecomposition
      });
    }
  }, [query, searchResults, selectedResult, decomposition, searchMetrics, previewDocument, searchMode, showDecomposition]);

  // Fetch query suggestions
  useEffect(() => {
    if (query.length > 2) {
      const fetchSuggestions = async () => {
        try {
          const response = await demoApi.getSuggestions(query);
          setSuggestions(response.suggestions || []);
        } catch (error) {
          console.error('Error fetching suggestions:', error);
        }
      };
      
      const debounceTimer = setTimeout(fetchSuggestions, 300);
      return () => clearTimeout(debounceTimer);
    } else {
      setSuggestions([]);
    }
  }, [query]);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setLoading(true);
    setSearchResults([]);
    setDecomposition(null);
    setSearchMetrics(null);

    try {
      const startTime = Date.now();

      // Perform enhanced search with explanations
      const searchResponse = await demoApi.search({
        query,
        mode: searchMode,
        includeExplanations: true,
        includeDecomposition: showDecomposition,
        maxResults: 10
      });

      const endTime = Date.now();
      const processingTime = endTime - startTime;

      setSearchResults(searchResponse.results || []);
      setDecomposition(searchResponse.decomposition || null);
      
      // Set search metrics
      setSearchMetrics({
        processingTime,
        totalResults: searchResponse.total_results,
        strategy: searchResponse.retrieval_strategy,
        vectorResults: searchResponse.metadata?.vector_results || 0,
        graphResults: searchResponse.metadata?.graph_results || 0,
        tokensUsed: searchResponse.metadata?.context_budgeting?.tokens_used || 0,
        confidence: searchResponse.average_confidence || 0
      });

      // Track search in history
      onSearchComplete({
        query,
        results: searchResponse.results,
        metrics: searchMetrics,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, searchMode, showDecomposition, onSearchComplete]);

  const handleResultSelect = (result: any) => {
    console.log('Result selected:', result);
    console.log('Looking for document_id:', result.document_id);
    console.log('Available documents:', documents.map(d => d.id));
    
    setSelectedResult(result);
    
    // Find and preview the document - try both document_id and doc_id
    const doc = documents.find(d => 
      d.id === result.document_id || 
      d.id === result.doc_id ||
      d.id === result.metadata?.document_id
    );
    
    if (doc) {
      console.log('Found document:', doc);
      setPreviewDocument(doc);
    } else {
      console.log('No matching document found');
      // If no document found, create a preview from the result content
      const previewDoc: Document = {
        id: result.document_id || result.id,
        title: result.document_title || 'Search Result',
        content: result.content || '',
        doc_type: 'text',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
      setPreviewDocument(previewDoc);
    }
  };

  const handleDocumentOpen = (doc: Document) => {
    onDocumentSelect(doc);
    setPreviewDocument(doc);
  };

  const handleQuerySelect = (selectedQuery: string) => {
    setQuery(selectedQuery);
    setSuggestions([]);
  };

  return (
    <div className="space-y-6">
      {/* Search Controls */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
        <div className="space-y-4">
          {/* Search Mode Selector */}
          <div className="flex items-center space-x-4 mb-4">
            <span className="text-sm font-medium text-gray-700">Search Mode:</span>
            <div className="flex space-x-2">
              {(['smart', 'vector', 'graph', 'hybrid'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setSearchMode(mode)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    searchMode === mode
                      ? 'bg-blue-600 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  {mode === 'smart' && ' 🧠'}
                </button>
              ))}
            </div>
            <label className="flex items-center space-x-2 ml-4">
              <input
                type="checkbox"
                checked={showDecomposition}
                onChange={(e) => setShowDecomposition(e.target.checked)}
                className="rounded text-blue-600"
              />
              <span className="text-sm text-gray-700">Show Query Decomposition</span>
            </label>
          </div>

          {/* Query Input */}
          <QueryInput
            query={query}
            onQueryChange={setQuery}
            onSearch={handleSearch}
            suggestions={suggestions}
            onSuggestionSelect={handleQuerySelect}
            loading={loading}
          />

          {/* Quick Examples */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">Try:</span>
            {[
              "How does the authentication system work?",
              "Compare vector search and graph retrieval",
              "What are the main entities in the documents?"
            ].map((example, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setQuery(example);
                  handleSearch();
                }}
                className="text-xs px-3 py-1 bg-white rounded-full hover:bg-blue-100 text-blue-600 transition-colors"
              >
                {example.substring(0, 30)}...
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Search Metrics */}
      {searchMetrics && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="grid grid-cols-6 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">{searchMetrics.processingTime}ms</div>
              <div className="text-xs text-gray-500">Response Time</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">{searchMetrics.totalResults}</div>
              <div className="text-xs text-gray-500">Total Results</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">{searchMetrics.vectorResults}</div>
              <div className="text-xs text-gray-500">Vector Matches</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-600">{searchMetrics.graphResults}</div>
              <div className="text-xs text-gray-500">Graph Matches</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-indigo-600">{searchMetrics.tokensUsed}</div>
              <div className="text-xs text-gray-500">Tokens Used</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-teal-600">
                {(searchMetrics.confidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500">Avg Confidence</div>
            </div>
          </div>
        </div>
      )}

      {/* Query Decomposition */}
      {showDecomposition && decomposition && (
        <QueryDecomposition decomposition={decomposition} />
      )}

      {/* Retrieval Summary */}
      {searchResults.length > 0 && (
        <RetrievalSummary 
          results={searchResults}
          query={query}
          processingTime={searchMetrics?.processingTime}
        />
      )}

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Results Panel - 2 columns on large screens */}
        <div className="lg:col-span-2">
          <ResultsPanel
            results={searchResults}
            loading={loading}
            onResultSelect={handleResultSelect}
            selectedResult={selectedResult}
            query={query}
          />
        </div>

        {/* Document/Graph Preview - 1 column */}
        <div className="lg:col-span-1">
          {selectedResult ? (
            // Show graph preview for graph results, document preview for others
            selectedResult.source === 'graph' ? (
              <GraphResultPreview
                result={selectedResult}
                query={query}
                onOpen={() => console.log('Open graph explorer')}
              />
            ) : (
              previewDocument ? (
                <DocumentPreview
                  document={previewDocument}
                  highlightedChunk={selectedResult?.chunk_id}
                  onOpen={() => handleDocumentOpen(previewDocument)}
                />
              ) : (
                <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-300">
                  <div className="text-gray-400 text-6xl mb-4">📄</div>
                  <p className="text-gray-600">Document preview unavailable</p>
                </div>
              )
            )
          ) : (
            <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-300">
              <div className="text-gray-400 text-6xl mb-4">📄</div>
              <p className="text-gray-600">Select a result to preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};