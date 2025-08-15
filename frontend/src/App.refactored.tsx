import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { ChunkVisualizer } from './components/ChunkVisualizer/ChunkVisualizer';
import { GraphViewer } from './components/GraphViewer/GraphViewer';
import { FusionControls } from './components/FusionControls/FusionControls';
import { DemoTab } from './components/Demo/DemoTab';
import { WeightRulesManager } from './components/WeightRules/WeightRulesManager';
import { useDocumentManager, useFileUpload, useQueryEngine, useDataLoader } from './hooks';

type TabType = 'upload' | 'chunks' | 'graph' | 'query' | 'stats' | 'demo' | 'weights';

function App() {
  // Tab navigation state
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [viewMode, setViewMode] = useState<'single' | 'all'>('single');
  const [selectedChunkId, setSelectedChunkId] = useState<string | undefined>();
  const [selectedEntityId, setSelectedEntityId] = useState<string | undefined>();

  // Use custom hooks
  const {
    documents,
    selectedDocument,
    chunks,
    entities,
    relationships,
    loading: docLoading,
    error: docError,
    loadDocuments,
    selectDocument,
    createDocument,
    updateDocument,
    setDocuments,
    setError: setDocError
  } = useDocumentManager();

  const {
    allChunks,
    allEntities,
    allRelationships,
    loading: dataLoading,
    error: dataError,
    loadAllData,
    clearAllData
  } = useDataLoader();

  const {
    uploadFiles,
    processingStatus,
    performanceMetrics,
    uploading
  } = useFileUpload(
    (doc) => {
      setDocuments(prev => [...prev, doc]);
      if (!selectedDocument) {
        selectDocument(doc);
      }
    },
    (docs) => {
      docs.forEach(doc => updateDocument(doc.id, doc));
    }
  );

  const {
    queryText,
    queryResults,
    fusionConfig,
    selectedPreset,
    loading: queryLoading,
    error: queryError,
    setQueryText,
    executeQuery,
    updateFusionConfig,
    selectPreset
  } = useQueryEngine();

  // Combine loading and error states
  const loading = docLoading || dataLoading || queryLoading || uploading;
  const error = docError || dataError || queryError;

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // Load all data when switching to 'all' view or demo tab
  useEffect(() => {
    if ((viewMode === 'all' || activeTab === 'demo') && documents.length > 0) {
      loadAllData(documents);
    }
  }, [viewMode, activeTab, documents.length, loadAllData, documents]);

  // Handlers
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      await uploadFiles(files);
    }
  };

  const handleCreateDocument = async () => {
    const title = prompt('Enter document title:');
    const content = prompt('Enter document content:');
    
    if (title && content) {
      await createDocument(title, content);
    }
  };

  const handleChunkSelect = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    const chunk = chunks.find(c => c.id === chunkId);
    if (chunk) {
      console.log('Selected chunk:', chunk);
    }
  };

  const handleEntitySelect = (entityId: string) => {
    setSelectedEntityId(entityId);
    const entity = entities.find(e => e.id === entityId);
    if (entity) {
      console.log('Selected entity:', entity);
    }
  };

  const handleClearAll = async () => {
    if (confirm('Are you sure you want to delete ALL data? This cannot be undone.')) {
      try {
        const response = await fetch('http://localhost:8001/api/clear-all', {
          method: 'DELETE',
        });
        if (response.ok) {
          alert('All data cleared successfully');
          setDocuments([]);
          clearAllData();
          await loadDocuments();
        } else {
          alert('Failed to clear data');
        }
      } catch (error) {
        console.error('Error clearing data:', error);
        alert('Error clearing data');
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                RAG Visualizer
              </h1>
              <p className="mt-2 text-gray-600 text-lg">Explore and understand your RAG pipeline</p>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={handleClearAll}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
              >
                Clear All Data
              </button>
              <div className="flex items-center space-x-2">
                <div className="h-3 w-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600">System Active</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="bg-white rounded-xl shadow-md p-2">
          <nav className="flex space-x-2">
            {(['upload', 'chunks', 'graph', 'query', 'stats', 'demo', 'weights'] as TabType[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`
                  flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                  ${activeTab === tab
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg transform scale-105'
                    : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }
                `}
              >
                {tab === 'demo' && '🚀 '}
                {tab === 'weights' && '⚖️ '}
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 pb-12">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow-md animate-slideIn">
            <div className="flex items-center">
              <span className="text-red-800 flex-1">{error}</span>
              <button 
                onClick={() => setDocError(null)} 
                className="ml-4 text-red-500 hover:text-red-700 text-2xl font-bold transition-colors"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Processing Status */}
        {processingStatus && (
          <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 rounded-lg shadow-md animate-slideIn">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                {!processingStatus.startsWith('✅') && (
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent mr-3"></div>
                )}
                <span className="text-blue-800 font-medium">{processingStatus}</span>
              </div>
              {performanceMetrics && (
                <div className="text-xs text-gray-600">
                  {performanceMetrics.total_time && (
                    <span className="ml-4">Total: {performanceMetrics.total_time}s</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'upload' && (
          <UploadTab
            documents={documents}
            selectedDocument={selectedDocument}
            loading={loading}
            onFileUpload={handleFileUpload}
            onCreateDocument={handleCreateDocument}
            onSelectDocument={selectDocument}
          />
        )}

        {activeTab === 'chunks' && (
          <ChunksTab
            chunks={chunks}
            allChunks={allChunks}
            viewMode={viewMode}
            setViewMode={setViewMode}
            selectedChunkId={selectedChunkId}
            onChunkSelect={handleChunkSelect}
            onRefresh={() => loadAllData(documents)}
            documents={documents}
          />
        )}

        {activeTab === 'graph' && (
          <GraphTab
            entities={entities}
            relationships={relationships}
            allEntities={allEntities}
            allRelationships={allRelationships}
            viewMode={viewMode}
            setViewMode={setViewMode}
            selectedEntityId={selectedEntityId}
            onEntitySelect={handleEntitySelect}
            onRefresh={() => loadAllData(documents)}
            loading={loading}
          />
        )}

        {activeTab === 'query' && (
          <QueryTab
            queryText={queryText}
            queryResults={queryResults}
            loading={loading}
            onQueryTextChange={setQueryText}
            onQuery={executeQuery}
            onFusionConfigChange={updateFusionConfig}
            onPresetSelect={selectPreset}
          />
        )}

        {activeTab === 'stats' && (
          <StatsTab
            documents={documents}
            selectedDocument={selectedDocument}
            chunks={chunks}
            entities={entities}
            relationships={relationships}
          />
        )}

        {activeTab === 'demo' && (
          <DemoTab
            documents={documents}
            chunks={Object.values(allChunks).flat()}
            entities={Object.values(allEntities).flat()}
            relationships={Object.values(allRelationships).flat()}
            loading={loading}
            onDocumentSelect={selectDocument}
          />
        )}

        {activeTab === 'weights' && (
          <WeightRulesManager documents={documents} />
        )}
      </main>
    </div>
  );
}

// Component placeholders - These will be extracted to separate files
const UploadTab = ({ documents, selectedDocument, loading, onFileUpload, onCreateDocument, onSelectDocument }: any) => (
  <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
    <h2 className="text-2xl font-bold text-gray-800 mb-6">Document Management</h2>
    {/* Upload section */}
    <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
      <label className="block text-lg font-semibold text-gray-700 mb-4">Upload Documents</label>
      <div className="flex items-center space-x-4">
        <label className="flex-1">
          <input
            type="file"
            accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.tiff,.bmp"
            onChange={onFileUpload}
            disabled={loading}
            className="hidden"
            id="file-upload"
            multiple
          />
          <label 
            htmlFor="file-upload"
            className={`cursor-pointer flex items-center justify-center px-6 py-3 rounded-lg shadow-md transition-all duration-200 transform hover:scale-105 ${
              loading ? 'bg-gradient-to-r from-gray-400 to-gray-500 cursor-not-allowed' : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
            } text-white min-w-[200px]`}
          >
            {loading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>}
            <span>{loading ? 'Processing...' : 'Choose Files'}</span>
          </label>
        </label>
        <span className="text-gray-500">or</span>
        <button
          onClick={onCreateDocument}
          disabled={loading}
          className="px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg shadow-md hover:from-green-600 hover:to-green-700 disabled:opacity-50 transition-all duration-200 transform hover:scale-105"
        >
          Create from Text
        </button>
      </div>
    </div>
    {/* Documents list */}
    <div>
      <h3 className="text-xl font-semibold text-gray-800 mb-4">Your Documents</h3>
      {documents.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="text-lg">No documents yet</p>
          <p className="text-sm mt-2">Upload a document or create one from text to get started</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {documents.map((doc: any) => (
            <div
              key={doc.id}
              onClick={() => onSelectDocument(doc)}
              className={`p-6 rounded-xl cursor-pointer transition-all duration-200 transform hover:scale-102 hover:shadow-lg ${
                selectedDocument?.id === doc.id 
                  ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-400 shadow-lg' 
                  : 'bg-white border-2 border-gray-200 hover:border-gray-300'
              }`}
            >
              <h4 className="font-semibold text-lg text-gray-800">{doc.title}</h4>
              <div className="text-sm text-gray-600 mt-2">
                {new Date(doc.created_at).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
);

const ChunksTab = ({ chunks, allChunks, viewMode, setViewMode, selectedChunkId, onChunkSelect, onRefresh, documents }: any) => (
  <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
    <div className="flex justify-between items-center mb-6">
      <h2 className="text-2xl font-bold text-gray-800">Chunk Visualization</h2>
      <div className="flex items-center space-x-2">
        <button
          onClick={() => setViewMode('single')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'single' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          Current Document
        </button>
        <button
          onClick={() => setViewMode('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          All Documents
        </button>
        {viewMode === 'all' && (
          <button onClick={onRefresh} className="px-3 py-2 rounded-lg bg-green-600 text-white">
            Refresh
          </button>
        )}
      </div>
    </div>
    {viewMode === 'single' ? (
      <ChunkVisualizer
        chunks={chunks}
        selectedChunkId={selectedChunkId}
        onChunkSelect={onChunkSelect}
      />
    ) : (
      <div className="space-y-6">
        {Object.entries(allChunks).map(([docId, docChunks]: [string, any]) => {
          const doc = documents.find((d: any) => d.id === docId);
          return (
            <div key={docId} className="border rounded-lg p-4">
              <h3 className="font-semibold text-lg mb-3">{doc?.title || 'Unknown'}</h3>
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
  </div>
);

const GraphTab = ({ entities, relationships, allEntities, allRelationships, viewMode, setViewMode, selectedEntityId, onEntitySelect, onRefresh, loading }: any) => (
  <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
    <div className="flex justify-between items-center mb-6">
      <h2 className="text-2xl font-bold text-gray-800">Knowledge Graph</h2>
      <div className="flex items-center space-x-2">
        <button
          onClick={() => setViewMode('single')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'single' ? 'bg-purple-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          Current Document
        </button>
        <button
          onClick={() => setViewMode('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'all' ? 'bg-purple-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          All Documents
        </button>
        {viewMode === 'all' && (
          <button onClick={onRefresh} className="px-3 py-2 rounded-lg bg-green-600 text-white">
            Refresh
          </button>
        )}
      </div>
    </div>
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
  </div>
);

const QueryTab = ({ queryText, queryResults, loading, onQueryTextChange, onQuery, onFusionConfigChange, onPresetSelect }: any) => (
  <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
    <h2 className="text-2xl font-bold text-gray-800 mb-6">Query Interface</h2>
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
      <div className="lg:col-span-2 p-6 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-200">
        <label className="block text-lg font-semibold text-gray-700 mb-4">What would you like to know?</label>
        <div className="flex gap-3 mb-6">
          <input
            type="text"
            value={queryText}
            onChange={(e) => onQueryTextChange(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && onQuery()}
            placeholder="Ask a question about your documents..."
            className="flex-1 px-5 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
          />
          <button
            onClick={onQuery}
            disabled={loading || !queryText.trim()}
            className="px-8 py-3 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-lg shadow-md hover:from-orange-600 hover:to-amber-600 disabled:opacity-50"
          >
            🔍 Search
          </button>
        </div>
        {/* Results */}
        <div className="h-[60vh] overflow-y-auto">
          {queryResults.length > 0 ? (
            <div className="space-y-3">
              {queryResults.map((result: any, idx: number) => (
                <div key={idx} className="p-4 bg-white rounded-lg border border-orange-200">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-lg font-bold text-orange-600">{idx + 1}.</span>
                    <span className="text-xs font-semibold text-green-800">{result.score.toFixed(1)}% Match</span>
                  </div>
                  <p className="text-gray-700 text-sm">{result.content}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <div className="text-4xl mb-3">🔍</div>
              <p className="text-sm">Enter your question above and click Search</p>
            </div>
          )}
        </div>
      </div>
      <div className="lg:col-span-1">
        <FusionControls
          onConfigChange={onFusionConfigChange}
          onPresetSelect={onPresetSelect}
        />
      </div>
    </div>
  </div>
);

const StatsTab = ({ documents, selectedDocument, chunks, entities, relationships }: any) => (
  <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
    <h2 className="text-2xl font-bold text-gray-800 mb-6">System Statistics</h2>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-100 rounded-xl border border-blue-200">
        <h3 className="font-medium text-blue-900 text-sm">Documents</h3>
        <p className="text-3xl font-bold text-blue-700 mt-2">{documents.length}</p>
      </div>
      <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl border border-green-200">
        <h3 className="font-medium text-green-900 text-sm">Total Chunks</h3>
        <p className="text-3xl font-bold text-green-700 mt-2">{chunks.length}</p>
      </div>
      <div className="p-6 bg-gradient-to-br from-purple-50 to-pink-100 rounded-xl border border-purple-200">
        <h3 className="font-medium text-purple-900 text-sm">Entities</h3>
        <p className="text-3xl font-bold text-purple-700 mt-2">{entities.length}</p>
      </div>
    </div>
    {selectedDocument && (
      <div className="p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
        <h3 className="font-semibold text-lg mb-4">Selected Document: {selectedDocument.title}</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-white rounded-lg">
            <span className="text-gray-600">Chunks</span>
            <p className="text-xl font-bold text-green-600">{chunks.length}</p>
          </div>
          <div className="p-3 bg-white rounded-lg">
            <span className="text-gray-600">Entities</span>
            <p className="text-xl font-bold text-purple-600">{entities.length}</p>
          </div>
        </div>
      </div>
    )}
  </div>
);

export default App;