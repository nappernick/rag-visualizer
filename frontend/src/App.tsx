import React, { useState, useEffect } from 'react';
import './App.css';
import { ChunkVisualizer } from './components/ChunkVisualizer/ChunkVisualizer';
import { GraphViewer } from './components/GraphViewer/GraphViewer';
import { FusionControls } from './components/FusionControls/FusionControls';
import { documentApi, chunkingApi, graphApi, queryApi } from './services/api';
import type { Document, Chunk, Entity, Relationship, ChunkingRequest } from './types';

type TabType = 'upload' | 'chunks' | 'graph' | 'query' | 'stats';

function App() {
  // State
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [allChunks, setAllChunks] = useState<{[docId: string]: Chunk[]}>({});
  const [allEntities, setAllEntities] = useState<{[docId: string]: Entity[]}>({});
  const [allRelationships, setAllRelationships] = useState<{[docId: string]: Relationship[]}>({});
  const [viewMode, setViewMode] = useState<'single' | 'all'>('single');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedChunkId, setSelectedChunkId] = useState<string | undefined>();
  const [selectedEntityId, setSelectedEntityId] = useState<string | undefined>();
  const [queryText, setQueryText] = useState('');
  const [queryResults, setQueryResults] = useState<any[]>([]);
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const [fusionConfig, setFusionConfig] = useState<any>(null);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);
  
  // Load all data when switching to 'all' view or when documents change
  useEffect(() => {
    if (viewMode === 'all' && documents.length > 0) {
      loadAllData();
    }
  }, [viewMode, documents.length]);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const docs = await documentApi.list();
      setDocuments(docs);
    } catch (err) {
      setError('Failed to load documents');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const loadAllData = async () => {
    try {
      setLoading(true);
      const newAllChunks: {[key: string]: Chunk[]} = {};
      const newAllEntities: {[key: string]: Entity[]} = {};
      const newAllRelationships: {[key: string]: Relationship[]} = {};
      
      for (const doc of documents) {
        try {
          const docChunks = await chunkingApi.getChunks(doc.id);
          newAllChunks[doc.id] = Array.isArray(docChunks) ? docChunks : [];
          
          const [docEntities, docRelationships] = await Promise.all([
            graphApi.getEntities(doc.id).catch(() => []),
            graphApi.getRelationships(doc.id).catch(() => [])
          ]);
          newAllEntities[doc.id] = docEntities;
          newAllRelationships[doc.id] = docRelationships;
        } catch (err) {
          console.error(`Failed to load data for document ${doc.id}:`, err);
        }
      }
      
      setAllChunks(newAllChunks);
      setAllEntities(newAllEntities);
      setAllRelationships(newAllRelationships);
    } catch (err) {
      setError('Failed to load all data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // File upload handler (supports multiple files)
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    try {
      setLoading(true);
      setError(null);
      
      const totalFiles = files.length;
      const newDocuments: Document[] = [];
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setProcessingStatus(`Uploading document ${i + 1} of ${totalFiles}: ${file.name}...`);
        
        const document = await documentApi.upload(file);
        newDocuments.push(document);
        setDocuments(prev => [...prev, document]);
        
        // Select the first uploaded document
        if (i === 0) {
          setSelectedDocument(document);
        }
        
        // Process the document
        setProcessingStatus(`Processing document ${i + 1} of ${totalFiles}: Chunking...`);
        const chunkingRequest: ChunkingRequest = {
          document_id: document.id,
          content: document.content,
          strategy: 'hierarchical',
          max_chunk_size: 800,
          chunk_overlap: 100
        };
        
        const chunkingResponse = await chunkingApi.chunkDocument(chunkingRequest);
        
        // Update chunks for the first document
        if (i === 0) {
          setChunks(Array.isArray(chunkingResponse.chunks) ? chunkingResponse.chunks : []);
        }
        
        // Update all chunks collection
        setAllChunks(prev => ({
          ...prev,
          [document.id]: Array.isArray(chunkingResponse.chunks) ? chunkingResponse.chunks : []
        }));
        
        // Extract graph
        setProcessingStatus(`Processing document ${i + 1} of ${totalFiles}: Extracting knowledge graph...`);
        const graphData = await graphApi.extractGraph(document.id, chunkingResponse.chunks);
        
        // Update entities/relationships for the first document
        if (i === 0) {
          setEntities(graphData.entities);
          setRelationships(graphData.relationships);
        }
        
        // Update all entities and relationships
        setAllEntities(prev => ({
          ...prev,
          [document.id]: graphData.entities
        }));
        setAllRelationships(prev => ({
          ...prev,
          [document.id]: graphData.relationships
        }));
        
        // Update document status to completed
        const updatedDoc = { ...document, status: 'completed' as const };
        setDocuments(prev => prev.map(d => d.id === document.id ? updatedDoc : d));
        
        if (i === 0) {
          setSelectedDocument(updatedDoc);
        }
      }
      
      setProcessingStatus(`Successfully processed ${totalFiles} document${totalFiles > 1 ? 's' : ''}!`);
      
      // Clear loading state immediately after processing
      setLoading(false);
      setTimeout(() => setProcessingStatus(''), 3000);
      
    } catch (err: any) {
      const errorMessage = err?.response?.data?.detail || err?.message || 'Failed to process document';
      setError(errorMessage);
      console.error('Document processing error:', err);
      
      // Update document status to failed
      if (document) {
        const updatedDoc = { ...document, status: 'failed' as const };
        setDocuments(prev => prev.map(d => d.id === document.id ? updatedDoc : d));
        setSelectedDocument(updatedDoc);
      }
      setLoading(false);
    }
  };

  // Create document from text
  const handleCreateDocument = async () => {
    const title = prompt('Enter document title:');
    const content = prompt('Enter document content:');
    
    if (!title || !content) return;

    try {
      setLoading(true);
      setError(null);
      
      const document = await documentApi.create(title, content);
      setDocuments([...documents, document]);
      setSelectedDocument(document);
      
      // Process the document
      const chunkingRequest: ChunkingRequest = {
        document_id: document.id,
        content: document.content,
        strategy: 'hierarchical',
        max_chunk_size: 800,
        chunk_overlap: 100
      };
      
      const chunkingResponse = await chunkingApi.chunkDocument(chunkingRequest);
      setChunks(Array.isArray(chunkingResponse.chunks) ? chunkingResponse.chunks : []);
      
    } catch (err) {
      setError('Failed to create document');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Load document data
  const handleSelectDocument = async (doc: Document) => {
    try {
      setLoading(true);
      setSelectedDocument(doc);
      
      // Load chunks
      const docChunks = await chunkingApi.getChunks(doc.id);
      setChunks(Array.isArray(docChunks) ? docChunks : []);
      
      // Load entities and relationships
      const [docEntities, docRelationships] = await Promise.all([
        graphApi.getEntities(doc.id).catch(() => []),
        graphApi.getRelationships(doc.id).catch(() => [])
      ]);
      
      setEntities(docEntities);
      setRelationships(docRelationships);
      
    } catch (err) {
      setError('Failed to load document data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Query handler
  const handleQuery = async () => {
    if (!queryText.trim()) return;

    try {
      setLoading(true);
      setError(null);
      
      const response = await queryApi.query(queryText, {
        max_results: fusionConfig?.final_top_k || 10,
        retrieval_strategy: fusionConfig?.auto_strategy ? undefined : 'hybrid',
        fusion_config: fusionConfig,
        preset: selectedPreset
      });
      
      setQueryResults(response.results);
      
    } catch (err) {
      setError('Failed to execute query');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Fusion configuration handlers
  const handleFusionConfigChange = (config: any) => {
    setFusionConfig(config);
  };

  const handlePresetSelect = (preset: string) => {
    setSelectedPreset(preset);
  };

  // Chunk selection handler
  const handleChunkSelect = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    const chunk = chunks.find(c => c.id === chunkId);
    if (chunk) {
      console.log('Selected chunk:', chunk);
    }
  };

  // Entity selection handler
  const handleEntitySelect = (entityId: string) => {
    setSelectedEntityId(entityId);
    const entity = entities.find(e => e.id === entityId);
    if (entity) {
      console.log('Selected entity:', entity);
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
                onClick={async () => {
                  if (confirm('Are you sure you want to delete ALL data? This cannot be undone.')) {
                    try {
                      const response = await fetch('http://localhost:8745/api/clear-all', {
                        method: 'DELETE',
                      });
                      if (response.ok) {
                        alert('All data cleared successfully');
                        setDocuments([]);
                        setSelectedDocument(null);
                        setChunks([]);
                        setEntities([]);
                        setRelationships([]);
                        setAllChunks({});
                        setAllEntities({});
                        setAllRelationships({});
                        await loadDocuments();
                      } else {
                        alert('Failed to clear data');
                      }
                    } catch (error) {
                      console.error('Error clearing data:', error);
                      alert('Error clearing data');
                    }
                  }
                }}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
              >
                🗑️ Clear All Data
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
            <button
              onClick={() => setActiveTab('upload')}
              className={`
                flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                ${activeTab === 'upload'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg transform scale-105'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              📤 Upload
            </button>
            <button
              onClick={() => setActiveTab('chunks')}
              className={`
                flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                ${activeTab === 'chunks'
                  ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg transform scale-105'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              📊 Chunks
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={`
                flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                ${activeTab === 'graph'
                  ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white shadow-lg transform scale-105'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              🔗 Graph
            </button>
            <button
              onClick={() => setActiveTab('query')}
              className={`
                flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                ${activeTab === 'query'
                  ? 'bg-gradient-to-r from-orange-500 to-orange-600 text-white shadow-lg transform scale-105'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              🔍 Query
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`
                flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                ${activeTab === 'stats'
                  ? 'bg-gradient-to-r from-pink-500 to-pink-600 text-white shadow-lg transform scale-105'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              📈 Stats
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 pb-12">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow-md animate-slideIn">
            <div className="flex items-center">
              <span className="text-2xl mr-3">⚠️</span>
              <span className="text-red-800 flex-1">{error}</span>
              <button 
                onClick={() => setError(null)} 
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
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent mr-3"></div>
              <span className="text-blue-800 font-medium">{processingStatus}</span>
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'upload' && (
          <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <span className="text-3xl mr-3">📁</span>
              Document Management
            </h2>
            
            {/* Upload Section */}
            <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
              <label className="block text-lg font-semibold text-gray-700 mb-4">
                Upload Documents
              </label>
              <div className="flex items-center space-x-4">
                <label className="flex-1">
                  <input
                    type="file"
                    accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.tiff,.bmp"
                    onChange={handleFileUpload}
                    disabled={loading}
                    className="hidden"
                    id="file-upload"
                    multiple
                  />
                  <label 
                    htmlFor="file-upload"
                    className="cursor-pointer flex items-center justify-center px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg shadow-md hover:from-blue-600 hover:to-blue-700 transition-all duration-200 transform hover:scale-105"
                  >
                    <span className="text-xl mr-2">📎</span>
                    Choose Files
                  </label>
                </label>
                <span className="text-gray-500">or</span>
                <button
                  onClick={handleCreateDocument}
                  disabled={loading}
                  className="px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg shadow-md hover:from-green-600 hover:to-green-700 disabled:opacity-50 transition-all duration-200 transform hover:scale-105"
                >
                  <span className="text-xl mr-2">✏️</span>
                  Create from Text
                </button>
              </div>
            </div>

            {/* Documents List */}
            <div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <span className="text-2xl mr-2">📚</span>
                Your Documents
              </h3>
              {documents.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <span className="text-6xl block mb-4">📭</span>
                  <p className="text-lg">No documents yet</p>
                  <p className="text-sm mt-2">Upload a document or create one from text to get started</p>
                </div>
              ) : (
                <div className="grid gap-4">
                  {documents.map((doc) => (
                    <div
                      key={doc.id}
                      onClick={() => handleSelectDocument(doc)}
                      className={`p-6 rounded-xl cursor-pointer transition-all duration-200 transform hover:scale-102 hover:shadow-lg ${
                        selectedDocument?.id === doc.id 
                          ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-400 shadow-lg' 
                          : 'bg-white border-2 border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div className="flex items-start space-x-3">
                          <span className="text-3xl">📄</span>
                          <div>
                            <h4 className="font-semibold text-lg text-gray-800">{doc.title}</h4>
                            <div className="flex items-center mt-2 space-x-3">
                              <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                                doc.status === 'completed' ? 'bg-green-100 text-green-800' :
                                doc.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                                doc.status === 'failed' ? 'bg-red-100 text-red-800' :
                                'bg-gray-100 text-gray-800'
                              }`}>
                                <span className="w-2 h-2 rounded-full mr-2 ${
                                  doc.status === 'completed' ? 'bg-green-500' :
                                  doc.status === 'processing' ? 'bg-yellow-500 animate-pulse' :
                                  doc.status === 'failed' ? 'bg-red-500' :
                                  'bg-gray-500'
                                }"></span>
                                {doc.status}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-medium text-gray-600">
                            {new Date(doc.created_at).toLocaleDateString()}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {new Date(doc.created_at).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'chunks' && (
          <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800 flex items-center">
                <span className="text-3xl mr-3">📊</span>
                Chunk Visualization
              </h2>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setViewMode('single')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    viewMode === 'single' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Current Document
                </button>
                <button
                  onClick={() => setViewMode('all')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    viewMode === 'all' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  All Documents
                </button>
                {viewMode === 'all' && (
                  <button
                    onClick={() => loadAllData()}
                    className="px-3 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
                    title="Refresh all documents data"
                  >
                    🔄 Refresh
                  </button>
                )}
              </div>
            </div>
            {viewMode === 'single' && chunks.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <span className="text-6xl block mb-4">📦</span>
                <p className="text-lg">No chunks to display</p>
                <p className="text-sm mt-2">Please select a document first</p>
              </div>
            ) : viewMode === 'all' && (Object.keys(allChunks).length === 0 || Object.values(allChunks).every(chunks => chunks.length === 0)) ? (
              <div className="text-center py-12 text-gray-500">
                <span className="text-6xl block mb-4">📦</span>
                <p className="text-lg">No chunks in the corpus</p>
                <p className="text-sm mt-2">Upload and process documents to see chunks</p>
              </div>
            ) : (
              <div>
                <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-6">
                      <div>
                        <span className="text-sm text-gray-600">Total Chunks</span>
                        <p className="text-2xl font-bold text-green-600">
                          {viewMode === 'single' 
                            ? chunks.length 
                            : Object.values(allChunks).reduce((sum, c) => sum + c.length, 0)}
                        </p>
                      </div>
                      {viewMode === 'all' && (
                        <>
                          <div className="h-8 w-px bg-gray-300"></div>
                          <div>
                            <span className="text-sm text-gray-600">Documents</span>
                            <p className="text-lg font-semibold text-gray-800">{Object.keys(allChunks).length}</p>
                          </div>
                        </>
                      )}
                      <div className="h-8 w-px bg-gray-300"></div>
                      <div>
                        <span className="text-sm text-gray-600">Selected</span>
                        <p className="text-lg font-semibold text-gray-800">
                          {selectedChunkId ? (() => {
                            const allChunksList = viewMode === 'single' 
                              ? chunks 
                              : Object.values(allChunks).flat();
                            const idx = allChunksList.findIndex(c => c.id === selectedChunkId);
                            return idx >= 0 ? `#${idx + 1}` : 'None';
                          })() : 'None'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                {viewMode === 'single' ? (
                  <ChunkVisualizer
                    chunks={chunks}
                    selectedChunkId={selectedChunkId}
                    onChunkSelect={handleChunkSelect}
                  />
                ) : (
                  <div className="space-y-6">
                    {Object.entries(allChunks).map(([docId, docChunks]) => {
                      const doc = documents.find(d => d.id === docId);
                      return (
                        <div key={docId} className="border rounded-lg p-4">
                          <h3 className="font-semibold text-lg mb-3">{doc?.title || 'Unknown'}</h3>
                          <ChunkVisualizer
                            chunks={docChunks}
                            selectedChunkId={selectedChunkId}
                            onChunkSelect={handleChunkSelect}
                          />
                        </div>
                      );
                    })}
                  </div>
                )}
                {selectedChunkId && (
                  <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
                    <h3 className="font-semibold text-lg mb-3 flex items-center">
                      <span className="text-xl mr-2">🔍</span>
                      Selected Chunk Content
                    </h3>
                    <pre className="text-sm whitespace-pre-wrap bg-white p-4 rounded-lg border border-gray-200 max-h-64 overflow-y-auto">
                      {chunks.find(c => c.id === selectedChunkId)?.content}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'graph' && (
          <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800 flex items-center">
                <span className="text-3xl mr-3">🔗</span>
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
                      onClick={() => loadAllData()}
                      className="px-3 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
                      title="Refresh all documents data"
                    >
                      🔄 Refresh
                    </button>
                    <button
                      onClick={async () => {
                        try {
                          setLoading(true);
                          const response = await fetch('http://localhost:8745/api/graph/link-documents', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' }
                          });
                          const result = await response.json();
                          if (response.ok) {
                            alert(`Successfully linked documents!\n${result.cross_relationships} cross-document relationships created\n${result.entity_matches} entity matches found`);
                            // Reload data to show new relationships
                            await loadAllData();
                          } else {
                            alert('Failed to link documents: ' + result.detail);
                          }
                        } catch (error) {
                          console.error('Error linking documents:', error);
                          alert('Error linking documents');
                        } finally {
                          setLoading(false);
                        }
                      }}
                      className="px-3 py-2 rounded-lg bg-purple-600 text-white hover:bg-purple-700 transition-colors"
                      title="Link entities across all documents"
                    >
                      🔗 Link Graphs
                    </button>
                  </>
                )}
              </div>
            </div>
            {viewMode === 'single' && entities.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <span className="text-6xl block mb-4">🌐</span>
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
                <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-6">
                      <div>
                        <span className="text-sm text-gray-600">Entities</span>
                        <p className="text-2xl font-bold text-purple-600">
                          {viewMode === 'single' 
                            ? entities.length 
                            : Object.values(allEntities).reduce((sum, e) => sum + e.length, 0)}
                        </p>
                      </div>
                      <div className="h-8 w-px bg-gray-300"></div>
                      <div>
                        <span className="text-sm text-gray-600">Relationships</span>
                        <p className="text-2xl font-bold text-pink-600">
                          {viewMode === 'single' 
                            ? relationships.length 
                            : Object.values(allRelationships).reduce((sum, r) => sum + r.length, 0)}
                        </p>
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
                {viewMode === 'single' ? (
                  <div style={{ height: '600px' }}>
                    <GraphViewer
                      entities={entities}
                      relationships={relationships}
                      selectedEntityId={selectedEntityId}
                      onNodeSelect={handleEntitySelect}
                    />
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div style={{ height: '800px' }} className="border rounded-lg p-4">
                      <h3 className="font-semibold text-lg mb-3">Combined Knowledge Graph</h3>
                      <GraphViewer
                        entities={Object.values(allEntities).flat()}
                        relationships={Object.values(allRelationships).flat()}
                        selectedEntityId={selectedEntityId}
                        onNodeSelect={handleEntitySelect}
                      />
                    </div>
                  </div>
                )}
                {selectedEntityId && (
                  <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
                    <h3 className="font-semibold text-lg mb-3 flex items-center">
                      <span className="text-xl mr-2">🎯</span>
                      Selected Entity
                    </h3>
                    <pre className="text-sm bg-white p-4 rounded-lg border border-gray-200 max-h-64 overflow-y-auto">
                      {JSON.stringify(entities.find(e => e.id === selectedEntityId), null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'query' && (
          <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <span className="text-3xl mr-3">🔍</span>
              Query Interface
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Query Input - Takes up 2 columns on large screens */}
              <div className="lg:col-span-2 p-6 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-200">
                <label className="block text-lg font-semibold text-gray-700 mb-4">
                  What would you like to know?
                </label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
                    placeholder="Ask a question about your documents..."
                    className="flex-1 px-5 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent text-lg"
                  />
                  <button
                    onClick={handleQuery}
                    disabled={loading || !queryText.trim()}
                    className="px-8 py-3 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-lg shadow-md hover:from-orange-600 hover:to-amber-600 disabled:opacity-50 transition-all duration-200 transform hover:scale-105 font-semibold"
                  >
                    <span className="text-xl mr-2">🔎</span>
                    Search
                  </button>
                </div>
              </div>
              
              {/* Fusion Controls - Takes up 1 column on large screens */}
              <div className="lg:col-span-1">
                <FusionControls
                  onConfigChange={handleFusionConfigChange}
                  onPresetSelect={handlePresetSelect}
                />
              </div>
            </div>

            {/* Query Results */}
            {queryResults.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <span className="text-2xl mr-2">📡</span>
                  Results ({queryResults.length})
                </h3>
                <div className="space-y-4">
                  {queryResults.map((result, idx) => (
                    <div key={idx} className="p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200 hover:shadow-lg transition-shadow duration-200">
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center space-x-3">
                          <span className="text-2xl">{idx + 1}.</span>
                          <div className="px-3 py-1 bg-gradient-to-r from-green-100 to-emerald-100 rounded-full">
                            <span className="text-sm font-semibold text-green-800">
                              {(result.score * 100).toFixed(1)}% Match
                            </span>
                          </div>
                        </div>
                        <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                          {result.source}
                        </span>
                      </div>
                      <p className="text-gray-700 leading-relaxed">{result.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <span className="text-3xl mr-3">📈</span>
              System Statistics
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-100 rounded-xl border border-blue-200 transform hover:scale-105 transition-transform duration-200">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-4xl">📄</span>
                  <span className="text-xs font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">Active</span>
                </div>
                <h3 className="font-medium text-blue-900 text-sm">Documents</h3>
                <p className="text-3xl font-bold text-blue-700 mt-2">{documents.length}</p>
              </div>
              
              <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl border border-green-200 transform hover:scale-105 transition-transform duration-200">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-4xl">📦</span>
                  <span className="text-xs font-medium text-green-600 bg-green-100 px-2 py-1 rounded-full">Processing</span>
                </div>
                <h3 className="font-medium text-green-900 text-sm">Total Chunks</h3>
                <p className="text-3xl font-bold text-green-700 mt-2">{chunks.length}</p>
              </div>
              
              <div className="p-6 bg-gradient-to-br from-purple-50 to-pink-100 rounded-xl border border-purple-200 transform hover:scale-105 transition-transform duration-200">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-4xl">🌐</span>
                  <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded-full">Extracted</span>
                </div>
                <h3 className="font-medium text-purple-900 text-sm">Entities</h3>
                <p className="text-3xl font-bold text-purple-700 mt-2">{entities.length}</p>
              </div>
            </div>

            {selectedDocument && (
              <div className="p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
                <h3 className="font-semibold text-lg mb-4 flex items-center">
                  <span className="text-xl mr-2">📊</span>
                  Selected Document Analytics
                </h3>
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-white rounded-lg">
                      <span className="text-gray-600 font-medium">Title</span>
                      <span className="font-semibold text-gray-800">{selectedDocument.title}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-lg">
                      <span className="text-gray-600 font-medium">Chunks</span>
                      <span className="font-bold text-green-600 text-lg">{chunks.length}</span>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-white rounded-lg">
                      <span className="text-gray-600 font-medium">Entities</span>
                      <span className="font-bold text-purple-600 text-lg">{entities.length}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-lg">
                      <span className="text-gray-600 font-medium">Relationships</span>
                      <span className="font-bold text-pink-600 text-lg">{relationships.length}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading...</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App
