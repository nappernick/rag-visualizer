import React, { useState, useEffect } from 'react';
import './App.css';

// Layout components
import { Header, NavigationTabs, ErrorDisplay, ProcessingStatus, TabType } from './components/Layout';

// Tab components
import { UploadTab, QueryTab, StatsTab, ChunksTab, GraphTab } from './components/Tabs';

// Other components
import { DemoTab } from './components/Demo/DemoTab';
import { WeightRulesManager } from './components/WeightRules/WeightRulesManager';

// Custom hooks
import { useDocumentManager, useFileUpload, useQueryEngine, useDataLoader } from './hooks';

function App() {
  // Tab navigation state
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [viewMode, setViewMode] = useState<'single' | 'all'>('single');
  const [selectedChunkId, setSelectedChunkId] = useState<string | undefined>();
  const [selectedEntityId, setSelectedEntityId] = useState<string | undefined>();

  // Use custom hooks for state management
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
  const error = docError || queryError;

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

  // Event handlers
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
      <Header onClearAll={handleClearAll} />
      <NavigationTabs activeTab={activeTab} onTabChange={setActiveTab} />
      
      <main className="max-w-7xl mx-auto px-6 pb-12">
        {error && <ErrorDisplay error={error} onDismiss={() => setDocError(null)} />}
        {processingStatus && <ProcessingStatus status={processingStatus} performanceMetrics={performanceMetrics} />}

        {/* Tab Content */}
        {activeTab === 'upload' && (
          <UploadTab
            documents={documents}
            selectedDocument={selectedDocument}
            loading={loading}
            onFileUpload={handleFileUpload}
            onCreateDocument={handleCreateDocument}
            onSelectDocument={selectDocument}
            processingStatus={processingStatus}
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

export default App;