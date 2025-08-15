import React, { useState, useEffect } from 'react';
import { SearchInterface } from './SearchInterface/SearchInterface';
import { DocumentExplorer } from './DocumentExplorer/DocumentExplorer';
import { KnowledgeNavigator } from './KnowledgeNavigator/KnowledgeNavigator';
import { Analytics } from './Analytics/Analytics';
import type { Document, Chunk, Entity, Relationship } from '../../types';

type DemoViewMode = 'search' | 'explore' | 'navigate' | 'analytics';

interface DemoTabProps {
  documents: Document[];
  chunks: Chunk[];
  entities: Entity[];
  relationships: Relationship[];
  loading: boolean;
  onDocumentSelect: (doc: Document) => void;
}

export const DemoTab: React.FC<DemoTabProps> = ({
  documents,
  chunks,
  entities,
  relationships,
  loading,
  onDocumentSelect
}) => {
  const [viewMode, setViewMode] = useState<DemoViewMode>('search');
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [searchHistory, setSearchHistory] = useState<any[]>([]);

  const handleDocumentSelect = (doc: Document) => {
    setSelectedDocument(doc);
    onDocumentSelect(doc);
  };

  const handleSearchComplete = (searchData: any) => {
    setSearchHistory(prev => [searchData, ...prev].slice(0, 20)); // Keep last 20 searches
  };

  return (
    <div className="bg-white rounded-xl shadow-xl animate-fadeIn">
      {/* Header with View Mode Switcher */}
      <div className="border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              RAG System Demo
            </h2>
            <p className="mt-2 text-gray-600">
              Experience the power of advanced document search and knowledge discovery
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500">Documents:</span>
            <span className="font-bold text-lg text-indigo-600">{documents.length}</span>
            <span className="text-gray-400 mx-2">|</span>
            <span className="text-sm text-gray-500">Entities:</span>
            <span className="font-bold text-lg text-purple-600">{entities.length}</span>
          </div>
        </div>

        {/* View Mode Tabs */}
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('search')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              viewMode === 'search'
                ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <span className="mr-2">🔍</span>
            Smart Search
          </button>
          <button
            onClick={() => setViewMode('explore')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              viewMode === 'explore'
                ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <span className="mr-2">📚</span>
            Document Explorer
          </button>
          <button
            onClick={() => setViewMode('navigate')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              viewMode === 'navigate'
                ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <span className="mr-2">🧭</span>
            Knowledge Navigator
          </button>
          <button
            onClick={() => setViewMode('analytics')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              viewMode === 'analytics'
                ? 'bg-gradient-to-r from-orange-500 to-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <span className="mr-2">📊</span>
            Analytics
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="p-6">
        {viewMode === 'search' && (
          <SearchInterface
            documents={documents}
            chunks={chunks}
            entities={entities}
            onSearchComplete={handleSearchComplete}
            onDocumentSelect={handleDocumentSelect}
          />
        )}

        {viewMode === 'explore' && (
          <DocumentExplorer
            documents={documents}
            chunks={chunks}
            entities={entities}
            onDocumentSelect={handleDocumentSelect}
            selectedDocument={selectedDocument}
          />
        )}

        {viewMode === 'navigate' && (
          <KnowledgeNavigator
            entities={entities}
            relationships={relationships}
            chunks={chunks}
            documents={documents}
          />
        )}

        {viewMode === 'analytics' && (
          <Analytics
            searchHistory={searchHistory}
            documents={documents}
            chunks={chunks}
            entities={entities}
            relationships={relationships}
          />
        )}
      </div>

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded-xl">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-500 border-t-transparent mx-auto"></div>
            <p className="mt-4 text-gray-600">Processing...</p>
          </div>
        </div>
      )}
    </div>
  );
};