import React, { useState, useEffect } from 'react';
import type { Document, Chunk, Entity } from '../../../types';
import { demoApi } from '../../../services/demoApi';

interface DocumentExplorerProps {
  documents: Document[];
  chunks: Chunk[];
  entities: Entity[];
  onDocumentSelect: (doc: Document) => void;
  selectedDocument: Document | null;
}

export const DocumentExplorer: React.FC<DocumentExplorerProps> = ({
  documents,
  chunks,
  entities,
  onDocumentSelect,
  selectedDocument
}) => {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [documentSummary, setDocumentSummary] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const filteredDocuments = documents.filter(doc =>
    doc.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.content?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleDocumentClick = async (doc: Document) => {
    setSelectedDocId(doc.id);
    onDocumentSelect(doc);
    
    // Load document summary
    setLoading(true);
    try {
      const summary = await demoApi.summarizeDocument(doc.id, 'brief');
      setDocumentSummary(summary);
    } catch (error) {
      console.error('Error loading summary:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDocumentIcon = (docType: string) => {
    switch (docType) {
      case 'pdf': return '📄';
      case 'text': return '📝';
      case 'markdown': return '📋';
      case 'code': return '💻';
      default: return '📄';
    }
  };

  return (
    <div className="space-y-6">
      {/* Search and View Controls */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200">
        <div className="flex items-center justify-between">
          <div className="flex-1 max-w-xl">
            <div className="relative">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search documents..."
                className="w-full px-4 py-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <span className="absolute left-3 top-2.5 text-gray-400">🔍</span>
            </div>
          </div>
          <div className="flex items-center space-x-2 ml-4">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-lg ${viewMode === 'grid' ? 'bg-green-600 text-white' : 'bg-white text-gray-700'}`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-lg ${viewMode === 'list' ? 'bg-green-600 text-white' : 'bg-white text-gray-700'}`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Document Grid/List */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Documents ({filteredDocuments.length})
            </h3>
            
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-2 gap-4">
                {filteredDocuments.map((doc) => (
                  <div
                    key={doc.id}
                    onClick={() => handleDocumentClick(doc)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                      selectedDocId === doc.id
                        ? 'border-green-500 bg-green-50'
                        : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <span className="text-3xl">{getDocumentIcon(doc.doc_type)}</span>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-gray-900 truncate">{doc.title}</h4>
                        <p className="text-sm text-gray-500 mt-1">
                          {doc.doc_type} • {new Date(doc.created_at).toLocaleDateString()}
                        </p>
                        <div className="mt-2">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            doc.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            {doc.status}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-2">
                {filteredDocuments.map((doc) => (
                  <div
                    key={doc.id}
                    onClick={() => handleDocumentClick(doc)}
                    className={`p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                      selectedDocId === doc.id
                        ? 'border-green-500 bg-green-50'
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{getDocumentIcon(doc.doc_type)}</span>
                        <div>
                          <h4 className="font-medium text-gray-900">{doc.title}</h4>
                          <p className="text-sm text-gray-500">
                            {doc.doc_type} • {new Date(doc.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        doc.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {doc.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Document Intelligence Panel */}
        <div className="lg:col-span-1">
          {selectedDocument ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Document Intelligence
              </h3>
              
              {loading ? (
                <div className="space-y-3 animate-pulse">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-4 bg-gray-200 rounded w-full"></div>
                  <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                </div>
              ) : documentSummary ? (
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Summary</h4>
                    <p className="text-sm text-gray-600">{documentSummary.summary}</p>
                  </div>
                  
                  {documentSummary.key_points && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Key Points</h4>
                      <ul className="space-y-1">
                        {documentSummary.key_points.map((point: string, idx: number) => (
                          <li key={idx} className="text-sm text-gray-600 flex items-start">
                            <span className="text-green-500 mr-2">•</span>
                            {point}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Statistics</h4>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-center p-2 bg-gray-50 rounded">
                        <div className="text-lg font-bold text-gray-700">
                          {chunks.filter(c => c.document_id === selectedDocument.id).length}
                        </div>
                        <div className="text-xs text-gray-500">Chunks</div>
                      </div>
                      <div className="text-center p-2 bg-gray-50 rounded">
                        <div className="text-lg font-bold text-gray-700">
                          {entities.filter(e => e.document_ids.includes(selectedDocument.id)).length}
                        </div>
                        <div className="text-xs text-gray-500">Entities</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-gray-500">Loading document intelligence...</p>
              )}
            </div>
          ) : (
            <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-300">
              <div className="text-gray-400 text-6xl mb-4">📊</div>
              <p className="text-gray-600">Select a document to view intelligence</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};