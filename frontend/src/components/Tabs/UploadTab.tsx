import React from 'react';
import type { Document } from '../../types';

interface UploadTabProps {
  documents: Document[];
  selectedDocument: Document | null;
  loading: boolean;
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onCreateDocument: () => void;
  onSelectDocument: (doc: Document) => void;
  processingStatus?: string;
}

export const UploadTab: React.FC<UploadTabProps> = ({
  documents,
  selectedDocument,
  loading,
  onFileUpload,
  onCreateDocument,
  onSelectDocument,
  processingStatus
}) => {
  return (
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
              onChange={onFileUpload}
              disabled={loading}
              className="hidden"
              id="file-upload"
              multiple
            />
            <label 
              htmlFor="file-upload"
              className={`cursor-pointer flex items-center justify-center px-6 py-3 rounded-lg shadow-md transition-all duration-200 transform hover:scale-105 ${
                loading 
                  ? 'bg-gradient-to-r from-gray-400 to-gray-500 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
              } text-white min-w-[200px]`}
            >
              {loading && (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              )}
              <span className="truncate">
                {loading ? (processingStatus ? processingStatus.substring(0, 50) : 'Processing...') : 'Choose Files'}
              </span>
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

      {/* Documents List */}
      <div>
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          Your Documents
        </h3>
        {documents.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <span className="text-6xl block mb-4">📄</span>
            <p className="text-lg">No documents yet</p>
            <p className="text-sm mt-2">Upload a document or create one from text to get started</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {documents.map((doc) => (
              <div
                key={doc.id}
                onClick={() => onSelectDocument(doc)}
                className={`p-6 rounded-xl cursor-pointer transition-all duration-200 transform hover:scale-102 hover:shadow-lg ${
                  selectedDocument?.id === doc.id 
                    ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-400 shadow-lg' 
                    : 'bg-white border-2 border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-start space-x-3">
                    <div>
                      <h4 className="font-semibold text-lg text-gray-800">{doc.title}</h4>
                      <div className="flex items-center mt-2 space-x-3">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                          doc.status === 'completed' ? 'bg-green-100 text-green-800' :
                          doc.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                          doc.status === 'failed' ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          <span className={`w-2 h-2 rounded-full mr-2 ${
                            doc.status === 'completed' ? 'bg-green-500' :
                            doc.status === 'processing' ? 'bg-yellow-500 animate-pulse' :
                            doc.status === 'failed' ? 'bg-red-500' :
                            'bg-gray-500'
                          }`}></span>
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
  );
};