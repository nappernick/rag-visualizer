import React from 'react';

interface ProcessingStatusProps {
  status: string;
  performanceMetrics?: any;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ status, performanceMetrics }) => {
  const isSuccess = status.startsWith('✅');
  
  return (
    <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 rounded-lg shadow-md animate-slideIn">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          {!isSuccess && (
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent mr-3"></div>
          )}
          <span className="text-blue-800 font-medium">{status}</span>
        </div>
        {performanceMetrics && (
          <div className="text-xs text-gray-600">
            {performanceMetrics.total_time && (
              <span className="ml-4">Total: {performanceMetrics.total_time}s</span>
            )}
          </div>
        )}
      </div>
      
      {performanceMetrics && !isSuccess && (
        <div className="mt-3 flex gap-4 text-xs text-gray-600">
          {performanceMetrics.file_read && (
            <span>📂 Read: {performanceMetrics.file_read}s</span>
          )}
          {performanceMetrics.content_processing && (
            <span>⚙️ Process: {performanceMetrics.content_processing}s</span>
          )}
          {performanceMetrics.chunking && (
            <span>✂️ Chunk: {performanceMetrics.chunking}s</span>
          )}
          {performanceMetrics.entity_extraction && (
            <span>🔍 Extract: {performanceMetrics.entity_extraction}s</span>
          )}
          {performanceMetrics.document_storage && (
            <span>💾 Store: {performanceMetrics.document_storage}s</span>
          )}
        </div>
      )}
    </div>
  );
};