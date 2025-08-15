import React from 'react';
import type { Document, Chunk, Entity, Relationship } from '../../types';

interface StatsTabProps {
  documents: Document[];
  selectedDocument: Document | null;
  chunks: Chunk[];
  entities: Entity[];
  relationships: Relationship[];
}

export const StatsTab: React.FC<StatsTabProps> = ({
  documents,
  selectedDocument,
  chunks,
  entities,
  relationships
}) => {
  // Calculate corpus-wide averages for documents with performance metrics
  const docsWithPerf = documents.filter(d => d.performance);
  const corpusStats = React.useMemo(() => {
    if (docsWithPerf.length === 0) return null;
    
    const avgTotalTime = docsWithPerf.reduce((sum, d) => sum + (d.performance?.total_time || 0), 0) / docsWithPerf.length;
    const avgChunkTime = docsWithPerf.reduce((sum, d) => sum + (d.performance?.chunking || 0), 0) / docsWithPerf.length;
    const avgExtractTime = docsWithPerf.reduce((sum, d) => sum + (d.performance?.entity_extraction || 0), 0) / docsWithPerf.length;
    const totalChars = docsWithPerf.reduce((sum, d) => sum + (d.performance?.content_length || 0), 0);
    const totalEntities = docsWithPerf.reduce((sum, d) => sum + (d.performance?.entity_count || 0), 0);
    const avgSpeed = totalChars / docsWithPerf.reduce((sum, d) => sum + (d.performance?.total_time || 0), 0) / 1000;
    
    return {
      avgTotalTime,
      avgChunkTime,
      avgExtractTime,
      totalChars,
      totalEntities,
      avgSpeed,
      entityDensity: (totalEntities / totalChars) * 1000,
      extractionPercent: (avgExtractTime / avgTotalTime) * 100
    };
  }, [docsWithPerf]);

  return (
    <div className="bg-white rounded-xl shadow-xl p-8 animate-fadeIn">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="text-3xl mr-3">📊</span>
        System Statistics
      </h2>
      
      {/* Overview Stats */}
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
            <span className="text-4xl">✂️</span>
            <span className="text-xs font-medium text-green-600 bg-green-100 px-2 py-1 rounded-full">Processing</span>
          </div>
          <h3 className="font-medium text-green-900 text-sm">Total Chunks</h3>
          <p className="text-3xl font-bold text-green-700 mt-2">{chunks.length}</p>
        </div>
        
        <div className="p-6 bg-gradient-to-br from-purple-50 to-pink-100 rounded-xl border border-purple-200 transform hover:scale-105 transition-transform duration-200">
          <div className="flex items-center justify-between mb-3">
            <span className="text-4xl">🔮</span>
            <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded-full">Extracted</span>
          </div>
          <h3 className="font-medium text-purple-900 text-sm">Entities</h3>
          <p className="text-3xl font-bold text-purple-700 mt-2">{entities.length}</p>
        </div>
      </div>

      {/* Selected Document Analytics */}
      {selectedDocument && (
        <>
          <div className="p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200 mb-6">
            <h3 className="font-semibold text-lg mb-4 flex items-center">
              📈 Selected Document Analytics
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

          {/* Performance Metrics */}
          {selectedDocument.performance && (
            <div className="p-6 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-200">
              <h3 className="font-semibold text-lg mb-4 flex items-center">
                🚀 Performance Metrics - {selectedDocument.title}
              </h3>
              
              {/* Processing Times */}
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Processing Times</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <MetricCard
                    icon="📂"
                    label="File Read"
                    value={`${selectedDocument.performance.file_read || 0}s`}
                    color="orange"
                  />
                  <MetricCard
                    icon="✂️"
                    label="Chunking"
                    value={`${selectedDocument.performance.chunking || 0}s`}
                    color="orange"
                  />
                  <MetricCard
                    icon="🔍"
                    label="Entity Extraction"
                    value={`${selectedDocument.performance.entity_extraction || 0}s`}
                    color="orange"
                  />
                  <MetricCard
                    icon="💾"
                    label="Storage"
                    value={`${selectedDocument.performance.document_storage || 0}s`}
                    color="orange"
                  />
                  <MetricCard
                    icon="⏱️"
                    label="Total Time"
                    value={`${selectedDocument.performance.total_time || 0}s`}
                    color="green"
                  />
                  <MetricCard
                    icon="📄"
                    label="Content Size"
                    value={`${((selectedDocument.performance.content_length || 0) / 1000).toFixed(1)}k`}
                    color="blue"
                  />
                </div>
              </div>

              {/* Efficiency Metrics */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">Efficiency Metrics</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <MetricCard
                    icon="⚡"
                    label="Processing Speed"
                    value={`${selectedDocument.performance.content_length && selectedDocument.performance.total_time
                      ? ((selectedDocument.performance.content_length / selectedDocument.performance.total_time) / 1000).toFixed(1)
                      : 0} k chars/s`}
                    color="blue"
                    gradient
                  />
                  <MetricCard
                    icon="📊"
                    label="Chunks per Second"
                    value={`${selectedDocument.performance.chunk_count && selectedDocument.performance.chunking
                      ? (selectedDocument.performance.chunk_count / selectedDocument.performance.chunking).toFixed(1)
                      : 0} chunks/s`}
                    color="green"
                    gradient
                  />
                  <MetricCard
                    icon="🎯"
                    label="Entities per Second"
                    value={`${selectedDocument.performance.entity_count && selectedDocument.performance.entity_extraction
                      ? (selectedDocument.performance.entity_count / selectedDocument.performance.entity_extraction).toFixed(1)
                      : 0} entities/s`}
                    color="purple"
                    gradient
                  />
                  <MetricCard
                    icon="📈"
                    label="Entity Density"
                    value={`${selectedDocument.performance.entity_count && selectedDocument.performance.content_length
                      ? ((selectedDocument.performance.entity_count / selectedDocument.performance.content_length) * 1000).toFixed(2)
                      : 0} per 1k chars`}
                    color="yellow"
                    gradient
                  />
                  <MetricCard
                    icon="⏳"
                    label="Extraction Overhead"
                    value={`${selectedDocument.performance.entity_extraction && selectedDocument.performance.total_time
                      ? ((selectedDocument.performance.entity_extraction / selectedDocument.performance.total_time) * 100).toFixed(0)
                      : 0}%`}
                    color="red"
                    gradient
                  />
                  <MetricCard
                    icon="📏"
                    label="Avg Chunk Size"
                    value={`${selectedDocument.performance.content_length && selectedDocument.performance.chunk_count
                      ? (selectedDocument.performance.content_length / selectedDocument.performance.chunk_count).toFixed(0)
                      : 0} chars`}
                    color="cyan"
                    gradient
                  />
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Corpus-wide Average Metrics */}
      {corpusStats && (
        <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
          <h3 className="font-semibold text-lg mb-4 flex items-center">
            📊 Corpus-wide Performance Averages ({docsWithPerf.length} documents)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-3 bg-white rounded-lg">
              <div className="text-xs text-gray-500">Avg Processing Time</div>
              <div className="text-lg font-bold text-gray-700">{corpusStats.avgTotalTime.toFixed(2)}s</div>
            </div>
            <div className="p-3 bg-white rounded-lg">
              <div className="text-xs text-gray-500">Avg Speed</div>
              <div className="text-lg font-bold text-blue-600">{corpusStats.avgSpeed.toFixed(1)} k chars/s</div>
            </div>
            <div className="p-3 bg-white rounded-lg">
              <div className="text-xs text-gray-500">Avg Entity Density</div>
              <div className="text-lg font-bold text-purple-600">
                {corpusStats.entityDensity.toFixed(2)} per 1k
              </div>
            </div>
            <div className="p-3 bg-white rounded-lg">
              <div className="text-xs text-gray-500">Extraction %</div>
              <div className="text-lg font-bold text-orange-600">
                {corpusStats.extractionPercent.toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper component for metric cards
const MetricCard: React.FC<{
  icon: string;
  label: string;
  value: string;
  color: string;
  gradient?: boolean;
}> = ({ icon, label, value, color, gradient }) => {
  const colorClasses = {
    orange: gradient ? 'bg-gradient-to-r from-orange-50 to-amber-50' : 'bg-white border-orange-100',
    green: gradient ? 'bg-gradient-to-r from-green-50 to-emerald-50' : 'bg-white border-green-100',
    blue: gradient ? 'bg-gradient-to-r from-blue-50 to-indigo-50' : 'bg-white border-blue-100',
    purple: gradient ? 'bg-gradient-to-r from-purple-50 to-pink-50' : 'bg-white border-purple-100',
    yellow: gradient ? 'bg-gradient-to-r from-yellow-50 to-amber-50' : 'bg-white border-yellow-100',
    red: gradient ? 'bg-gradient-to-r from-red-50 to-pink-50' : 'bg-white border-red-100',
    cyan: gradient ? 'bg-gradient-to-r from-cyan-50 to-blue-50' : 'bg-white border-cyan-100',
    gray: 'bg-white border-gray-100'
  };

  const textColors = {
    orange: 'text-orange-600',
    green: 'text-green-600',
    blue: 'text-blue-600',
    purple: 'text-purple-600',
    yellow: 'text-yellow-600',
    red: 'text-red-600',
    cyan: 'text-cyan-600',
    gray: 'text-gray-600'
  };

  return (
    <div className={`p-3 rounded-lg border ${colorClasses[color as keyof typeof colorClasses] || colorClasses.gray}`}>
      <div className="text-xs text-gray-500">
        <span className="mr-1">{icon}</span>
        {label}
      </div>
      <div className={`text-lg font-bold ${textColors[color as keyof typeof textColors] || textColors.gray}`}>
        {value}
      </div>
    </div>
  );
};