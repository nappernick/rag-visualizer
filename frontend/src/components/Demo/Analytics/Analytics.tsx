import React, { useState, useEffect } from 'react';
import type { Document, Chunk, Entity, Relationship } from '../../../types';

interface AnalyticsProps {
  searchHistory: any[];
  documents: Document[];
  chunks: Chunk[];
  entities: Entity[];
  relationships: Relationship[];
}

export const Analytics: React.FC<AnalyticsProps> = ({
  searchHistory,
  documents,
  chunks,
  entities,
  relationships
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'performance' | 'quality' | 'usage'>('performance');
  
  // Calculate statistics
  const avgResponseTime = searchHistory.length > 0
    ? searchHistory.reduce((sum, s) => sum + (s.metrics?.processingTime || 0), 0) / searchHistory.length
    : 0;
  
  const avgResultCount = searchHistory.length > 0
    ? searchHistory.reduce((sum, s) => sum + (s.results?.length || 0), 0) / searchHistory.length
    : 0;
  
  const searchStrategies = searchHistory.reduce((acc, s) => {
    const strategy = s.metrics?.strategy || 'unknown';
    acc[strategy] = (acc[strategy] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  const topQueries = searchHistory
    .slice(0, 5)
    .map(s => ({
      query: s.query,
      time: s.timestamp,
      results: s.results?.length || 0,
      processingTime: s.metrics?.processingTime || 0
    }));

  return (
    <div className="space-y-6">
      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border border-blue-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-3xl">📊</span>
            <span className="text-xs font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">System</span>
          </div>
          <div className="text-2xl font-bold text-blue-700">{documents.length}</div>
          <div className="text-sm text-blue-600">Total Documents</div>
          <div className="mt-2 text-xs text-blue-500">
            {chunks.length} chunks • {entities.length} entities
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border border-green-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-3xl">⚡</span>
            <span className="text-xs font-medium text-green-600 bg-green-100 px-2 py-1 rounded-full">Performance</span>
          </div>
          <div className="text-2xl font-bold text-green-700">{avgResponseTime.toFixed(0)}ms</div>
          <div className="text-sm text-green-600">Avg Response Time</div>
          <div className="mt-2 text-xs text-green-500">
            {searchHistory.length} total searches
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border border-purple-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-3xl">🎯</span>
            <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded-full">Quality</span>
          </div>
          <div className="text-2xl font-bold text-purple-700">{avgResultCount.toFixed(1)}</div>
          <div className="text-sm text-purple-600">Avg Results/Query</div>
          <div className="mt-2 text-xs text-purple-500">
            {relationships.length} relationships
          </div>
        </div>

        <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-6 border border-orange-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-3xl">🔍</span>
            <span className="text-xs font-medium text-orange-600 bg-orange-100 px-2 py-1 rounded-full">Usage</span>
          </div>
          <div className="text-2xl font-bold text-orange-700">{searchHistory.length}</div>
          <div className="text-sm text-orange-600">Total Searches</div>
          <div className="mt-2 text-xs text-orange-500">
            {Object.keys(searchStrategies).length} strategies used
          </div>
        </div>
      </div>

      {/* Metric Selector */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Detailed Analytics</h3>
          <div className="flex space-x-2">
            {(['performance', 'quality', 'usage'] as const).map((metric) => (
              <button
                key={metric}
                onClick={() => setSelectedMetric(metric)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedMetric === metric
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {metric.charAt(0).toUpperCase() + metric.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Performance Metrics */}
        {selectedMetric === 'performance' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Response Time Distribution</h4>
              <div className="space-y-2">
                {['< 100ms', '100-500ms', '500-1000ms', '> 1000ms'].map((range, idx) => {
                  const count = searchHistory.filter(s => {
                    const time = s.metrics?.processingTime || 0;
                    if (idx === 0) return time < 100;
                    if (idx === 1) return time >= 100 && time < 500;
                    if (idx === 2) return time >= 500 && time < 1000;
                    return time >= 1000;
                  }).length;
                  const percentage = searchHistory.length > 0 ? (count / searchHistory.length) * 100 : 0;
                  
                  return (
                    <div key={range} className="flex items-center space-x-3">
                      <span className="text-sm text-gray-600 w-24">{range}</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-6 relative">
                        <div
                          className="bg-gradient-to-r from-blue-400 to-blue-600 h-full rounded-full flex items-center justify-end pr-2"
                          style={{ width: `${percentage}%` }}
                        >
                          {percentage > 10 && (
                            <span className="text-xs text-white font-medium">{percentage.toFixed(0)}%</span>
                          )}
                        </div>
                      </div>
                      <span className="text-sm text-gray-500 w-12 text-right">{count}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Strategy Performance</h4>
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(searchStrategies).map(([strategy, count]) => (
                  <div key={strategy} className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-lg font-bold text-gray-700">{count}</div>
                    <div className="text-xs text-gray-500">{strategy}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Quality Metrics */}
        {selectedMetric === 'quality' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Result Quality Distribution</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="text-2xl font-bold text-green-700">
                    {(searchHistory.filter(s => (s.results?.length || 0) > 5).length / Math.max(searchHistory.length, 1) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-green-600">High Result Count (&gt;5)</div>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                  <div className="text-2xl font-bold text-orange-700">
                    {(searchHistory.filter(s => (s.results?.length || 0) === 0).length / Math.max(searchHistory.length, 1) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-orange-600">No Results</div>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Entity Coverage</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Total Entities</span>
                  <span className="font-medium text-gray-900">{entities.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Avg Frequency</span>
                  <span className="font-medium text-gray-900">
                    {(entities.reduce((sum, e) => sum + e.frequency, 0) / Math.max(entities.length, 1)).toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Documents with Entities</span>
                  <span className="font-medium text-gray-900">
                    {new Set(entities.flatMap(e => e.document_ids)).size}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Usage Metrics */}
        {selectedMetric === 'usage' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Recent Searches</h4>
              <div className="space-y-2">
                {topQueries.map((query, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900 line-clamp-1">{query.query}</p>
                        <div className="mt-1 flex items-center space-x-3 text-xs text-gray-500">
                          <span>{new Date(query.time).toLocaleTimeString()}</span>
                          <span>•</span>
                          <span>{query.results} results</span>
                          <span>•</span>
                          <span>{query.processingTime}ms</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">System Utilization</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <div className="text-lg font-bold text-blue-700">
                    {((chunks.length / Math.max(documents.length, 1))).toFixed(0)}
                  </div>
                  <div className="text-xs text-blue-600">Avg Chunks/Doc</div>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <div className="text-lg font-bold text-purple-700">
                    {((entities.length / Math.max(documents.length, 1))).toFixed(0)}
                  </div>
                  <div className="text-xs text-purple-600">Avg Entities/Doc</div>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <div className="text-lg font-bold text-green-700">
                    {((relationships.length / Math.max(entities.length, 1))).toFixed(1)}
                  </div>
                  <div className="text-xs text-green-600">Avg Relations/Entity</div>
                </div>
                <div className="p-3 bg-orange-50 rounded-lg">
                  <div className="text-lg font-bold text-orange-700">
                    {searchHistory.length > 0 ? (searchHistory.length / documents.length).toFixed(1) : 0}
                  </div>
                  <div className="text-xs text-orange-600">Searches/Doc</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};