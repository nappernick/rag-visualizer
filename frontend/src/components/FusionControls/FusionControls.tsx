import React, { useState, useEffect } from 'react';

interface FusionConfig {
  vector_weight: number;
  graph_weight: number;
  vector_top_k: number;
  graph_top_k: number;
  final_top_k: number;
  chunk_size: number;
  chunk_overlap: number;
  graph_confidence_threshold: number;
  graph_expansion_depth: number;
  entity_relevance_threshold: number;
  use_reranker: boolean;
  reranker_weight: number;
  context_budget: number;
  prioritize_summaries: boolean;
  summary_boost: number;
  auto_strategy: boolean;
  force_hybrid_threshold: number;
}

interface FusionControlsProps {
  onConfigChange: (config: FusionConfig) => void;
  onPresetSelect: (preset: string) => void;
}

const presets = {
  balanced: {
    name: 'Balanced',
    description: 'Equal weight to vector and graph',
    icon: '⚖️',
    config: {
      vector_weight: 0.5,
      graph_weight: 0.5,
      graph_expansion_depth: 2,
      use_reranker: true
    }
  },
  technical_documentation: {
    name: 'Technical Docs',
    description: 'Optimized for code and documentation',
    icon: '📚',
    config: {
      vector_weight: 0.8,
      graph_weight: 0.2,
      chunk_size: 600,
      prioritize_summaries: false
    }
  },
  conceptual_learning: {
    name: 'Conceptual',
    description: 'Focus on relationships and concepts',
    icon: '🧠',
    config: {
      vector_weight: 0.3,
      graph_weight: 0.7,
      graph_expansion_depth: 3,
      prioritize_summaries: true
    }
  },
  code_search: {
    name: 'Code Search',
    description: 'Precise code snippet retrieval',
    icon: '💻',
    config: {
      vector_weight: 0.9,
      graph_weight: 0.1,
      chunk_size: 400,
      use_reranker: false
    }
  },
  research_papers: {
    name: 'Research',
    description: 'Academic papers and research',
    icon: '🔬',
    config: {
      vector_weight: 0.6,
      graph_weight: 0.4,
      chunk_size: 800,
      context_budget: 12000,
      summary_boost: 1.5
    }
  },
  qa_chatbot: {
    name: 'Q&A Chat',
    description: 'Conversational question answering',
    icon: '💬',
    config: {
      vector_weight: 0.5,
      graph_weight: 0.5,
      final_top_k: 5,
      context_budget: 4000,
      auto_strategy: true
    }
  }
};

export const FusionControls: React.FC<FusionControlsProps> = ({ 
  onConfigChange, 
  onPresetSelect 
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  
  const [config, setConfig] = useState<FusionConfig>({
    vector_weight: 0.7,
    graph_weight: 0.3,
    vector_top_k: 20,
    graph_top_k: 15,
    final_top_k: 10,
    chunk_size: 400,
    chunk_overlap: 50,
    graph_confidence_threshold: 0.6,
    graph_expansion_depth: 2,
    entity_relevance_threshold: 0.7,
    use_reranker: true,
    reranker_weight: 0.5,
    context_budget: 8000,
    prioritize_summaries: true,
    summary_boost: 1.2,
    auto_strategy: true,
    force_hybrid_threshold: 0.5
  });

  useEffect(() => {
    // Update graph_weight when vector_weight changes
    setConfig(prev => ({
      ...prev,
      graph_weight: 1 - prev.vector_weight
    }));
  }, [config.vector_weight]);

  const handleConfigUpdate = (field: keyof FusionConfig, value: any) => {
    const updatedConfig = { ...config, [field]: value };
    
    // Ensure vector and graph weights sum to 1
    if (field === 'vector_weight') {
      updatedConfig.graph_weight = 1 - value;
    } else if (field === 'graph_weight') {
      updatedConfig.vector_weight = 1 - value;
    }
    
    setConfig(updatedConfig);
    onConfigChange(updatedConfig);
    setSelectedPreset(null); // Clear preset when manual changes are made
  };

  const applyPreset = (presetKey: string) => {
    const preset = presets[presetKey as keyof typeof presets];
    if (preset) {
      const updatedConfig = { ...config, ...preset.config };
      setConfig(updatedConfig);
      onConfigChange(updatedConfig);
      onPresetSelect(presetKey);
      setSelectedPreset(presetKey);
    }
  };

  return (
    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 border border-indigo-200">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center">
          <span className="text-2xl mr-2">🎛️</span>
          Fusion Controls
        </h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-indigo-600 hover:text-indigo-800 font-medium flex items-center"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced
          <span className="ml-1">{showAdvanced ? '▲' : '▼'}</span>
        </button>
      </div>

      {/* Preset Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Quick Presets
        </label>
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(presets).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => applyPreset(key)}
              className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                selectedPreset === key
                  ? 'border-indigo-500 bg-indigo-100'
                  : 'border-gray-200 bg-white hover:border-indigo-300 hover:bg-indigo-50'
              }`}
            >
              <div className="text-2xl mb-1">{preset.icon}</div>
              <div className="text-sm font-medium text-gray-800">{preset.name}</div>
              <div className="text-xs text-gray-500 mt-1">{preset.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Core Controls */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Vector vs Graph Weight
          </label>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">Vector</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.vector_weight}
              onChange={(e) => handleConfigUpdate('vector_weight', parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm text-gray-600">Graph</span>
            <div className="ml-2 px-3 py-1 bg-white rounded-lg border border-gray-300">
              <span className="text-sm font-mono">
                {config.vector_weight.toFixed(1)} / {config.graph_weight.toFixed(1)}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Final Results
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={config.final_top_k}
              onChange={(e) => handleConfigUpdate('final_top_k', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Context Budget
            </label>
            <input
              type="number"
              min="1000"
              max="20000"
              step="1000"
              value={config.context_budget}
              onChange={(e) => handleConfigUpdate('context_budget', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>

        <div className="flex items-center space-x-6">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.use_reranker}
              onChange={(e) => handleConfigUpdate('use_reranker', e.target.checked)}
              className="mr-2 h-4 w-4 text-indigo-600 rounded"
            />
            <span className="text-sm text-gray-700">Use Reranker</span>
          </label>
          
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.auto_strategy}
              onChange={(e) => handleConfigUpdate('auto_strategy', e.target.checked)}
              className="mr-2 h-4 w-4 text-indigo-600 rounded"
            />
            <span className="text-sm text-gray-700">Auto Strategy</span>
          </label>
          
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.prioritize_summaries}
              onChange={(e) => handleConfigUpdate('prioritize_summaries', e.target.checked)}
              className="mr-2 h-4 w-4 text-indigo-600 rounded"
            />
            <span className="text-sm text-gray-700">Prioritize Summaries</span>
          </label>
        </div>
      </div>

      {/* Advanced Controls */}
      {showAdvanced && (
        <div className="border-t border-indigo-200 pt-6 space-y-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Advanced Settings</h4>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Vector Top K
              </label>
              <input
                type="number"
                min="5"
                max="100"
                value={config.vector_top_k}
                onChange={(e) => handleConfigUpdate('vector_top_k', parseInt(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Graph Top K
              </label>
              <input
                type="number"
                min="5"
                max="100"
                value={config.graph_top_k}
                onChange={(e) => handleConfigUpdate('graph_top_k', parseInt(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Chunk Size
              </label>
              <input
                type="number"
                min="100"
                max="2000"
                step="100"
                value={config.chunk_size}
                onChange={(e) => handleConfigUpdate('chunk_size', parseInt(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Chunk Overlap
              </label>
              <input
                type="number"
                min="0"
                max="500"
                step="10"
                value={config.chunk_overlap}
                onChange={(e) => handleConfigUpdate('chunk_overlap', parseInt(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Graph Depth
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={config.graph_expansion_depth}
                onChange={(e) => handleConfigUpdate('graph_expansion_depth', parseInt(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Summary Boost
              </label>
              <input
                type="number"
                min="1"
                max="2"
                step="0.1"
                value={config.summary_boost}
                onChange={(e) => handleConfigUpdate('summary_boost', parseFloat(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Graph Confidence Threshold
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.graph_confidence_threshold}
                onChange={(e) => handleConfigUpdate('graph_confidence_threshold', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 text-center">
                {config.graph_confidence_threshold.toFixed(1)}
              </div>
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Entity Relevance Threshold
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.entity_relevance_threshold}
                onChange={(e) => handleConfigUpdate('entity_relevance_threshold', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 text-center">
                {config.entity_relevance_threshold.toFixed(1)}
              </div>
            </div>
          </div>
          
          {config.use_reranker && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Reranker Weight
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.reranker_weight}
                onChange={(e) => handleConfigUpdate('reranker_weight', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 text-center">
                {config.reranker_weight.toFixed(1)}
              </div>
            </div>
          )}
          
          {config.auto_strategy && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Force Hybrid Threshold
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.force_hybrid_threshold}
                onChange={(e) => handleConfigUpdate('force_hybrid_threshold', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 text-center">
                {config.force_hybrid_threshold.toFixed(1)}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Current Strategy Indicator */}
      <div className="mt-6 p-3 bg-white rounded-lg border border-gray-200">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Active Strategy:</span>
          <div className="flex items-center space-x-2">
            {config.auto_strategy ? (
              <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                Auto-detect
              </span>
            ) : config.vector_weight > 0.7 ? (
              <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                Vector-focused
              </span>
            ) : config.graph_weight > 0.7 ? (
              <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs font-medium rounded-full">
                Graph-focused
              </span>
            ) : (
              <span className="px-2 py-1 bg-orange-100 text-orange-800 text-xs font-medium rounded-full">
                Hybrid
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};