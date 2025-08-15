import { useState, useCallback } from 'react';
import { queryApi } from '../services/api';

export const useQueryEngine = () => {
  const [queryText, setQueryText] = useState('');
  const [queryResults, setQueryResults] = useState<any[]>([]);
  const [fusionConfig, setFusionConfig] = useState<any>(null);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeQuery = useCallback(async () => {
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
      return response;
      
    } catch (err) {
      setError('Failed to execute query');
      console.error(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [queryText, fusionConfig, selectedPreset]);

  const updateFusionConfig = useCallback((config: any) => {
    setFusionConfig(config);
  }, []);

  const selectPreset = useCallback((preset: string) => {
    setSelectedPreset(preset);
  }, []);

  const clearResults = useCallback(() => {
    setQueryResults([]);
    setError(null);
  }, []);

  return {
    // State
    queryText,
    queryResults,
    fusionConfig,
    selectedPreset,
    loading,
    error,
    
    // Actions
    setQueryText,
    executeQuery,
    updateFusionConfig,
    selectPreset,
    clearResults,
    setError
  };
};