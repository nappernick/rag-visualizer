import React, { useState, useRef, useEffect } from 'react';

interface QueryInputProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: () => void;
  suggestions: string[];
  onSuggestionSelect: (suggestion: string) => void;
  loading: boolean;
}

export const QueryInput: React.FC<QueryInputProps> = ({
  query,
  onQueryChange,
  onSearch,
  suggestions,
  onSuggestionSelect,
  loading
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setShowSuggestions(suggestions.length > 0 && query.length > 2);
    setSelectedSuggestionIndex(-1);
  }, [suggestions, query]);

  // Handle click outside to close suggestions
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(event.target as Node) &&
          inputRef.current && !inputRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (selectedSuggestionIndex >= 0 && suggestions[selectedSuggestionIndex]) {
        onSuggestionSelect(suggestions[selectedSuggestionIndex]);
        setShowSuggestions(false);
      } else {
        onSearch();
      }
    } else if (e.key === 'ArrowDown' && showSuggestions) {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => 
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp' && showSuggestions) {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => prev > 0 ? prev - 1 : -1);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
      setSelectedSuggestionIndex(-1);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onQueryChange(e.target.value);
  };

  const handleSuggestionClick = (suggestion: string) => {
    onSuggestionSelect(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  return (
    <div className="relative">
      <div className="relative">
        <textarea
          ref={inputRef}
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(suggestions.length > 0)}
          placeholder="Ask anything about your documents... (e.g., 'How does authentication work?', 'What are the main components?')"
          className="w-full px-5 py-4 pr-24 border-2 border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg resize-none"
          rows={3}
        />
        <div className="absolute right-3 bottom-3 flex items-center space-x-2">
          <span className="text-xs text-gray-400">
            {query.length > 0 && `${query.length} chars`}
          </span>
          <button
            onClick={onSearch}
            disabled={loading || !query.trim()}
            className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 ${
              loading || !query.trim()
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 transform hover:scale-105'
            }`}
          >
            {loading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                <span>Searching...</span>
              </div>
            ) : (
              <>
                <span>Search</span>
                <span className="ml-2">⏎</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Suggestions Dropdown */}
      {showSuggestions && (
        <div
          ref={suggestionsRef}
          className="absolute z-10 w-full mt-2 bg-white rounded-lg shadow-lg border border-gray-200 max-h-60 overflow-y-auto"
        >
          <div className="p-2">
            <div className="text-xs text-gray-500 px-3 py-1 font-medium">Suggestions</div>
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                onMouseEnter={() => setSelectedSuggestionIndex(index)}
                className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                  selectedSuggestionIndex === index
                    ? 'bg-blue-50 text-blue-700'
                    : 'hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-gray-400">🔍</span>
                  <span className="flex-1">{suggestion}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};