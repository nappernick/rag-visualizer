import React from 'react';

interface HeaderProps {
  onClearAll: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onClearAll }) => {
  return (
    <header className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              RAG Visualizer
            </h1>
            <p className="mt-2 text-gray-600 text-lg">Explore and understand your RAG pipeline</p>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={onClearAll}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
            >
              Clear All Data
            </button>
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">System Active</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};