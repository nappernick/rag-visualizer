import React from 'react';

interface ErrorDisplayProps {
  error: string;
  onDismiss: () => void;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, onDismiss }) => {
  return (
    <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow-md animate-slideIn">
      <div className="flex items-center">
        <span className="text-red-800 flex-1">{error}</span>
        <button 
          onClick={onDismiss} 
          className="ml-4 text-red-500 hover:text-red-700 text-2xl font-bold transition-colors"
        >
          ×
        </button>
      </div>
    </div>
  );
};