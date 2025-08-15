import React from 'react';

export type TabType = 'upload' | 'chunks' | 'graph' | 'query' | 'stats' | 'demo' | 'weights';

interface NavigationTabsProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

const tabConfig: { id: TabType; label: string; icon?: string; gradient: string }[] = [
  { id: 'upload', label: 'Upload', icon: '📁', gradient: 'from-blue-500 to-blue-600' },
  { id: 'chunks', label: 'Chunks', icon: '✂️', gradient: 'from-green-500 to-green-600' },
  { id: 'graph', label: 'Graph', icon: '🔮', gradient: 'from-purple-500 to-purple-600' },
  { id: 'query', label: 'Query', icon: '🔍', gradient: 'from-orange-500 to-orange-600' },
  { id: 'stats', label: 'Stats', icon: '📊', gradient: 'from-pink-500 to-pink-600' },
  { id: 'demo', label: 'Demo', icon: '🚀', gradient: 'from-indigo-500 to-indigo-600' },
  { id: 'weights', label: 'Weights', icon: '⚖️', gradient: 'from-purple-500 to-purple-600' }
];

export const NavigationTabs: React.FC<NavigationTabsProps> = ({ activeTab, onTabChange }) => {
  return (
    <div className="max-w-7xl mx-auto px-6 py-6">
      <div className="bg-white rounded-xl shadow-md p-2">
        <nav className="flex space-x-2">
          {tabConfig.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`
                flex-1 py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200
                ${activeTab === tab.id
                  ? `bg-gradient-to-r ${tab.gradient} text-white shadow-lg transform scale-105`
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              {tab.icon && <span className="mr-1">{tab.icon}</span>}
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
};