import React from 'react';

interface WeightDistributionChartProps {
  distribution: {
    distribution: Record<string, number>;
    total_documents: number;
  };
}

export const WeightDistributionChart: React.FC<WeightDistributionChartProps> = ({ distribution }) => {
  const ranges = [
    { key: '0.1-0.5', label: '0.1-0.5', color: '#dc3545' },
    { key: '0.5-1.0', label: '0.5-1.0', color: '#ffc107' },
    { key: '1.0-2.0', label: '1.0-2.0', color: '#28a745' },
    { key: '2.0-3.0', label: '2.0-3.0', color: '#17a2b8' },
    { key: '3.0-5.0', label: '3.0-5.0', color: '#6610f2' },
    { key: '5.0-10.0', label: '5.0-10', color: '#e83e8c' }
  ];

  const maxCount = Math.max(...Object.values(distribution.distribution || {}), 1);

  return (
    <div className="weight-distribution-chart">
      <div className="chart-title">Document Weight Distribution</div>
      <div className="chart-bars">
        {ranges.map(range => {
          const count = distribution.distribution?.[range.key] || 0;
          const percentage = distribution.total_documents > 0 
            ? (count / distribution.total_documents * 100).toFixed(1)
            : '0';
          const barHeight = (count / maxCount) * 100;

          return (
            <div key={range.key} className="bar-container">
              <div className="bar-wrapper">
                <div 
                  className="bar" 
                  style={{ 
                    height: `${barHeight}%`,
                    backgroundColor: range.color 
                  }}
                >
                  {count > 0 && (
                    <div className="bar-value">{count}</div>
                  )}
                </div>
              </div>
              <div className="bar-label">{range.label}</div>
              <div className="bar-percentage">{percentage}%</div>
            </div>
          );
        })}
      </div>

      <style jsx>{`
        .weight-distribution-chart {
          padding: 15px;
          background: #f8f9fa;
          border-radius: 8px;
        }

        .chart-title {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 20px;
          text-align: center;
          color: #495057;
        }

        .chart-bars {
          display: flex;
          justify-content: space-around;
          align-items: flex-end;
          height: 150px;
          padding: 0 10px;
        }

        .bar-container {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 5px;
        }

        .bar-wrapper {
          height: 120px;
          width: 100%;
          display: flex;
          align-items: flex-end;
          justify-content: center;
        }

        .bar {
          width: 30px;
          min-height: 5px;
          border-radius: 4px 4px 0 0;
          position: relative;
          transition: all 0.3s ease;
          display: flex;
          align-items: flex-start;
          justify-content: center;
        }

        .bar:hover {
          opacity: 0.8;
          transform: translateY(-2px);
        }

        .bar-value {
          position: absolute;
          top: -20px;
          font-size: 12px;
          font-weight: bold;
          color: #495057;
        }

        .bar-label {
          font-size: 11px;
          color: #6c757d;
          text-align: center;
        }

        .bar-percentage {
          font-size: 10px;
          color: #6c757d;
          text-align: center;
        }
      `}</style>
    </div>
  );
};