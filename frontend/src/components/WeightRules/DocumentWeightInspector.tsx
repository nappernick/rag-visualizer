import React from 'react';
import type { Document, WeightCalculation } from '../../types';

interface DocumentWeightInspectorProps {
  document: Document;
  calculation: WeightCalculation;
  onClose: () => void;
}

export const DocumentWeightInspector: React.FC<DocumentWeightInspectorProps> = ({
  document,
  calculation,
  onClose
}) => {
  const getWeightColor = (weight: number) => {
    if (weight < 0.5) return '#dc3545';
    if (weight < 1.0) return '#ffc107';
    if (weight < 2.0) return '#28a745';
    if (weight < 3.0) return '#17a2b8';
    if (weight < 5.0) return '#6610f2';
    return '#e83e8c';
  };

  const getRuleIcon = (ruleType: string) => {
    switch (ruleType) {
      case 'document_type': return '📄';
      case 'title_pattern': return '🔤';
      case 'temporal': return '📅';
      case 'content': return '📝';
      case 'manual': return '✋';
      default: return '📋';
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Weight Calculation Inspector</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="document-info">
          <h4>{document.title}</h4>
          <div className="document-meta">
            <span className="doc-type">Type: {document.doc_type}</span>
            <span className="doc-date">Created: {new Date(document.created_at).toLocaleDateString()}</span>
          </div>
        </div>

        <div className="weight-summary">
          <div className="weight-display">
            <div className="base-weight">
              <label>Base Weight:</label>
              <span>{calculation.base_weight.toFixed(2)}</span>
            </div>
            <div className="arrow">→</div>
            <div className="final-weight" style={{ color: getWeightColor(calculation.final_weight) }}>
              <label>Final Weight:</label>
              <span className="weight-value">{calculation.final_weight.toFixed(2)}</span>
            </div>
          </div>
        </div>

        <div className="calculation-breakdown">
          <h5>Applied Rules</h5>
          {calculation.applied_rules.length > 0 ? (
            <div className="rules-list">
              {calculation.applied_rules.map((rule, idx) => (
                <div key={idx} className="applied-rule">
                  <div className="rule-header">
                    <span className="rule-icon">{getRuleIcon(rule.rule_type)}</span>
                    <span className="rule-name">{rule.rule_name}</span>
                    <span className="weight-modifier">×{rule.weight_applied.toFixed(2)}</span>
                  </div>
                  <div className="rule-reason">{rule.reason}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-rules">No rules applied - using base weight</div>
          )}
        </div>

        <div className="calculation-path">
          <h5>Calculation Path</h5>
          <div className="path-display">{calculation.calculation_path}</div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-primary" onClick={onClose}>Close</button>
        </div>

        <style jsx>{`
          .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
          }

          .modal-content {
            background: white;
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
          }

          .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
          }

          .modal-header h3 {
            margin: 0;
            font-size: 18px;
          }

          .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #6c757d;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
          }

          .close-btn:hover {
            color: #000;
          }

          .document-info {
            padding: 20px;
            background: #f8f9fa;
          }

          .document-info h4 {
            margin: 0 0 10px 0;
            font-size: 16px;
          }

          .document-meta {
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #6c757d;
          }

          .weight-summary {
            padding: 20px;
            background: white;
          }

          .weight-display {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
          }

          .base-weight,
          .final-weight {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
          }

          .base-weight label,
          .final-weight label {
            font-size: 12px;
            color: #6c757d;
          }

          .weight-value {
            font-size: 32px;
            font-weight: bold;
          }

          .arrow {
            font-size: 24px;
            color: #6c757d;
          }

          .calculation-breakdown {
            padding: 20px;
            border-top: 1px solid #dee2e6;
          }

          .calculation-breakdown h5,
          .calculation-path h5 {
            margin: 0 0 15px 0;
            font-size: 14px;
            color: #495057;
          }

          .rules-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
          }

          .applied-rule {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            border-left: 3px solid #007bff;
          }

          .rule-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 5px;
          }

          .rule-icon {
            font-size: 16px;
          }

          .rule-name {
            flex: 1;
            font-weight: 500;
            font-size: 14px;
          }

          .weight-modifier {
            background: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
          }

          .rule-reason {
            font-size: 12px;
            color: #6c757d;
            margin-left: 26px;
          }

          .no-rules {
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-style: italic;
          }

          .calculation-path {
            padding: 20px;
            border-top: 1px solid #dee2e6;
          }

          .path-display {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.5;
            color: #495057;
            word-break: break-all;
          }

          .modal-footer {
            padding: 20px;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: flex-end;
          }

          .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
          }

          .btn-primary {
            background: #007bff;
            color: white;
          }

          .btn-primary:hover {
            background: #0056b3;
          }
        `}</style>
      </div>
    </div>
  );
};