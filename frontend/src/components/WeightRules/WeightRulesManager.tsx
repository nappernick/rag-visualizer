import React, { useState, useEffect } from 'react';
import type { WeightRule, WeightCalculation, Document, WeightDistribution } from '../../types';
import { RuleEditor } from './RuleEditor';
import { WeightDistributionChart } from './WeightDistributionChart';
import { DocumentWeightInspector } from './DocumentWeightInspector';

interface WeightRulesManagerProps {
  documents: Document[];
}

export const WeightRulesManager: React.FC<WeightRulesManagerProps> = ({ documents }) => {
  const [rules, setRules] = useState<WeightRule[]>([]);
  const [selectedRule, setSelectedRule] = useState<WeightRule | null>(null);
  const [distribution, setDistribution] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [showRuleEditor, setShowRuleEditor] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [documentCalculation, setDocumentCalculation] = useState<WeightCalculation | null>(null);

  // Load weight rules on mount
  useEffect(() => {
    loadWeightRules();
    if (documents.length > 0) {
      loadWeightDistribution();
    }
  }, [documents]);

  const loadWeightRules = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8001/api/weight-rules');
      const data = await response.json();
      setRules(data);
    } catch (error) {
      console.error('Error loading weight rules:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadWeightDistribution = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/weight-rules/distribution');
      const data = await response.json();
      setDistribution(data);
    } catch (error) {
      console.error('Error loading weight distribution:', error);
    }
  };

  const loadDocumentCalculation = async (documentId: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/weight-rules/document/${documentId}/calculation`);
      const data = await response.json();
      setDocumentCalculation(data);
    } catch (error) {
      console.error('Error loading document calculation:', error);
    }
  };

  const handleRuleToggle = async (ruleId: string, enabled: boolean) => {
    try {
      const rule = rules.find(r => r.id === ruleId);
      if (!rule) return;

      const updatedRule = { ...rule, enabled };
      
      const response = await fetch(`http://localhost:8001/api/weight-rules/${ruleId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedRule)
      });

      if (response.ok) {
        setRules(rules.map(r => r.id === ruleId ? updatedRule : r));
        loadWeightDistribution();
      }
    } catch (error) {
      console.error('Error updating rule:', error);
    }
  };

  const handleDeleteRule = async (ruleId: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/weight-rules/${ruleId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setRules(rules.filter(r => r.id !== ruleId));
        loadWeightDistribution();
      }
    } catch (error) {
      console.error('Error deleting rule:', error);
    }
  };

  const handleDocumentSelect = (doc: Document) => {
    setSelectedDocument(doc);
    loadDocumentCalculation(doc.id);
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

  const formatRuleConditions = (rule: WeightRule) => {
    const conditions = [];
    
    if (rule.conditions.type_weights) {
      Object.entries(rule.conditions.type_weights).forEach(([type, weight]) => {
        conditions.push(`${type} → ${weight}x`);
      });
    }
    
    if (rule.conditions.patterns) {
      rule.conditions.patterns.forEach(pattern => {
        conditions.push(`${pattern.match} "${pattern.value}" → ${pattern.weight}x`);
      });
    }
    
    if (rule.conditions.ranges) {
      rule.conditions.ranges.forEach(range => {
        if (range.within) conditions.push(`Within ${range.within} → ${range.weight}x`);
        if (range.older_than) conditions.push(`Older than ${range.older_than} → ${range.weight}x`);
      });
    }
    
    return conditions;
  };

  return (
    <div className="weight-rules-manager">
      <div className="header">
        <h2>📊 Document Weight Rules</h2>
        <button 
          className="btn btn-primary"
          onClick={() => setShowRuleEditor(true)}
        >
          + Add Rule
        </button>
      </div>

      <div className="rules-grid">
        <div className="rules-section">
          <h3>Active Rules ({rules.filter(r => r.enabled).length})</h3>
          <div className="rules-controls">
            <button 
              className="btn btn-sm"
              onClick={() => rules.forEach(r => handleRuleToggle(r.id, false))}
            >
              Disable All
            </button>
          </div>

          <div className="rules-list">
            {rules.map(rule => (
              <div key={rule.id} className={`rule-card ${!rule.enabled ? 'disabled' : ''}`}>
                <div className="rule-header">
                  <div className="rule-title">
                    <span className="rule-icon">{getRuleIcon(rule.rule_type)}</span>
                    <h4>{rule.name}</h4>
                    <span className="rule-priority">Priority: {rule.priority}</span>
                  </div>
                  <div className="rule-actions">
                    <button onClick={() => setSelectedRule(rule)}>Edit</button>
                    <button onClick={() => handleRuleToggle(rule.id, !rule.enabled)}>
                      {rule.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button onClick={() => handleDeleteRule(rule.id)}>×</button>
                  </div>
                </div>
                
                <div className="rule-conditions">
                  {formatRuleConditions(rule).map((condition, idx) => (
                    <div key={idx} className="condition-item">
                      • {condition}
                    </div>
                  ))}
                </div>
                
                <div className="rule-footer">
                  <span className="affected-count">
                    Affects: {rule.affected_count} documents
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="visualization-section">
          {distribution && (
            <>
              <h3>Weight Distribution</h3>
              <WeightDistributionChart distribution={distribution} />
              
              <div className="distribution-stats">
                <div className="stat">
                  <label>Average Weight:</label>
                  <span>{distribution.average_weight?.toFixed(2)}</span>
                </div>
                <div className="stat">
                  <label>Median Weight:</label>
                  <span>{distribution.median_weight?.toFixed(2)}</span>
                </div>
                <div className="stat">
                  <label>Total Documents:</label>
                  <span>{distribution.total_documents}</span>
                </div>
              </div>

              {distribution.top_weighted && (
                <div className="top-weighted">
                  <h4>Top Weighted Documents</h4>
                  {distribution.top_weighted.map((doc: any, idx: number) => (
                    <div 
                      key={doc.document_id} 
                      className="weighted-doc-item"
                      onClick={() => {
                        const fullDoc = documents.find(d => d.id === doc.document_id);
                        if (fullDoc) handleDocumentSelect(fullDoc);
                      }}
                    >
                      <span className="rank">#{idx + 1}</span>
                      <span className="doc-title">{doc.title}</span>
                      <span className="weight">Weight: {doc.weight.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {selectedDocument && documentCalculation && (
        <DocumentWeightInspector
          document={selectedDocument}
          calculation={documentCalculation}
          onClose={() => {
            setSelectedDocument(null);
            setDocumentCalculation(null);
          }}
        />
      )}

      {showRuleEditor && (
        <RuleEditor
          rule={selectedRule}
          onSave={async (rule) => {
            if (selectedRule) {
              // Update existing rule
              await fetch(`http://localhost:8001/api/weight-rules/${selectedRule.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(rule)
              });
            } else {
              // Create new rule
              await fetch('http://localhost:8001/api/weight-rules', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(rule)
              });
            }
            loadWeightRules();
            loadWeightDistribution();
            setShowRuleEditor(false);
            setSelectedRule(null);
          }}
          onCancel={() => {
            setShowRuleEditor(false);
            setSelectedRule(null);
          }}
        />
      )}

      <style jsx>{`
        .weight-rules-manager {
          padding: 20px;
          background: #f8f9fa;
          min-height: 100vh;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
        }

        .header h2 {
          font-size: 24px;
          margin: 0;
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

        .rules-grid {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 30px;
        }

        .rules-section h3,
        .visualization-section h3 {
          margin-bottom: 20px;
        }

        .rules-controls {
          margin-bottom: 15px;
        }

        .rule-card {
          background: white;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 15px;
          border: 1px solid #dee2e6;
          transition: all 0.2s;
        }

        .rule-card:hover {
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .rule-card.disabled {
          opacity: 0.6;
          background: #f5f5f5;
        }

        .rule-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }

        .rule-title {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .rule-icon {
          font-size: 20px;
        }

        .rule-title h4 {
          margin: 0;
          font-size: 16px;
        }

        .rule-priority {
          background: #e9ecef;
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 12px;
        }

        .rule-actions {
          display: flex;
          gap: 5px;
        }

        .rule-actions button {
          padding: 4px 8px;
          border: 1px solid #dee2e6;
          background: white;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }

        .rule-actions button:hover {
          background: #f8f9fa;
        }

        .rule-conditions {
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          margin-bottom: 10px;
        }

        .condition-item {
          font-size: 13px;
          color: #495057;
          margin-bottom: 4px;
        }

        .rule-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .affected-count {
          font-size: 12px;
          color: #6c757d;
        }

        .visualization-section {
          background: white;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #dee2e6;
        }

        .distribution-stats {
          display: grid;
          grid-template-columns: 1fr;
          gap: 10px;
          margin: 20px 0;
        }

        .stat {
          display: flex;
          justify-content: space-between;
          padding: 8px;
          background: #f8f9fa;
          border-radius: 4px;
        }

        .stat label {
          font-weight: 500;
          color: #495057;
        }

        .stat span {
          font-weight: bold;
        }

        .top-weighted {
          margin-top: 20px;
        }

        .top-weighted h4 {
          margin-bottom: 10px;
          font-size: 14px;
        }

        .weighted-doc-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px;
          background: #f8f9fa;
          border-radius: 4px;
          margin-bottom: 5px;
          cursor: pointer;
          font-size: 13px;
        }

        .weighted-doc-item:hover {
          background: #e9ecef;
        }

        .rank {
          font-weight: bold;
          color: #6c757d;
        }

        .doc-title {
          flex: 1;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .weight {
          font-weight: 500;
          color: #28a745;
        }
      `}</style>
    </div>
  );
};