import React, { useState, useEffect } from 'react';
import type { WeightRule, RuleType, PatternMatch, TemporalRange } from '../../types';

interface RuleEditorProps {
  rule: WeightRule | null;
  onSave: (rule: any) => void;
  onCancel: () => void;
}

export const RuleEditor: React.FC<RuleEditorProps> = ({ rule, onSave, onCancel }) => {
  const [name, setName] = useState(rule?.name || '');
  const [ruleType, setRuleType] = useState<RuleType>(rule?.rule_type || 'document_type');
  const [priority, setPriority] = useState(rule?.priority || 50);
  const [enabled, setEnabled] = useState(rule?.enabled ?? true);
  
  // Document type conditions
  const [typeWeights, setTypeWeights] = useState<Record<string, number>>(
    rule?.conditions?.type_weights || {
      pdf: 1.0,
      text: 1.0,
      markdown: 1.0,
      code: 1.0
    }
  );
  
  // Title pattern conditions
  const [patterns, setPatterns] = useState<PatternMatch[]>(
    rule?.conditions?.patterns || []
  );
  
  // Temporal conditions
  const [ranges, setRanges] = useState<TemporalRange[]>(
    rule?.conditions?.ranges || []
  );

  const handleSave = () => {
    const conditions: any = {};
    
    switch (ruleType) {
      case 'document_type':
        conditions.type_weights = typeWeights;
        break;
      case 'title_pattern':
        conditions.patterns = patterns;
        break;
      case 'temporal':
        conditions.ranges = ranges;
        break;
    }
    
    onSave({
      name,
      rule_type: ruleType,
      priority,
      enabled,
      conditions,
      weight_modifier: 1.0
    });
  };

  const addPattern = () => {
    setPatterns([...patterns, { match: 'contains', value: '', weight: 1.0 }]);
  };

  const updatePattern = (index: number, field: string, value: any) => {
    const updated = [...patterns];
    updated[index] = { ...updated[index], [field]: value };
    setPatterns(updated);
  };

  const removePattern = (index: number) => {
    setPatterns(patterns.filter((_, i) => i !== index));
  };

  const addRange = () => {
    setRanges([...ranges, { within: '7d', weight: 1.0 }]);
  };

  const updateRange = (index: number, field: string, value: any) => {
    const updated = [...ranges];
    updated[index] = { ...updated[index], [field]: value };
    setRanges(updated);
  };

  const removeRange = (index: number) => {
    setRanges(ranges.filter((_, i) => i !== index));
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h3>{rule ? 'Edit Rule' : 'New Weight Rule'}</h3>
          <button className="close-btn" onClick={onCancel}>×</button>
        </div>

        <div className="form-group">
          <label>Rule Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter rule name"
          />
        </div>

        <div className="form-group">
          <label>Rule Type</label>
          <select value={ruleType} onChange={(e) => setRuleType(e.target.value as RuleType)}>
            <option value="document_type">Document Type</option>
            <option value="title_pattern">Title Pattern</option>
            <option value="temporal">Temporal (Date-based)</option>
            <option value="content">Content Pattern</option>
            <option value="manual">Manual Override</option>
          </select>
        </div>

        <div className="form-group">
          <label>Priority (higher = applied first)</label>
          <input
            type="number"
            value={priority}
            onChange={(e) => setPriority(Number(e.target.value))}
            min="0"
            max="1000"
          />
        </div>

        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            Enabled
          </label>
        </div>

        <div className="conditions-section">
          <h4>Conditions</h4>
          
          {ruleType === 'document_type' && (
            <div className="type-weights">
              {Object.entries(typeWeights).map(([type, weight]) => (
                <div key={type} className="weight-row">
                  <label>{type}:</label>
                  <input
                    type="number"
                    value={weight}
                    onChange={(e) => setTypeWeights({
                      ...typeWeights,
                      [type]: Number(e.target.value)
                    })}
                    min="0.1"
                    max="10"
                    step="0.1"
                  />
                </div>
              ))}
            </div>
          )}

          {ruleType === 'title_pattern' && (
            <div className="patterns">
              {patterns.map((pattern, idx) => (
                <div key={idx} className="pattern-row">
                  <select
                    value={pattern.match}
                    onChange={(e) => updatePattern(idx, 'match', e.target.value)}
                  >
                    <option value="contains">Contains</option>
                    <option value="startsWith">Starts With</option>
                    <option value="endsWith">Ends With</option>
                    <option value="regex">Regex</option>
                    <option value="exact">Exact Match</option>
                  </select>
                  <input
                    type="text"
                    value={pattern.value}
                    onChange={(e) => updatePattern(idx, 'value', e.target.value)}
                    placeholder="Pattern value"
                  />
                  <input
                    type="number"
                    value={pattern.weight}
                    onChange={(e) => updatePattern(idx, 'weight', Number(e.target.value))}
                    min="0.1"
                    max="10"
                    step="0.1"
                  />
                  <button onClick={() => removePattern(idx)}>Remove</button>
                </div>
              ))}
              <button onClick={addPattern}>Add Pattern</button>
            </div>
          )}

          {ruleType === 'temporal' && (
            <div className="ranges">
              {ranges.map((range, idx) => (
                <div key={idx} className="range-row">
                  <input
                    type="text"
                    value={range.within || range.older_than || ''}
                    onChange={(e) => updateRange(idx, 'within', e.target.value)}
                    placeholder="e.g., 7d, 30d, 1y"
                  />
                  <input
                    type="number"
                    value={range.weight}
                    onChange={(e) => updateRange(idx, 'weight', Number(e.target.value))}
                    min="0.1"
                    max="10"
                    step="0.1"
                  />
                  <button onClick={() => removeRange(idx)}>Remove</button>
                </div>
              ))}
              <button onClick={addRange}>Add Range</button>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
          <button className="btn btn-primary" onClick={handleSave}>Save Rule</button>
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
          }

          .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #6c757d;
          }

          .form-group {
            padding: 15px 20px;
          }

          .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #495057;
          }

          .form-group input[type="text"],
          .form-group input[type="number"],
          .form-group select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
          }

          .form-group input[type="checkbox"] {
            margin-right: 8px;
          }

          .conditions-section {
            padding: 20px;
            background: #f8f9fa;
          }

          .conditions-section h4 {
            margin: 0 0 15px 0;
          }

          .weight-row,
          .pattern-row,
          .range-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
          }

          .weight-row label {
            min-width: 80px;
          }

          .weight-row input {
            width: 100px;
          }

          .pattern-row select,
          .range-row input[type="text"] {
            flex: 1;
          }

          .pattern-row input[type="number"],
          .range-row input[type="number"] {
            width: 80px;
          }

          .pattern-row button,
          .range-row button {
            padding: 4px 8px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
          }

          .patterns button,
          .ranges button {
            margin-top: 10px;
            padding: 6px 12px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
          }

          .modal-footer {
            padding: 20px;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: flex-end;
            gap: 10px;
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

          .btn-secondary {
            background: #6c757d;
            color: white;
          }

          .btn:hover {
            opacity: 0.9;
          }
        `}</style>
      </div>
    </div>
  );
};