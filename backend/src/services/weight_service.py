"""
Weight calculation and rules management service
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from ..models.schemas import (
    Document, WeightRule, WeightCalculation, AppliedRule,
    RuleType, PatternMatch, TemporalRange, WeightRuleConditions
)
from ..services.storage import get_storage_service

logger = logging.getLogger(__name__)


class WeightService:
    """Service for managing document weights and weight rules"""
    
    def __init__(self):
        self.storage = get_storage_service()
        self.rules_cache = {}
        self.calculation_cache = {}
    
    async def calculate_document_weight(
        self, 
        document: Document, 
        rules: Optional[List[WeightRule]] = None
    ) -> WeightCalculation:
        """
        Calculate the final weight for a document based on all active rules
        """
        # Get active rules if not provided
        if rules is None:
            rules = await self.get_active_rules()
        
        # Start with base weight
        base_weight = document.weight if hasattr(document, 'weight') else 1.0
        current_weight = base_weight
        applied_rules = []
        calculation_steps = []
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check if rule applies to this document
            rule_weight = await self._apply_rule(document, rule)
            
            if rule_weight != 1.0:  # Rule applies
                # Apply the weight modification
                previous_weight = current_weight
                current_weight *= rule_weight
                
                # Track the applied rule
                applied_rules.append(AppliedRule(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    weight_applied=rule_weight,
                    reason=self._get_rule_reason(document, rule, rule_weight)
                ))
                
                calculation_steps.append(
                    f"{rule.name}: {previous_weight:.2f} × {rule_weight:.2f} = {current_weight:.2f}"
                )
        
        # Clamp final weight to valid range
        final_weight = max(0.1, min(10.0, current_weight))
        
        # Build calculation path
        if calculation_steps:
            calculation_path = f"Base: {base_weight:.2f} → " + " → ".join(calculation_steps) + f" → Final: {final_weight:.2f}"
        else:
            calculation_path = f"Base weight: {base_weight:.2f} (no rules applied)"
        
        return WeightCalculation(
            document_id=document.id,
            document_title=document.title,
            base_weight=base_weight,
            applied_rules=applied_rules,
            final_weight=final_weight,
            calculation_path=calculation_path,
            calculated_at=datetime.utcnow()
        )
    
    async def _apply_rule(self, document: Document, rule: WeightRule) -> float:
        """Apply a single rule to a document and return the weight modifier"""
        
        if rule.rule_type == RuleType.DOCUMENT_TYPE:
            return await self._apply_document_type_rule(document, rule)
        elif rule.rule_type == RuleType.TITLE_PATTERN:
            return await self._apply_title_pattern_rule(document, rule)
        elif rule.rule_type == RuleType.TEMPORAL:
            return await self._apply_temporal_rule(document, rule)
        elif rule.rule_type == RuleType.CONTENT:
            return await self._apply_content_rule(document, rule)
        elif rule.rule_type == RuleType.MANUAL:
            return await self._apply_manual_rule(document, rule)
        
        return 1.0  # No modification
    
    async def _apply_document_type_rule(self, document: Document, rule: WeightRule) -> float:
        """Apply document type based weight rule"""
        if not rule.conditions.type_weights:
            return 1.0
        
        doc_type = document.doc_type.lower()
        
        # Check for exact match
        if doc_type in rule.conditions.type_weights:
            return rule.conditions.type_weights[doc_type]
        
        # Check for partial matches (e.g., "pdf" matches "application/pdf")
        for type_pattern, weight in rule.conditions.type_weights.items():
            if type_pattern.lower() in doc_type or doc_type in type_pattern.lower():
                return weight
        
        return 1.0
    
    async def _apply_title_pattern_rule(self, document: Document, rule: WeightRule) -> float:
        """Apply title pattern based weight rule"""
        if not rule.conditions.patterns:
            return 1.0
        
        title = document.title
        
        for pattern in rule.conditions.patterns:
            if self._match_pattern(title, pattern):
                return pattern.weight
        
        return 1.0
    
    async def _apply_temporal_rule(self, document: Document, rule: WeightRule) -> float:
        """Apply temporal (date-based) weight rule"""
        if not rule.conditions.ranges:
            return 1.0
        
        # Parse document creation date
        try:
            if hasattr(document, 'created_at'):
                if isinstance(document.created_at, str):
                    doc_date = datetime.fromisoformat(document.created_at.replace('Z', '+00:00'))
                else:
                    doc_date = document.created_at
            else:
                return 1.0
        except:
            return 1.0
        
        now = datetime.utcnow()
        age = now - doc_date
        
        for time_range in rule.conditions.ranges:
            if time_range.within:
                max_age = self._parse_time_duration(time_range.within)
                if age <= max_age:
                    return time_range.weight
            
            if time_range.older_than:
                min_age = self._parse_time_duration(time_range.older_than)
                if age > min_age:
                    return time_range.weight
            
            if time_range.newer_than:
                max_age = self._parse_time_duration(time_range.newer_than)
                if age < max_age:
                    return time_range.weight
        
        return 1.0
    
    async def _apply_content_rule(self, document: Document, rule: WeightRule) -> float:
        """Apply content-based weight rule"""
        content = document.content or ""
        
        # Check content patterns
        if rule.conditions.content_patterns:
            for pattern in rule.conditions.content_patterns:
                if self._match_pattern(content, pattern):
                    return pattern.weight
        
        # Check content length
        content_length = len(content)
        
        if rule.conditions.min_length and content_length < rule.conditions.min_length:
            return 0.5  # Penalize too-short content
        
        if rule.conditions.max_length and content_length > rule.conditions.max_length:
            return 0.8  # Slightly penalize too-long content
        
        return 1.0
    
    async def _apply_manual_rule(self, document: Document, rule: WeightRule) -> float:
        """Apply manual override rule"""
        # Check specific document IDs
        if rule.conditions.document_ids and document.id in rule.conditions.document_ids:
            return rule.weight_modifier
        
        # Check document patterns (e.g., path patterns)
        if rule.conditions.document_patterns:
            for pattern in rule.conditions.document_patterns:
                if re.match(pattern, document.title) or re.match(pattern, document.id):
                    return rule.weight_modifier
        
        return 1.0
    
    def _match_pattern(self, text: str, pattern: PatternMatch) -> bool:
        """Check if text matches a pattern"""
        if not pattern.case_sensitive:
            text = text.lower()
            pattern_value = pattern.value.lower()
        else:
            pattern_value = pattern.value
        
        if pattern.match == "contains":
            return pattern_value in text
        elif pattern.match == "startsWith":
            return text.startswith(pattern_value)
        elif pattern.match == "endsWith":
            return text.endswith(pattern_value)
        elif pattern.match == "exact":
            return text == pattern_value
        elif pattern.match == "regex":
            try:
                return bool(re.search(pattern_value, text))
            except:
                return False
        
        return False
    
    def _parse_time_duration(self, duration_str: str) -> timedelta:
        """Parse duration string like '7d', '30d', '1y' into timedelta"""
        match = re.match(r'(\d+)([dhmy])', duration_str.lower())
        if not match:
            return timedelta(days=0)
        
        value, unit = match.groups()
        value = int(value)
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(days=value * 30)  # Approximate
        elif unit == 'y':
            return timedelta(days=value * 365)  # Approximate
        
        return timedelta(days=0)
    
    def _get_rule_reason(self, document: Document, rule: WeightRule, weight: float) -> str:
        """Generate a human-readable reason for why a rule was applied"""
        if rule.rule_type == RuleType.DOCUMENT_TYPE:
            return f"Document type '{document.doc_type}' → weight {weight:.1f}x"
        elif rule.rule_type == RuleType.TITLE_PATTERN:
            return f"Title matches pattern → weight {weight:.1f}x"
        elif rule.rule_type == RuleType.TEMPORAL:
            return f"Document age → weight {weight:.1f}x"
        elif rule.rule_type == RuleType.CONTENT:
            return f"Content matches criteria → weight {weight:.1f}x"
        elif rule.rule_type == RuleType.MANUAL:
            return f"Manual override → weight {weight:.1f}x"
        
        return f"Rule applied → weight {weight:.1f}x"
    
    async def get_active_rules(self) -> List[WeightRule]:
        """Get all active weight rules from storage"""
        # This would normally query the database
        # For now, return default rules
        return [
            WeightRule(
                id="default_doc_type",
                name="Document Type Weights",
                rule_type=RuleType.DOCUMENT_TYPE,
                enabled=True,
                priority=100,
                conditions=WeightRuleConditions(
                    type_weights={
                        "pdf": 1.5,
                        "markdown": 1.2,
                        "text": 1.0,
                        "code": 0.8,
                        "image": 0.5
                    }
                ),
                weight_modifier=1.0
            ),
            WeightRule(
                id="default_temporal",
                name="Recency Boost",
                rule_type=RuleType.TEMPORAL,
                enabled=True,
                priority=90,
                conditions=WeightRuleConditions(
                    ranges=[
                        TemporalRange(within="7d", weight=2.0),
                        TemporalRange(within="30d", weight=1.5),
                        TemporalRange(within="90d", weight=1.2),
                        TemporalRange(older_than="365d", weight=0.7)
                    ]
                ),
                weight_modifier=1.0
            ),
            WeightRule(
                id="default_title_important",
                name="Important Documents",
                rule_type=RuleType.TITLE_PATTERN,
                enabled=True,
                priority=80,
                conditions=WeightRuleConditions(
                    patterns=[
                        PatternMatch(match="contains", value="important", weight=2.0),
                        PatternMatch(match="contains", value="critical", weight=2.5),
                        PatternMatch(match="contains", value="policy", weight=1.8),
                        PatternMatch(match="contains", value="draft", weight=0.5)
                    ]
                ),
                weight_modifier=1.0
            )
        ]
    
    async def calculate_weight_distribution(
        self, 
        documents: List[Document],
        rules: Optional[List[WeightRule]] = None
    ) -> Dict:
        """Calculate weight distribution statistics for a set of documents"""
        
        calculations = []
        for doc in documents:
            calc = await self.calculate_document_weight(doc, rules)
            calculations.append(calc)
        
        # Calculate distribution
        weights = [calc.final_weight for calc in calculations]
        
        # Create distribution buckets
        distribution = defaultdict(int)
        ranges = [
            (0.1, 0.5, "0.1-0.5"),
            (0.5, 1.0, "0.5-1.0"),
            (1.0, 2.0, "1.0-2.0"),
            (2.0, 3.0, "2.0-3.0"),
            (3.0, 5.0, "3.0-5.0"),
            (5.0, 10.0, "5.0-10.0")
        ]
        
        for weight in weights:
            for min_val, max_val, label in ranges:
                if min_val <= weight <= max_val:
                    distribution[label] += 1
                    break
        
        # Calculate statistics
        avg_weight = statistics.mean(weights) if weights else 0
        median_weight = statistics.median(weights) if weights else 0
        
        # Find top weighted documents
        top_documents = sorted(
            zip(documents, calculations),
            key=lambda x: x[1].final_weight,
            reverse=True
        )[:5]
        
        return {
            "total_documents": len(documents),
            "distribution": dict(distribution),
            "average_weight": round(avg_weight, 2),
            "median_weight": round(median_weight, 2),
            "min_weight": round(min(weights), 2) if weights else 0,
            "max_weight": round(max(weights), 2) if weights else 0,
            "top_weighted": [
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "weight": calc.final_weight,
                    "calculation_path": calc.calculation_path
                }
                for doc, calc in top_documents
            ]
        }
    
    async def simulate_rules(
        self,
        documents: List[Document],
        rules: List[WeightRule]
    ) -> Dict:
        """Simulate the effect of rules on documents without saving"""
        
        # Calculate weights with simulated rules
        distribution = await self.calculate_weight_distribution(documents, rules)
        
        # Count affected documents per rule
        rule_impacts = {}
        for rule in rules:
            if not rule.enabled:
                continue
            
            affected_count = 0
            for doc in documents:
                weight = await self._apply_rule(doc, rule)
                if weight != 1.0:
                    affected_count += 1
            
            rule_impacts[rule.id] = {
                "name": rule.name,
                "affected_count": affected_count,
                "percentage": (affected_count / len(documents) * 100) if documents else 0
            }
        
        return {
            "distribution": distribution,
            "rule_impacts": rule_impacts,
            "simulation_timestamp": datetime.utcnow().isoformat()
        }


# Singleton instance
_weight_service = None

def get_weight_service() -> WeightService:
    """Get or create the weight service singleton"""
    global _weight_service
    if _weight_service is None:
        _weight_service = WeightService()
    return _weight_service