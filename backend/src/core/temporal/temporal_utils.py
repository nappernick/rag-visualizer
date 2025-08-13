"""
Minimal temporal utilities for time-aware document retrieval.
Provides decay functions and type detection with minimal footprint.
"""
import math
import time
from typing import Optional, Dict


def detect_doc_type(path: str, content: str = "") -> str:
    """
    Simple document type detection based on path and content.
    
    Args:
        path: File path or document name
        content: Optional content for additional detection
        
    Returns:
        Document type: 'values', 'project', 'meeting', or 'default'
    """
    path_lower = path.lower()
    
    # Time-invariant documents (no decay)
    if any(x in path_lower for x in ['values', 'charter', 'legal', 'contract', 'policy']):
        return 'values'
    
    # Project documents (120 days slow, then rapid decay)
    elif any(x in path_lower for x in ['spec', 'architecture', 'design', 'workflow', 'technical']):
        return 'project'
    
    # Meeting notes (rapid decay)
    elif any(x in path_lower for x in ['meeting', 'notes', 'minutes', 'standup']):
        return 'meeting'
    
    # Check content if no path match (only first 500 chars for efficiency)
    elif content:
        content_lower = content[:500].lower()
        if any(x in content_lower for x in ['milestone', 'sprint', 'deadline', 'specification']):
            return 'project'
        elif any(x in content_lower for x in ['meeting', 'action items', 'attendees']):
            return 'meeting'
    
    return 'default'


def get_temporal_score(created_at_ms: Optional[int], doc_type: str = 'default') -> float:
    """
    Calculate temporal relevance score using Weibull decay function.
    
    Weibull provides flexible decay curves with shape and scale parameters:
    - Shape (k): Controls decay curve shape (k<1: rapid initial, k>1: slow initial)
    - Scale (λ): Controls when decay reaches ~37% (characteristic life)
    
    Decay patterns:
    - 'values': No decay (always 1.0)
    - 'project': Weibull with k=2.5, λ=120 (slow initial, accelerates after 120 days)
    - 'meeting': Weibull with k=0.8, λ=14 (rapid initial decay)
    - 'default': Weibull with k=1.5, λ=60 (moderate decay)
    
    Args:
        created_at_ms: Creation timestamp in milliseconds (epoch)
        doc_type: Document type determining decay pattern
        
    Returns:
        Temporal score between 0.0 and 1.0
    """
    if created_at_ms is None:
        return 1.0  # No timestamp = assume current
    
    # Calculate age in days
    current_ms = time.time() * 1000
    age_days = max(0, (current_ms - created_at_ms) / 86_400_000)
    
    # Time-invariant documents
    if doc_type in ['values', 'legal', 'charter', 'policy']:
        return 1.0
    
    # Weibull decay function: exp(-(t/λ)^k)
    def weibull_decay(t: float, shape: float, scale: float, baseline: float = 0.1) -> float:
        """Apply Weibull decay with baseline"""
        score = math.exp(-math.pow(t / scale, shape))
        return max(baseline, score)
    
    # Project documents: Slow initial decay, accelerates after 120 days
    if doc_type == 'project':
        # Shape=2.5: Slow start, accelerates later (reliability curve)
        # Scale=120: Characteristic life at 120 days (~37% relevance)
        # At day 0: 100%, day 120: ~37%, day 150: ~15%, day 180: ~5%
        return weibull_decay(age_days, shape=2.5, scale=120, baseline=0.1)
    
    # Meeting notes: Rapid initial decay
    elif doc_type == 'meeting':
        # Shape=0.8: Rapid initial decay (early failure curve)
        # Scale=14: Characteristic life at 14 days
        # At day 0: 100%, day 7: ~55%, day 14: ~37%, day 30: ~15%
        return weibull_decay(age_days, shape=0.8, scale=14, baseline=0.05)
    
    # Default: Moderate decay
    else:
        # Shape=1.5: Moderate curve between exponential and slow
        # Scale=60: Characteristic life at 60 days
        # At day 0: 100%, day 30: ~65%, day 60: ~37%, day 120: ~10%
        return weibull_decay(age_days, shape=1.5, scale=60, baseline=0.1)


def enrich_with_temporal_metadata(doc_metadata: Dict, content: str, filename: str = "") -> Dict:
    """
    Enrich document metadata with temporal information extracted from content.
    
    Args:
        doc_metadata: Existing document metadata
        content: Document content for date extraction
        filename: Optional filename for additional context
        
    Returns:
        Enriched metadata with temporal fields
    """
    from .date_extractor import extract_temporal_metadata
    
    # Extract temporal data from content
    extracted = extract_temporal_metadata(content, filename)
    
    # Merge with existing metadata
    if extracted:
        # Preserve existing created_at if present
        if 'created_at_ms' not in doc_metadata and 'created_at_ms' in extracted:
            doc_metadata['created_at_ms'] = extracted['created_at_ms']
        
        # Add expiry information
        if 'expires_at_ms' in extracted:
            doc_metadata['expires_at_ms'] = extracted['expires_at_ms']
        
        # Add other temporal metadata
        for key in ['milestone', 'sprint', 'version', 'lifecycle_stage']:
            if key in extracted:
                doc_metadata[key] = extracted[key]
    
    # Auto-detect doc type if not present
    if 'doc_type' not in doc_metadata:
        doc_metadata['doc_type'] = detect_doc_type(filename, content[:500])
    
    # Ensure created_at_ms exists (use current time as fallback)
    if 'created_at_ms' not in doc_metadata:
        doc_metadata['created_at_ms'] = int(time.time() * 1000)
    
    return doc_metadata


def apply_temporal_boost(query: str) -> float:
    """
    Determine temporal weight adjustment based on query intent.
    
    Args:
        query: User query text
        
    Returns:
        Weight multiplier for temporal scoring (0.1 to 0.5)
    """
    query_lower = query.lower()
    
    # Strong recency preference
    if any(x in query_lower for x in ['latest', 'current', 'recent', 'new']):
        return 0.5  # Increase temporal weight
    
    # Historical preference
    elif any(x in query_lower for x in ['historical', 'archive', 'old', 'previous']):
        return 0.1  # Reduce temporal weight
    
    # Default balanced weight
    return 0.3