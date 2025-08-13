"""
Minimal date extraction from document content.
Extracts creation dates, expiry dates, and validity periods.
"""
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import time


def extract_temporal_metadata(content: str, filename: str = "") -> Dict[str, any]:
    """
    Extract temporal metadata from document content and filename.
    
    Looks for:
    - Creation/updated dates in content
    - Expiry/valid-until dates
    - Version numbers and dates in filenames
    - Meeting dates, milestone dates, sprint periods
    
    Args:
        content: Document text content
        filename: Optional filename for additional context
        
    Returns:
        Dictionary with extracted temporal metadata
    """
    metadata = {}
    
    # Common date patterns
    date_patterns = [
        # ISO format: 2024-01-15
        (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
        # US format: 01/15/2024 or 1/15/2024
        (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),
        # Written: January 15, 2024
        (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})', '%B %d, %Y'),
        # Short written: Jan 15, 2024
        (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2},? \d{4})', '%b %d, %Y'),
    ]
    
    # Extract creation/updated date
    created_patterns = [
        r'(?:created|written|prepared|dated?)[:\s]+([^\n]+)',
        r'(?:date|updated)[:\s]+([^\n]+)',
        r'(?:as of|effective)[:\s]+([^\n]+)',
    ]
    
    for pattern in created_patterns:
        match = re.search(pattern, content[:1000], re.IGNORECASE)
        if match:
            date_str = match.group(1)
            parsed_date = _parse_date(date_str, date_patterns)
            if parsed_date:
                metadata['created_at_ms'] = int(parsed_date.timestamp() * 1000)
                break
    
    # Extract expiry/valid-until date
    expiry_patterns = [
        r'(?:expires?|valid until|effective until)[:\s]+([^\n]+)',
        r'(?:deadline|due date|milestone date)[:\s]+([^\n]+)',
        r'(?:end date|completion date)[:\s]+([^\n]+)',
    ]
    
    for pattern in expiry_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            parsed_date = _parse_date(date_str, date_patterns)
            if parsed_date:
                metadata['expires_at_ms'] = int(parsed_date.timestamp() * 1000)
                break
    
    # Extract sprint/milestone periods
    sprint_match = re.search(r'sprint[:\s#]+(\d+)', content[:500], re.IGNORECASE)
    if sprint_match:
        sprint_num = int(sprint_match.group(1))
        metadata['sprint'] = sprint_num
        # Assume 2-week sprints, calculate approximate date
        if 'created_at_ms' not in metadata:
            # Rough estimate: sprint started (sprint_num - 1) * 14 days ago
            sprint_start = datetime.now() - timedelta(days=(sprint_num - 1) * 14)
            metadata['created_at_ms'] = int(sprint_start.timestamp() * 1000)
    
    # Extract milestone from content
    milestone_match = re.search(r'milestone[:\s]+([^\n]+)', content[:500], re.IGNORECASE)
    if milestone_match:
        milestone_text = milestone_match.group(1)
        metadata['milestone'] = milestone_text.strip()
        # Check if milestone has a date
        milestone_date = _parse_date(milestone_text, date_patterns)
        if milestone_date and 'expires_at_ms' not in metadata:
            metadata['expires_at_ms'] = int(milestone_date.timestamp() * 1000)
    
    # Extract version and date from filename
    if filename:
        # Version patterns: v1.2, v2024.01, _v3, -v2
        version_match = re.search(r'[_\-\.]v?(\d+(?:\.\d+)*)', filename)
        if version_match:
            metadata['version'] = version_match.group(1)
        
        # Date in filename: 2024-01-15, 20240115, 2024_01_15
        filename_date_patterns = [
            (r'(\d{4}[-_]\d{2}[-_]\d{2})', '%Y-%m-%d'),
            (r'(\d{8})', '%Y%m%d'),
            (r'(\d{4}_\d{2}_\d{2})', '%Y_%m_%d'),
        ]
        
        for pattern, fmt in filename_date_patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    date_str = match.group(1).replace('_', '-')
                    if fmt == '%Y%m%d':
                        parsed = datetime.strptime(match.group(1), fmt)
                    else:
                        parsed = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if 'created_at_ms' not in metadata:
                        metadata['created_at_ms'] = int(parsed.timestamp() * 1000)
                    break
                except:
                    continue
    
    # Determine document lifecycle based on metadata
    if 'expires_at_ms' in metadata:
        if metadata['expires_at_ms'] < time.time() * 1000:
            metadata['lifecycle_stage'] = 'expired'
        else:
            metadata['lifecycle_stage'] = 'active'
    elif 'created_at_ms' in metadata:
        age_days = (time.time() * 1000 - metadata['created_at_ms']) / 86_400_000
        if age_days < 30:
            metadata['lifecycle_stage'] = 'current'
        elif age_days < 90:
            metadata['lifecycle_stage'] = 'recent'
        else:
            metadata['lifecycle_stage'] = 'historical'
    
    return metadata


def _parse_date(date_str: str, patterns: list) -> Optional[datetime]:
    """Helper to parse date string with multiple patterns"""
    date_str = date_str.strip()
    
    for pattern, fmt in patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt)
            except:
                continue
    
    return None


def calculate_relevance_window(doc_metadata: Dict) -> Tuple[float, str]:
    """
    Calculate relevance based on document's temporal window.
    
    Returns:
        (relevance_score, reason)
    """
    now_ms = time.time() * 1000
    
    # Check if document has expired
    if 'expires_at_ms' in doc_metadata:
        if doc_metadata['expires_at_ms'] < now_ms:
            days_expired = (now_ms - doc_metadata['expires_at_ms']) / 86_400_000
            if days_expired > 30:
                return (0.1, "expired_old")
            else:
                return (0.3, "recently_expired")
        else:
            # Document still valid
            days_until_expiry = (doc_metadata['expires_at_ms'] - now_ms) / 86_400_000
            if days_until_expiry < 7:
                return (0.9, "expiring_soon")
            else:
                return (1.0, "valid")
    
    # Check milestone relevance
    if 'milestone' in doc_metadata:
        # Milestone documents are highly relevant near their date
        return (0.8, "milestone_document")
    
    # Check sprint relevance
    if 'sprint' in doc_metadata:
        # Assume current sprint is most relevant
        return (0.7, "sprint_document")
    
    # Default to age-based relevance
    if 'created_at_ms' in doc_metadata:
        age_days = (now_ms - doc_metadata['created_at_ms']) / 86_400_000
        if age_days < 7:
            return (1.0, "very_recent")
        elif age_days < 30:
            return (0.8, "recent")
        elif age_days < 90:
            return (0.5, "moderate_age")
        else:
            return (0.3, "old")
    
    return (0.5, "no_temporal_data")