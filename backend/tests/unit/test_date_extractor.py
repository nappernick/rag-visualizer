"""
Comprehensive tests for date extraction from document content.
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.temporal.date_extractor import (
    extract_temporal_metadata,
    _parse_date,
    calculate_relevance_window
)


class TestDatePatternExtraction:
    """Test extraction of various date patterns from content."""
    
    def test_iso_date_extraction(self):
        """Test ISO format date extraction (YYYY-MM-DD)."""
        content = "Created: 2024-01-15\nThis document was prepared on this date."
        metadata = extract_temporal_metadata(content)
        
        # Should extract the date
        assert 'created_at_ms' in metadata
        expected = datetime(2024, 1, 15).timestamp() * 1000
        assert abs(metadata['created_at_ms'] - expected) < 86400000  # Within a day
    
    def test_us_date_extraction(self):
        """Test US format date extraction (MM/DD/YYYY)."""
        content = "Date: 01/15/2024\nMeeting scheduled for this date."
        metadata = extract_temporal_metadata(content)
        
        assert 'created_at_ms' in metadata
        expected = datetime(2024, 1, 15).timestamp() * 1000
        assert abs(metadata['created_at_ms'] - expected) < 86400000
    
    def test_written_date_extraction(self):
        """Test written date format extraction."""
        test_cases = [
            "Created: January 15, 2024",
            "Date: Jan 15, 2024",
            "Updated: January 15 2024",  # Without comma
            "Prepared: Feb 28, 2024",
            "Written: December 31, 2023"
        ]
        
        for content in test_cases:
            metadata = extract_temporal_metadata(content)
            assert 'created_at_ms' in metadata, f"Failed to extract date from: {content}"
    
    def test_expiry_date_extraction(self):
        """Test extraction of expiry/deadline dates."""
        test_cases = [
            ("Expires: 2024-12-31", datetime(2024, 12, 31)),
            ("Valid until: 2024-06-30", datetime(2024, 6, 30)),
            ("Deadline: 03/15/2024", datetime(2024, 3, 15)),
            ("Due date: April 1, 2024", datetime(2024, 4, 1)),
            ("Milestone date: 2024-Q2", None),  # Should not parse Q2
            ("End date: May 31, 2024", datetime(2024, 5, 31))
        ]
        
        for content, expected_date in test_cases:
            metadata = extract_temporal_metadata(content)
            if expected_date:
                assert 'expires_at_ms' in metadata
                expected_ms = expected_date.timestamp() * 1000
                assert abs(metadata['expires_at_ms'] - expected_ms) < 86400000
            else:
                assert 'expires_at_ms' not in metadata
    
    def test_multiple_dates_priority(self):
        """Test that creation date is found before expiry date."""
        content = """
        Created: 2024-01-15
        Updated: 2024-01-20
        Expires: 2024-12-31
        """
        metadata = extract_temporal_metadata(content)
        
        # Should use "Created" date
        created_date = datetime(2024, 1, 15).timestamp() * 1000
        assert abs(metadata['created_at_ms'] - created_date) < 86400000
        
        # Should also have expiry
        expiry_date = datetime(2024, 12, 31).timestamp() * 1000
        assert abs(metadata['expires_at_ms'] - expiry_date) < 86400000
    
    def test_date_in_filename(self):
        """Test date extraction from filename."""
        test_cases = [
            ("report_2024-01-15.md", datetime(2024, 1, 15)),
            ("meeting_20240115.txt", datetime(2024, 1, 15)),
            ("doc_2024_01_15.pdf", datetime(2024, 1, 15)),
            ("notes-2024-03-22.docx", datetime(2024, 3, 22)),
            ("file_without_date.txt", None)
        ]
        
        for filename, expected_date in test_cases:
            metadata = extract_temporal_metadata("", filename)
            if expected_date:
                assert 'created_at_ms' in metadata
                expected_ms = expected_date.timestamp() * 1000
                assert abs(metadata['created_at_ms'] - expected_ms) < 86400000
            else:
                # Should not have extracted date
                assert 'created_at_ms' not in metadata or metadata['created_at_ms'] is None


class TestSprintMilestoneExtraction:
    """Test extraction of sprint and milestone information."""
    
    def test_sprint_number_extraction(self):
        """Test extraction of sprint numbers."""
        test_cases = [
            ("Sprint 23 objectives", 23),
            ("sprint #42 planning", 42),
            ("SPRINT: 5", 5),
            ("Current sprint: 100", 100),
            ("Sprint23 (no space)", 23),
            ("Multiple Sprint 1 and Sprint 2", 1),  # Gets first
        ]
        
        for content, expected_sprint in test_cases:
            metadata = extract_temporal_metadata(content)
            assert metadata.get('sprint') == expected_sprint, \
                f"Expected sprint {expected_sprint} from '{content}', got {metadata.get('sprint')}"
    
    def test_sprint_date_estimation(self):
        """Test that sprint number creates estimated date."""
        content = "Sprint 10 planning meeting"
        metadata = extract_temporal_metadata(content)
        
        assert 'sprint' in metadata
        assert metadata['sprint'] == 10
        
        # Should estimate created_at based on sprint (10-1)*14 days ago
        assert 'created_at_ms' in metadata
        expected_days_ago = (10 - 1) * 14
        expected_ms = (datetime.now() - timedelta(days=expected_days_ago)).timestamp() * 1000
        
        # Should be roughly in the right range (within a day)
        assert abs(metadata['created_at_ms'] - expected_ms) < 86400000
    
    def test_milestone_extraction(self):
        """Test extraction of milestone information."""
        test_cases = [
            ("Milestone: Q1 2024 Release", "Q1 2024 Release"),
            ("MILESTONE: Beta Launch", "Beta Launch"),
            ("milestone: v2.0 deployment", "v2.0 deployment"),
            ("Milestone:\nPhase 1 Complete", "Phase 1 Complete"),
        ]
        
        for content, expected_milestone in test_cases:
            metadata = extract_temporal_metadata(content)
            assert metadata.get('milestone') == expected_milestone.strip(), \
                f"Expected milestone '{expected_milestone}' from '{content}'"
    
    def test_milestone_with_date(self):
        """Test milestone with embedded date."""
        content = "Milestone: Release on 2024-06-30"
        metadata = extract_temporal_metadata(content)
        
        assert metadata.get('milestone') == "Release on 2024-06-30"
        # Should also extract the date as expiry
        assert 'expires_at_ms' in metadata
        expected_ms = datetime(2024, 6, 30).timestamp() * 1000
        assert abs(metadata['expires_at_ms'] - expected_ms) < 86400000


class TestVersionExtraction:
    """Test version extraction from filenames."""
    
    def test_version_patterns(self):
        """Test various version patterns in filenames."""
        test_cases = [
            ("spec_v1.2.md", "1.2"),
            ("doc_v3.md", "3"),
            ("report-v2.1.3.pdf", "2.1.3"),
            ("file_v2024.01.txt", "2024.01"),
            ("document.v5.docx", "5"),
            ("api_spec_1.0.yaml", "1.0"),
            ("noversion.txt", None),
        ]
        
        for filename, expected_version in test_cases:
            metadata = extract_temporal_metadata("", filename)
            assert metadata.get('version') == expected_version, \
                f"Expected version '{expected_version}' from '{filename}', got {metadata.get('version')}"


class TestLifecycleStage:
    """Test lifecycle stage determination."""
    
    @patch('src.core.temporal.date_extractor.time')
    def test_expired_lifecycle(self, mock_time):
        """Test detection of expired documents."""
        # Set current time
        current_time = datetime(2024, 6, 15).timestamp()
        mock_time.time.return_value = current_time
        
        # Document that expired
        content = "Expires: 2024-06-01"
        metadata = extract_temporal_metadata(content)
        
        assert metadata.get('lifecycle_stage') == 'expired'
    
    @patch('src.core.temporal.date_extractor.time')
    def test_active_lifecycle(self, mock_time):
        """Test detection of active documents."""
        current_time = datetime(2024, 6, 15).timestamp()
        mock_time.time.return_value = current_time
        
        # Document still valid
        content = "Expires: 2024-12-31"
        metadata = extract_temporal_metadata(content)
        
        assert metadata.get('lifecycle_stage') == 'active'
    
    @patch('src.core.temporal.date_extractor.time')
    def test_age_based_lifecycle(self, mock_time):
        """Test lifecycle based on document age."""
        current_time = datetime(2024, 6, 15).timestamp()
        mock_time.time.return_value = current_time
        
        # Current document (< 30 days)
        content = "Created: 2024-06-01"
        metadata = extract_temporal_metadata(content)
        assert metadata.get('lifecycle_stage') == 'current'
        
        # Recent document (30-90 days)
        content = "Created: 2024-04-01"
        metadata = extract_temporal_metadata(content)
        assert metadata.get('lifecycle_stage') == 'recent'
        
        # Historical document (> 90 days)
        content = "Created: 2024-01-01"
        metadata = extract_temporal_metadata(content)
        assert metadata.get('lifecycle_stage') == 'historical'


class TestRelevanceWindow:
    """Test relevance window calculation."""
    
    @patch('src.core.temporal.date_extractor.time')
    def test_expired_relevance(self, mock_time):
        """Test relevance for expired documents."""
        current_time = datetime(2024, 6, 15).timestamp()
        mock_time.time.return_value = current_time
        
        # Recently expired (< 30 days)
        metadata = {
            'expires_at_ms': datetime(2024, 6, 1).timestamp() * 1000
        }
        score, reason = calculate_relevance_window(metadata)
        assert score == 0.3
        assert reason == "recently_expired"
        
        # Long expired (> 30 days)
        metadata = {
            'expires_at_ms': datetime(2024, 1, 1).timestamp() * 1000
        }
        score, reason = calculate_relevance_window(metadata)
        assert score == 0.1
        assert reason == "expired_old"
    
    @patch('src.core.temporal.date_extractor.time')
    def test_expiring_soon_relevance(self, mock_time):
        """Test relevance for documents expiring soon."""
        current_time = datetime(2024, 6, 15).timestamp()
        mock_time.time.return_value = current_time
        
        # Expiring in 5 days
        metadata = {
            'expires_at_ms': datetime(2024, 6, 20).timestamp() * 1000
        }
        score, reason = calculate_relevance_window(metadata)
        assert score == 0.9
        assert reason == "expiring_soon"
        
        # Valid for longer
        metadata = {
            'expires_at_ms': datetime(2024, 12, 31).timestamp() * 1000
        }
        score, reason = calculate_relevance_window(metadata)
        assert score == 1.0
        assert reason == "valid"
    
    def test_milestone_relevance(self):
        """Test relevance for milestone documents."""
        metadata = {'milestone': 'Q1 Release'}
        score, reason = calculate_relevance_window(metadata)
        assert score == 0.8
        assert reason == "milestone_document"
    
    def test_sprint_relevance(self):
        """Test relevance for sprint documents."""
        metadata = {'sprint': 23}
        score, reason = calculate_relevance_window(metadata)
        assert score == 0.7
        assert reason == "sprint_document"
    
    @patch('src.core.temporal.date_extractor.time')
    def test_age_based_relevance(self, mock_time):
        """Test age-based relevance calculation."""
        current_time = datetime(2024, 6, 15).timestamp()
        mock_time.time.return_value = current_time
        
        test_cases = [
            (datetime(2024, 6, 14), 1.0, "very_recent"),  # 1 day old
            (datetime(2024, 6, 1), 0.8, "recent"),  # 14 days old
            (datetime(2024, 4, 1), 0.5, "moderate_age"),  # ~75 days old
            (datetime(2024, 1, 1), 0.3, "old"),  # > 90 days old
        ]
        
        for created_date, expected_score, expected_reason in test_cases:
            metadata = {'created_at_ms': created_date.timestamp() * 1000}
            score, reason = calculate_relevance_window(metadata)
            assert score == expected_score, \
                f"Expected score {expected_score} for {created_date}, got {score}"
            assert reason == expected_reason
    
    def test_no_temporal_data(self):
        """Test handling of documents with no temporal data."""
        metadata = {}
        score, reason = calculate_relevance_window(metadata)
        assert score == 0.5
        assert reason == "no_temporal_data"


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling in date extraction."""
    
    def test_malformed_dates(self):
        """Test handling of malformed date strings."""
        test_cases = [
            "Created: 2024-13-45",  # Invalid month/day
            "Date: 99/99/9999",  # Invalid date
            "Updated: February 30, 2024",  # Invalid day for February
            "Expires: 2024",  # Incomplete date
            "Date: tomorrow",  # Relative date
        ]
        
        for content in test_cases:
            metadata = extract_temporal_metadata(content)
            # Should not crash, might not extract date
            assert isinstance(metadata, dict)
    
    def test_ambiguous_dates(self):
        """Test handling of ambiguous date formats."""
        # Could be MM/DD or DD/MM
        content = "Date: 01/02/2024"
        metadata = extract_temporal_metadata(content)
        
        # Should parse as MM/DD (US format)
        assert 'created_at_ms' in metadata
        expected = datetime(2024, 1, 2).timestamp() * 1000  # Jan 2, not Feb 1
        assert abs(metadata['created_at_ms'] - expected) < 86400000
    
    def test_empty_content(self):
        """Test handling of empty content."""
        metadata = extract_temporal_metadata("")
        assert isinstance(metadata, dict)
        assert len(metadata) == 0 or all(v is None for v in metadata.values())
        
        metadata = extract_temporal_metadata("", "")
        assert isinstance(metadata, dict)
    
    def test_very_long_content(self):
        """Test extraction from very long content."""
        # Date at beginning
        content = "Created: 2024-01-15\n" + "x" * 10000
        metadata = extract_temporal_metadata(content)
        assert 'created_at_ms' in metadata
        
        # Date beyond first 1000 chars (should be found for expiry)
        content = "x" * 1500 + "\nExpires: 2024-12-31"
        metadata = extract_temporal_metadata(content)
        assert 'expires_at_ms' in metadata
    
    def test_multiple_date_formats(self):
        """Test content with multiple date formats."""
        content = """
        Created: 2024-01-15
        Updated: Jan 20, 2024
        Review date: 02/15/2024
        Expires: December 31, 2024
        """
        metadata = extract_temporal_metadata(content)
        
        # Should get created and expires
        assert 'created_at_ms' in metadata
        assert 'expires_at_ms' in metadata
        
        # Created should be Jan 15
        created_expected = datetime(2024, 1, 15).timestamp() * 1000
        assert abs(metadata['created_at_ms'] - created_expected) < 86400000
        
        # Expires should be Dec 31
        expires_expected = datetime(2024, 12, 31).timestamp() * 1000
        assert abs(metadata['expires_at_ms'] - expires_expected) < 86400000
    
    def test_case_insensitive_keywords(self):
        """Test case-insensitive keyword detection."""
        test_cases = [
            "CREATED: 2024-01-15",
            "created: 2024-01-15",
            "Created: 2024-01-15",
            "CrEaTeD: 2024-01-15",
        ]
        
        for content in test_cases:
            metadata = extract_temporal_metadata(content)
            assert 'created_at_ms' in metadata
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        content = "Created: 2024-01-15\n创建日期：2024年1月15日"
        metadata = extract_temporal_metadata(content)
        assert 'created_at_ms' in metadata
        
        # With unicode in filename
        metadata = extract_temporal_metadata("", "文档_2024-01-15.md")
        assert 'created_at_ms' in metadata


class TestPrivateFunctions:
    """Test private helper functions."""
    
    def test_parse_date_helper(self):
        """Test the _parse_date helper function."""
        patterns = [
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),
        ]
        
        # Valid dates
        assert _parse_date("2024-01-15", patterns) == datetime(2024, 1, 15)
        assert _parse_date("1/15/2024", patterns) == datetime(2024, 1, 15)
        assert _parse_date("Text before 2024-01-15 text after", patterns) == datetime(2024, 1, 15)
        
        # Invalid dates
        assert _parse_date("not a date", patterns) is None
        assert _parse_date("2024-13-45", patterns) is None
        assert _parse_date("", patterns) is None


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])