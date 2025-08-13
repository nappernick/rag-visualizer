"""
Comprehensive tests for temporal utilities with edge cases and deep coverage.
"""
import pytest
import time
import math
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.temporal.temporal_utils import (
    detect_doc_type,
    get_temporal_score,
    apply_temporal_boost,
    enrich_with_temporal_metadata
)


class TestDocTypeDetection:
    """Test document type detection with various edge cases."""
    
    def test_values_detection_path_patterns(self):
        """Test detection of values/legal documents from path."""
        assert detect_doc_type("company_values.md") == "values"
        assert detect_doc_type("legal/contract.pdf") == "values"
        assert detect_doc_type("docs/charter_2024.docx") == "values"
        assert detect_doc_type("policy/security_policy.md") == "values"
        assert detect_doc_type("/path/to/VALUES.txt") == "values"
        assert detect_doc_type("LEGAL_NOTICE.md") == "values"
    
    def test_project_detection_path_patterns(self):
        """Test detection of project documents from path."""
        assert detect_doc_type("project_spec.md") == "project"
        assert detect_doc_type("architecture/design.md") == "project"
        assert detect_doc_type("workflow_diagram.png") == "project"
        assert detect_doc_type("technical_requirements.pdf") == "project"
        assert detect_doc_type("system_ARCHITECTURE.docx") == "project"
        assert detect_doc_type("API_SPEC_v2.yaml") == "project"
    
    def test_meeting_detection_path_patterns(self):
        """Test detection of meeting notes from path."""
        assert detect_doc_type("meeting_notes.md") == "meeting"
        assert detect_doc_type("2024_01_standup.txt") == "meeting"
        assert detect_doc_type("team/minutes_jan.docx") == "meeting"
        assert detect_doc_type("MEETING_NOTES_Q1.pdf") == "meeting"
        assert detect_doc_type("standup-2024-01-15.md") == "meeting"
    
    def test_content_based_detection(self):
        """Test detection based on content when path doesn't match."""
        # Project content
        content = "Milestone: Q1 Release\nSprint 23 objectives..."
        assert detect_doc_type("random.txt", content) == "project"
        
        # Meeting content
        content = "Meeting notes\nAttendees: John, Jane\nAction items:..."
        assert detect_doc_type("doc.md", content) == "meeting"
        
        # No matching content
        content = "Random text without keywords"
        assert detect_doc_type("file.txt", content) == "default"
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        assert detect_doc_type("") == "default"
        assert detect_doc_type("", "") == "default"
        assert detect_doc_type("file.txt", "") == "default"
    
    def test_case_insensitivity(self):
        """Test case-insensitive detection."""
        assert detect_doc_type("VALUES.MD") == "values"
        assert detect_doc_type("Meeting_Notes.TXT") == "meeting"
        assert detect_doc_type("SPEC_DOC.PDF") == "project"
        
        content = "MILESTONE: Q1\nSPRINT 23"
        assert detect_doc_type("doc.txt", content) == "project"
    
    def test_special_characters_in_path(self):
        """Test paths with special characters."""
        assert detect_doc_type("company-values_2024.md") == "values"
        assert detect_doc_type("meeting.notes@jan.txt") == "meeting"
        assert detect_doc_type("spec~v2.1.md") == "project"
        assert detect_doc_type("../../../legal/doc.pdf") == "values"
    
    def test_priority_order(self):
        """Test that path patterns take priority over content."""
        meeting_content = "This is about meeting and action items"
        assert detect_doc_type("company_values.md", meeting_content) == "values"
        
        project_content = "Sprint 23 milestone deadline"
        assert detect_doc_type("meeting_notes.md", project_content) == "meeting"


class TestTemporalScoring:
    """Test temporal scoring with Weibull decay functions."""
    
    def test_values_no_decay(self):
        """Test that values documents never decay."""
        now_ms = time.time() * 1000
        
        # Test various ages
        for days_old in [0, 30, 120, 365, 1000]:
            created_at = now_ms - (days_old * 86_400_000)
            score = get_temporal_score(created_at, "values")
            assert score == 1.0, f"Values doc should always be 1.0, got {score} at {days_old} days"
        
        # Also test other time-invariant types
        assert get_temporal_score(now_ms - 365 * 86_400_000, "legal") == 1.0
        assert get_temporal_score(now_ms - 365 * 86_400_000, "charter") == 1.0
        assert get_temporal_score(now_ms - 365 * 86_400_000, "policy") == 1.0
    
    def test_project_weibull_decay(self):
        """Test project document Weibull decay (k=2.5, λ=120)."""
        now_ms = time.time() * 1000
        
        # At 0 days: should be 1.0
        score = get_temporal_score(now_ms, "project")
        assert abs(score - 1.0) < 0.01
        
        # At 60 days: should be ~0.84
        score = get_temporal_score(now_ms - 60 * 86_400_000, "project")
        assert 0.82 < score < 0.86, f"Expected ~0.84 at 60 days, got {score}"
        
        # At 120 days (characteristic life): should be ~0.37
        score = get_temporal_score(now_ms - 120 * 86_400_000, "project")
        assert 0.35 < score < 0.39, f"Expected ~0.37 at 120 days, got {score}"
        
        # At 150 days: should be ~0.17
        score = get_temporal_score(now_ms - 150 * 86_400_000, "project")
        assert 0.15 < score < 0.19, f"Expected ~0.17 at 150 days, got {score}"
        
        # Very old: should hit baseline of 0.1
        score = get_temporal_score(now_ms - 365 * 86_400_000, "project")
        assert score == 0.1, f"Expected baseline 0.1, got {score}"
    
    def test_meeting_rapid_decay(self):
        """Test meeting notes rapid decay (k=0.8, λ=14)."""
        now_ms = time.time() * 1000
        
        # At 0 days: should be 1.0
        score = get_temporal_score(now_ms, "meeting")
        assert abs(score - 1.0) < 0.01
        
        # At 7 days: should be ~0.56
        score = get_temporal_score(now_ms - 7 * 86_400_000, "meeting")
        assert 0.54 < score < 0.58, f"Expected ~0.56 at 7 days, got {score}"
        
        # At 14 days (characteristic life): should be ~0.37
        score = get_temporal_score(now_ms - 14 * 86_400_000, "meeting")
        assert 0.35 < score < 0.39, f"Expected ~0.37 at 14 days, got {score}"
        
        # At 30 days: should be close to baseline (0.05)
        score = get_temporal_score(now_ms - 30 * 86_400_000, "meeting")
        assert 0.05 <= score < 0.20, f"Expected low score at 30 days, got {score}"
        
        # Very old: should be baseline 0.05
        score = get_temporal_score(now_ms - 100 * 86_400_000, "meeting")
        assert score == 0.05, f"Expected baseline 0.05, got {score}"
    
    def test_default_moderate_decay(self):
        """Test default document decay (k=1.5, λ=60)."""
        now_ms = time.time() * 1000
        
        # At 30 days: should be ~0.65
        score = get_temporal_score(now_ms - 30 * 86_400_000, "default")
        assert 0.63 < score < 0.67, f"Expected ~0.65 at 30 days, got {score}"
        
        # At 60 days: should be ~0.37
        score = get_temporal_score(now_ms - 60 * 86_400_000, "default")
        assert 0.35 < score < 0.39, f"Expected ~0.37 at 60 days, got {score}"
        
        # At 120 days: should be close to baseline
        score = get_temporal_score(now_ms - 120 * 86_400_000, "default")
        assert score <= 0.15, f"Expected low score at 120 days, got {score}"
    
    def test_none_timestamp(self):
        """Test handling of None timestamp."""
        assert get_temporal_score(None, "project") == 1.0
        assert get_temporal_score(None, "meeting") == 1.0
        assert get_temporal_score(None, "default") == 1.0
    
    def test_future_timestamps(self):
        """Test handling of future timestamps (negative age)."""
        future_ms = time.time() * 1000 + 86_400_000  # 1 day in future
        
        # Should treat as current (score = 1.0)
        assert get_temporal_score(future_ms, "project") == 1.0
        assert get_temporal_score(future_ms, "meeting") == 1.0
    
    def test_very_old_documents(self):
        """Test extremely old documents hit baseline."""
        ancient_ms = time.time() * 1000 - (10000 * 86_400_000)  # 10000 days old
        
        assert get_temporal_score(ancient_ms, "project") == 0.1
        assert get_temporal_score(ancient_ms, "meeting") == 0.05
        assert get_temporal_score(ancient_ms, "default") == 0.1
        assert get_temporal_score(ancient_ms, "values") == 1.0  # Still no decay
    
    def test_edge_case_timestamps(self):
        """Test edge case timestamps."""
        # Zero timestamp
        assert get_temporal_score(0, "project") == 0.1  # Very old
        
        # Negative timestamp (invalid)
        assert get_temporal_score(-1000, "project") == 0.1
        
        # Very large timestamp (far future)
        huge_ms = 9999999999999
        assert get_temporal_score(huge_ms, "project") == 1.0


class TestQueryTemporalBoost:
    """Test query-based temporal weight adjustment."""
    
    def test_high_recency_keywords(self):
        """Test queries that should boost temporal weight."""
        queries = [
            "What is the latest status?",
            "Show current architecture",
            "Get recent updates",
            "Find new documentation",
            "Latest project spec",
            "LATEST meeting notes",
            "most recent and current docs"
        ]
        
        for query in queries:
            weight = apply_temporal_boost(query)
            assert weight == 0.5, f"Query '{query}' should have weight 0.5, got {weight}"
    
    def test_historical_keywords(self):
        """Test queries that should reduce temporal weight."""
        queries = [
            "Find historical data",
            "Archive of old docs",
            "Previous version",
            "Historical meeting notes",
            "ARCHIVE search",
            "old project specs"
        ]
        
        for query in queries:
            weight = apply_temporal_boost(query)
            assert weight == 0.1, f"Query '{query}' should have weight 0.1, got {weight}"
    
    def test_default_weight(self):
        """Test queries without temporal hints."""
        queries = [
            "Find project documentation",
            "Search for API specs",
            "Meeting notes about design",
            "Technical requirements",
            "",
            "   ",
            "Random search query"
        ]
        
        for query in queries:
            weight = apply_temporal_boost(query)
            assert weight == 0.3, f"Query '{query}' should have default weight 0.3, got {weight}"
    
    def test_mixed_keywords(self):
        """Test queries with conflicting temporal hints."""
        # "latest" takes precedence over "historical"
        query = "latest historical data"
        assert apply_temporal_boost(query) == 0.5
        
        # "archive" takes precedence when it appears first
        query = "archive of current docs"
        assert apply_temporal_boost(query) == 0.1
    
    def test_case_insensitivity(self):
        """Test case-insensitive keyword detection."""
        assert apply_temporal_boost("LATEST docs") == 0.5
        assert apply_temporal_boost("Historical DATA") == 0.1
        assert apply_temporal_boost("CuRrEnT status") == 0.5


class TestTemporalMetadataEnrichment:
    """Test metadata enrichment with temporal information."""
    
    @patch('src.core.temporal.temporal_utils.time')
    def test_basic_enrichment(self, mock_time):
        """Test basic metadata enrichment."""
        mock_time.time.return_value = 1700000000  # Fixed timestamp
        
        from src.core.temporal.temporal_utils import enrich_with_temporal_metadata
        
        metadata = {}
        content = "Project specification document"
        filename = "spec.md"
        
        enriched = enrich_with_temporal_metadata(metadata, content, filename)
        
        assert 'created_at_ms' in enriched
        assert enriched['created_at_ms'] == 1700000000000
        assert enriched['doc_type'] == 'project'
    
    @patch('src.core.temporal.date_extractor.extract_temporal_metadata')
    def test_extracted_metadata_merge(self, mock_extract):
        """Test merging of extracted temporal metadata."""
        from src.core.temporal.temporal_utils import enrich_with_temporal_metadata
        
        mock_extract.return_value = {
            'created_at_ms': 1600000000000,
            'expires_at_ms': 1800000000000,
            'milestone': 'Q1 Release',
            'sprint': 23,
            'version': '2.1',
            'lifecycle_stage': 'active'
        }
        
        metadata = {'existing_field': 'value'}
        enriched = enrich_with_temporal_metadata(metadata, "content", "file.md")
        
        assert enriched['created_at_ms'] == 1600000000000
        assert enriched['expires_at_ms'] == 1800000000000
        assert enriched['milestone'] == 'Q1 Release'
        assert enriched['sprint'] == 23
        assert enriched['version'] == '2.1'
        assert enriched['lifecycle_stage'] == 'active'
        assert enriched['existing_field'] == 'value'
    
    def test_preserve_existing_created_at(self):
        """Test that existing created_at is preserved."""
        from src.core.temporal.temporal_utils import enrich_with_temporal_metadata
        
        metadata = {'created_at_ms': 1500000000000}
        
        with patch('src.core.temporal.date_extractor.extract_temporal_metadata') as mock:
            mock.return_value = {'created_at_ms': 1600000000000}
            enriched = enrich_with_temporal_metadata(metadata, "content", "file.md")
        
        # Should preserve original
        assert enriched['created_at_ms'] == 1500000000000
    
    def test_doc_type_detection_in_enrichment(self):
        """Test doc type detection during enrichment."""
        from src.core.temporal.temporal_utils import enrich_with_temporal_metadata
        
        # Should detect from filename
        metadata = {}
        enriched = enrich_with_temporal_metadata(metadata, "", "meeting_notes.md")
        assert enriched['doc_type'] == 'meeting'
        
        # Should detect from content if filename doesn't match
        metadata = {}
        content = "Sprint 23 objectives and milestones"
        enriched = enrich_with_temporal_metadata(metadata, content, "doc.txt")
        assert enriched['doc_type'] == 'project'
        
        # Should preserve existing doc_type
        metadata = {'doc_type': 'custom'}
        enriched = enrich_with_temporal_metadata(metadata, "", "meeting.md")
        assert enriched['doc_type'] == 'custom'
    
    @patch('src.core.temporal.date_extractor.extract_temporal_metadata')
    def test_empty_extraction_handling(self, mock_extract):
        """Test handling when date extraction returns nothing."""
        from src.core.temporal.temporal_utils import enrich_with_temporal_metadata
        
        mock_extract.return_value = {}
        
        with patch('src.core.temporal.temporal_utils.time') as mock_time:
            mock_time.time.return_value = 1700000000
            
            metadata = {}
            enriched = enrich_with_temporal_metadata(metadata, "content", "file.md")
            
            # Should still have created_at_ms from current time
            assert enriched['created_at_ms'] == 1700000000000
            assert enriched['doc_type'] == 'default'


class TestWeibullDecayMath:
    """Test the mathematical properties of Weibull decay."""
    
    def test_weibull_shape_parameter_effects(self):
        """Test that shape parameter affects decay curve correctly."""
        now_ms = time.time() * 1000
        
        # For project docs (k=2.5), decay should accelerate
        # Early decay should be slower than late decay
        age_30 = now_ms - 30 * 86_400_000
        age_60 = now_ms - 60 * 86_400_000
        age_90 = now_ms - 90 * 86_400_000
        age_120 = now_ms - 120 * 86_400_000
        
        score_30 = get_temporal_score(age_30, "project")
        score_60 = get_temporal_score(age_60, "project")
        score_90 = get_temporal_score(age_90, "project")
        score_120 = get_temporal_score(age_120, "project")
        
        # Calculate decay rates
        decay_rate_early = (score_30 - score_60) / 30  # Decay per day from 30-60
        decay_rate_late = (score_90 - score_120) / 30  # Decay per day from 90-120
        
        # Late decay should be faster (more negative) than early
        assert abs(decay_rate_late) > abs(decay_rate_early), \
            f"Project docs should decay faster later. Early: {decay_rate_early}, Late: {decay_rate_late}"
    
    def test_weibull_characteristic_life(self):
        """Test that characteristic life (scale parameter) works correctly."""
        now_ms = time.time() * 1000
        
        # At characteristic life, Weibull should be e^(-1) ≈ 0.368
        expected = math.exp(-1)
        
        # Project: λ=120 days
        score = get_temporal_score(now_ms - 120 * 86_400_000, "project")
        assert abs(score - expected) < 0.02, f"At λ=120, expected {expected}, got {score}"
        
        # Meeting: λ=14 days
        score = get_temporal_score(now_ms - 14 * 86_400_000, "meeting")
        assert abs(score - expected) < 0.02, f"At λ=14, expected {expected}, got {score}"
        
        # Default: λ=60 days
        score = get_temporal_score(now_ms - 60 * 86_400_000, "default")
        assert abs(score - expected) < 0.02, f"At λ=60, expected {expected}, got {score}"
    
    def test_baseline_enforcement(self):
        """Test that baseline values are enforced."""
        very_old = time.time() * 1000 - 1000 * 86_400_000
        
        # Check baselines
        assert get_temporal_score(very_old, "project") == 0.1
        assert get_temporal_score(very_old, "meeting") == 0.05
        assert get_temporal_score(very_old, "default") == 0.1
        
        # Values should never hit baseline (no decay)
        assert get_temporal_score(very_old, "values") == 1.0


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""
    
    def test_unicode_and_special_chars(self):
        """Test handling of unicode and special characters."""
        # Unicode in paths
        assert detect_doc_type("会议记录.md", "") == "default"
        assert detect_doc_type("café_meeting_notes.txt", "") == "meeting"
        assert detect_doc_type("spec™.pdf", "") == "project"
        
        # Unicode in content
        content = "Meeting notes: 会議 📝\nAction items: ✓ Complete"
        assert detect_doc_type("doc.md", content) == "meeting"
    
    def test_very_long_inputs(self):
        """Test handling of very long paths and content."""
        # Very long path
        long_path = "meeting_" + "x" * 1000 + "_notes.md"
        assert detect_doc_type(long_path) == "meeting"
        
        # Very long content (should only check first 500 chars)
        content = "Sprint 23 " + "x" * 10000
        assert detect_doc_type("doc.txt", content) == "project"
    
    def test_concurrent_scoring(self):
        """Test thread safety of temporal scoring."""
        import threading
        import random
        
        results = []
        errors = []
        
        def score_random():
            try:
                for _ in range(100):
                    age_days = random.randint(0, 365)
                    created_at = time.time() * 1000 - age_days * 86_400_000
                    doc_type = random.choice(['project', 'meeting', 'values', 'default'])
                    score = get_temporal_score(created_at, doc_type)
                    results.append(score)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=score_random) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors and valid scores
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 1000
        assert all(0 <= s <= 1.0 for s in results)
    
    def test_precision_and_rounding(self):
        """Test numerical precision and rounding."""
        now_ms = time.time() * 1000
        
        # Very small time differences
        score1 = get_temporal_score(now_ms - 0.001, "project")
        score2 = get_temporal_score(now_ms - 0.002, "project")
        assert abs(score1 - score2) < 0.0001  # Should be nearly identical
        
        # Check scores are in valid range
        for days in range(0, 400, 10):
            for doc_type in ['project', 'meeting', 'default', 'values']:
                score = get_temporal_score(now_ms - days * 86_400_000, doc_type)
                assert 0 <= score <= 1.0, f"Score out of range: {score} for {doc_type} at {days} days"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])