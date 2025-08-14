#!/usr/bin/env python3
"""
Test and demonstrate temporal RAG functionality.
Shows how different document types decay over time.
"""
import time
from datetime import datetime, timedelta
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Note: Matplotlib not available, skipping visualizations")

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.temporal.temporal_utils import get_temporal_score, detect_doc_type, apply_temporal_boost


def test_doc_type_detection():
    """Test document type detection"""
    print("\n" + "="*60)
    print("TEST 1: Document Type Detection")
    print("="*60)
    
    test_cases = [
        ("company_values.md", "Our company values guide everything we do"),
        ("project_spec_v2.3.md", "Technical specification for the new API"),
        ("meeting_notes_2024_01.md", "Meeting notes with action items"),
        ("legal/contract_template.pdf", ""),
        ("architecture/system_design.md", "System architecture overview"),
        ("sprint_23_planning.md", "Sprint 23 goals and milestones"),
    ]
    
    for path, content in test_cases:
        doc_type = detect_doc_type(path, content)
        print(f"  {path:35} → {doc_type:10}")


def test_temporal_scoring():
    """Test temporal scoring for different document types"""
    print("\n" + "="*60)
    print("TEST 2: Temporal Decay Patterns")
    print("="*60)
    
    # Current time in milliseconds
    now_ms = time.time() * 1000
    
    # Test different ages
    test_ages = [0, 7, 14, 30, 60, 90, 120, 135, 150, 180, 365]
    doc_types = ['project', 'meeting', 'values', 'default']
    
    print("\n  Age (days) |  Project  | Meeting  |  Values  | Default")
    print("  " + "-"*56)
    
    for age_days in test_ages:
        created_at_ms = now_ms - (age_days * 86_400_000)
        scores = []
        
        for doc_type in doc_types:
            score = get_temporal_score(created_at_ms, doc_type)
            scores.append(score)
        
        print(f"  {age_days:9} | {scores[0]:8.2f} | {scores[1]:7.2f} | {scores[2]:7.2f} | {scores[3]:7.2f}")


def plot_decay_curves():
    """Plot temporal decay curves for visualization"""
    print("\n" + "="*60)
    print("TEST 3: Visualizing Decay Curves (Weibull Distribution)")
    print("="*60)
    
    if not PLOT_AVAILABLE:
        print("  Skipping visualization - matplotlib not installed")
        return
    
    # Generate age range
    ages = np.linspace(0, 200, 500)
    now_ms = time.time() * 1000
    
    # Calculate scores for each doc type
    project_scores = []
    meeting_scores = []
    values_scores = []
    default_scores = []
    
    for age in ages:
        created_at_ms = now_ms - (age * 86_400_000)
        project_scores.append(get_temporal_score(created_at_ms, 'project'))
        meeting_scores.append(get_temporal_score(created_at_ms, 'meeting'))
        values_scores.append(get_temporal_score(created_at_ms, 'values'))
        default_scores.append(get_temporal_score(created_at_ms, 'default'))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(ages, project_scores, 'b-', linewidth=2, label='Project Docs (120d slow → 30d rapid)')
    plt.plot(ages, meeting_scores, 'r-', linewidth=2, label='Meeting Notes (rapid decay)')
    plt.plot(ages, values_scores, 'g-', linewidth=2, label='Values/Legal (no decay)')
    plt.plot(ages, default_scores, 'gray', linestyle='--', linewidth=1, label='Default (gradual)')
    
    # Add key points for project docs
    plt.axvline(x=120, color='blue', linestyle=':', alpha=0.5)
    plt.text(120, 0.85, '120 days\n(80% relevance)', ha='center', fontsize=9)
    plt.axvline(x=150, color='blue', linestyle=':', alpha=0.5)
    plt.text(150, 0.15, '150 days\n(10% relevance)', ha='center', fontsize=9)
    
    plt.xlabel('Document Age (days)', fontsize=12)
    plt.ylabel('Temporal Relevance Score', fontsize=12)
    plt.title('Temporal Decay Patterns for Consulting Documents', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim(0, 200)
    plt.ylim(0, 1.05)
    
    # Save plot
    plt.savefig('temporal_decay_curves.png', dpi=150, bbox_inches='tight')
    print("\n  Decay curves saved to: temporal_decay_curves.png")
    
    # Show plot (if running interactively)
    try:
        plt.show()
    except:
        pass


def test_query_temporal_boost():
    """Test how queries affect temporal weighting"""
    print("\n" + "="*60)
    print("TEST 4: Query-Based Temporal Weighting")
    print("="*60)
    
    test_queries = [
        "What is the latest project status?",
        "Show me current architecture",
        "Find historical meeting notes",
        "What are our company values?",
        "Get me the newest technical spec",
        "Archive of old project documents",
        "Standard query without time hints",
    ]
    
    print("\n  Query | Temporal Weight | Semantic Weight")
    print("  " + "-"*50)
    
    for query in test_queries:
        temporal_weight = apply_temporal_boost(query)
        semantic_weight = 1.0 - temporal_weight
        print(f"  {query[:40]:40} | {temporal_weight:6.1%} | {semantic_weight:6.1%}")


def simulate_retrieval():
    """Simulate how temporal scoring affects document ranking"""
    print("\n" + "="*60)
    print("TEST 5: Simulated Retrieval with Temporal Scoring")
    print("="*60)
    
    # Mock documents with different ages and types
    now_ms = time.time() * 1000
    
    documents = [
        {"name": "project_spec_v3.md", "age_days": 5, "type": "project", "semantic_score": 0.85},
        {"name": "project_spec_v2.md", "age_days": 125, "type": "project", "semantic_score": 0.90},
        {"name": "meeting_2024_01.md", "age_days": 20, "type": "meeting", "semantic_score": 0.75},
        {"name": "company_values.md", "age_days": 365, "type": "values", "semantic_score": 0.70},
        {"name": "architecture_old.md", "age_days": 180, "type": "project", "semantic_score": 0.88},
        {"name": "sprint_current.md", "age_days": 3, "type": "project", "semantic_score": 0.82},
    ]
    
    # Test with different queries
    queries = [
        ("Find latest project documentation", 0.5),  # High temporal weight
        ("Show project specifications", 0.3),         # Balanced
        ("Historical project overview", 0.1),         # Low temporal weight
    ]
    
    for query, temporal_weight in queries:
        print(f"\n  Query: '{query}' (temporal weight: {temporal_weight:.0%})")
        print("  " + "-"*70)
        print("  Document                    | Semantic | Temporal | Combined | Rank")
        print("  " + "-"*70)
        
        semantic_weight = 1.0 - temporal_weight
        results = []
        
        for doc in documents:
            created_at_ms = now_ms - (doc['age_days'] * 86_400_000)
            temporal_score = get_temporal_score(created_at_ms, doc['type'])
            combined_score = (semantic_weight * doc['semantic_score']) + (temporal_weight * temporal_score)
            
            results.append({
                'name': doc['name'],
                'semantic': doc['semantic_score'],
                'temporal': temporal_score,
                'combined': combined_score
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined'], reverse=True)
        
        for i, result in enumerate(results, 1):
            print(f"  {result['name']:27} | {result['semantic']:8.2f} | {result['temporal']:8.2f} | "
                  f"{result['combined']:8.2f} | #{i}")


def main():
    """Run all temporal tests"""
    print("\n🕐 TEMPORAL RAG SYSTEM TESTS 🕐")
    
    test_doc_type_detection()
    test_temporal_scoring()
    test_query_temporal_boost()
    simulate_retrieval()
    
    # Generate visualization
    try:
        plot_decay_curves()
    except ImportError:
        print("\n  ⚠️  Matplotlib not installed - skipping visualization")
        print("     Install with: pip install matplotlib")
    
    print("\n" + "="*60)
    print("✅ TEMPORAL TESTING COMPLETE")
    print("="*60)
    
    print("\n📝 Summary:")
    print("  • Document types are auto-detected from paths and content")
    print("  • Project docs: 120 days slow decay → 30 days rapid → 10% baseline")
    print("  • Meeting notes: Rapid exponential decay (50% at 14 days)")
    print("  • Values/Legal: No decay (always 100% relevant)")
    print("  • Query keywords ('latest', 'current') boost temporal weight")
    print("  • Final score = (semantic * weight) + (temporal * weight)")
    
    print("\n💡 Integration Notes:")
    print("  • Add 'created_at_ms' to chunk metadata during ingestion")
    print("  • Add 'doc_type' or let system auto-detect from path")
    print("  • Temporal scoring automatically applied in fusion_controller")
    print("  • No schema migrations required - uses existing metadata field")


if __name__ == "__main__":
    main()