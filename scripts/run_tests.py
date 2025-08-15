#!/usr/bin/env python3
"""
Comprehensive test runner for the enhanced RAG system.
Runs all unit and integration tests with coverage reporting.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        return False
    
    print(f"✅ {description} completed successfully")
    return True


def main():
    """Run all tests with various configurations."""
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    print("🧪 Enhanced RAG System - Comprehensive Test Suite")
    print("="*60)
    
    all_passed = True
    
    # 1. Run unit tests for temporal features
    if not run_command(
        ["python", "-m", "pytest", "tests/unit/test_temporal_utils.py", "-v"],
        "Temporal Utilities Tests"
    ):
        all_passed = False
    
    if not run_command(
        ["python", "-m", "pytest", "tests/unit/test_date_extractor.py", "-v"],
        "Date Extraction Tests"
    ):
        all_passed = False
    
    # 2. Run unit tests for enhanced features
    if not run_command(
        ["python", "-m", "pytest", "tests/unit/test_hybrid_search.py", "-v"],
        "Hybrid Search with RRF Tests"
    ):
        all_passed = False
    
    if not run_command(
        ["python", "-m", "pytest", "tests/unit/test_query_enhancer.py", "-v"],
        "Query Enhancement Tests"
    ):
        all_passed = False
    
    if not run_command(
        ["python", "-m", "pytest", "tests/unit/test_graph_rag.py", "-v"],
        "GraphRAG Multi-hop Tests"
    ):
        all_passed = False
    
    # 3. Run integration tests
    if not run_command(
        ["python", "-m", "pytest", "tests/integration/test_fusion_controller.py", "-v"],
        "Fusion Controller Integration Tests"
    ):
        all_passed = False
    
    # 4. Run all tests with coverage
    print("\n" + "="*60)
    print("Running all tests with coverage report...")
    print("="*60)
    
    coverage_cmd = [
        "python", "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]
    
    if not run_command(coverage_cmd, "Full Test Suite with Coverage"):
        all_passed = False
    
    # 5. Run specific test categories
    print("\n" + "="*60)
    print("Test Summary by Category")
    print("="*60)
    
    categories = [
        ("Temporal", "tests/unit/test_temporal*.py"),
        ("Search", "tests/unit/test_*search*.py"),
        ("Query", "tests/unit/test_query*.py"),
        ("Graph", "tests/unit/test_graph*.py"),
        ("Integration", "tests/integration/*.py")
    ]
    
    for category_name, pattern in categories:
        cmd = ["python", "-m", "pytest", pattern, "--tb=no", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse pytest output for pass/fail counts
        output_lines = result.stdout.strip().split('\n')
        if output_lines:
            summary = output_lines[-1]
            print(f"{category_name:15} | {summary}")
    
    # Final report
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed successfully!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("❌ Some tests failed. Please review the output above.")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()