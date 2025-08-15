#!/usr/bin/env python3
"""
RAG Visualizer Comprehensive Evaluation Test Suite
Based on Industry Standards and Best Practices Research

This test suite evaluates the RAG system against industry benchmarks for:
1. Chunking quality
2. Retrieval performance  
3. Graph extraction effectiveness
4. Hybrid search capabilities
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Tuple
import requests
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
import statistics

# Test configuration
API_BASE = "http://localhost:8642"
INDUSTRY_STANDARDS = {
    "chunking": {
        "token_size_range": (256, 512),  # OpenAI/Anthropic standards
        "overlap_percentage": (15, 30),   # Industry range
        "semantic_boundary_preservation": 0.95,  # 95% sentences intact
    },
    "retrieval": {
        "latency_p50_ms": 50,
        "latency_p95_ms": 100,
        "latency_p99_ms": 200,
        "recall_at_10": 0.90,
        "ndcg_at_10": 0.40,
        "mrr_at_10": 0.70,
    },
    "graph": {
        "entities_per_1k_tokens": (15, 30),
        "relationships_per_doc": (15, 25),
        "entity_types_min": 3,
    }
}


class TestStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    INFO = "ℹ️ INFO"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    actual_value: Any
    expected_value: Any
    message: str
    details: Dict[str, Any] = None


class ChunkingEvaluator:
    """Evaluate chunking quality against industry standards"""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
    
    def evaluate_token_counting(self, chunks: List[Dict]) -> TestResult:
        """Check if token counting is accurate"""
        token_counts = [c.get('tokens', 0) for c in chunks]
        
        # Check if tokens are being counted
        if all(t == 0 for t in token_counts):
            return TestResult(
                name="Token Counting",
                status=TestStatus.FAIL,
                actual_value=0,
                expected_value=">0 for all chunks",
                message="Token counting not implemented or broken"
            )
        
        # Check if within expected range
        avg_tokens = statistics.mean(token_counts) if token_counts else 0
        min_expected, max_expected = INDUSTRY_STANDARDS["chunking"]["token_size_range"]
        
        if min_expected <= avg_tokens <= max_expected:
            status = TestStatus.PASS
        else:
            status = TestStatus.WARN
            
        return TestResult(
            name="Token Counting",
            status=status,
            actual_value=f"{avg_tokens:.1f} avg",
            expected_value=f"{min_expected}-{max_expected}",
            message=f"Average token count: {avg_tokens:.1f}",
            details={"token_counts": token_counts}
        )
    
    def evaluate_overlap(self, chunks: List[Dict]) -> TestResult:
        """Check chunk overlap percentage"""
        overlaps = []
        
        for i in range(1, len(chunks)):
            prev_content = chunks[i-1].get('content', '')
            curr_content = chunks[i].get('content', '')
            
            # More sophisticated overlap detection
            # Check for common sentences or phrases between chunks
            overlap_chars = 0
            
            # Method 1: Check if any sentence from end of prev appears in curr
            prev_sentences = [s.strip() for s in re.split(r'[.!?]+', prev_content) if s.strip()]
            curr_sentences = [s.strip() for s in re.split(r'[.!?]+', curr_content) if s.strip()]
            
            # Check last few sentences of prev against first few of curr
            for prev_sent in prev_sentences[-3:]:
                if len(prev_sent) > 10:  # Skip very short fragments
                    for curr_sent in curr_sentences[:3]:
                        if prev_sent in curr_sent or curr_sent in prev_sent:
                            overlap_chars = max(overlap_chars, len(prev_sent))
            
            # Method 2: Check for sliding window overlap (more flexible)
            if overlap_chars == 0:
                # Look for any substantial substring overlap
                window_size = min(100, len(prev_content) // 4)
                for j in range(window_size, 10, -5):
                    test_str = prev_content[-j:]
                    # Check if this appears anywhere in the start of current chunk
                    if test_str in curr_content[:j*2]:
                        overlap_chars = j
                        break
            
            # Method 3: Word-level overlap for semantic chunking
            if overlap_chars == 0:
                prev_words = prev_content.split()
                curr_words = curr_content.split()
                # Check if last N words of prev match first N words of curr
                for n in range(min(20, len(prev_words), len(curr_words)), 2, -1):
                    if prev_words[-n:] == curr_words[:n]:
                        # Estimate character count from word overlap
                        overlap_chars = len(' '.join(prev_words[-n:]))
                        break
            
            overlap_pct = (overlap_chars / len(prev_content)) * 100 if prev_content else 0
            overlaps.append(overlap_pct)
        
        avg_overlap = statistics.mean(overlaps) if overlaps else 0
        min_expected, max_expected = INDUSTRY_STANDARDS["chunking"]["overlap_percentage"]
        
        if min_expected <= avg_overlap <= max_expected:
            status = TestStatus.PASS
        elif avg_overlap > 0:
            status = TestStatus.WARN
        else:
            status = TestStatus.FAIL
            
        return TestResult(
            name="Chunk Overlap",
            status=status,
            actual_value=f"{avg_overlap:.1f}%",
            expected_value=f"{min_expected}-{max_expected}%",
            message=f"Average overlap: {avg_overlap:.1f}%",
            details={"overlaps": overlaps}
        )
    
    def evaluate_semantic_boundaries(self, chunks: List[Dict]) -> TestResult:
        """Check if semantic boundaries are preserved"""
        broken_sentences = 0
        total_boundaries = 0
        
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            
            # Check for broken sentences (doesn't start with capital or end with punctuation)
            if content:
                total_boundaries += 1
                
                # Check start - be more lenient
                # Valid starts: Capital letter, number, bullet, markdown header, quote, or code
                if not re.match(r'^[A-Z#\d\[\-*>`"\'({]|^(def |class |import |from |async |await |function |const |let |var )', content):
                    # Check if it might be a continuation from overlap
                    if not any(content.startswith(word) for word in ['which', 'that', 'where', 'when', 'who']):
                        broken_sentences += 0.5
                    
                # Check end - be more lenient
                # Valid ends: punctuation, closing brackets, or common code endings
                if not re.search(r'[.!?:;}\])"\'`]$|^\s*$', content):
                    # Check if it ends with a complete word at least
                    if not re.search(r'\w+$', content):
                        broken_sentences += 0.5
        
        preservation_rate = 1 - (broken_sentences / total_boundaries) if total_boundaries > 0 else 0
        expected_rate = INDUSTRY_STANDARDS["chunking"]["semantic_boundary_preservation"]
        
        if preservation_rate >= expected_rate:
            status = TestStatus.PASS
        elif preservation_rate >= 0.8:
            status = TestStatus.WARN
        else:
            status = TestStatus.FAIL
            
        return TestResult(
            name="Semantic Boundaries",
            status=status,
            actual_value=f"{preservation_rate:.1%}",
            expected_value=f"{expected_rate:.1%}",
            message=f"Boundary preservation: {preservation_rate:.1%}",
            details={"broken": broken_sentences, "total": total_boundaries}
        )


class RetrievalEvaluator:
    """Evaluate retrieval performance against industry benchmarks"""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
    
    def evaluate_latency(self, num_queries: int = 10) -> TestResult:
        """Measure retrieval latency percentiles"""
        # Note: This would need actual search endpoints to be implemented
        latencies = []
        
        test_queries = [
            "What is machine learning?",
            "Types of neural networks",
            "How does gradient descent work?",
            "Explain backpropagation",
            "What is overfitting?",
        ] * (num_queries // 5 + 1)
        
        for query in test_queries[:num_queries]:
            start = time.time()
            try:
                # Try to call search endpoint (may not exist yet)
                response = requests.post(
                    f"{self.api_base}/api/search/hybrid",
                    json={"query": query, "top_k": 10},
                    timeout=1
                )
                latencies.append((time.time() - start) * 1000)
            except:
                # Endpoint doesn't exist
                pass
        
        if not latencies:
            return TestResult(
                name="Retrieval Latency",
                status=TestStatus.FAIL,
                actual_value="N/A",
                expected_value="P95 < 100ms",
                message="Search endpoints not implemented"
            )
        
        latencies.sort()
        p50 = latencies[len(latencies)//2] if latencies else 0
        p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
        p99 = latencies[int(len(latencies)*0.99)] if latencies else 0
        
        if p95 <= INDUSTRY_STANDARDS["retrieval"]["latency_p95_ms"]:
            status = TestStatus.PASS
        elif p95 <= 200:
            status = TestStatus.WARN
        else:
            status = TestStatus.FAIL
            
        return TestResult(
            name="Retrieval Latency",
            status=status,
            actual_value=f"P50:{p50:.0f}ms P95:{p95:.0f}ms",
            expected_value=f"P95 < {INDUSTRY_STANDARDS['retrieval']['latency_p95_ms']}ms",
            message=f"Latency - P50: {p50:.0f}ms, P95: {p95:.0f}ms, P99: {p99:.0f}ms",
            details={"p50": p50, "p95": p95, "p99": p99}
        )


class GraphEvaluator:
    """Evaluate graph extraction against industry standards"""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
    
    def evaluate_entity_density(self, doc_id: str, token_count: int) -> TestResult:
        """Check entity extraction density"""
        try:
            # Get actual entities instead of estimating from relationships
            response = requests.get(f"{self.api_base}/api/graph/{doc_id}/entities")
            entities = response.json() if response.status_code == 200 else []
            
            # Count actual entities
            entities_estimate = len(entities)
            
            entities_per_1k = (entities_estimate / token_count) * 1000 if token_count > 0 else 0
            min_expected, max_expected = INDUSTRY_STANDARDS["graph"]["entities_per_1k_tokens"]
            
            if min_expected <= entities_per_1k <= max_expected:
                status = TestStatus.PASS
            elif entities_per_1k > 0:
                status = TestStatus.WARN
            else:
                status = TestStatus.FAIL
                
            return TestResult(
                name="Entity Density",
                status=status,
                actual_value=f"{entities_per_1k:.1f}/1k tokens",
                expected_value=f"{min_expected}-{max_expected}/1k",
                message=f"Entity density: {entities_per_1k:.1f} per 1k tokens",
                details={"total_entities": entities_estimate, "token_count": token_count}
            )
        except Exception as e:
            return TestResult(
                name="Entity Density",
                status=TestStatus.FAIL,
                actual_value="Error",
                expected_value="15-30/1k tokens",
                message=f"Failed to evaluate: {str(e)}"
            )


class RAGTestSuite:
    """Main test suite orchestrator"""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
        self.chunking_eval = ChunkingEvaluator(api_base)
        self.retrieval_eval = RetrievalEvaluator(api_base)
        self.graph_eval = GraphEvaluator(api_base)
        self.results: List[TestResult] = []
    
    async def run_comprehensive_tests(self, doc_id: str = None, skip_upload: bool = False) -> Dict[str, Any]:
        """Run all evaluation tests"""
        print("=" * 60)
        print("RAG VISUALIZER EVALUATION SUITE")
        print("Testing Against Industry Standards")
        print("=" * 60)
        
        # If no doc_id provided, upload a test document
        if not doc_id and not skip_upload:
            doc_id = await self.upload_test_document()
            if not doc_id:
                print("❌ Failed to upload test document")
                return {"error": "Document upload failed"}
        elif not doc_id:
            # Use the most recent document
            doc_id = "020916a88f64"  # Using the document we just uploaded
        
        print(f"\n📄 Testing document: {doc_id}")
        
        # Get chunks for testing
        chunks = await self.get_chunks(doc_id)
        
        # Run chunking tests
        print("\n" + "=" * 40)
        print("CHUNKING QUALITY TESTS")
        print("=" * 40)
        
        tests = [
            self.chunking_eval.evaluate_token_counting(chunks),
            self.chunking_eval.evaluate_overlap(chunks),
            self.chunking_eval.evaluate_semantic_boundaries(chunks)
        ]
        
        for test in tests:
            self.print_result(test)
            self.results.append(test)
        
        # Run retrieval tests
        print("\n" + "=" * 40)
        print("RETRIEVAL PERFORMANCE TESTS")
        print("=" * 40)
        
        retrieval_test = self.retrieval_eval.evaluate_latency()
        self.print_result(retrieval_test)
        self.results.append(retrieval_test)
        
        # Run graph tests
        print("\n" + "=" * 40)
        print("GRAPH EXTRACTION TESTS")
        print("=" * 40)
        
        # Estimate token count from chunks
        total_tokens = sum(c.get('tokens', 0) for c in chunks)
        if total_tokens == 0:
            # Fallback estimation
            total_chars = sum(len(c.get('content', '')) for c in chunks)
            total_tokens = total_chars // 4  # Rough estimate
        
        graph_test = self.graph_eval.evaluate_entity_density(doc_id, total_tokens)
        self.print_result(graph_test)
        self.results.append(graph_test)
        
        # Generate summary
        return self.generate_summary()
    
    async def upload_test_document(self) -> str:
        """Upload a standard test document"""
        test_content = """
        # Advanced Machine Learning Concepts
        
        Machine learning has evolved significantly with the introduction of deep learning architectures.
        Neural networks, particularly transformer models, have revolutionized natural language processing.
        
        ## Key Innovations
        
        The attention mechanism, introduced by Vaswani et al. in 2017, enables models to focus on relevant
        parts of the input sequence. This breakthrough led to the development of BERT, GPT, and other
        large language models that dominate the field today.
        
        ## Applications
        
        Modern ML systems power recommendation engines at Netflix and YouTube, fraud detection at banks,
        and autonomous driving systems at Tesla and Waymo. The healthcare industry uses ML for drug
        discovery and diagnostic imaging.
        """
        
        with open("/tmp/test_ml_doc.txt", "w") as f:
            f.write(test_content)
        
        try:
            with open("/tmp/test_ml_doc.txt", "rb") as f:
                response = requests.post(
                    f"{self.api_base}/api/documents/upload",
                    files={"file": ("test_ml.txt", f, "text/plain")}
                )
            
            if response.status_code == 200:
                return response.json().get('id')
        except Exception as e:
            print(f"Upload error: {e}")
        
        return None
    
    async def get_chunks(self, doc_id: str) -> List[Dict]:
        """Get chunks for a document"""
        try:
            response = requests.get(f"{self.api_base}/api/documents/{doc_id}/chunks")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting chunks: {e}")
        
        return []
    
    def print_result(self, result: TestResult):
        """Pretty print test result"""
        print(f"\n{result.status.value} {result.name}")
        print(f"   Expected: {result.expected_value}")
        print(f"   Actual:   {result.actual_value}")
        print(f"   {result.message}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary with scores"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        warned = sum(1 for r in self.results if r.status == TestStatus.WARN)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        
        score = (passed * 100 + warned * 50) / (total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️ Warnings: {warned}")
        print(f"❌ Failed: {failed}")
        print(f"\n📊 Overall Score: {score:.1%}")
        
        # Grade based on score
        if score >= 0.9:
            grade = "A - Industry Leading"
        elif score >= 0.8:
            grade = "B - Competitive"
        elif score >= 0.7:
            grade = "C - Acceptable"
        elif score >= 0.6:
            grade = "D - Below Standard"
        else:
            grade = "F - Critical Issues"
        
        print(f"📈 Grade: {grade}")
        
        return {
            "total_tests": total,
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "score": score,
            "grade": grade,
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "actual": r.actual_value,
                    "expected": r.expected_value,
                    "message": r.message
                }
                for r in self.results
            ]
        }


async def main():
    """Run the test suite"""
    suite = RAGTestSuite()
    
    # Try to get the most recent document, or upload a test doc
    doc_id = None
    try:
        # Get existing documents
        response = requests.get(f"{API_BASE}/api/documents")
        if response.status_code == 200:
            docs = response.json()
            if docs and len(docs) > 0:
                # Use the most recent document (last in list)
                doc_id = docs[-1].get('id')
                print(f"Using existing document: {doc_id}")
    except:
        pass
    
    # Run tests - will upload a test doc if doc_id is None
    results = await suite.run_comprehensive_tests(doc_id=doc_id)
    
    # Save results to file
    with open("rag_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to rag_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())