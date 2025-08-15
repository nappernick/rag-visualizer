#!/usr/bin/env python3
"""
Comprehensive UI functionality test - tests the actual user workflow
"""
import requests
import json
import time
import os
from pathlib import Path

# Test configuration
BACKEND_URL = "http://localhost:8642"
TEST_DOCUMENT_CONTENT = """
Artificial Intelligence in Medical Diagnosis

AI systems are revolutionizing healthcare by providing accurate medical diagnoses. 
Machine learning algorithms can analyze medical images to detect cancer, fractures, 
and other conditions with high precision. Deep learning models trained on vast 
datasets of medical scans can identify patterns invisible to human radiologists.

Key applications include:
- Radiology image analysis for tumor detection
- Pathology slide examination for cancer screening  
- Retinal imaging for diabetic retinopathy diagnosis
- ECG analysis for cardiac abnormalities

These AI diagnostic tools reduce human error, speed up diagnosis time, and improve 
patient outcomes through early detection of diseases.
"""

class UIFunctionalityTester:
    def __init__(self):
        self.session = requests.Session()
        self.document_id = None
        self.test_results = []
    
    def log_result(self, test_name, success, details=""):
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_backend_health(self):
        """Test if backend is responding"""
        try:
            response = self.session.get(f"{BACKEND_URL}/health", timeout=5)
            success = response.status_code == 200
            self.log_result("Backend Health Check", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_result("Backend Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_clear_existing_data(self):
        """Clear all existing data to start fresh"""
        try:
            response = self.session.delete(f"{BACKEND_URL}/api/clear-all", timeout=10)
            success = response.status_code == 200
            self.log_result("Clear Existing Data", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_result("Clear Existing Data", False, f"Error: {str(e)}")
            return False
    
    def test_document_upload(self):
        """Test document upload through API (simulates UI upload)"""
        try:
            # Create test file
            test_file_path = "/tmp/test_ai_medical.txt"
            with open(test_file_path, 'w') as f:
                f.write(TEST_DOCUMENT_CONTENT)
            
            # Upload file
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_ai_medical.txt', f, 'text/plain')}
                data = {'filename': 'test_ai_medical.txt'}
                response = self.session.post(f"{BACKEND_URL}/api/documents/upload", 
                                           files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.document_id = result.get('id')
                self.log_result("Document Upload", True, f"Document ID: {self.document_id}")
                return True
            else:
                self.log_result("Document Upload", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Document Upload", False, f"Error: {str(e)}")
            return False
    
    def test_document_list(self):
        """Test that uploaded document appears in document list"""
        try:
            response = self.session.get(f"{BACKEND_URL}/api/documents", timeout=10)
            if response.status_code == 200:
                documents = response.json()
                found = any(doc.get('id') == self.document_id for doc in documents)
                self.log_result("Document List", found, f"Found {len(documents)} documents, target found: {found}")
                return found
            else:
                self.log_result("Document List", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Document List", False, f"Error: {str(e)}")
            return False
    
    def test_chunking_process(self):
        """Test that document gets chunked properly"""
        if not self.document_id:
            self.log_result("Chunking Process", False, "No document ID available")
            return False
        
        try:
            # Trigger chunking with required content parameter
            response = self.session.post(f"{BACKEND_URL}/api/chunking", 
                                       json={
                                           "document_id": self.document_id,
                                           "content": TEST_DOCUMENT_CONTENT,
                                           "strategy": "semantic"
                                       }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                chunk_count = result.get('chunk_count', 0)
                success = chunk_count > 0
                self.log_result("Chunking Process", success, f"Created {chunk_count} chunks")
                return success
            else:
                self.log_result("Chunking Process", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Chunking Process", False, f"Error: {str(e)}")
            return False
    
    def test_chunks_retrieval(self):
        """Test that chunks can be retrieved (All Documents functionality)"""
        if not self.document_id:
            self.log_result("Chunks Retrieval", False, "No document ID available")
            return False
        
        try:
            response = self.session.get(f"{BACKEND_URL}/api/documents/{self.document_id}/chunks", timeout=10)
            if response.status_code == 200:
                chunks = response.json()
                success = len(chunks) > 0
                self.log_result("Chunks Retrieval", success, f"Retrieved {len(chunks)} chunks")
                return success
            else:
                self.log_result("Chunks Retrieval", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Chunks Retrieval", False, f"Error: {str(e)}")
            return False
    
    def test_graph_extraction(self):
        """Test graph extraction functionality"""
        if not self.document_id:
            self.log_result("Graph Extraction", False, "No document ID available")
            return False
        
        try:
            # First get chunks for the document
            chunks_response = self.session.get(f"{BACKEND_URL}/api/documents/{self.document_id}/chunks")
            chunks = chunks_response.json() if chunks_response.status_code == 200 else []
            
            # Trigger graph extraction with chunks
            response = self.session.post(f"{BACKEND_URL}/api/graph/extract", 
                                       json={
                                           "document_id": self.document_id,
                                           "chunks": chunks,
                                           "extract_entities": True,
                                           "extract_relationships": True
                                       }, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                entity_count = result.get('entity_count', 0)
                relationship_count = result.get('relationship_count', 0)
                success = entity_count > 0
                self.log_result("Graph Extraction", success, 
                              f"Entities: {entity_count}, Relationships: {relationship_count}")
                return success
            else:
                self.log_result("Graph Extraction", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Graph Extraction", False, f"Error: {str(e)}")
            return False
    
    def test_search_functionality(self):
        """Test search with multiple relevant queries"""
        search_queries = [
            "artificial intelligence medical diagnosis",
            "machine learning healthcare",
            "AI radiology imaging",
            "deep learning cancer detection",
            "medical image analysis"
        ]
        
        successful_searches = 0
        
        for query in search_queries:
            try:
                response = self.session.post(f"{BACKEND_URL}/api/query", 
                                           json={"query": query, "k": 5}, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    results = result.get('results', [])
                    # Check if we got actual results (not just "no results" message)
                    real_results = [r for r in results if r.get('score', 0) > 0]
                    
                    if real_results:
                        successful_searches += 1
                        self.log_result(f"Search: '{query}'", True, 
                                      f"Found {len(real_results)} results, best score: {max(r.get('score', 0) for r in real_results):.3f}")
                    else:
                        self.log_result(f"Search: '{query}'", False, "No relevant results found")
                else:
                    self.log_result(f"Search: '{query}'", False, f"Status: {response.status_code}")
            except Exception as e:
                self.log_result(f"Search: '{query}'", False, f"Error: {str(e)}")
        
        overall_success = successful_searches >= len(search_queries) // 2
        self.log_result("Overall Search Functionality", overall_success, 
                       f"{successful_searches}/{len(search_queries)} queries successful")
        return overall_success
    
    def test_state_persistence(self):
        """Test that data persists between requests (simulates browser reload)"""
        try:
            # Wait a moment then re-check document list
            time.sleep(2)
            response = self.session.get(f"{BACKEND_URL}/api/documents", timeout=10)
            
            if response.status_code == 200:
                documents = response.json()
                found = any(doc.get('id') == self.document_id for doc in documents)
                self.log_result("State Persistence", found, 
                              f"Document still exists after delay: {found}")
                return found
            else:
                self.log_result("State Persistence", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("State Persistence", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run complete UI functionality test suite"""
        print("🚀 Starting Comprehensive UI Functionality Tests")
        print("=" * 60)
        
        tests = [
            self.test_backend_health,
            self.test_clear_existing_data,
            self.test_document_upload,
            self.test_document_list,
            self.test_chunking_process,
            self.test_chunks_retrieval,
            self.test_graph_extraction,
            self.test_search_functionality,
            self.test_state_persistence
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                print()  # Empty line between tests
            except Exception as e:
                print(f"💥 Test {test.__name__} crashed: {str(e)}")
                print()
        
        print("=" * 60)
        print(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - UI functionality is working correctly!")
        elif passed >= total * 0.7:
            print("⚠️  MOST TESTS PASSED - Minor issues detected")
        else:
            print("🚨 MULTIPLE FAILURES - Major UI issues detected")
        
        # Print detailed summary
        print("\n📝 Detailed Results:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['test']}: {result['details']}")
        
        return passed / total

if __name__ == "__main__":
    tester = UIFunctionalityTester()
    success_rate = tester.run_all_tests()
    exit(0 if success_rate == 1.0 else 1)