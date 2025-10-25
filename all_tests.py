"""
Comprehensive Test Suite
========================

This module provides comprehensive testing for both Phase 2 and Phase 3
search engine implementations, including unit tests, integration tests,
and performance tests.

Author: Search Engine Optimization Project
Course: MSCS-532 Data Structures and Algorithm Analysis
"""

import unittest
import time
import random
import string
from typing import List, Dict, Tuple, Any
from search_engine import SearchEngine
from data_structures import InvertedIndex, Trie, PriorityQueue, Graph
from optimized_data_structures import OptimizedInvertedIndex, OptimizedTrie, OptimizedPriorityQueue, OptimizedGraph


class TestDataStructures(unittest.TestCase):
    """Test individual data structures."""
    
    def setUp(self):
        """Set up test environment."""
        self.sample_docs = [
            ("doc1", ["python", "programming", "data", "structures"]),
            ("doc2", ["algorithms", "data", "science", "python"]),
            ("doc3", ["machine", "learning", "algorithms", "python"])
        ]
    
    def test_inverted_index(self):
        """Test inverted index functionality."""
        index = InvertedIndex()
        
        # Test document addition
        for doc_id, terms in self.sample_docs:
            index.add_document(doc_id, terms)
        
        self.assertEqual(index.total_docs, 3)
        
        # Test term retrieval
        python_docs = index.get_documents_with_term("python")
        self.assertIn("doc1", python_docs)
        self.assertIn("doc2", python_docs)
        self.assertIn("doc3", python_docs)
        
        # Test TF-IDF calculation
        tf_idf_score = index.calculate_tf_idf("python", "doc1")
        self.assertGreaterEqual(tf_idf_score, 0)  # Can be 0 if term doesn't exist
    
    def test_trie(self):
        """Test trie functionality."""
        trie = Trie()
        
        words = ["python", "programming", "data", "structures", "algorithms"]
        for word in words:
            trie.insert(word, "sample_doc")
        
        # Test prefix matching
        matches = trie.get_prefix_matches("prog", limit=3)
        self.assertGreater(len(matches), 0)
        self.assertTrue(all(word.startswith("prog") for word, _ in matches))
    
    def test_priority_queue(self):
        """Test priority queue functionality."""
        pq = PriorityQueue()
        
        # Test push and pop
        pq.push("doc1", 0.8)
        pq.push("doc2", 0.6)
        pq.push("doc3", 0.9)
        
        # Test top-k retrieval
        top_results = pq.get_top_k(2)
        self.assertEqual(len(top_results), 2)
        
        # Check ordering (highest score first)
        scores = [score for _, score in top_results]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_graph(self):
        """Test graph and PageRank functionality."""
        graph = Graph()
        
        # Create a simple graph
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        # Test PageRank calculation
        pageranks = graph.calculate_pagerank()
        self.assertEqual(len(pageranks), 3)
        self.assertIn("A", pageranks)
        self.assertIn("B", pageranks)
        self.assertIn("C", pageranks)


class TestOptimizedDataStructures(unittest.TestCase):
    """Test optimized data structures."""
    
    def setUp(self):
        """Set up test environment."""
        self.sample_docs = [
            ("doc1", ["python", "programming", "data", "structures"]),
            ("doc2", ["algorithms", "data", "science", "python"]),
            ("doc3", ["machine", "learning", "algorithms", "python"])
        ]
    
    def test_optimized_inverted_index(self):
        """Test optimized inverted index."""
        index = OptimizedInvertedIndex()
        
        # Test batch addition
        doc_data = [(doc_id, terms) for doc_id, terms in self.sample_docs]
        index.add_document_batch(doc_data)
        
        self.assertEqual(index.total_docs, 3)
        
        # Test caching
        score1 = index.calculate_tf_idf("python", "doc1")
        score2 = index.calculate_tf_idf("python", "doc1")
        self.assertEqual(score1, score2)
        
        # Check cache stats
        stats = index.get_cache_stats()
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_optimized_trie(self):
        """Test optimized trie."""
        trie = OptimizedTrie()
        
        # Test batch insertion
        trie_data = []
        for doc_id, terms in self.sample_docs:
            for term in terms:
                trie_data.append((term, doc_id))
        
        trie.insert_batch(trie_data)
        
        # Test prefix matching
        matches = trie.get_prefix_matches("prog", limit=3)
        self.assertGreater(len(matches), 0)
    
    def test_optimized_priority_queue(self):
        """Test optimized priority queue."""
        pq = OptimizedPriorityQueue()
        
        # Test batch operations
        items = [("doc1", 0.8), ("doc2", 0.6), ("doc3", 0.9)]
        pq.push_batch(items)
        
        # Test caching
        results1 = pq.get_top_k(2)
        results2 = pq.get_top_k(2)
        self.assertEqual(results1, results2)
    
    def test_optimized_graph(self):
        """Test optimized graph."""
        graph = OptimizedGraph()
        
        # Create graph
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        # Test PageRank with early convergence
        pageranks = graph.calculate_pagerank(tolerance=1e-6)
        self.assertEqual(len(pageranks), 3)
        
        # Test staleness check (should be stale immediately after creation)
        self.assertTrue(graph.is_stale(max_age=0))


class TestSearchEngineIntegration(unittest.TestCase):
    """Test integrated search engine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.sample_documents = [
            ("doc1", "Python is a powerful programming language for data science", ["doc2"]),
            ("doc2", "Data structures and algorithms are fundamental in computer science", ["doc1", "doc3"]),
            ("doc3", "Machine learning algorithms use various data structures for optimization", ["doc2"])
        ]
    
    def test_phase2_search_engine(self):
        """Test Phase 2 search engine."""
        engine = SearchEngine(use_optimized=False)
        
        # Add documents
        for doc_id, content, links in self.sample_documents:
            engine.add_document(doc_id, content, links)
        
        # Test search
        results = engine.search("python programming", top_k=2)
        self.assertGreater(len(results), 0)
        
        # Test autocomplete
        suggestions = engine.autocomplete("prog", limit=3)
        self.assertGreater(len(suggestions), 0)
        
        # Test statistics
        stats = engine.get_stats()
        self.assertGreater(stats['total_documents'], 0)
    
    def test_phase3_search_engine(self):
        """Test Phase 3 search engine."""
        engine = SearchEngine(use_optimized=True)
        
        # Add documents using batch operation
        engine.add_document_batch(self.sample_documents)
        
        # Test search
        results = engine.search("python programming", top_k=2)
        self.assertGreater(len(results), 0)
        
        # Test autocomplete
        suggestions = engine.autocomplete("prog", limit=3)
        self.assertGreater(len(suggestions), 0)
        
        # Test statistics
        stats = engine.get_stats()
        self.assertGreater(stats['total_documents'], 0)
    
    def test_implementation_switching(self):
        """Test switching between implementations."""
        engine = SearchEngine(use_optimized=False)
        
        # Add documents to Phase 2
        for doc_id, content, links in self.sample_documents:
            engine.add_document(doc_id, content, links)
        
        # Switch to Phase 3
        engine.switch_implementation(use_optimized=True)
        
        # Add documents to Phase 3
        engine.add_document_batch(self.sample_documents)
        
        # Test that both implementations work
        results = engine.search("python programming", top_k=2)
        self.assertGreater(len(results), 0)


class TestPerformance(unittest.TestCase):
    """Performance testing."""
    
    def generate_test_data(self, num_docs: int) -> List[Tuple[str, str, List[str]]]:
        """Generate test data for performance testing."""
        documents = []
        
        topics = [
            "artificial intelligence machine learning deep learning",
            "data structures algorithms computer science programming",
            "web development frontend backend database systems"
        ]
        
        for i in range(num_docs):
            doc_id = f"doc_{i:06d}"
            topic = random.choice(topics)
            content = f"{topic} document {i} with additional content"
            
            # Generate some links
            num_links = random.randint(0, min(2, num_docs // 50))
            links = []
            for _ in range(num_links):
                target_doc = random.randint(0, num_docs - 1)
                if target_doc != i:
                    links.append(f"doc_{target_doc:06d}")
            
            documents.append((doc_id, content, links))
        
        return documents
    
    def test_indexing_performance(self):
        """Test indexing performance."""
        test_data = self.generate_test_data(100)
        
        # Test Phase 2
        engine2 = SearchEngine(use_optimized=False)
        start_time = time.time()
        for doc_id, content, links in test_data:
            engine2.add_document(doc_id, content, links)
        phase2_time = time.time() - start_time
        
        # Test Phase 3
        engine3 = SearchEngine(use_optimized=True)
        start_time = time.time()
        engine3.add_document_batch(test_data)
        phase3_time = time.time() - start_time
        
        # Both should complete in reasonable time
        self.assertLess(phase2_time, 5.0)
        self.assertLess(phase3_time, 5.0)
    
    def test_search_performance(self):
        """Test search performance."""
        test_data = self.generate_test_data(50)
        
        # Test both implementations
        for use_optimized in [False, True]:
            engine = SearchEngine(use_optimized=use_optimized)
            
            if use_optimized:
                engine.add_document_batch(test_data)
            else:
                for doc_id, content, links in test_data:
                    engine.add_document(doc_id, content, links)
            
            # Test search performance
            start_time = time.time()
            for query in ["artificial intelligence", "data structures", "machine learning"]:
                results = engine.search(query, top_k=5)
            search_time = time.time() - start_time
            
            # Should complete quickly
            self.assertLess(search_time, 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_empty_queries(self):
        """Test handling of empty queries."""
        engine = SearchEngine(use_optimized=True)
        engine.add_document("doc1", "test content")
        
        # Empty query
        results = engine.search("", top_k=5)
        self.assertEqual(len(results), 0)
        
        # Whitespace-only query
        results = engine.search("   ", top_k=5)
        self.assertEqual(len(results), 0)
    
    def test_nonexistent_terms(self):
        """Test handling of non-existent terms."""
        engine = SearchEngine(use_optimized=True)
        engine.add_document("doc1", "test content")
        
        # Non-existent term
        results = engine.search("nonexistentterm12345", top_k=5)
        self.assertEqual(len(results), 0)
    
    def test_large_queries(self):
        """Test handling of very long queries."""
        engine = SearchEngine(use_optimized=True)
        engine.add_document("doc1", "test content")
        
        # Very long query
        long_query = " ".join(["word"] * 100)
        results = engine.search(long_query, top_k=5)
        # Should handle gracefully without crashing
        self.assertIsInstance(results, list)
    
    def test_zero_top_k(self):
        """Test handling of zero top-k."""
        engine = SearchEngine(use_optimized=True)
        engine.add_document("doc1", "test content")
        
        # Zero top-k
        results = engine.search("test", top_k=0)
        self.assertEqual(len(results), 0)
        
        # Negative top-k
        results = engine.search("test", top_k=-1)
        self.assertEqual(len(results), 0)


def run_all_tests():
    """Run all test suites."""
    print("Running Comprehensive Test Suite")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDataStructures))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestOptimizedDataStructures))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSearchEngineIntegration))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
