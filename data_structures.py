"""
Search Engine Data Structures - Phase 2 Implementation
======================================================

This module contains the original Phase 2 implementation of search engine
data structures including inverted index, trie, priority queue, and graph.

Author: Search Engine Optimization Project
Course: MSCS-532 Data Structures and Algorithm Analysis
"""

import heapq
import math
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any


class InvertedIndex:
    """Inverted index for efficient document-term mapping."""
    
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list))
        self.doc_lengths = {}
        self.total_docs = 0
        self.doc_frequencies = defaultdict(int)
    
    def add_document(self, doc_id: str, terms: List[str]) -> None:
        """Add a document to the inverted index."""
        self.doc_lengths[doc_id] = len(terms)
        self.total_docs += 1
        unique_terms = set()
        
        for position, term in enumerate(terms):
            self.index[term][doc_id].append(position)
            if term not in unique_terms:
                self.doc_frequencies[term] += 1
                unique_terms.add(term)
    
    def get_documents_with_term(self, term: str) -> List[str]:
        """Get all documents containing a specific term."""
        return list(self.index[term].keys())
    
    def calculate_tf_idf(self, term: str, doc_id: str) -> float:
        """Calculate TF-IDF score for a term in a document."""
        if term not in self.index or doc_id not in self.index[term]:
            return 0.0
        tf = len(self.index[term][doc_id]) / self.doc_lengths[doc_id]
        idf = math.log(self.total_docs / self.doc_frequencies[term])
        return tf * idf


class TrieNode:
    """Node in the trie data structure."""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0
        self.documents = set()


class Trie:
    """Trie data structure for prefix-based search."""
    
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str, doc_id: str) -> None:
        """Insert a word into the trie."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word_count += 1
        node.documents.add(doc_id)
        self.total_words += 1
    
    def get_prefix_matches(self, prefix: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Get words matching the given prefix."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        self._collect_words(node, prefix, results)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _collect_words(self, node: TrieNode, current_word: str, results: List[Tuple[str, int]]) -> None:
        """Collect all words from a node."""
        if node.is_end_of_word:
            results.append((current_word, node.word_count))
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, results)


class PriorityQueue:
    """Priority queue for ranking search results."""
    
    def __init__(self):
        self.heap = []
        self.entry_count = 0
    
    def push(self, doc_id: str, score: float) -> None:
        """Add a document with its score to the queue."""
        heapq.heappush(self.heap, (-score, self.entry_count, doc_id))
        self.entry_count += 1
    
    def pop(self) -> Optional[Tuple[str, float]]:
        """Remove and return the highest priority item."""
        if not self.heap:
            return None
        neg_score, _, doc_id = heapq.heappop(self.heap)
        return doc_id, -neg_score
    
    def get_top_k(self, k: int) -> List[Tuple[str, float]]:
        """Get top k results without modifying the heap."""
        if k <= 0:
            return []
        temp_heap = self.heap.copy()
        results = []
        for _ in range(min(k, len(temp_heap))):
            if not temp_heap:
                break
            neg_score, _, doc_id = heapq.heappop(temp_heap)
            results.append((doc_id, -neg_score))
        return results


class Graph:
    """Graph data structure for PageRank calculation."""
    
    def __init__(self):
        self.outgoing_links = defaultdict(set)
        self.incoming_links = defaultdict(set)
        self.page_ranks = {}
        self.nodes = set()
    
    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)
        if node not in self.page_ranks:
            self.page_ranks[node] = 0.0
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge between two nodes."""
        self.add_node(from_node)
        self.add_node(to_node)
        self.outgoing_links[from_node].add(to_node)
        self.incoming_links[to_node].add(from_node)
    
    def calculate_pagerank(self, damping_factor: float = 0.85, 
                          max_iterations: int = 100, 
                          tolerance: float = 1e-6) -> Dict[str, float]:
        """Calculate PageRank scores using iterative algorithm."""
        if not self.nodes:
            return {}
        
        n = len(self.nodes)
        initial_rank = 1.0 / n
        
        for node in self.nodes:
            self.page_ranks[node] = initial_rank
        
        for iteration in range(max_iterations):
            new_ranks = {}
            for node in self.nodes:
                rank_sum = 0.0
                for incoming_node in self.incoming_links[node]:
                    outgoing_count = len(self.outgoing_links[incoming_node])
                    if outgoing_count > 0:
                        rank_sum += self.page_ranks[incoming_node] / outgoing_count
                
                new_ranks[node] = (1 - damping_factor) / n + damping_factor * rank_sum
            
            max_change = max(abs(new_ranks[node] - self.page_ranks[node]) 
                           for node in self.nodes)
            self.page_ranks.update(new_ranks)
            
            if max_change < tolerance:
                break
        
        return self.page_ranks.copy()
    
    def get_page_rank(self, node: str) -> float:
        """Get PageRank score for a specific node."""
        return self.page_ranks.get(node, 0.0)


class SearchEngineDataStructures:
    """Main search engine class integrating all data structures."""
    
    def __init__(self):
        self.inverted_index = InvertedIndex()
        self.trie = Trie()
        self.priority_queue = PriorityQueue()
        self.page_graph = Graph()
        self.documents = {}
    
    def add_document(self, doc_id: str, content: str, links: List[str] = None) -> None:
        """Add a document to the search engine."""
        self.documents[doc_id] = content
        terms = self._tokenize(content)
        
        self.inverted_index.add_document(doc_id, terms)
        for term in set(terms):
            self.trie.insert(term, doc_id)
        
        if links:
            for linked_doc in links:
                self.page_graph.add_edge(doc_id, linked_doc)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            docs_with_term = self.inverted_index.get_documents_with_term(term)
            for doc_id in docs_with_term:
                tf_idf_score = self.inverted_index.calculate_tf_idf(term, doc_id)
                pagerank_score = self.page_graph.get_page_rank(doc_id)
                combined_score = tf_idf_score + 0.1 * pagerank_score
                doc_scores[doc_id] += combined_score
        
        for doc_id, score in doc_scores.items():
            self.priority_queue.push(doc_id, score)
        
        return self.priority_queue.get_top_k(top_k)
    
    def autocomplete(self, prefix: str, limit: int = 5) -> List[str]:
        """Get autocomplete suggestions for a prefix."""
        matches = self.trie.get_prefix_matches(prefix, limit)
        return [word for word, _ in matches]
    
    def calculate_page_ranks(self) -> Dict[str, float]:
        """Calculate PageRank scores for all documents."""
        return self.page_graph.calculate_pagerank()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function."""
        import re
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        return {
            'total_documents': len(self.documents),
            'total_terms_in_index': len(self.inverted_index.index),
            'total_words_in_trie': self.trie.total_words,
            'total_pages_in_graph': len(self.page_graph.nodes),
            'average_document_length': sum(self.inverted_index.doc_lengths.values()) / len(self.inverted_index.doc_lengths) if self.inverted_index.doc_lengths else 0
        }