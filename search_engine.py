"""
Search Engine Implementation with Command Line Interface
======================================================

This module provides a unified interface for both Phase 2 and Phase 3
search engine implementations, with an interactive command line interface
for adding documents and searching.

Author: Search Engine Optimization Project
Course: MSCS-532 Data Structures and Algorithm Analysis
"""

import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional
from data_structures import SearchEngineDataStructures
from optimized_data_structures import OptimizedSearchEngine


class SearchEngine:
    """Unified search engine interface supporting both implementations."""
    
    def __init__(self, use_optimized: bool = True):
        """
        Initialize search engine.
        
        Args:
            use_optimized: If True, use Phase 3 optimized implementation.
                          If False, use Phase 2 original implementation.
        """
        self.use_optimized = use_optimized
        
        if use_optimized:
            self.engine = OptimizedSearchEngine()
        else:
            self.engine = SearchEngineDataStructures()
    
    def add_document(self, doc_id: str, content: str, links: List[str] = None) -> None:
        """Add a document to the search engine."""
        self.engine.add_document(doc_id, content, links)
    
    def add_document_batch(self, documents: List[Tuple[str, str, List[str]]]) -> None:
        """Add multiple documents in batch (optimized implementation only)."""
        if self.use_optimized:
            self.engine.add_document_batch(documents)
        else:
            # Fallback to individual additions for Phase 2
            for doc_id, content, links in documents:
                self.engine.add_document(doc_id, content, links)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query."""
        if self.use_optimized:
            return self.engine.search(query, top_k)
        else:
            return self.engine.search(query, top_k)
    
    def autocomplete(self, prefix: str, limit: int = 5) -> List[str]:
        """Get autocomplete suggestions for a prefix."""
        return self.engine.autocomplete(prefix, limit)
    
    def calculate_page_ranks(self) -> Dict[str, float]:
        """Calculate PageRank scores for all documents."""
        return self.engine.calculate_page_ranks()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine."""
        if self.use_optimized:
            return self.engine.get_performance_stats()
        else:
            return self.engine.get_document_stats()
    
    def switch_implementation(self, use_optimized: bool) -> None:
        """Switch between Phase 2 and Phase 3 implementations."""
        self.use_optimized = use_optimized
        
        if use_optimized:
            self.engine = OptimizedSearchEngine()
        else:
            self.engine = SearchEngineDataStructures()
    
    def get_implementation_info(self) -> str:
        """Get information about current implementation."""
        if self.use_optimized:
            return "Phase 3 - Optimized Implementation (Caching, Batch Operations, Sparse Matrix)"
        else:
            return "Phase 2 - Original Implementation (Basic Data Structures)"


class SearchEngineCLI:
    """Command Line Interface for the Search Engine."""
    
    def __init__(self, use_optimized: bool = True):
        """Initialize CLI with specified implementation."""
        self.engine = SearchEngine(use_optimized=use_optimized)
        self.documents = {}  # Store document content for display
        self.doc_counter = 0  # Counter for auto-generating document IDs
    
    def print_welcome(self):
        """Print welcome message and help."""
        print("=" * 60)
        print("🔍 SEARCH ENGINE COMMAND LINE INTERFACE")
        print("=" * 60)
        print(f"Implementation: {self.engine.get_implementation_info()}")
        print()
        print("Available Commands:")
        print("  add `<content>` [links]          - Add a document (auto-generates ID)")
        print("  search `<query>` [top_k]        - Search for documents")
        print("  suggest `<prefix>` [limit]       - Get autocomplete suggestions")
        print("  stats                           - Show system statistics")
        print("  switch                          - Switch between Phase 2/3")
        print("  help                            - Show this help message")
        print("  quit/exit                       - Exit the program")
        print()
        print("Examples:")
        print("  add `Python is a programming language`")
        print("  add `Data structures are important` doc1")
        print("  add `He said \"Hello world\" to me` doc2")
        print("  search `python programming` 5")
        print("  suggest `prog` 3")
        print()
        print("Note: Content must be wrapped in backticks (`) for robust handling")
        print("      of quotes and special characters.")
        print("=" * 60)
    
    def add_document(self, args: List[str]) -> None:
        """Add a document to the search engine."""
        if len(args) < 1:
            print("❌ Error: Usage: add `<content>` [links]")
            return
        
        # Auto-generate document ID
        self.doc_counter += 1
        doc_id = f"doc_{self.doc_counter:03d}"
        
        # Parse content (everything except potential links)
        # Look for quoted content first
        content = ""
        links = []
        
        # Check if content is wrapped in backticks for robust quote handling
        if args[0].startswith("`"):
            content_parts = []
            i = 0
            while i < len(args):
                if i == 0:
                    # Remove opening backtick
                    content_parts.append(args[i][1:])
                else:
                    content_parts.append(args[i])
                
                # Check if this part ends with backtick (and has content before it)
                if args[i].endswith("`") and len(args[i]) > 1:
                    # Remove closing backtick
                    content_parts[-1] = content_parts[-1][:-1]
                    content = " ".join(content_parts)
                    # Remaining args are links
                    links = args[i+1:] if i+1 < len(args) else []
                    break
                i += 1
            
            # If no closing backtick found, treat all as content
            if not content:
                content = " ".join(content_parts)
        else:
            # No backticks, treat all args as content (no links in this case)
            content = " ".join(args)
        
        # Store document content for display
        self.documents[doc_id] = content
        
        # Add to search engine
        self.engine.add_document(doc_id, content, links)
        
        print(f"✅ Document '{doc_id}' added successfully!")
        print(f"   Content: {content[:50]}{'...' if len(content) > 50 else ''}")
        if links:
            print(f"   Links: {', '.join(links)}")
    
    def search_documents(self, args: List[str]) -> None:
        """Search for documents."""
        if len(args) < 1:
            print("❌ Error: Usage: search `<query>` [top_k]")
            return
        
        query = args[0]
        
        # Parse top_k with error handling
        top_k = 10  # default
        if len(args) > 1:
            try:
                top_k = int(args[1])
            except ValueError:
                print(f"❌ Error: Invalid top_k value '{args[1]}'. Using default value 10.")
                top_k = 10
        
        print(f"🔍 Searching for: '{query}' (top {top_k} results)")
        print("-" * 50)
        
        # Perform search
        results = self.engine.search(query, top_k)
        
        if not results:
            print("❌ No documents found matching your query.")
            return
        
        print(f"📊 Found {len(results)} results:")
        print()
        
        for i, (doc_id, score) in enumerate(results, 1):
            content = self.documents.get(doc_id, "Content not available")
            print(f"{i:2d}. Document: {doc_id}")
            print(f"    Relevance Score: {score:.4f}")
            print(f"    Content: {content[:80]}{'...' if len(content) > 80 else ''}")
            print()
    
    def get_suggestions(self, args: List[str]) -> None:
        """Get autocomplete suggestions."""
        if len(args) < 1:
            print("❌ Error: Usage: suggest `<prefix>` [limit]")
            return
        
        prefix = args[0]
        
        # Parse limit with error handling
        limit = 5  # default
        if len(args) > 1:
            try:
                limit = int(args[1])
            except ValueError:
                print(f"❌ Error: Invalid limit value '{args[1]}'. Using default value 5.")
                limit = 5
        
        print(f"💡 Autocomplete suggestions for: '{prefix}'")
        print("-" * 40)
        
        suggestions = self.engine.autocomplete(prefix, limit)
        
        if not suggestions:
            print("❌ No suggestions found.")
            return
        
        print(f"📝 Found {len(suggestions)} suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    def show_stats(self) -> None:
        """Show system statistics."""
        stats = self.engine.get_stats()
        
        print("📊 SYSTEM STATISTICS")
        print("-" * 30)
        print(f"Implementation: {self.engine.get_implementation_info()}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Terms in Index: {stats['total_terms_in_index']}")
        print(f"Total Words in Trie: {stats['total_words_in_trie']}")
        print(f"Total Pages in Graph: {stats['total_pages_in_graph']}")
        
        if 'hit_rate' in stats:
            print(f"Cache Hit Rate: {stats['hit_rate']:.2%}")
        
        if 'current_memory_mb' in stats:
            print(f"Memory Usage: {stats['current_memory_mb']:.1f}MB")
    
    def switch_implementation(self) -> None:
        """Switch between Phase 2 and Phase 3 implementations."""
        current = "Phase 3 (Optimized)" if self.engine.use_optimized else "Phase 2 (Original)"
        new = "Phase 2 (Original)" if self.engine.use_optimized else "Phase 3 (Optimized)"
        
        print(f"🔄 Switching from {current} to {new}")
        
        # Get current documents
        current_docs = []
        for doc_id, content in self.documents.items():
            current_docs.append((doc_id, content, []))  # No links for simplicity
        
        # Switch implementation
        self.engine.switch_implementation(not self.engine.use_optimized)
        
        # Re-add documents to new implementation
        if current_docs:
            if self.engine.use_optimized:
                self.engine.add_document_batch(current_docs)
            else:
                for doc_id, content, links in current_docs:
                    self.engine.add_document(doc_id, content, links)
        
        print(f"✅ Switched to {self.engine.get_implementation_info()}")
    
    def run_interactive(self) -> None:
        """Run the interactive CLI."""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 Search Engine> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:]
                
                # Handle commands
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif command == 'help':
                    self.print_welcome()
                elif command == 'add':
                    self.add_document(args)
                elif command == 'search':
                    self.search_documents(args)
                elif command == 'suggest':
                    self.get_suggestions(args)
                elif command == 'stats':
                    self.show_stats()
                elif command == 'switch':
                    self.switch_implementation()
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_batch_mode(self, commands: List[str]) -> None:
        """Run commands in batch mode."""
        for command_line in commands:
            print(f"🔍 Executing: {command_line}")
            
            # Handle backtick-wrapped content properly
            parts = []
            current_part = ""
            in_backticks = False
            
            for char in command_line:
                if char == '`' and not in_backticks:
                    in_backticks = True
                    if current_part.strip():
                        parts.append(current_part.strip())
                        current_part = ""
                elif char == '`' and in_backticks:
                    in_backticks = False
                    if current_part.strip():
                        parts.append(current_part.strip())
                        current_part = ""
                elif char == ' ' and not in_backticks:
                    if current_part.strip():
                        parts.append(current_part.strip())
                        current_part = ""
                else:
                    current_part += char
            
            if current_part.strip():
                parts.append(current_part.strip())
            
            if not parts:
                continue
                
            command = parts[0].lower()
            args = parts[1:]
            
            try:
                if command == 'add':
                    self.add_document(args)
                elif command == 'search':
                    self.search_documents(args)
                elif command == 'suggest':
                    self.get_suggestions(args)
                elif command == 'stats':
                    self.show_stats()
                else:
                    print(f"❌ Unknown command: {command}")
            except Exception as e:
                print(f"❌ Error executing '{command_line}': {e}")


def create_search_engine(use_optimized: bool = True) -> SearchEngine:
    """
    Factory function to create a search engine instance.
    
    Args:
        use_optimized: If True, use Phase 3 optimized implementation.
        
    Returns:
        SearchEngine instance
    """
    return SearchEngine(use_optimized=use_optimized)


def demo_search_engine(use_optimized: bool = True) -> None:
    """
    Demonstrate search engine functionality.
    
    Args:
        use_optimized: If True, use Phase 3 optimized implementation.
    """
    print(f"Search Engine Demo - {'Phase 3 (Optimized)' if use_optimized else 'Phase 2 (Original)'}")
    print("=" * 60)
    
    # Create search engine
    engine = create_search_engine(use_optimized)
    
    # Add sample documents
    sample_docs = [
        ("doc1", "Python is a powerful programming language for data science", ["doc2"]),
        ("doc2", "Data structures and algorithms are fundamental in computer science", ["doc1", "doc3"]),
        ("doc3", "Machine learning algorithms use various data structures for optimization", ["doc2"]),
        ("doc4", "Web development with Python involves frameworks like Django and Flask", ["doc1"]),
        ("doc5", "Database systems use sophisticated indexing structures for fast retrieval", ["doc2", "doc4"])
    ]
    
    print("\n1. Adding Documents:")
    print("-" * 20)
    
    if use_optimized:
        engine.add_document_batch(sample_docs)
        print("✓ Added 5 documents using batch operation")
    else:
        for doc_id, content, links in sample_docs:
            engine.add_document(doc_id, content, links)
            print(f"✓ Added document {doc_id}")
    
    # Calculate PageRank
    print("\n2. Calculating PageRank:")
    print("-" * 25)
    pageranks = engine.calculate_page_ranks()
    print("✓ PageRank calculation completed")
    
    # Test search
    print("\n3. Search Results:")
    print("-" * 18)
    test_queries = ["python programming", "data structures", "machine learning"]
    
    for query in test_queries:
        results = engine.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"  {i}. Document {doc_id} (relevance: {score:.4f})")
    
    # Test autocomplete
    print("\n4. Autocomplete Suggestions:")
    print("-" * 30)
    test_prefixes = ["prog", "data", "mach"]
    
    for prefix in test_prefixes:
        suggestions = engine.autocomplete(prefix, limit=3)
        if suggestions:
            print(f"'{prefix}' -> {suggestions}")
        else:
            print(f"'{prefix}' -> No suggestions found")
    
    # Display statistics
    print("\n5. System Statistics:")
    print("-" * 22)
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n{engine.get_implementation_info()}")
    print("\n" + "=" * 60)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Search Engine CLI')
    parser.add_argument('--phase2', action='store_true', 
                       help='Use Phase 2 (original) implementation')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration mode')
    parser.add_argument('--batch', nargs='+',
                       help='Run commands in batch mode')
    
    args = parser.parse_args()
    
    use_optimized = not args.phase2
    
    if args.demo:
        # Run demo for both implementations
        demo_search_engine(use_optimized=False)  # Phase 2
        print("\n")
        demo_search_engine(use_optimized=True)   # Phase 3
    elif args.batch:
        # Run batch mode
        cli = SearchEngineCLI(use_optimized=use_optimized)
        cli.run_batch_mode(args.batch)
    else:
        # Run interactive mode
        cli = SearchEngineCLI(use_optimized=use_optimized)
        cli.run_interactive()


if __name__ == "__main__":
    main()
