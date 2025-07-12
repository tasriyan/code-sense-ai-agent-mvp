from datetime import datetime
from typing import Dict, Any

from rag.report_utils import calculate_performance_metrics
from vectorization.semantic_vector_database import SemanticVectorDatabase


class RAGTester:
    """Test and evaluate RAG retrieval quality for CodeSense"""

    def __init__(self, vector_db: SemanticVectorDatabase):
        self._vector_db = vector_db

    def run_test_suite(self, collection_name: str) -> Dict[str, Any]:
        """Run comprehensive RAG test suite"""

        print("\n" + "=" * 60)
        print("COMPREHENSIVE RAG TEST SUITE")
        print("=" * 60)

        collection = self._vector_db.get_collection(collection_name)

        test_results = {
            'timestamp': datetime.now().isoformat(),
            'collection_stats': collection.get_collection_stats_v2(),
            'basic_tests': {},
            'filtered_tests': {},
            'edge_case_tests': {},
            'performance_metrics': {}
        }

        # Test 1: Basic semantic queries
        print("\n1. BASIC SEMANTIC RETRIEVAL TESTS")
        # Query testing
        # Distance analysis for semantic similarity quality
        # Result ranking and relevance assessment
        # File and project coverage analysis

        basic_queries = [
            "loyalty points calculation rules",
            "order processing workflow",
            "customer data integration",
            "payment service integration",
            "business rule patterns",
            "event handlers",
            "database operations",
            "service dependencies",
            "configuration settings",
            "loyalty point rewards"
        ]

        for query in basic_queries:
            result = collection.semantic_search(query, n_results=3)
            test_results['basic_tests'][query] = result

        # Test 2: Filtered retrieval
        print("\n2. FILTERED RETRIEVAL TESTS")
        # Metadata filtering by file type, project, etc.
        # Filter effectiveness measurement

        filtered_tests = [
            {
                'query': "loyalty points calculation",
                'filters': {'file_type': 'cs'},
                'description': 'C# files only'
            },
            {
                'query': "service integration",
                'filters': {'file_type': 'appsettings'},
                'description': 'Configuration files only'
            },
            {
                'query': "business rules",
                'filters': {'project_name': 'LoyaltyPoints'},
                'description': 'Main project only'
            }
        ]

        for test in filtered_tests:
            print(f"\nTesting: {test['description']}")
            result = collection.filtered_semantic_search(
                test['query'],
                test['filters'],
                n_results=3
            )
            test_results['filtered_tests'][test['query']] = result

        # Test 3: Edge cases
        print("\n3. EDGE CASE TESTS")
         # Empty queries and malformed input
         # Non-existent terms for robustness
         # Single character and stop words handling
         # Very long queries for boundary testing

        edge_cases = [
            "",  # Empty query
            "xyzabc123nonexistent",  # Non-existent terms
            "a",  # Single character
            "the and or but",  # Stop words only
            "loyalty" * 50,  # Very long query
        ]

        for query in edge_cases:
            result = collection.semantic_search(query, n_results=1)
            test_results['edge_case_tests'][query] = result

        # Test 4: Performance metrics
        print("\n4. PERFORMANCE METRICS")
        test_results['performance_metrics'] = calculate_performance_metrics(test_results)

        return test_results