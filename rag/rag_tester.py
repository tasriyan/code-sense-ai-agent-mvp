import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "loyalty_code_semantics"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class RAGTester:
    """Test and evaluate RAG retrieval quality for CodeSense"""

    def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Get collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Connected to collection '{self.collection_name}' with {self.collection.count()} documents")
        except ValueError as e:
            raise ValueError(f"Collection '{self.collection_name}' not found. Run Notebook 2 first. Error: {e}")

    def test_basic_retrieval(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Test basic semantic retrieval for a query"""

        print(f"\n=== Basic Retrieval Test ===")
        print(f"Query: '{query}'")
        print(f"Retrieving top {n_results} results...")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            if not results['ids'][0]:
                return {
                    'query': query,
                    'results': [],
                    'summary': {'total_results': 0, 'error': 'No results found'}
                }

            print(f"\nFound {len(results['ids'][0])} results:")

            retrieved_docs = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                print(f"\n--- Result {i + 1} (Distance: {distance:.4f}) ---")
                print(f"File: {metadata.get('file_path', 'Unknown')}")
                print(f"Project: {metadata.get('project_name', 'Unknown')}")
                print(f"Business Purpose: {metadata.get('business_purpose', 'Unknown')[:100]}...")
                print(f"Technical Pattern: {metadata.get('technical_pattern', 'Unknown')}")

                retrieved_docs.append({
                    'id': doc_id,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'project_name': metadata.get('project_name', 'Unknown'),
                    'business_purpose': metadata.get('business_purpose', 'Unknown'),
                    'technical_pattern': metadata.get('technical_pattern', 'Unknown'),
                    'distance': distance,
                    'document': document
                })

            return {
                'query': query,
                'results': retrieved_docs,
                'summary': {
                    'total_results': len(results['ids'][0]),
                    'avg_distance': np.mean(results['distances'][0]),
                    'min_distance': np.min(results['distances'][0]),
                    'max_distance': np.max(results['distances'][0]),
                    'files_found': [doc['file_path'] for doc in retrieved_docs]
                }
            }

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return {
                'query': query,
                'error': str(e),
                'results': [],
                'summary': {'total_results': 0, 'error': str(e)}
            }

    def test_filtered_retrieval(self, query: str, filters: Dict[str, Any], n_results: int = 5) -> Dict[str, Any]:
        """Test retrieval with metadata filters"""

        print(f"\n=== Filtered Retrieval Test ===")
        print(f"Query: '{query}'")
        print(f"Filters: {filters}")
        print(f"Retrieving top {n_results} results...")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            if not results['ids'][0]:
                return {
                    'query': query,
                    'filters': filters,
                    'results': [],
                    'summary': {'total_results': 0, 'message': 'No results found with filters'}
                }

            print(f"\nFound {len(results['ids'][0])} results with filters:")

            retrieved_docs = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                print(f"\n--- Filtered Result {i + 1} (Distance: {distance:.4f}) ---")
                print(f"File: {metadata.get('file_path', 'Unknown')}")
                print(f"Project: {metadata.get('project_name', 'Unknown')}")
                print(f"File Type: {metadata.get('file_type', 'Unknown')}")
                print(f"Business Purpose: {metadata.get('business_purpose', 'Unknown')[:100]}...")

                retrieved_docs.append({
                    'id': doc_id,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'project_name': metadata.get('project_name', 'Unknown'),
                    'file_type': metadata.get('file_type', 'Unknown'),
                    'business_purpose': metadata.get('business_purpose', 'Unknown'),
                    'distance': distance,
                    'document': document
                })

            return {
                'query': query,
                'filters': filters,
                'results': retrieved_docs,
                'summary': {
                    'total_results': len(results['ids'][0]),
                    'avg_distance': np.mean(results['distances'][0]),
                    'files_found': [doc['file_path'] for doc in retrieved_docs]
                }
            }

        except Exception as e:
            print(f"Error during filtered retrieval: {e}")
            return {
                'query': query,
                'filters': filters,
                'error': str(e),
                'results': [],
                'summary': {'total_results': 0, 'error': str(e)}
            }

    def run_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive RAG test suite"""

        print("\n" + "=" * 60)
        print("COMPREHENSIVE RAG TEST SUITE")
        print("=" * 60)

        test_results = {
            'timestamp': datetime.now().isoformat(),
            'collection_stats': self._get_collection_stats(),
            'basic_tests': {},
            'filtered_tests': {},
            'edge_case_tests': {},
            'performance_metrics': {}
        }

        # Test 1: Basic semantic queries
        print("\n1. BASIC SEMANTIC RETRIEVAL TESTS")
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
            result = self.test_basic_retrieval(query, n_results=3)
            test_results['basic_tests'][query] = result

        # Test 2: Filtered retrieval
        print("\n2. FILTERED RETRIEVAL TESTS")
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
            result = self.test_filtered_retrieval(
                test['query'],
                test['filters'],
                n_results=3
            )
            test_results['filtered_tests'][test['query']] = result

        # Test 3: Edge cases
        print("\n3. EDGE CASE TESTS")
        edge_cases = [
            "",  # Empty query
            "xyzabc123nonexistent",  # Non-existent terms
            "a",  # Single character
            "the and or but",  # Stop words only
            "loyalty" * 50,  # Very long query
        ]

        for query in edge_cases:
            result = self.test_basic_retrieval(query, n_results=1)
            test_results['edge_case_tests'][query] = result

        # Test 4: Performance metrics
        print("\n4. PERFORMANCE METRICS")
        test_results['performance_metrics'] = self._calculate_performance_metrics(test_results)

        return test_results

    def _get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            total_count = self.collection.count()
            sample_docs = self.collection.get(limit=100, include=["metadatas"])

            if not sample_docs['metadatas']:
                return {'total_documents': total_count, 'error': 'No metadata available'}

            # Analyze metadata distribution
            projects = {}
            file_types = {}
            patterns = {}

            for metadata in sample_docs['metadatas']:
                project = metadata.get('project_name', 'Unknown')
                projects[project] = projects.get(project, 0) + 1

                file_type = metadata.get('file_type', 'Unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1

                pattern = metadata.get('technical_pattern', 'Unknown')
                patterns[pattern] = patterns.get(pattern, 0) + 1

            return {
                'total_documents': total_count,
                'sample_size': len(sample_docs['metadatas']),
                'project_distribution': projects,
                'file_type_distribution': file_types,
                'pattern_distribution': patterns
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics from test results"""

        metrics = {
            'retrieval_success_rate': 0,
            'average_distance': 0,
            'query_coverage': 0,
            'filter_effectiveness': 0
        }

        try:
            # Basic retrieval success rate
            basic_tests = test_results['basic_tests']
            successful_queries = sum(1 for result in basic_tests.values()
                                     if result['summary']['total_results'] > 0)
            metrics['retrieval_success_rate'] = successful_queries / len(basic_tests) if basic_tests else 0

            # Average distance across all queries
            all_distances = []
            for result in basic_tests.values():
                if 'avg_distance' in result['summary']:
                    all_distances.append(result['summary']['avg_distance'])
            metrics['average_distance'] = np.mean(all_distances) if all_distances else 0

            # Query coverage (unique files returned)
            all_files = set()
            for result in basic_tests.values():
                all_files.update(result['summary'].get('files_found', []))

            total_docs = test_results['collection_stats']['total_documents']
            metrics['query_coverage'] = len(all_files) / total_docs if total_docs > 0 else 0

            # Filter effectiveness
            filtered_tests = test_results['filtered_tests']
            filter_improvements = []
            for result in filtered_tests.values():
                if result['summary']['total_results'] > 0:
                    filter_improvements.append(1)
                else:
                    filter_improvements.append(0)
            metrics['filter_effectiveness'] = np.mean(filter_improvements) if filter_improvements else 0

        except Exception as e:
            metrics['error'] = str(e)

        return metrics

    def analyze_query_patterns(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in query performance"""

        print("\n=== Query Pattern Analysis ===")

        analysis = {
            'best_performing_queries': [],
            'worst_performing_queries': [],
            'distance_distribution': {},
            'file_type_preferences': {},
            'project_coverage': {}
        }

        try:
            basic_tests = test_results['basic_tests']

            # Sort queries by performance (distance)
            query_performance = []
            for query, result in basic_tests.items():
                if 'avg_distance' in result['summary']:
                    query_performance.append({
                        'query': query,
                        'avg_distance': result['summary']['avg_distance'],
                        'total_results': result['summary']['total_results']
                    })

            query_performance.sort(key=lambda x: x['avg_distance'])

            # Best and worst performing queries
            analysis['best_performing_queries'] = query_performance[:3]
            analysis['worst_performing_queries'] = query_performance[-3:]

            # Distance distribution
            distances = [q['avg_distance'] for q in query_performance]
            analysis['distance_distribution'] = {
                'min': np.min(distances),
                'max': np.max(distances),
                'mean': np.mean(distances),
                'std': np.std(distances)
            }

            # File type preferences
            file_type_hits = {}
            for result in basic_tests.values():
                for doc in result['results']:
                    file_path = doc.get('file_path', '')
                    if file_path.endswith('.cs'):
                        file_type_hits['cs'] = file_type_hits.get('cs', 0) + 1
                    elif 'appsettings' in file_path:
                        file_type_hits['appsettings'] = file_type_hits.get('appsettings', 0) + 1

            analysis['file_type_preferences'] = file_type_hits

            # Project coverage
            project_hits = {}
            for result in basic_tests.values():
                for doc in result['results']:
                    project = doc.get('project_name', 'Unknown')
                    project_hits[project] = project_hits.get(project, 0) + 1

            analysis['project_coverage'] = project_hits

        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""

        report = []
        report.append("=" * 60)
        report.append("CODESENSE RAG TESTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {test_results['timestamp']}")
        report.append("")

        # Collection Stats
        stats = test_results['collection_stats']
        report.append("COLLECTION STATISTICS:")
        report.append(f"  Total Documents: {stats.get('total_documents', 'Unknown')}")
        report.append(f"  Sample Size: {stats.get('sample_size', 'Unknown')}")
        report.append(f"  Projects: {stats.get('project_distribution', {})}")
        report.append(f"  File Types: {stats.get('file_type_distribution', {})}")
        report.append("")

        # Performance Metrics
        metrics = test_results['performance_metrics']
        report.append("PERFORMANCE METRICS:")
        report.append(f"  Retrieval Success Rate: {metrics.get('retrieval_success_rate', 0):.2%}")
        report.append(f"  Average Distance: {metrics.get('average_distance', 0):.4f}")
        report.append(f"  Query Coverage: {metrics.get('query_coverage', 0):.2%}")
        report.append(f"  Filter Effectiveness: {metrics.get('filter_effectiveness', 0):.2%}")
        report.append("")

        # Best/Worst Queries
        analysis = self.analyze_query_patterns(test_results)
        report.append("QUERY ANALYSIS:")
        report.append("  Best Performing Queries:")
        for q in analysis.get('best_performing_queries', []):
            report.append(f"    '{q['query']}' (distance: {q['avg_distance']:.4f})")

        report.append("  Worst Performing Queries:")
        for q in analysis.get('worst_performing_queries', []):
            report.append(f"    '{q['query']}' (distance: {q['avg_distance']:.4f})")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        success_rate = metrics.get('retrieval_success_rate', 0)
        if success_rate < 0.8:
            report.append("  - Consider improving document preparation or embedding model")
        if metrics.get('average_distance', 1) > 0.5:
            report.append("  - Query-document semantic alignment could be improved")
        if metrics.get('query_coverage', 0) < 0.5:
            report.append("  - Some documents may not be well-represented in queries")

        return "\n".join(report)

    def save_test_results(self, test_results: Dict[str, Any], output_path: str = "rag_test_results.json"):
        """Save test results to file"""

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Test results saved to: {output_path}")

        # Also save readable report
        report = self.generate_test_report(test_results)
        report_path = output_path.replace('.json', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Test report saved to: {report_path}")