from typing import Dict, Any, List

import numpy as np

from vectorization.semantic_match import SemanticMatch


def analyze_query_patterns(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """ Best/worst performing queries identification
        Distance distribution statistics
        File type preferences in retrieval
        Project coverage analysis
    """

    print("\n=== Query Pattern Analysis ===")

    analysis = {
        'best_performing_queries': [],
        'worst_performing_queries': [],
        'distance_distribution': {},
        'file_type_preferences': {},
        'project_coverage': {}
    }

    try:
        basic_tests: list[SemanticMatch] = test_results['basic_tests']

        # Sort queries by performance (distance)
        query_performance = []
        for basic_test in basic_tests:
            query = basic_test.query
            summary = basic_test.summary
            if 'avg_distance' in summary:
                query_performance.append({
                    'query': query,
                    'avg_distance': summary['avg_distance'],
                    'total_results': summary['total_results']
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
        for basic_test in basic_tests:
            for doc in basic_test.results:
                file_path = doc.get('file_path', '')
                if file_path.endswith('.cs'):
                    file_type_hits['cs'] = file_type_hits.get('cs', 0) + 1
                elif 'appsettings' in file_path:
                    file_type_hits['appsettings'] = file_type_hits.get('appsettings', 0) + 1

        analysis['file_type_preferences'] = file_type_hits

        # Project coverage
        project_hits = {}
        for basic_test in basic_tests:
            for doc in basic_test.results:
                project = doc.get('project_name', 'Unknown')
                project_hits[project] = project_hits.get(project, 0) + 1

        analysis['project_coverage'] = project_hits

    except Exception as e:
        analysis['error'] = str(e)

    return analysis

def calculate_performance_metrics(test_results: Dict[str, List[SemanticMatch]], collection_stats: Dict[str, Any]) -> Dict[str, Any]:
    """ Retrieval success rate: % of queries returning results
        Average distance: Semantic similarity quality
        Query coverage: How many documents are discoverable
        Filter effectiveness: How well filters work
    """

    metrics = {
        'retrieval_success_rate': 0,
        'average_distance': 0,
        'query_coverage': 0,
        'filter_effectiveness': 0
    }

    try:
        # Basic retrieval success rate
        basic_tests = test_results['basic_tests']
        successful_queries = sum(1 for semantic_match in basic_tests
                                 if semantic_match.summary['total_results'] > 0)
        metrics['retrieval_success_rate'] = successful_queries / len(basic_tests) if basic_tests else 0

        # Average distance across all queries
        all_distances = []
        for semantic_match in basic_tests:
            if 'avg_distance' in semantic_match.summary:
                all_distances.append(semantic_match.summary['avg_distance'])
        metrics['average_distance'] = np.mean(all_distances) if all_distances else 0

        # Query coverage (unique files returned)
        all_files = set()
        for semantic_match in basic_tests:
            all_files.update(semantic_match.summary.get('files_found', []))

        total_docs = collection_stats['total_documents']
        metrics['query_coverage'] = len(all_files) / total_docs if total_docs > 0 else 0

        # Filter effectiveness
        filtered_tests = test_results['filtered_tests']
        filter_improvements = []
        for semantic_match in filtered_tests:
            if semantic_match.summary['total_results'] > 0:
                filter_improvements.append(1)
            else:
                filter_improvements.append(0)
        metrics['filter_effectiveness'] = np.mean(filter_improvements) if filter_improvements else 0

    except Exception as e:
        metrics['error'] = str(e)

    return metrics

def generate_rag_report(test_results: Dict[str, Any]) -> str:
    """ Detailed performance report with recommendations
        JSON results for programmatic analysis
    """

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
    analysis = analyze_query_patterns(test_results)
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