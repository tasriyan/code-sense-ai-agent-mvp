
import re
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple


class RAGReportAnalyzer:
    """Analyzes RAG testing reports and generates performance comparison tables."""

    def __init__(self, reports_directory: str = "."):
        self.reports_directory = Path(reports_directory)
        self.results = []

    def load_reports(self, file_pattern: str = "*.txt") -> List[Dict]:
        """Load and parse all RAG reports matching the pattern."""
        report_files = list(self.reports_directory.glob(file_pattern))

        if not report_files:
            print(f"No report files found matching pattern: {file_pattern}")
            return []

        results = []
        for file_path in report_files:
            if 'report' in file_path.name.lower():
                parsed_data = self._parse_report_file(file_path)
                if parsed_data:
                    results.append(parsed_data)

        self.results = results
        return results

    def generate_performance_table(self, sort_by: str = 'avg_distance') -> pd.DataFrame:
        """Generate a performance comparison table."""
        if not self.results:
            print("No results loaded. Please run load_reports() first.")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Clean up model names
        df['embedding_model'] = df['embedding_model'].str.replace('_', '-')
        df['llm_model'] = df['llm_model'].str.replace('_', '.')

        # Create display columns
        display_df = df[
            ['embedding_model', 'llm_model', 'avg_distance', 'query_coverage', 'best_query_distance']].copy()

        # Sort by specified metric (lower is better for distance metrics)
        if sort_by in ['avg_distance', 'best_query_distance']:
            display_df = display_df.sort_values(sort_by, ascending=True)
        else:
            display_df = display_df.sort_values(sort_by, ascending=False)

        display_df.columns = ['Embedding Model', 'LLM', 'Avg Distance', 'Query Coverage (%)', 'Best Query Distance']
        # Format numbers for better display
        display_df['Avg Distance'] = display_df['Avg Distance'].round(4)
        display_df['Best Query Distance'] = display_df['Best Query Distance'].round(4)
        display_df['Query Coverage (%)'] = display_df['Query Coverage (%)'].round(2)

        return display_df

    def print_analysis_summary(self):
        """Print a comprehensive analysis summary."""
        if not self.results:
            print("No results loaded. Please run load_reports() first.")
            return

        df = pd.DataFrame(self.results)

        print("=" * 60)
        print("RAG PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)

        # Overall statistics
        print(f"Total Reports Analyzed: {len(self.results)}")
        print(f"Embedding Models: {df['embedding_model'].nunique()}")
        print(f"LLM Models: {df['llm_model'].nunique()}")
        print()

        # Performance table
        print("PERFORMANCE COMPARISON TABLE:")
        print(self._generate_markdown_table())
        print()

        # Best performers
        best_overall = df.loc[df['avg_distance'].idxmin()]
        best_coverage = df.loc[df['query_coverage'].idxmax()]
        best_query = df.loc[df['best_query_distance'].idxmin()]

        print("KEY FINDINGS:")
        print(
            f"• Best Overall Performance: {best_overall['embedding_model']} + {best_overall['llm_model']} (Avg Distance: {best_overall['avg_distance']:.4f})")
        print(
            f"• Best Query Coverage: {best_coverage['embedding_model']} + {best_coverage['llm_model']} ({best_coverage['query_coverage']:.2f}%)")
        print(
            f"• Best Individual Query: {best_query['embedding_model']} + {best_query['llm_model']} (Distance: {best_query['best_query_distance']:.4f})")

        # Model comparison
        print("\nEMBEDDING MODEL COMPARISON:")
        model_stats = df.groupby('embedding_model').agg({
            'avg_distance': 'mean',
            'query_coverage': 'mean',
            'best_query_distance': 'mean'
        }).round(4)
        print(model_stats.to_string())

        print("\nLLM MODEL COMPARISON:")
        llm_stats = df.groupby('llm_model').agg({
            'avg_distance': 'mean',
            'query_coverage': 'mean',
            'best_query_distance': 'mean'
        }).round(4)
        print(llm_stats.to_string())

    def _parse_report_file(self, file_path: Path) -> Dict:
        """Parse a single RAG report file and extract key metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract embedding model and LLM from filename
            filename = file_path.stem
            parts = filename.split('.')

            # Pattern: embedding_model.dataset.llm_report
            if len(parts) >= 2:
                embedding_model = parts[0]
            else:
                # Fallback parsing
                embedding_model = "unknown"

            llm_model = self._extract_model_from_filename(filename)

            # Extract metrics using regex
            metrics = {}

            # Average Distance
            avg_distance_match = re.search(r'Average Distance:\s*([\d.]+)', content)
            metrics['avg_distance'] = float(avg_distance_match.group(1)) if avg_distance_match else None

            # Query Coverage
            query_coverage_match = re.search(r'Query Coverage:\s*([\d.]+)%', content)
            metrics['query_coverage'] = float(query_coverage_match.group(1)) if query_coverage_match else None

            # Best performing query distance
            best_query_match = re.search(r'Best Performing Queries:\s*\n\s*\'[^\']*\'\s*\(distance:\s*([\d.]+)\)',
                                         content)
            metrics['best_query_distance'] = float(best_query_match.group(1)) if best_query_match else None

            # Retrieval Success Rate
            success_rate_match = re.search(r'Retrieval Success Rate:\s*([\d.]+)%', content)
            metrics['success_rate'] = float(success_rate_match.group(1)) if success_rate_match else None

            # Filter Effectiveness
            filter_eff_match = re.search(r'Filter Effectiveness:\s*([\d.]+)%', content)
            metrics['filter_effectiveness'] = float(filter_eff_match.group(1)) if filter_eff_match else None

            return {
                'embedding_model': embedding_model,
                'llm_model': llm_model,
                'filename': filename,
                **metrics
            }

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def _extract_model_from_filename(self, filename):
        """Extract model name if it contains claude3.5"""
        if "claude3.5" in filename:
            return "claude3.5"
        if "claude3.7" in filename:
            return "claude3.7"
        if "claude4.0" in filename:
            return "claude4.0"
        elif "gpt4.1" in filename:
            return "gpt4.1"
        elif "codellama" in filename:
            return "codellama"
        else:
            return "unknown"

    def _generate_markdown_table(self, sort_by: str = 'avg_distance') -> str:
        """Generate a markdown-formatted performance table."""
        df = self.generate_performance_table(sort_by)

        if df.empty:
            return "No data available to generate table."

        # Convert to markdown
        markdown_table = df.to_markdown(index=False, floatfmt='.4f')

        # Add bold formatting for best performers
        lines = markdown_table.split('\n')
        if len(lines) > 2:  # Header + separator + at least one data row
            # Bold the first data row (best performer)
            data_row = lines[2]
            lines[2] = '| **' + data_row[1:-1].replace('|', '** | **') + '** |'

        return '\n'.join(lines)