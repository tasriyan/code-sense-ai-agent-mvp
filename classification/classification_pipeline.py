import csv
import json
from pathlib import Path
from typing import List

from classification.classification_result import ClassificationResult as cr, ClassificationResult
from classification.code_scanner import CodeScanner
from llms.providers import LLMClassifier


class ClassificationPipeline:
    """Main pipeline for code classification"""

    def __init__(self,
                 provider: LLMClassifier,
                 scanner: CodeScanner,
                 output_csv: str, intermediate_dir: str):
        self.output_csv = output_csv
        self.intermediate_dir = Path(intermediate_dir)
        self.intermediate_dir.mkdir(exist_ok=True)

        self.provider = provider
        self.scanner = scanner

    def run_classification(self) -> List[ClassificationResult]:
        """Run the complete classification pipeline"""
        provider_name = self.provider.get_provider_name()
        print(f"Starting classification with provider: {provider_name}")

        # Scan files
        print("Scanning code files...")
        code_files = self.scanner.scan_files()
        print(f"Found {len(code_files)} files to classify")

        # Classify files
        print("Classifying files...")
        results = []
        for i, code_file in enumerate(code_files):
            print(f"Classifying {i + 1}/{len(code_files)}: {code_file.relative_path}")

            try:
                classification = self.provider.classify_code(code_file)
                result = cr.create_classification_result(code_file, classification, provider_name)
                results.append(result)

                # Save intermediate result
                self._save_intermediate_result(result, i)

            except Exception as e:
                print(f"Error classifying {code_file.file_path}: {e}")
                continue

        # Save final results
        self._save_results_to_csv(results)
        print(f"Classification complete. Results saved to {self.output_csv}")

        return results

    def _save_intermediate_result(self, result: ClassificationResult, index: int):
        """Save intermediate result for debugging"""
        filename = f"result_{index:04d}_{result.project_name}_{Path(result.file_path).stem}.json"
        filepath = self.intermediate_dir / filename

        result_dict = {
            'file_path': result.file_path,
            'project_name': result.project_name,
            'file_type': result.file_type,
            'business_purpose': result.business_purpose,
            'business_rules': result.business_rules,
            'business_triggers': result.business_triggers,
            'business_data': result.business_data,
            'integration_points': result.integration_points,
            'business_workflow': result.business_workflow,
            'technical_pattern': result.technical_pattern,
            'llm_provider': result.llm_provider,
            'classification_confidence': result.classification_confidence
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    def _save_results_to_csv(self, results: List[ClassificationResult]):
        """Save classification results to CSV"""
        fieldnames = [
            'file_path', 'project_name', 'file_type', 'business_purpose',
            'business_rules', 'business_triggers', 'business_data',
            'integration_points', 'business_workflow', 'technical_pattern',
            'llm_provider', 'classification_confidence'
        ]

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow({
                    'file_path': result.file_path,
                    'project_name': result.project_name,
                    'file_type': result.file_type,
                    'business_purpose': result.business_purpose,
                    'business_rules': '|'.join(result.business_rules),
                    'business_triggers': '|'.join(result.business_triggers),
                    'business_data': '|'.join(result.business_data),
                    'integration_points': '|'.join(result.integration_points),
                    'business_workflow': result.business_workflow,
                    'technical_pattern': result.technical_pattern,
                    'llm_provider': result.llm_provider,
                    'classification_confidence': result.classification_confidence
                })