from dataclasses import dataclass
from typing import Dict, Any

from chromadb import QueryResult

@dataclass
class SemanticMatch:
    query: str
    results: QueryResult
    filters: Dict[str, Any]
    summary: Dict[str, Any]

    def build_business_context_summary(self) -> str:
        """Build business context summary"""
        summary_parts = []

        for i, (doc_id, document, metadata, distance) in enumerate(zip(
                self.results['ids'][0],
                self.results['documents'][0],
                self.results['metadatas'][0],
                self.results['distances'][0]
        )):
            summary_parts.append(f"\n--- Relevant File {i}: {metadata.get('file_path', 'Unknown')} ---")
            summary_parts.append(f"Project Name: {metadata.get('project_name', 'Unknown')}")
            summary_parts.append(f"File Type: {metadata.get('file_type', 'Unknown')}")
            summary_parts.append(f"Business Purpose: {metadata.get('business_purpose', 'Unknown')}")
            summary_parts.append(f"Technical Pattern: {metadata.get('technical_pattern', 'Unknown')}")
            summary_parts.append(f"Business Workflow: {metadata.get('business_workflow', 'Unknown')}")
            summary_parts.append(f"Business Triggers: {metadata.get('business_triggers', '')}")
            summary_parts.append(f"Business Data: {metadata.get('business_data', '')}")

            print(f"\n--- Result {i + 1} (Distance: {distance:.4f}) ---")
            print(f"File: {metadata.get('file_path', 'Unknown')}")
            print(f"Business Purpose: {metadata.get('business_purpose', 'Unknown')}")
            print(f"Technical Pattern: {metadata.get('technical_pattern', 'Unknown')}")
            print(f"Business Workflow: {metadata.get('business_workflow', 'Unknown')[:100]}...")

            if metadata['business_rules']:
                summary_parts.append(f"Business Rules: {', '.join(metadata['business_rules'][:3])}")

            if metadata['integration_points']:
                summary_parts.append(f"Integration Points: {', '.join(metadata['integration_points'][:3])}")

            summary_parts.append(f"Semantic Distance: {distance:.4f}")
            summary_parts.append(f"Confidence: {metadata.get('classification_confidence', 0.0)}")
            summary_parts.append("")

        return "\n".join(summary_parts)

    def get_context_docs(self):
        context_docs = []
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
                self.results['ids'][0],
                self.results['documents'][0],
                self.results['metadatas'][0],
                self.results['distances'][0]
        )):
            context_doc = {
                'rank': i + 1,
                'file_path': metadata.get('file_path', 'Unknown'),
                'project_name': metadata.get('project_name', 'Unknown'),
                'file_type': metadata.get('file_type', 'Unknown'),
                'business_purpose': metadata.get('business_purpose', 'Unknown'),
                'business_rules': metadata.get('business_rules', '').split(' | ') if metadata.get(
                    'business_rules') else [],
                'business_triggers': metadata.get('business_triggers', '').split(' | ') if metadata.get(
                    'business_triggers') else [],
                'business_data': metadata.get('business_data', '').split(' | ') if metadata.get(
                    'business_data') else [],
                'integration_points': metadata.get('integration_points', '').split(' | ') if metadata.get(
                    'integration_points') else [],
                'business_workflow': metadata.get('business_workflow', 'Unknown'),
                'technical_pattern': metadata.get('technical_pattern', 'Unknown'),
                'distance': distance,
                'full_document': document,
                'confidence': metadata.get('classification_confidence', 0.0)
            }
            context_docs.append(context_doc)

        print(f"Retrieved {len(context_docs)} relevant documents")
        for doc in context_docs:
            print(f"  - {doc['file_path']} (distance: {doc['distance']:.4f})")

        return context_docs