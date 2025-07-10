import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

def prepare_documents_for_embedding(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Semantic Document Preparation
        - Rich text creation from classification results for better embedding
        - Metadata preservation for filtering and context
        - Business-focused text combining purpose, rules, workflows, and integration points
    """

    documents = []

    for _, row in df.iterrows():
        # Create rich text document for embedding
        document_text = _create_document_text(row)

        # Create document record
        doc_record = {
            'id': str(uuid.uuid4()),
            'text': document_text,
            'metadata': {
                'file_path': str(row.get('file_path', '')),
                'project_name': str(row.get('project_name', '')),
                'file_type': str(row.get('file_type', '')),
                'business_purpose': str(row.get('business_purpose', '')),
                'business_rules': str(row.get('business_rules', '')),
                'business_triggers': str(row.get('business_triggers', '')),
                'business_data': str(row.get('business_data', '')),
                'integration_points': str(row.get('integration_points', '')),
                'business_workflow': str(row.get('business_workflow', '')),
                'technical_pattern': str(row.get('technical_pattern', '')),
                'llm_provider': str(row.get('llm_provider', '')),
                'classification_confidence': float(row.get('classification_confidence', 0.0)),
                'created_at': datetime.now().isoformat()
            }
        }

        documents.append(doc_record)

    print(f"Prepared {len(documents)} documents for embedding")
    return documents

def load_classification_data(csv_path: str) -> pd.DataFrame:
    """Load classification results from CSV"""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Classification CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} classification records from {csv_path}")

    # Validate required columns
    required_columns = [
        'file_path', 'project_name', 'file_type', 'business_purpose',
        'business_rules', 'business_triggers', 'business_data',
        'integration_points', 'business_workflow', 'technical_pattern'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df

def _parse_list_field(field_value: Any) -> List[str]:
    """Parse pipe-separated list fields from CSV"""
    if pd.isna(field_value) or field_value == '':
        return []

    if isinstance(field_value, str):
        return [item.strip() for item in field_value.split('|') if item.strip()]

    return []

def _create_document_text(row: pd.Series) -> str:
    """Create rich text representation for semantic embedding"""

    # Combine semantic information into searchable text
    text_parts = []

    # Business purpose (most important for semantic search)
    if pd.notna(row.get('business_purpose')):
        text_parts.append(f"Business Purpose: {row['business_purpose']}")

    # Business workflow
    if pd.notna(row.get('business_workflow')):
        text_parts.append(f"Business Workflow: {row['business_workflow']}")

    # Business rules
    business_rules = _parse_list_field(row.get('business_rules', ''))
    if business_rules:
        text_parts.append(f"Business Rules: {', '.join(business_rules)}")

    # Business triggers
    business_triggers = _parse_list_field(row.get('business_triggers', ''))
    if business_triggers:
        text_parts.append(f"Business Triggers: {', '.join(business_triggers)}")

    # Business data
    business_data = _parse_list_field(row.get('business_data', ''))
    if business_data:
        text_parts.append(f"Business Data: {', '.join(business_data)}")

    # Integration points
    integration_points = _parse_list_field(row.get('integration_points', ''))
    if integration_points:
        text_parts.append(f"Integration Points: {', '.join(integration_points)}")

    # Technical pattern
    if pd.notna(row.get('technical_pattern')):
        text_parts.append(f"Technical Pattern: {row['technical_pattern']}")

    # File context
    text_parts.append(f"File: {row.get('file_path', 'unknown')}")
    text_parts.append(f"Project: {row.get('project_name', 'unknown')}")

    return " | ".join(text_parts)
