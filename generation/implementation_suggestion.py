import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class ImplementationSuggestion:
    """Represents a complete implementation suggestion"""
    user_request: str
    retrieved_context: List[Dict[str, Any]]
    suggested_service: str
    suggested_files: List[Dict[str, str]]
    implementation_steps: List[str]
    business_rationale: str
    integration_points: List[str]
    code_examples: List[Dict[str, str]]
    confidence_score: float
    llm_provider: str
    generated_at: str

    def display_suggestion(self):
        """Display implementation suggestion in readable format"""

        print("\n" + "=" * 60)
        print("CODESENSE IMPLEMENTATION SUGGESTION")
        print("=" * 60)
        print(f"Request: {self.user_request}")
        print(f"Generated: {self.generated_at}")
        print(f"Provider: {self.llm_provider}")
        print(f"Confidence: {self.confidence_score:.2%}")

        print(f"\n--- SUGGESTED SERVICE ---")
        print(f"{self.suggested_service}")

        print(f"\n--- SUGGESTED FILES ---")
        for file_info in self.suggested_files:
            print(f"  {file_info.get('action', 'unknown').upper()}: {file_info.get('file_path', 'unknown')}")
            print(f"    Purpose: {file_info.get('purpose', 'unknown')}")

        print(f"\n--- IMPLEMENTATION STEPS ---")
        for i, step in enumerate(self.implementation_steps, 1):
            print(f"  {i}. {step}")

        print(f"\n--- BUSINESS RATIONALE ---")
        print(f"{self.business_rationale}")

        print(f"\n--- INTEGRATION POINTS ---")
        for integration in self.integration_points:
            print(f"  - {integration}")

        if self.code_examples:
            print(f"\n--- CODE EXAMPLES ---")
            for example in self.code_examples:
                print(f"  File: {example.get('file', 'unknown')}")
                print(f"  Code: {example.get('code', 'no code provided')}")

        print(f"\n--- RETRIEVED CONTEXT ---")
        print(f"Found {len(self.retrieved_context)} relevant documents:")
        for doc in self.retrieved_context:
            print(f"  - {doc['file_path']} (distance: {doc['distance']:.4f})")

    def save_suggestion(self, output_path: str = None):
        """Save implementation suggestion to file"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"implementation_suggestion_{timestamp}.json"

        # Convert to dictionary for JSON serialization
        suggestion_dict = {
            'user_request': self.user_request,
            'suggested_service': self.suggested_service,
            'suggested_files': self.suggested_files,
            'implementation_steps': self.implementation_steps,
            'business_rationale': self.business_rationale,
            'integration_points': self.integration_points,
            'code_examples': self.code_examples,
            'confidence_score': self.confidence_score,
            'llm_provider': self.llm_provider,
            'generated_at': self.generated_at,
            'retrieved_context': self.retrieved_context
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(suggestion_dict, f, indent=2, ensure_ascii=False, default=str)

        print(f"Implementation suggestion saved to: {output_path}")

class NoSuggestion(ImplementationSuggestion):
    def __init__(self):
        self.user_request = "no implementation"
        self.retrieved_context = []
        self.suggested_service = "no implementation"
        self.suggested_files = []
        self.implementation_steps = []
        self.business_rationale = "no implementation"
        self.integration_points = []
        self.code_examples = []
        self.confidence_score = 0.0
        self.llm_provider = "no implementation"
        self.generated_at = "no implementation"