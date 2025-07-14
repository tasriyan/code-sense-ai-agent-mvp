from typing import Dict, Any, List


def get_validated_answer(llm_answer: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize implementation response"""
    required_fields = [
        'suggested_service', 'suggested_files', 'implementation_steps',
        'business_rationale', 'integration_points', 'code_examples', 'confidence_score'
    ]

    # Ensure all required fields exist
    for field in required_fields:
        if field not in llm_answer:
            llm_answer[field] = [] if field.endswith(
                's') and field != 'confidence_score' else "Unknown" if field != 'confidence_score' else 0.0

    # Validate confidence score
    confidence = llm_answer.get('confidence_score', 0.0)
    if isinstance(confidence, (int, float)):
        llm_answer['confidence_score'] = max(0.0, min(1.0, float(confidence)))
    else:
        llm_answer['confidence_score'] = 0.0

    return llm_answer

def get_fallback_answer(user_request: str) -> Dict[str, Any]:
    """Fallback implementation when LLM fails"""
    return {
        "suggested_service": "LoyaltyPoints",
        "suggested_files": [
            {
                "action": "create",
                "file_path": "/loyalty/points/NewFeature.cs",
                "purpose": "Implementation for: " + user_request
            }
        ],
        "implementation_steps": [
            "LLM generation failed - manual implementation required",
            "Review existing patterns in the codebase",
            "Follow established architectural conventions"
        ],
        "business_rationale": "Fallback response due to LLM error",
        "integration_points": ["Manual analysis required"],
        "code_examples": [],
        "confidence_score": 0.0
    }