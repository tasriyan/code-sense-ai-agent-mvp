# CodeSense MVP

An AI-powered feature development assistant that bridges the gap between business requirements and implementation details through semantic code understanding.

## The Problem

When developers receive user stories like "As a customer, I want to earn 100 loyalty points when my order amount is greater than $10", they spend significant time understanding:

- Which services and projects are involved
- What existing patterns and architectures to follow  
- How the new feature integrates with existing business logic
- What files need to be created or modified
- Which business rules and workflows apply

**This semantic understanding and architectural reasoning is where developers spend most of their time** - not on the actual coding.

## Our Solution

CodeSense analyzes your existing codebase to understand **business context and semantic meaning**, then uses this knowledge to generate intelligent implementation suggestions when new features are requested.

### How It Works

1. **Semantic Classification**: LLM-powered analysis extracts business purpose, rules, workflows, and integration points from your existing code
2. **Knowledge Base Creation**: Semantic understanding is stored in a vector database for fast retrieval
3. **RAG-Powered Suggestions**: When developers request new features, the system retrieves relevant business patterns and generates specific implementation guidance

### Example Input/Output

**Input**: "I want to add a new rule to calculate loyalty points"

**Output**:
```
Service: Loyalty.Points.Service
- Create: /loyalty.points/rules/NewLoyaltyRule.cs  
- Implement: ILoyaltyRule interface
- Pattern: Inherit from BaseLoyaltyRule
- Registration: Add to Program.cs DI container
- Event Handler: Subscribe to OrderCompletedEvent
- Dependencies: Inject IOrderService, ICustomerService
```

## Project Context

**Target System**: Plant Based Pizza, Inc. - a microservices-based pizza ordering and delivery platform

**Focus Domain**: Loyalty Points Service (MVP scope)
- LoyaltyPoints.csproj
- LoyaltyPoints.Internal.csproj  
- LoyaltyPoints.Shared.csproj
- LoyaltyPoints.Worker.csproj

**Current Phase**: MVP development for developer feedback and validation

## Design Decisions

### Why Semantic Understanding Over Static Analysis

**Static analysis** (AST parsing, dependency graphs) gives you **technical structure** - class names, method signatures, inheritance relationships. Developers can already get this from their IDEs.

**Semantic analysis** gives you **business context** - what problems the code solves, what business rules it implements, how services collaborate for business workflows. **This is what takes developers time to understand.**

### Why LLM-Based Classification

We evaluated several approaches:

- **Zero-shot classification**: Limited to predefined categories, doesn't generate rich semantic descriptions
- **Fine-tuned models**: Requires training data we don't have yet, more complex for MVP
- **Rule-based extraction**: Can't understand business semantics from code patterns

**LLM-based classification** provides:
- Rich semantic understanding out-of-the-box
- Flexibility to discover unexpected business patterns  
- Natural language descriptions that developers can easily understand
- Ability to reason about business context from technical implementation

### Why Multiple LLM Providers

**Strategy Pattern Implementation** allows runtime switching between:
- **OpenAI GPT-4**: High-quality semantic understanding, cloud-based
- **Anthropic Claude 3.5 Sonnet**: Excellent code reasoning, structured output
- **CodeLlama (via Ollama)**: Local deployment, code-specialized, security/privacy

This enables **performance comparison** and **deployment flexibility** based on:
- Security requirements (local vs cloud)
- Cost considerations  
- Quality benchmarking
- Organizational constraints

### Why Direct API Calls Over LangChain

For this MVP, we chose direct HTTP API calls over LangChain because:

- **Minimal dependencies**: Only requires `requests`, faster to get started
- **Precise control**: Full visibility into prompts and response parsing  
- **Custom output parsing**: Need specific JSON schema for classification results
- **Performance**: No abstraction overhead for batch processing hundreds of files
- **Debugging**: Easier to troubleshoot when classification goes wrong

*Note: Can be refactored to LangChain later if chain composition becomes needed.*

### Why Chroma for Vector Storage

Selected Chroma over alternatives (Pinecone, Weaviate, etc.) for:
- **Local deployment**: Matches security requirements
- **Python-native**: Integrates well with our ML pipeline
- **Lightweight**: Perfect for MVP scale
- **Embedding flexibility**: Easy to test different sentence-transformer models

## Current MVP Scope

### What's Included
- **Single domain focus**: Loyalty Points service only
- **Semantic classification**: Business purpose, rules, workflows, integration points
- **Multi-provider LLM support**: OpenAI, Anthropic, Ollama/CodeLlama
- **Vector database**: Searchable knowledge base of business semantics
- **File type support**: C# code files and appsettings.json configurations

### What's Not Included (Yet)
- **Cross-service integration**: Multi-service feature suggestions
- **Database schema changes**: Focus on business logic only  
- **Deployment automation**: Code generation only
- **Full RAG pipeline**: Implementation suggestion generation (next phase)
- **User interface**: Jupyter notebook testing only

## Success Metrics

**Primary Goal**: Does this actually save developers time in understanding business context and architectural patterns?

**Key Questions We're Answering**:
1. Can LLMs accurately extract business semantics from our codebase?
2. Does semantic search return relevant patterns for new feature requests?
3. Do developers find the implementation suggestions useful and accurate?
4. Which LLM provider gives the best semantic understanding for our domain?

## Next Steps

1. **Complete MVP pipeline**: RAG query system + implementation suggestion generation
2. **Developer testing**: Get feedback from actual loyalty service developers  
3. **Quality evaluation**: Compare LLM provider accuracy on our specific codebase
4. **Scope expansion**: Additional services if MVP proves valuable

---

*This is an MVP focused on proving the core concept: can AI understand business semantics well enough to accelerate feature development? We're optimizing for learning and feedback, not production deployment.*