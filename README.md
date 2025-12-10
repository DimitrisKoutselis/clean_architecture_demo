# Clean Architecture RAG Demo

A demonstration of **Clean Architecture** principles applied to a **Retrieval Augmented Generation (RAG)** system in Python. This project showcases how to structure a production-quality application with complete separation of concerns across four distinct layers.

## What is This Project?

This is an educational demo showing how a simple RAG application can be structured using Clean Architecture. RAG combines document retrieval with LLM-based text generation to answer questions based on indexed documents.

**Key Features:**
- Full Clean Architecture implementation with Domain, Application, Infrastructure, and Presentation layers
- RAG pipeline: embed documents → vector similarity search → context-augmented LLM generation
- Multiple LLM provider support (Google Gemini, OpenAI)
- Fake services for testing without API costs
- Both REST API and CLI interfaces sharing the same business logic

## Project Structure

```
src/
├── domain/                 # Core business logic (innermost layer)
│   ├── entities/           # Document entity
│   ├── value_objects/      # Query, RetrievalResult
│   ├── repositories/       # Repository interfaces
│   ├── services/           # Service interfaces (EmbeddingService, LLMService)
│   └── exceptions/         # Domain exceptions
│
├── application/            # Use cases and orchestration
│   ├── use_cases/          # IndexDocument, QueryDocuments
│   ├── dtos/               # Data transfer objects
│   └── mappers/            # Entity ↔ DTO transformations
│
├── infrastructure/         # External integrations
│   ├── configuration/      # Settings from environment
│   ├── persistence/        # In-memory repository implementation
│   ├── external_services/  # Gemini, OpenAI, Fake service implementations
│   └── dependencies.py     # Dependency injection container
│
├── presentation/           # User interfaces
│   ├── api/                # FastAPI routes, controllers, schemas
│   └── cli/                # Command-line interface
│
└── main.py                 # Application entry point
```

## Installation

### Prerequisites
- Python 3.10+

### Setup

```bash
# Clone the repository
git clone https://github.com/DimitrisKoutselis/clean_architecture_demo
cd clean_architecture_demo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file from template
cp .env.example .env
```

### Configuration

Edit `.env` to configure the application:

```env
# API Keys (required for real services)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
GEMINI_LLM_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Feature Flags
USE_FAKE_SERVICES=true  # Set to false to use real API services
```

## Usage

### Running the API Server

```bash
# Start the FastAPI server
python src/main.py

# Or with Uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: `http://localhost:8000`
- OpenAPI Docs: `http://localhost:8000/docs`

### API Endpoints

#### Index a Document

```bash
curl -X POST http://localhost:8000/documents/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Clean Architecture separates concerns into layers with dependencies pointing inward.",
    "metadata": {"source": "architecture-book"}
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "Clean Architecture separates concerns into layers...",
  "metadata": {"source": "architecture-book"}
}
```

#### Query Documents (RAG)

```bash
curl -X POST http://localhost:8000/documents/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is clean architecture?",
    "top_k": 3
  }'
```

Response:
```json
{
  "answer": "Based on the provided context, clean architecture is...",
  "sources": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

### CLI Usage

```bash
# Index a document
python -m src.presentation.cli.cli index "Your document content here"

# Query documents
python -m src.presentation.cli.cli query "What is RAG?" --top-k 3
```

## Architecture Overview

### The Dependency Rule

Dependencies only point **inward**. Outer layers depend on inner layers, never the reverse.

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION                             │
│              (FastAPI Routes, CLI)                          │
├─────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE                           │
│    (Repositories, External Services, Configuration)         │
├─────────────────────────────────────────────────────────────┤
│                    APPLICATION                              │
│              (Use Cases, DTOs, Mappers)                     │
├─────────────────────────────────────────────────────────────┤
│                      DOMAIN                                 │
│     (Entities, Value Objects, Interfaces, Exceptions)       │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline

```
User Question
    ↓
[Embed Query] → Vector representation
    ↓
[Similarity Search] → Find relevant documents
    ↓
[Build Context] → Concatenate document contents
    ↓
[Generate Answer] → LLM generates response with context
    ↓
Answer + Source IDs
```

### Key Design Patterns

- **Dependency Injection**: All dependencies injected via constructors
- **Repository Pattern**: Abstract data access behind interfaces
- **Use Case Pattern**: Single business workflow per class
- **DTO Pattern**: Isolate domain entities from external layers
- **Factory Pattern**: Centralized dependency wiring

## Testing

Use fake services for fast tests without API costs:

```env
USE_FAKE_SERVICES=true
```

The fake services provide:
- **FakeEmbeddingService**: Deterministic hash-based embeddings
- **FakeLLMService**: Mock LLM responses

## Extending the Project

### Add a New LLM Provider

1. Create a new service in `src/infrastructure/external_services/`:

```python
from src.domain.services.llm_service import LLMService

class AnthropicLLMService(LLMService):
    def generate(self, prompt: str, context: str) -> str:
        # Implementation here
        pass
```

2. Update `src/infrastructure/dependencies.py` to wire the new service

### Add a New Storage Backend

1. Create a new repository in `src/infrastructure/persistence/repositories/`:

```python
from src.domain.repositories.document_repository import DocumentRepository

class PostgresDocumentRepository(DocumentRepository):
    # Implement all abstract methods
    pass
```

2. Update the dependency container to use the new repository

## Technology Stack

- **Framework**: FastAPI 0.100.0+
- **Server**: Uvicorn 0.23.0+
- **Validation**: Pydantic 2.0.0+
- **LLM Integration**: Google Generative AI, OpenAI
- **Configuration**: python-dotenv 1.0.0+
