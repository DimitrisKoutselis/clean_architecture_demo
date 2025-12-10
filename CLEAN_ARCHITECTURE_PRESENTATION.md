# Clean Architecture
## Building Scalable, Maintainable, and Team-Friendly Software

---

# What is Clean Architecture?

Clean Architecture is a software design philosophy that organizes code into **concentric layers** with strict dependency rules.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION                               │
│   (API Controllers, Routes, CLI, UI)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   INFRASTRUCTURE                        │    │
│  │   (Database, External APIs, Frameworks)                 │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │                 APPLICATION                     │    │    │
│  │  │   (Use Cases, DTOs, Mappers)                    │    │    │
│  │  │  ┌─────────────────────────────────────────┐    │    │    │
│  │  │  │              DOMAIN                     │    │    │    │
│  │  │  │   (Entities, Value Objects, Interfaces) │    │    │    │
│  │  │  └─────────────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

        ⬅️ DEPENDENCIES POINT INWARD (The Dependency Rule)
```

**The Golden Rule:** Source code dependencies can only point **inward**. Inner layers know nothing about outer layers.

---

# The Four Layers Explained

## 1. Domain Layer (The Core)
**What it contains:** Business entities, value objects, repository interfaces, service interfaces, domain exceptions

**Purpose:** Pure business logic with zero framework dependencies

## 2. Application Layer
**What it contains:** Use cases, DTOs (Data Transfer Objects), mappers

**Purpose:** Orchestrates business workflows by coordinating domain objects

## 3. Infrastructure Layer
**What it contains:** Database implementations, external API clients, configuration

**Purpose:** Provides concrete implementations of domain interfaces

## 4. Presentation Layer
**What it contains:** API controllers, routes, request/response schemas, CLI

**Purpose:** Handles user interaction (HTTP, CLI, etc.)

---

# Our Project Structure

```
src/
├── domain/                              # 🎯 Core Business Logic
│   ├── entities/
│   │   └── document.py                 # Document entity
│   ├── value_objects/
│   │   ├── query.py                    # Query value object
│   │   └── retrieval_result.py         # Result value object
│   ├── repositories/
│   │   └── document_repository.py      # Repository interface (ABC)
│   ├── services/
│   │   ├── embedding_service.py        # Embedding interface (ABC)
│   │   └── llm_service.py              # LLM interface (ABC)
│   └── exceptions/
│       └── domain_exceptions.py        # Business exceptions
│
├── application/                         # 📋 Use Case Orchestration
│   ├── use_cases/
│   │   ├── index_document.py           # Index document workflow
│   │   └── query_documents.py          # Query workflow (RAG)
│   ├── dtos/
│   │   ├── document_dto.py             # Document DTOs
│   │   └── query_dto.py                # Query DTOs
│   └── mappers/
│       └── document_mapper.py          # Entity ↔ DTO conversion
│
├── infrastructure/                      # 🔧 Technical Implementations
│   ├── persistence/
│   │   ├── models/document_model.py
│   │   └── repositories/
│   │       └── in_memory_document_repository.py
│   ├── external_services/
│   │   ├── fake_embedding_service.py   # For testing
│   │   ├── fake_llm_service.py
│   │   ├── openai_embedding_service.py # Real OpenAI
│   │   ├── openai_llm_service.py
│   │   ├── gemini_embedding_service.py # Real Gemini
│   │   └── gemini_llm_service.py
│   ├── configuration/settings.py
│   └── dependencies.py                  # Dependency injection
│
└── presentation/                        # 🖥️ User Interfaces
    ├── api/
    │   ├── controllers/document_controller.py
    │   ├── routes/document_routes.py
    │   └── schemas/
    │       ├── request_schemas.py
    │       └── response_schemas.py
    └── cli/cli.py                       # Command-line interface
```

---

# Why Clean Architecture? The Problems It Solves

## ❌ Without Clean Architecture

```python
# Everything mixed together - the "Big Ball of Mud"
@app.post("/documents")
def index_document(request: Request):
    content = request.json["content"]

    # Direct OpenAI call in controller
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding = client.embeddings.create(input=content, model="text-embedding-3-small")

    # Direct database call
    db.execute("INSERT INTO documents VALUES (?, ?, ?)",
               (uuid4(), content, embedding))

    return {"status": "ok"}
```

**Problems:**
- 🔴 Can't test without real OpenAI API (costs money, slow)
- 🔴 Can't switch to Gemini without changing every file
- 🔴 Business logic buried in HTTP handling code
- 🔴 Database changes require modifying controllers
- 🔴 Team members step on each other's toes

---

## ✅ With Clean Architecture

### Domain Layer: Pure Business Concepts

```python
# src/domain/entities/document.py
@dataclass
class Document:
    """Domain entity - represents our core business concept"""
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: Optional[dict] = None
```

```python
# src/domain/repositories/document_repository.py
class DocumentRepository(ABC):
    """Interface - defined in domain, implemented elsewhere"""

    @abstractmethod
    def save(self, document: Document) -> None:
        pass

    @abstractmethod
    def find_similar(self, embedding: list[float], top_k: int) -> list[Document]:
        pass
```

```python
# src/domain/services/embedding_service.py
class EmbeddingService(ABC):
    """Interface for any embedding provider"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass
```

**Key Insight:** The domain defines **what** operations are needed, not **how** they work.

---

### Application Layer: Business Workflows

```python
# src/application/use_cases/index_document.py
class IndexDocumentUseCase:
    """Orchestrates the document indexing workflow"""

    def __init__(
        self,
        document_repository: DocumentRepository,  # Abstraction!
        embedding_service: EmbeddingService       # Abstraction!
    ):
        self._repository = document_repository
        self._embedding_service = embedding_service

    def execute(self, dto: CreateDocumentDTO) -> DocumentResponseDTO:
        # 1. Convert DTO to entity
        document = DocumentMapper.to_entity(dto)

        # 2. Generate embedding (doesn't know if it's OpenAI or Gemini!)
        document.embedding = self._embedding_service.embed(document.content)

        # 3. Save (doesn't know if it's in-memory or PostgreSQL!)
        self._repository.save(document)

        # 4. Return DTO (never expose domain entities)
        return DocumentMapper.to_response_dto(document)
```

**Key Insight:** Use cases depend on **abstractions**, not concrete implementations.

---

### Infrastructure Layer: Pluggable Implementations

```python
# src/infrastructure/external_services/fake_embedding_service.py
class FakeEmbeddingService(EmbeddingService):
    """Fake implementation for testing - no API calls!"""

    def embed(self, text: str) -> list[float]:
        # Deterministic hash-based embeddings for testing
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [(b / 255.0) * 2 - 1 for b in hash_bytes[:128]]
```

```python
# src/infrastructure/external_services/openai_embedding_service.py
class OpenAIEmbeddingService(EmbeddingService):
    """Real OpenAI implementation"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._api_key = api_key
        self._model = model

    def embed(self, text: str) -> list[float]:
        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding
```

```python
# src/infrastructure/external_services/gemini_embedding_service.py
class GeminiEmbeddingService(EmbeddingService):
    """Real Gemini implementation"""

    def embed(self, text: str) -> list[float]:
        result = genai.embed_content(model=self._model, content=text)
        return result['embedding']
```

**Key Insight:** Three implementations, **same interface**. Switch with zero code changes!

---

### Presentation Layer: User Interface

```python
# src/presentation/api/controllers/document_controller.py
class DocumentController:
    """Handles HTTP concerns, delegates to use cases"""

    def __init__(
        self,
        index_use_case: IndexDocumentUseCase,
        query_use_case: QueryDocumentsUseCase
    ):
        self._index_use_case = index_use_case
        self._query_use_case = query_use_case

    def index_document(self, request: IndexDocumentRequest) -> DocumentResponse:
        # Convert HTTP schema to DTO
        dto = CreateDocumentDTO(content=request.content, metadata=request.metadata)

        # Execute use case (all business logic is there)
        result = self._index_use_case.execute(dto)

        # Convert DTO to HTTP response
        return DocumentResponse(id=result.id, content=result.content)
```

**Key Insight:** Controllers are thin - they just translate and delegate.

---

# Dependency Injection: Wiring It All Together

```python
# src/infrastructure/dependencies.py
def create_container() -> Container:
    """The composition root - where all dependencies are wired"""

    settings = Settings.from_env()

    # Choose repository implementation
    document_repository = InMemoryDocumentRepository()
    # Could be: PostgresDocumentRepository(), MongoDocumentRepository()

    # Choose service implementations based on configuration
    if settings.use_fake_services:
        embedding_service = FakeEmbeddingService()
        llm_service = FakeLLMService()
    else:
        embedding_service = GeminiEmbeddingService(settings.gemini_api_key)
        llm_service = GeminiLLMService(settings.gemini_api_key)
        # Or: OpenAIEmbeddingService(), OpenAILLMService()

    # Wire up use cases with their dependencies
    index_use_case = IndexDocumentUseCase(document_repository, embedding_service)
    query_use_case = QueryDocumentsUseCase(
        document_repository, embedding_service, llm_service
    )

    # Wire up controller
    controller = DocumentController(index_use_case, query_use_case)

    return Container(settings=settings, document_controller=controller)
```

**Key Insight:** One place controls all dependency decisions. Change here, change everywhere.

---

# Benefit 1: Effortless Scaling

## Scenario: Switching from In-Memory to PostgreSQL

### Without Clean Architecture
- Hunt through entire codebase
- Modify controllers, services, tests
- Risk breaking unrelated features
- Weeks of work, high risk

### With Clean Architecture

**Step 1:** Create new implementation
```python
# src/infrastructure/persistence/repositories/postgres_document_repository.py
class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, connection_string: str):
        self._db = create_engine(connection_string)

    def save(self, document: Document) -> None:
        # PostgreSQL implementation

    def find_similar(self, embedding: list[float], top_k: int) -> list[Document]:
        # pgvector similarity search
```

**Step 2:** Update one line in dependencies.py
```python
# Before
document_repository = InMemoryDocumentRepository()

# After
document_repository = PostgresDocumentRepository(settings.db_connection_string)
```

**That's it!** All use cases, controllers, and tests continue working unchanged.

---

## Scenario: Adding Anthropic as New LLM Provider

**Step 1:** Create implementation
```python
# src/infrastructure/external_services/anthropic_llm_service.py
class AnthropicLLMService(LLMService):
    def generate(self, prompt: str, context: str) -> str:
        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": f"Context: {context}\n\n{prompt}"}]
        )
        return message.content[0].text
```

**Step 2:** Add to dependency container
```python
elif settings.llm_provider == "anthropic":
    llm_service = AnthropicLLMService(settings.anthropic_api_key)
```

**Zero changes to:**
- ✅ Domain layer
- ✅ Use cases
- ✅ Controllers
- ✅ Existing tests

---

# Benefit 2: Better Teamwork

## Clear Ownership & Parallel Development

```
┌─────────────────────────────────────────────────────────────┐
│ TEAM ORGANIZATION                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Alice (Domain Expert)                                      │
│  └── Works on: domain/entities, domain/value_objects        │
│      Focus: Business rules, entity validation               │
│                                                             │
│  Bob (Backend Developer)                                    │
│  └── Works on: application/use_cases                        │
│      Focus: Business workflows, orchestration               │
│                                                             │
│  Carol (Infrastructure Engineer)                            │
│  └── Works on: infrastructure/persistence, external_services│
│      Focus: Database optimization, API integrations         │
│                                                             │
│  Dave (Frontend/API Developer)                              │
│  └── Works on: presentation/api                             │
│      Focus: REST endpoints, schemas, validation             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

| Without Clean Architecture | With Clean Architecture |
|---------------------------|------------------------|
| "I changed the database, now 15 tests fail" | Changes isolated to one layer |
| "We need to coordinate before every commit" | Teams work independently |
| "Who owns this code?" | Clear boundaries and ownership |
| "My changes broke the API" | Layers communicate through interfaces |

---

## Code Review Benefits

### Reviewing a Use Case Change
```python
# PR: Add document validation to indexing
class IndexDocumentUseCase:
    def execute(self, dto: CreateDocumentDTO) -> DocumentResponseDTO:
        # NEW: Business rule validation
        if len(dto.content) < 10:
            raise InvalidQueryException("Document too short")
```

**Reviewer knows:**
- ✅ This is business logic (Application layer)
- ✅ No database concerns
- ✅ No HTTP concerns
- ✅ Easy to verify business rule correctness

### Reviewing an Infrastructure Change
```python
# PR: Add Redis caching to repository
class CachedDocumentRepository(DocumentRepository):
    def find_by_id(self, document_id: str) -> Document | None:
        cached = self._redis.get(document_id)
        if cached:
            return self._deserialize(cached)
        return self._delegate.find_by_id(document_id)
```

**Reviewer knows:**
- ✅ This is infrastructure (caching)
- ✅ Doesn't affect business logic
- ✅ Interface unchanged
- ✅ Can focus on caching strategy

---

# Benefit 3: Superior Testability

## Unit Testing Use Cases (Fast, No External Dependencies)

```python
# tests/application/test_index_document.py
class TestIndexDocumentUseCase:
    def test_indexes_document_with_embedding(self):
        # Arrange - use fake implementations
        fake_repo = InMemoryDocumentRepository()
        fake_embedder = FakeEmbeddingService()
        use_case = IndexDocumentUseCase(fake_repo, fake_embedder)

        dto = CreateDocumentDTO(content="Test content")

        # Act
        result = use_case.execute(dto)

        # Assert
        assert result.id is not None
        assert result.content == "Test content"

        # Verify document was saved with embedding
        saved_doc = fake_repo.find_by_id(result.id)
        assert saved_doc is not None
        assert saved_doc.embedding is not None
```

**No mocking frameworks needed!** Just inject the fake implementations.

---

## Testing Pyramid with Clean Architecture

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲           Few, slow, expensive
                 ╱──────╲         (Real API, Real DB)
                ╱        ╲
               ╱Integration╲      Some, medium speed
              ╱────────────╲     (Real DB, Fake APIs)
             ╱              ╲
            ╱   Unit Tests   ╲    Many, fast, cheap
           ╱──────────────────╲  (All fakes)
          ╱                    ╲
         ╱  Domain & Use Cases  ╲
        ╱________________________╲

```

### Our Testing Strategy

| Layer | Test Type | Dependencies | Speed |
|-------|-----------|--------------|-------|
| Domain | Unit | None | ⚡ Instant |
| Use Cases | Unit | Fake services | ⚡ Instant |
| Infrastructure | Integration | Real DB/Redis | 🐢 Seconds |
| Controllers | Integration | Fake use cases | ⚡ Fast |
| Full System | E2E | Everything real | 🐌 Minutes |

---

# Data Flow Example: Query Documents (RAG)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. HTTP Request arrives                                         │
│    POST /documents/query                                        │
│    {"question": "What is RAG?", "top_k": 3}                     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Presentation Layer: Route → Controller                       │
│    QueryRequest (Pydantic) → QueryRequestDTO                    │
│    Validates input, converts to application format              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Application Layer: QueryDocumentsUseCase.execute()           │
│                                                                 │
│    a) Embed the question                                        │
│       embedding_service.embed("What is RAG?")                   │
│                     ▼                                           │
│       [0.1, -0.3, 0.8, ...]  (vector representation)            │
│                                                                 │
│    b) Find similar documents                                    │
│       document_repository.find_similar(embedding, top_k=3)      │
│                     ▼                                           │
│       [Document1, Document2, Document3]                         │
│                                                                 │
│    c) Generate answer using context                             │
│       context = "Doc1 content\n\nDoc2 content\n\nDoc3 content"  │
│       llm_service.generate("What is RAG?", context)             │
│                     ▼                                           │
│       "RAG is Retrieval-Augmented Generation, which..."         │
│                                                                 │
│    d) Return QueryResponseDTO(answer, sources)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Presentation Layer: DTO → Response                           │
│    QueryResponseDTO → QueryResponse (Pydantic)                  │
│    {"answer": "RAG is...", "sources": ["id1", "id2", "id3"]}    │
└─────────────────────────────────────────────────────────────────┘
```

---

# The Dependency Rule Visualized

```
                         DEPENDS ON
    ┌────────────────────────────────────────────────────┐
    │                                                    │
    │   PRESENTATION          INFRASTRUCTURE             │
    │   ┌──────────┐          ┌──────────────┐           │
    │   │Controller│          │OpenAI Service│           │
    │   │  Routes  │          │Gemini Service│           │
    │   │ Schemas  │          │  Repository  │           │
    │   └────┬─────┘          └──────┬───────┘           │
    │        │                       │                   │
    │        │    ┌──────────────────┘                   │
    │        │    │                                      │
    │        ▼    ▼                                      │
    │   ┌─────────────────────────────────┐              │
    │   │         APPLICATION             │              │
    │   │    ┌─────────────────┐          │              │
    │   │    │    Use Cases    │          │              │
    │   │    │   DTOs/Mappers  │          │              │
    │   │    └────────┬────────┘          │              │
    │   └─────────────┼───────────────────┘              │
    │                 │                                  │
    │                 ▼                                  │
    │   ┌─────────────────────────────────┐              │
    │   │           DOMAIN                │              │
    │   │    ┌─────────────────┐          │              │
    │   │    │    Entities     │          │              │
    │   │    │ Value Objects   │          │              │
    │   │    │   Interfaces    │ ◀────────┼── Implements |
    │   │    └─────────────────┘          │              │
    │   └─────────────────────────────────┘              │
    │                                                    │
    └────────────────────────────────────────────────────┘

    ⬆️ Outer layers depend on inner layers
    ⬆️ Inner layers NEVER depend on outer layers
    ⬆️ Domain is completely independent
```

---

# What Goes Where? Quick Reference

## Domain Layer
| Component | Purpose | Example from Project |
|-----------|---------|---------------------|
| **Entities** | Core business objects with identity | `Document` with id, content, embedding |
| **Value Objects** | Immutable concepts without identity | `Query`, `RetrievalResult` |
| **Repository Interfaces** | Data access contracts | `DocumentRepository(ABC)` |
| **Service Interfaces** | External capability contracts | `EmbeddingService(ABC)`, `LLMService(ABC)` |
| **Exceptions** | Domain-specific errors | `DocumentNotFoundException` |

## Application Layer
| Component | Purpose | Example from Project |
|-----------|---------|---------------------|
| **Use Cases** | Business workflow orchestration | `IndexDocumentUseCase`, `QueryDocumentsUseCase` |
| **DTOs** | Data transfer between layers | `CreateDocumentDTO`, `QueryResponseDTO` |
| **Mappers** | Entity ↔ DTO conversion | `DocumentMapper.to_entity()` |

## Infrastructure Layer
| Component | Purpose | Example from Project |
|-----------|---------|---------------------|
| **Repository Impl** | Concrete data access | `InMemoryDocumentRepository` |
| **External Services** | Third-party integrations | `OpenAIEmbeddingService`, `GeminiLLMService` |
| **Configuration** | Environment/settings | `Settings.from_env()` |
| **DI Container** | Dependency wiring | `create_container()` |

## Presentation Layer
| Component | Purpose | Example from Project |
|-----------|---------|---------------------|
| **Controllers** | Request handling & delegation | `DocumentController` |
| **Routes** | Endpoint definitions | `create_document_router()` |
| **Schemas** | Request/response validation | `IndexDocumentRequest`, `QueryResponse` |
| **CLI** | Command-line interface | `CLI.run()` |

---

# Common Mistakes to Avoid

## ❌ Mistake 1: Leaking Domain Entities

```python
# BAD - Exposing domain entity in API response
@router.get("/documents/{id}")
def get_document(id: str) -> Document:  # ❌ Returns entity!
    return repository.find_by_id(id)
```

```python
# GOOD - Use DTOs
@router.get("/documents/{id}")
def get_document(id: str) -> DocumentResponse:  # ✅ Returns DTO
    entity = repository.find_by_id(id)
    return DocumentMapper.to_response_dto(entity)
```

## ❌ Mistake 2: Business Logic in Controllers

```python
# BAD - Business logic in controller
def index_document(self, request):
    if len(request.content) < 10:  # ❌ Business rule here
        raise HTTPException(400, "Too short")
    embedding = openai.embed(request.content)  # ❌ Direct service call
```

```python
# GOOD - Delegate to use case
def index_document(self, request):
    dto = CreateDocumentDTO(content=request.content)
    return self._index_use_case.execute(dto)  # ✅ Use case handles rules
```

## ❌ Mistake 3: Infrastructure in Domain

```python
# BAD - Domain depends on framework
from sqlalchemy import Column, String
class Document(Base):  # ❌ SQLAlchemy in domain!
    __tablename__ = 'documents'
```

```python
# GOOD - Pure domain, separate ORM model in infrastructure
@dataclass
class Document:  # ✅ Pure Python
    id: str
    content: str
```

---

# Summary: The Value of Clean Architecture

## 🚀 **Scalability**
- Swap implementations without touching business logic
- Add new providers (OpenAI → Gemini → Anthropic) easily
- Scale database (In-Memory → PostgreSQL → Distributed) with one line change

## 👥 **Team Collaboration**
- Clear ownership boundaries
- Parallel development without conflicts
- Easier code reviews with focused changes
- New team members understand structure quickly

## 🧪 **Testability**
- Unit test business logic without external services
- Fast test execution with fake implementations
- Isolated testing of each layer
- Confidence to refactor

## 📦 **Maintainability**
- Changes are localized to one layer
- Business rules survive technology changes
- Framework updates don't break business logic
- Technical debt is contained

## 🔄 **Flexibility**
- Multiple interfaces (API + CLI) share same logic
- Easy to add new features
- Technology decisions can be deferred
- Future-proof architecture

---

# Getting Started Checklist

1. **Define your domain first**
   - What are your core business entities?
   - What operations do you need?
   - Define interfaces for external dependencies

2. **Create use cases for each workflow**
   - One use case per business operation
   - Depend only on abstractions
   - Use DTOs for input/output

3. **Implement infrastructure separately**
   - Start with fake/simple implementations
   - Swap to real implementations when ready
   - Keep all framework code here

4. **Build thin presentation layer**
   - Controllers just convert and delegate
   - Validate input with schemas
   - Never put business logic here

5. **Wire dependencies in one place**
   - Create a composition root
   - Configuration-driven implementation selection
   - Easy to switch for testing

---

# Questions?

### Resources
- [Clean Architecture by Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Hexagonal Architecture by Alistair Cockburn](https://alistair.cockburn.us/hexagonal-architecture/)
- This project: Your working reference implementation!

### Project Structure Recap
```
src/
├── domain/           → Business rules (innermost, no dependencies)
├── application/      → Use cases (depends on domain)
├── infrastructure/   → Implementations (depends on domain interfaces)
└── presentation/     → UI/API (depends on application)
```

**Remember:** Dependencies always point inward! 🎯
