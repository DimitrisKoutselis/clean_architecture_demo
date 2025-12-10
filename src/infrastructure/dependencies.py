from dataclasses import dataclass

from src.infrastructure.configuration.settings import Settings

from src.infrastructure.persistence.repositories.in_memory_document_repository import InMemoryDocumentRepository

from src.infrastructure.external_services.fake_embedding_service import FakeEmbeddingService
from src.infrastructure.external_services.fake_llm_service import FakeLLMService

from src.infrastructure.external_services.gemini_embedding_service import GeminiEmbeddingService
from src.infrastructure.external_services.gemini_llm_service import GeminiLLMService

from src.application.use_cases.index_document import IndexDocumentUseCase
from src.application.use_cases.query_documents import QueryDocumentsUseCase

from src.presentation.api.controllers.document_controller import DocumentController


@dataclass
class Container:
    """Dependency injection container holding all application dependencies."""
    settings: Settings
    document_controller: DocumentController


def create_container() -> Container:
    """Wire up all dependencies and return the container."""
    settings = Settings.from_env()

    document_repository = InMemoryDocumentRepository()

    if settings.use_fake_services:
        embedding_service = FakeEmbeddingService()
        llm_service = FakeLLMService()
    else:
        embedding_service = GeminiEmbeddingService(settings.gemini_api_key, settings.gemini_embedding_model)
        llm_service = GeminiLLMService(settings.gemini_api_key, settings.gemini_llm_model)

    index_use_case = IndexDocumentUseCase(document_repository, embedding_service)
    query_use_case = QueryDocumentsUseCase(document_repository, embedding_service, llm_service)

    controller = DocumentController(index_use_case, query_use_case)

    return Container(
        settings=settings,
        document_controller=controller,
    )
