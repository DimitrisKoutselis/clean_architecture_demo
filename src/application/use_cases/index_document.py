from src.domain.repositories.document_repository import DocumentRepository
from src.domain.services.embedding_service import EmbeddingService
from src.application.dtos.document_dto import CreateDocumentDTO, DocumentResponseDTO
from src.application.mappers.document_mapper import DocumentMapper


class IndexDocumentUseCase:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService
    ):
        self._repository = document_repository
        self._embedding_service = embedding_service

    def execute(self, dto: CreateDocumentDTO) -> DocumentResponseDTO:
        document = DocumentMapper.to_entity(dto)
        document.embedding = self._embedding_service.embed(document.content)
        self._repository.save(document)
        return DocumentMapper.to_response_dto(document)
