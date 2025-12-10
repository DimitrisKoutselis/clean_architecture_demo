import uuid
from src.domain.entities.document import Document
from src.application.dtos.document_dto import CreateDocumentDTO, DocumentResponseDTO


class DocumentMapper:
    @staticmethod
    def to_entity(dto: CreateDocumentDTO) -> Document:
        return Document(
            id=str(uuid.uuid4()),
            content=dto.content,
            metadata=dto.metadata
        )

    @staticmethod
    def to_response_dto(entity: Document) -> DocumentResponseDTO:
        return DocumentResponseDTO(
            id=entity.id,
            content=entity.content,
            metadata=entity.metadata
        )
