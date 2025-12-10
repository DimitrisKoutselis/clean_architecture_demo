from src.domain.repositories.document_repository import DocumentRepository
from src.domain.services.embedding_service import EmbeddingService
from src.domain.services.llm_service import LLMService
from src.application.dtos.query_dto import QueryRequestDTO, QueryResponseDTO


class QueryDocumentsUseCase:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
        llm_service: LLMService
    ):
        self._repository = document_repository
        self._embedding_service = embedding_service
        self._llm_service = llm_service

    def execute(self, dto: QueryRequestDTO) -> QueryResponseDTO:
        query_embedding = self._embedding_service.embed(dto.question)
        similar_docs = self._repository.find_similar(query_embedding, dto.top_k)

        context = "\n\n".join([doc.content for doc in similar_docs])
        answer = self._llm_service.generate(dto.question, context)
        sources = [doc.id for doc in similar_docs]

        return QueryResponseDTO(answer=answer, sources=sources)
