from src.application.use_cases.index_document import IndexDocumentUseCase
from src.application.use_cases.query_documents import QueryDocumentsUseCase
from src.application.dtos.document_dto import CreateDocumentDTO
from src.application.dtos.query_dto import QueryRequestDTO
from src.presentation.api.schemas.request_schemas import IndexDocumentRequest, QueryRequest
from src.presentation.api.schemas.response_schemas import DocumentResponse, QueryResponse


class DocumentController:
    def __init__(
        self,
        index_use_case: IndexDocumentUseCase,
        query_use_case: QueryDocumentsUseCase
    ):
        self._index_use_case = index_use_case
        self._query_use_case = query_use_case

    def index_document(self, request: IndexDocumentRequest) -> DocumentResponse:
        dto = CreateDocumentDTO(content=request.content, metadata=request.metadata)
        result = self._index_use_case.execute(dto)
        return DocumentResponse(id=result.id, content=result.content, metadata=result.metadata)

    def query(self, request: QueryRequest) -> QueryResponse:
        dto = QueryRequestDTO(question=request.question, top_k=request.top_k)
        result = self._query_use_case.execute(dto)
        return QueryResponse(answer=result.answer, sources=result.sources)
