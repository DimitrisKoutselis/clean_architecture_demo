from fastapi import APIRouter, Depends
from src.presentation.api.schemas.request_schemas import IndexDocumentRequest, QueryRequest
from src.presentation.api.schemas.response_schemas import DocumentResponse, QueryResponse
from src.presentation.api.controllers.document_controller import DocumentController


def create_document_router(controller: DocumentController) -> APIRouter:
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post("/", response_model=DocumentResponse)
    def index_document(request: IndexDocumentRequest):
        return controller.index_document(request)

    @router.post("/query", response_model=QueryResponse)
    def query_documents(request: QueryRequest):
        return controller.query(request)

    return router
