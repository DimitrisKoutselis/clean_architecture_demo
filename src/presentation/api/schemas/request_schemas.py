from pydantic import BaseModel
from typing import Optional


class IndexDocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
