from pydantic import BaseModel
from typing import Optional


class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
