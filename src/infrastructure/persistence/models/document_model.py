from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentModel:
    id: str
    content: str
    embedding: list[float]
    metadata: Optional[dict] = None
