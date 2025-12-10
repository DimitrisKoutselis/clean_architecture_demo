from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: Optional[dict] = None
