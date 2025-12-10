from dataclasses import dataclass
from typing import Optional


@dataclass
class CreateDocumentDTO:
    content: str
    metadata: Optional[dict] = None


@dataclass
class DocumentResponseDTO:
    id: str
    content: str
    metadata: Optional[dict] = None
