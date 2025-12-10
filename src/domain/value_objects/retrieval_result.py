from dataclasses import dataclass
from src.domain.entities.document import Document


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    query_text: str
