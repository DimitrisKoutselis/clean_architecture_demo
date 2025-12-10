import math
from src.domain.entities.document import Document
from src.domain.repositories.document_repository import DocumentRepository


class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self):
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> None:
        self._documents[document.id] = document

    def find_by_id(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def find_similar(self, embedding: list[float], top_k: int) -> list[Document]:
        scored = []
        for doc in self._documents.values():
            if doc.embedding:
                score = self._cosine_similarity(embedding, doc.embedding)
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def delete(self, document_id: str) -> None:
        self._documents.pop(document_id, None)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
