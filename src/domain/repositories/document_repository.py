from abc import ABC, abstractmethod
from src.domain.entities.document import Document


class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        pass

    @abstractmethod
    def find_by_id(self, document_id: str) -> Document | None:
        pass

    @abstractmethod
    def find_similar(self, embedding: list[float], top_k: int) -> list[Document]:
        pass

    @abstractmethod
    def delete(self, document_id: str) -> None:
        pass
