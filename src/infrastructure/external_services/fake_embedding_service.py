import hashlib
from src.domain.services.embedding_service import EmbeddingService


class FakeEmbeddingService(EmbeddingService):
    def __init__(self, dimensions: int = 128):
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        hash_bytes = hashlib.sha256(text.lower().encode()).digest()
        values = []
        for i in range(self._dimensions):
            byte_val = hash_bytes[i % len(hash_bytes)]
            values.append((byte_val / 255.0) * 2 - 1)
        return values
