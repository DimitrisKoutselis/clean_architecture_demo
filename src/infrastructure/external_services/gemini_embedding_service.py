from src.domain.services.embedding_service import EmbeddingService
import google.generativeai as genai


class GeminiEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self._api_key = api_key
        self._model = model
        genai.configure(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        result = genai.embed_content(model=self._model, content=text)
        return result['embedding']
