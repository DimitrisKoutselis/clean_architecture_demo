from src.domain.services.embedding_service import EmbeddingService
import openai


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._api_key = api_key
        self._model = model

    def embed(self, text: str) -> list[float]:
        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding
