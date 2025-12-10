from src.domain.services.llm_service import LLMService
import google.generativeai as genai


class GeminiLLMService(LLMService):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._api_key = api_key
        self._model = model
        genai.configure(api_key=api_key)

    def generate(self, prompt: str, context: str) -> str:
        model = genai.GenerativeModel(self._model)
        full_prompt = f"Answer the user's question using this context:\n\n{context}\n\nQuestion: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text
