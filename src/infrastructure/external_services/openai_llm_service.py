from src.domain.services.llm_service import LLMService
import openai


class OpenAILLMService(LLMService):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._api_key = api_key
        self._model = model

    def generate(self, prompt: str, context: str) -> str:
        client = openai.OpenAI(api_key=self._api_key)
        messages = [
            {"role": "system", "content": f"Answer the user's question using this context:\n\n{context}"},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(model=self._model, messages=messages)
        return response.choices[0].message.content
