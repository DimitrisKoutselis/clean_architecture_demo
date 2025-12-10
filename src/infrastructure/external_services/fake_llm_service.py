from src.domain.services.llm_service import LLMService


class FakeLLMService(LLMService):
    def generate(self, prompt: str, context: str) -> str:
        return f"Based on the provided context, here is the answer to '{prompt}': The context mentions: {context[:200]}..."
