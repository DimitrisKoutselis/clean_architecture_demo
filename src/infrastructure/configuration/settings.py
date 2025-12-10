import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""

    # Model names
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Feature flags
    use_fake_services: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gemini_llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
            gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
            openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            use_fake_services=os.getenv("USE_FAKE_SERVICES", "true").lower() == "true"
        )
