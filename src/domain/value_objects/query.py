from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    text: str
    top_k: int = 3
