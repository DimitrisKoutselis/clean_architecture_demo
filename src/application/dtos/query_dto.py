from dataclasses import dataclass


@dataclass
class QueryRequestDTO:
    question: str
    top_k: int = 3


@dataclass
class QueryResponseDTO:
    answer: str
    sources: list[str]
