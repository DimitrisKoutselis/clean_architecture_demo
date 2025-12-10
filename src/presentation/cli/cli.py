import argparse
from src.application.use_cases.index_document import IndexDocumentUseCase
from src.application.use_cases.query_documents import QueryDocumentsUseCase
from src.application.dtos.document_dto import CreateDocumentDTO
from src.application.dtos.query_dto import QueryRequestDTO


class CLI:
    def __init__(
        self,
        index_use_case: IndexDocumentUseCase,
        query_use_case: QueryDocumentsUseCase
    ):
        self._index_use_case = index_use_case
        self._query_use_case = query_use_case

    def run(self):
        parser = argparse.ArgumentParser(description="RAG System CLI")
        subparsers = parser.add_subparsers(dest="command")

        index_parser = subparsers.add_parser("index", help="Index a document")
        index_parser.add_argument("content", help="Document content")

        query_parser = subparsers.add_parser("query", help="Query documents")
        query_parser.add_argument("question", help="Question to ask")
        query_parser.add_argument("--top-k", type=int, default=3, help="Number of results")

        args = parser.parse_args()

        if args.command == "index":
            result = self._index_use_case.execute(CreateDocumentDTO(content=args.content))
            print(f"Indexed document with ID: {result.id}")

        elif args.command == "query":
            result = self._query_use_case.execute(QueryRequestDTO(question=args.question, top_k=args.top_k))
            print(f"Answer: {result.answer}")
            print(f"Sources: {result.sources}")

        else:
            parser.print_help()
