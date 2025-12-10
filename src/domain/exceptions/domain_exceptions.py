class DomainException(Exception):
    pass


class DocumentNotFoundException(DomainException):
    def __init__(self, document_id: str):
        super().__init__(f"Document with id '{document_id}' not found")
        self.document_id = document_id


class InvalidQueryException(DomainException):
    def __init__(self, reason: str):
        super().__init__(f"Invalid query: {reason}")
