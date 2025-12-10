from fastapi import FastAPI

from src.infrastructure.dependencies import create_container
from src.presentation.api.routes.document_routes import create_document_router


def create_app() -> FastAPI:
    container = create_container()

    app = FastAPI(title="RAG Demo - Clean Architecture")
    app.include_router(create_document_router(container.document_controller))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
