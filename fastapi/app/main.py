# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.docs import configure_docs
from app.api.v1.endpoints import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Debate Chatbot",
        version="1.0.0",
        docs_url=None,            
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )

    configure_docs(app)

    app.include_router(api_router)

    return app


app = create_app()
