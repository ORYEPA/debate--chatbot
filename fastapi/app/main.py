import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.docs import configure_docs
from app.api.v1.endpoints import router as api_v1
from app.config import LLM_BASE_URL

def create_app() -> FastAPI:
    app = FastAPI(
        title="Debate Chatbot",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins.split(",") if allow_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )

    @app.on_event("startup")
    def _warmup() -> None:
        """
        Non-blocking warmup:
        - If LLM_BASE_URL is set and points to an Ollama-compatible endpoint,
          we try hitting /api/tags. If it fails, we silently ignore it.
        - If LLM_BASE_URL is empty or it's a non-Ollama provider, we skip.
        """
        base = (LLM_BASE_URL or "").rstrip("/")
        if not base:
            return
        try:
            requests.get(f"{base}/api/tags", timeout=3)
        except Exception:
            pass  
    configure_docs(app)
    app.include_router(api_v1, prefix="/api/v1")
    return app

app = create_app()
