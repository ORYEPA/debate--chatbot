import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.docs import configure_docs
from app.api.v1.endpoints import router as api_v1
from app.config import OLLAMA_BASE_URL 

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
    def _warmup():
        """
        Warmup no bloqueante: si hay OLLAMA_BASE_URL, intentamos listar tags.
        Si falla o no hay URL, no pasa nada – el fallback a OpenAI ocurre en runtime.
        """
        base = (OLLAMA_BASE_URL or "").rstrip("/")
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
