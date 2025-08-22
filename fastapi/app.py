import os
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@app.get("/")
def home():
    return {"hello": "World"}

@app.get("/health")
def health():
    return {"status": "ok", "ollama_base_url": OLLAMA_BASE_URL}

@app.get("/ask")
def ask(prompt: str):
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"prompt": prompt, "stream": False, "model": "llama3"},
            timeout=240,
        )
        return JSONResponse(status_code=r.status_code, content=r.json())
    except requests.RequestException as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": "could_not_reach_ollama",
                "detail": str(e),
                "base_url": OLLAMA_BASE_URL,
            },
        )
    except ValueError:
        # respuesta no-JSON
        return JSONResponse(
            status_code=502,
            content={
                "error": "invalid_ollama_response",
                "base_url": OLLAMA_BASE_URL,
            },
        )
