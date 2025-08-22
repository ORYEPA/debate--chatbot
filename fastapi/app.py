import os, requests 
from fastapi import FastAPI, Response


app = FastAPI()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

@app.get("/")
def home():
    return {"hello":"World"}

@app.get("/ask")
def ask(prompt: str):
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "prompt": prompt, 
            "stream": False, 
            "model": "llama3"},
        timeout=120
    )
    r.raise_for_status()
    return Response(content=r.text, media_type="application/json")
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
        return JSONResponse(
            status_code=502,
            content={
                "error": "invalid_ollama_response",
                "base_url": OLLAMA_BASE_URL,
            },
        )