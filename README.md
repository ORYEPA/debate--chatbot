## Debate Chatbot

Servicio API que mantiene conversaciones de debate. Define un **topic** en el primer mensaje (sin `conversation_id`) y, a partir de ahí, **se mantiene en la postura inicial** para convencer a la otra parte.

📄 **Documentación interactiva del API (Swagger UI):** [https://debate-api.fly.dev/docs](https://debate-api.fly.dev/docs)

---

### 🧱 Arquitectura (alto nivel)

```
Cliente (curl / App / Postman)
        │
        ▼
   FastAPI (app)
        │                       ┌───────────────────────────────┐
        │───────────────▶ (1) ▶ │  Servidor LLM (Ollama)        │
        │                       └───────────────────────────────┘
        │                                     │ fallo/timeout
        │                                     ▼
        │                       ┌───────────────────────────────┐
        │───────────────▶ (2) ▶ │  Proveedor LLM (OpenAI)       │
        │                       └───────────────────────────────┘
        ▼
  Lógica de conversación (memoria corta, reglas)
```

> **Tolerancia a fallas:** El backend intenta primero **Ollama**. Si no responde (p. ej. fuera de servicio o timeout), **hace failover automático a OpenAI** siempre que exista `OPENAI_API_KEY` configurada. El endpoint **/health** indica la base LLM preferida y cuál está activa.

* **/health**: estado del servicio y base LLM.
* **/ask** (POST): `{ conversation_id, message }` 

---

## 1) Requisitos previos

> Puedes ejecutar **sin Docker** (entorno local con Python) o **con Docker Compose** (recomendado para reproducibilidad).

### Sistema

* **Windows 10/11**, **macOS 12+** o **Linux**.
* **Conexión a Internet** si usarás un servidor LLM remoto (p. ej. `OLLAMA_BASE_URL`) o **OpenAI** como fallback.

### Herramientas

* **make** (GNU Make).

  * Windows: instala **Git for Windows** (trae Git Bash) o `choco install make` / `winget install GnuWin32.Make`.
  * macOS: `xcode-select --install` o `brew install make`.
  * Linux (Debian/Ubuntu): `sudo apt-get install make`.
* **Docker** + **Docker Compose v2** (subcomando `docker compose`).

  * Windows/macOS: **Docker Desktop**.
  * Linux: `sudo apt-get install docker.io` y luego seguir la guía oficial para habilitar `docker compose`.
* **Python 3.11+** (solo para ejecución local sin Docker).

  * Windows: `winget install Python.Python.3.11` o desde python.org.
  * macOS (brew): `brew install python@3.11`.
  * Linux: `sudo apt-get install python3 python3-venv python3-pip`.

> El **Makefile** verifica dependencias mínimas y, si faltan, te imprime instrucciones.

---

## 2) Variables de entorno

Crea un archivo **`.env`** en la raíz del repo (o usa variables del sistema). Ejemplo:

```env
# App
APP_NAME=debate-chatbot
APP_ENV=development            # development | production
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info                 # debug | info | warning | error
ENABLE_CORS=true

# LLM / Ollama (proveedor primario)
OLLAMA_BASE_URL=http://ollama:11434  # Apunta a tu servidor Ollama o proveedor LLM
MODEL_NAME=llama3:8b
HTTP_TIMEOUT_SECONDS=60
KEEP_ALIVE=5m

# Conversación
MAX_HISTORY_PAIRS=5
REPLY_CHAR_LIMIT=1200
NUM_PREDICT_CAP=350
NUM_CTX=8192

# Reglas universales (opcional; JSON o texto)
UNIVERSAL_RULES=Keep responses on topic and persuasive.

# OpenAI (fallback automático si Ollama falla)
# Si OPENAI_API_KEY está configurada, el backend puede cambiar a OpenAI cuando Ollama no esté disponible.
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=60
```

**Descripción breve:**

* `OLLAMA_BASE_URL`: URL base del servidor LLM (p. ej., Ollama local: `http://localhost:11434`).
* `MODEL_NAME`: modelo a usar con Ollama (p. ej. `llama3:8b`).
* `OPENAI_API_KEY`: clave de OpenAI para habilitar el **fallback**.
* `OPENAI_BASE_URL`: base URL de OpenAI; por defecto `https://api.openai.com/v1` (usa tu endpoint si es Azure OpenAI compatible).
* `OPENAI_MODEL`: modelo de OpenAI a usar como fallback (ej. `gpt-4o-mini`).
* `MAX_HISTORY_PAIRS`, `REPLY_CHAR_LIMIT`, `NUM_PREDICT_CAP`, `NUM_CTX`: controles de tamaño y contexto.
* `KEEP_ALIVE`, `HTTP_TIMEOUT_SECONDS` / `OPENAI_TIMEOUT_SECONDS`: parámetros para timeouts/conexiones.

> **Orden de preferencia:** por defecto se intenta **Ollama**. Si hay **timeout** o **conexión rechazada**, se usa **OpenAI** (si `OPENAI_API_KEY` está presente). Esto es transparente para el cliente.

---

## 3) Ejecución con Docker (recomendada)

### 3.1 Estructura esperada

```
.
├─ app/
│  ├─ main.py         # FastAPI
│  ├─ config.py       # Lee env vars
│  ├─ models/ ...
│  └─ tests/          # Pytest
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ Makefile
├─ .env               # tus variables
└─ README.md
```

### 3.2 `docker-compose.yml` (referencia)

> El repositorio ya debe incluirlo; en caso de necesitar uno mínimo, este ejemplo funciona:

```yaml
version: "3.8"

services:
  ollama:
    build:
      context: ./ollama
      dockerfile: Dockerfile
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - chatbot-net
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - MODEL_NAME=${MODEL_NAME:-llama3.2:1b}
    restart: unless-stopped
    # Opcional: healthcheck simple (no bloquea el arranque del API)
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 3s
      retries: 10

  web:
    build: ./fastapi
    container_name: debate-api
    environment:
      # ---- Redis / perfil / modelo ----
      - REDIS_URL=redis://redis:6379/0
      - PROFILE_DEFAULT=${PROFILE_DEFAULT:-smart_shy}
      - MODEL_NAME=${MODEL_NAME:-llama3.2:1b}
      - DOCS_VERSION=${DOCS_VERSION:-dev}
      - OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-10m}
      - NUM_CTX=${NUM_CTX:-512}
      - HTTP_TIMEOUT_SECONDS=${HTTP_TIMEOUT_SECONDS:-45}
      - REPLY_CHAR_LIMIT=${REPLY_CHAR_LIMIT:-550}
      - CORS_ALLOW_ORIGINS=${CORS_ALLOW_ORIGINS:-*}

      # ---- Proveedor LLM primario: Ollama ----
      # Si quieres forzar OpenAI, deja esto vacío en tu .env: OLLAMA_BASE_URL=
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://ollama:11434}

      # ---- Fallback a OpenAI ----
      - PROVIDER_PREFERENCE=${PROVIDER_PREFERENCE:-ollama_first}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

    depends_on:
      - ollama
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./fastapi:/app
    networks:
      - chatbot-net

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: ["redis-server","--appendonly","yes"]
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD","redis-cli","ping"]
      interval: 5s
      timeout: 3s
      retries: 20
    networks:
      - chatbot-net

networks:
  chatbot-net:
    driver: bridge

volumes:
  ollama_models:
    driver: local
  redis_data:
    driver: local
```

> Para usar **modelo local**, entra al contenedor `ollama` y ejecuta `ollama pull llama3:8b`. El Makefile incluye un atajo.

### 3.3 Comandos principales (Makefile)

* **`make`** — lista de comandos disponibles.
* **`make run`** — levanta API + dependencias con Docker.
* **`make down`** — apaga servicios.
* **`make clean`** — teardown completo (volúmenes e imágenes locales).

> También tienes `make install` (setup local) y `make test` (pytest).

---

## 4) Ejecución local (sin Docker)

1. Crear entorno:

```bash
make install
```

> En Windows (PowerShell) manualmente:
>
> ```pwsh
> py -3 -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install -r requirements.txt
> ```

2. Exportar variables (o `.env`). Si quieres **fallback** a OpenAI, asegúrate de establecer `OPENAI_API_KEY`.

3. Correr servidor:

```bash
. .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5) Endpoints y pruebas rápidas

### Salud

```bash
curl -s http://localhost:8000/health
# Devuelve también información sobre el proveedor activo y la base preferida.
```

### Conversación

```bash
# Primer mensaje: define topic y postura
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id": null, "message": "Hola, ¿de qué vamos a debatir?"}' | jq

# Respuesta subsiguiente: reutiliza conversation_id devuelto
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id": "<id devuelto>", "message": "No estoy de acuerdo por X razón"}' | jq
```

**Probar fallback a OpenAI**

1) Configura `OPENAI_API_KEY` (y opcionalmente `OPENAI_MODEL`).  
2) Detén Ollama o apunta `OLLAMA_BASE_URL` a un host no disponible.  
3) Realiza una petición a `/ask`; el servicio responderá usando OpenAI.  
4) `/health` debe reflejar que el proveedor activo es OpenAI.

---

## 6) Resolución de problemas (FAQ)

* **`could_not_reach_ollama` / timeouts**: verifica `OLLAMA_BASE_URL` y que el servidor LLM esté activo; si además configuraste `OPENAI_API_KEY`, el backend hará **fallback** a **OpenAI** de forma automática.
* **`docker compose` no existe**: usa `docker-compose` clásico o actualiza Docker Desktop; el Makefile detecta ambas variantes.
* **Windows y `source`**: usa Git Bash para `make`; en PowerShell, activa el venv con `./.venv/Scripts/Activate.ps1`.
* **Modelo no encontrado (Ollama)**: entra al contenedor `ollama` y ejecuta `ollama pull <MODEL_NAME>`.
* **Costos y cuotas (OpenAI)**: en producción, controla `REPLY_CHAR_LIMIT` y `NUM_PREDICT_CAP`. No expongas tu API sin autenticación si usas proveedores de pago.

---

## 7) Estándares de código

* **Formateo**: puedes usar *black* y *ruff* (opcional).
* **Tests**: *pytest* en `app/tests`.

---

## 8) Seguridad y despliegue

* No expongas la API pública sin autenticación si usas modelos locales o proveedores de pago.
* Valida límites (`REPLY_CHAR_LIMIT`, `NUM_PREDICT_CAP`).
* Logs: usa `LOG_LEVEL` adecuado en producción.
* Gestiona las claves (`OPENAI_API_KEY`) con un **secret manager** en tu plataforma de despliegue.

---

# Makefile

```makefile
SHELL := /bin/sh

.PHONY: help
help: 
	@echo "make help        - lista de comandos"
	@echo "make install     - instala requisitos locales (Python 3.11+, pip)"
	@echo "make test        - ejecuta tests (local)"
	@echo "make run         - levanta docker compose (api + redis + ollama)"
	@echo "make down        - baja servicios"
	@echo "make clean       - baja y borra volúmenes"

.PHONY: install
install:
	@python --version || (echo "❗ Instala Python 3.11+ primero"; exit 1)
	@pip --version || (echo "❗ Instala pip primero"; exit 1)
	@pip install -r fastapi/requirements.txt

.PHONY: test
test:
	@python -m pytest -q fastapi/tests

.PHONY: run
run:
	docker compose up -d --build

.PHONY: down
down:
	docker compose down

.PHONY: clean
clean:
	docker compose down -v --remove-orphans
```
