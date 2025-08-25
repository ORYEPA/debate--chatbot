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
        │              ┌───────────────────────────────┐
        │────────────▶ │  Servidor LLM (Ollama) 	   │
        │              └───────────────────────────────┘
        │
        ▼
  Lógica de conversación (memoria corta, reglas)
```

* **/health**: estado del servicio y base LLM.
* **/ask** (POST): `{ conversation_id, message }` → Responde y mantiene histórico breve (hasta `MAX_HISTORY_PAIRS`).

---

## 1) Requisitos previos

> Puedes ejecutar **sin Docker** (entorno local con Python) o **con Docker Compose** (recomendado para reproducibilidad).

### Sistema

* **Windows 10/11**, **macOS 12+** o **Linux**.
* **Conexión a Internet** si usarás un servidor LLM remoto (p. ej. `OLLAMA_BASE_URL`).

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

# LLM / Ollama
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
```

**Descripción breve:**

* `OLLAMA_BASE_URL`: URL base del servidor LLM (por ej., Ollama local: `http://localhost:11434`).
* `MODEL_NAME`: modelo a usar (p. ej. `llama3:8b`).
* `MAX_HISTORY_PAIRS`: pares (usuario/bot) recientes que se conservan.
* `REPLY_CHAR_LIMIT`, `NUM_PREDICT_CAP`, `NUM_CTX`: controles de tamaño y contexto.
* `KEEP_ALIVE`, `HTTP_TIMEOUT_SECONDS`: parámetros para timeouts/conexiones.

> Si usas **Ollama en Docker** con el `docker-compose.yml` provisto, deja `OLLAMA_BASE_URL=http://ollama:11434` para que el contenedor `api` resuelva el nombre de servicio `ollama`.

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
version: "3.9"
services:
  api:
    build: .
    env_file:
      - .env
    environment:
      PORT: "8080"
      REDIS_URL: "redis://redis:6379"
      OLLAMA_URL: "http://ollama:11434"
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
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

2. Exportar variables (o `.env`).

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

### Tests

```bash
make test
```

---

## 6) Resolución de problemas (FAQ)

* **`could_not_reach_ollama` / timeouts**: verifica `OLLAMA_BASE_URL` y que el servidor LLM esté activo; si usas Docker, `make run` debe levantar `ollama`.
* **`docker compose` no existe**: usa `docker-compose` clásico o actualiza Docker Desktop; el Makefile detecta ambas variantes.
* **Windows y `source`**: usa Git Bash para `make`; en PowerShell, activa el venv con `./.venv/Scripts/Activate.ps1`.
* **Modelo no encontrado**: entra al contenedor `ollama` y ejecuta `ollama pull <MODEL_NAME>`.

---

## 7) Estándares de código

* **Formateo**: puedes usar *black* y *ruff* (opcional).
* **Tests**: *pytest* en `app/tests`.

---

## 8) Seguridad y despliegue

* No expongas la API pública sin autenticación si usas modelos locales.
* Valida límites (`REPLY_CHAR_LIMIT`, `NUM_PREDICT_CAP`).
* Logs: usa `LOG_LEVEL` adecuado en producción.

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
