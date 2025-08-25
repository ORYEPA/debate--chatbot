set -e

PRIMARY="${MODEL_NAME:-llama3.2:1b}"
CANDIDATES="$PRIMARY llama3.2:1b llama3.1:8b"

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
SKIP_PULL="${SKIP_PULL:-0}"

echo "[ollama] starting server on $OLLAMA_HOST ..."
ollama serve &
SERVER_PID=$!
trap "echo '[ollama] stopping...'; kill -TERM $SERVER_PID; wait $SERVER_PID" INT TERM

echo "[ollama] waiting readiness..."
i=0
while [ $i -lt 60 ]; do
  if ollama list >/dev/null 2>&1; then break; fi
  i=$((i+1)); sleep 1
done

if [ "$SKIP_PULL" = "1" ]; then
  echo "[ollama] SKIP_PULL=1 -> no pull inicial"
  wait "$SERVER_PID"
  exit $?
fi

pulled=0
for tag in $CANDIDATES; do
  [ -z "$tag" ] && continue
  echo "[ollama] trying to pull: $tag"
  n=0
  while [ $n -lt 2 ]; do
    if ollama pull "$tag"; then pulled=1; echo "[ollama] model ready: $tag"; break; fi
    n=$((n+1)); echo "[ollama] pull failed (retry $n)"; sleep 2
  done
  [ $pulled -eq 1 ] && break
done

[ $pulled -ne 1 ] && echo "[ollama] WARNING: no model pulled; server sigue arriba"

echo "[ollama] server ready"
wait "$SERVER_PID"
