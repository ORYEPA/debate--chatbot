set -e

PRIMARY="${MODEL_NAME:-llama3.2:1b}"
CANDIDATES="$PRIMARY,llama3.2:1b,llama3.1:8b"

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"

echo "[ollama] starting server on $OLLAMA_HOST ..."
ollama serve &
SERVER_PID=$!

trap "kill -TERM $SERVER_PID; wait $SERVER_PID" INT TERM

echo "[ollama] waiting for readiness..."
i=0
while [ $i -lt 60 ]; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  i=$((i+1))
  sleep 1
done

pulled=0
IFS=',' 
for tag in $CANDIDATES; do
  echo "[ollama] trying to pull: $tag"
  if ollama pull "$tag"; then
    echo "[ollama] model ready: $tag"
    pulled=1
    break
  else
    echo "[ollama] pull failed for $tag, trying next..."
  fi
done
unset IFS

if [ "$pulled" -ne 1 ]; then
  echo "[ollama] WARNING: no model could be pulled now (network/tag)."
  echo "[ollama] Server stays up; try later: 'ollama pull <model>'"
fi

echo "[ollama] server ready"
wait "$SERVER_PID"
