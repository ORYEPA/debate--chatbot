set -e
export OLLAMA_HOST=0.0.0.0:${PORT:-11434}   
ollama serve &                               
sleep 3
echo "Pulling llama3 model"
ollama pull llama3 || true
wait
