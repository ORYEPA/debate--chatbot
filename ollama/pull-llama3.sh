set -e
ollama serve &           
sleep 3
echo "Pulling llama3 model"
ollama pull llama3 || true
wait                     
