FROM ollama/ollama:latest
COPY pull-llama3.sh /pull-llama3.sh
RUN chmod +x /pull-llama3.sh
ENTRYPOINT ["/bin/sh","/pull-llama3.sh"]
