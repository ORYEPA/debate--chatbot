FROM ollama/ollama:latest
ENV OLLAMA_HOST=0.0.0.0:11434

COPY pull-llama3.sh /pull-llama3.sh
RUN sed -i 's/\r$//' /pull-llama3.sh && chmod +x /pull-llama3.sh

ENTRYPOINT ["/bin/sh","/pull-llama3.sh"]
