FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[llm,monkdb,api]"
EXPOSE 8000
