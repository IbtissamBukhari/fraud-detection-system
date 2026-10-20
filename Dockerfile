FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# libgomp is required by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY frontend ./frontend
COPY models ./models
RUN mkdir -p /app/data

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Render and most hosts pass the port in $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
