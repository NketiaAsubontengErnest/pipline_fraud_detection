FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/     ./src/
COPY static/  ./static/
COPY metrics.json ./

# Ensure Data/ and Models/archive/ directories exist at runtime
RUN mkdir -p Data Models/archive

ENV PYTHONPATH=/app/src

# When running inside Docker Compose alongside the kafka service,
# set KAFKA_BROKER=kafka:9092 via docker-compose environment or .env.
# When running standalone against a host Kafka, use localhost:9092.
ENV KAFKA_BROKER=localhost:9092
ENV DB_PATH=Data/transactions.db

EXPOSE 8000

# Default: run the FastAPI dashboard.
# Override CMD to run other services, e.g.:
#   docker run ... python src/producer.py
#   docker run ... python src/preprocessing.py
#   docker run ... python src/decision_engine.py
CMD ["uvicorn", "reporting_api:app", "--host", "0.0.0.0", "--port", "8000"]
