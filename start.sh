#!/bin/bash
set -e

# Make src/ importable so all scripts can resolve 'from features import ...'
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "Starting Kafka and Zookeeper..."
docker compose up -d

echo "Waiting for Kafka to be ready (healthcheck)..."
until docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; do
    echo "  Kafka not ready yet — retrying in 3s..."
    sleep 3
done
echo "Kafka is ready!"

echo "Starting Producer Service..."
python src/producer.py &
PRODUCER_PID=$!

echo "Starting Preprocessing Service..."
python src/preprocessing.py &
PREPROCESS_PID=$!

echo "Starting Decision Engine Service..."
python src/decision_engine.py &
DECISION_PID=$!

echo "Starting Reporting API / Dashboard..."
uvicorn reporting_api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "==========================================================="
echo "Pipeline is Running! Access the dashboards in your browser:"
echo " - Live Transaction Monitor:  http://127.0.0.1:8000"
echo " - Model Training Metrics:    http://127.0.0.1:8000/metrics"
echo "==========================================================="
echo "PIDs: producer=$PRODUCER_PID  preprocessing=$PREPROCESS_PID  decision=$DECISION_PID  api=$API_PID"
echo "Press Ctrl+C to stop all services."

trap "echo 'Stopping all services...'; kill $PRODUCER_PID $PREPROCESS_PID $DECISION_PID $API_PID 2>/dev/null; docker compose down" INT TERM
wait
