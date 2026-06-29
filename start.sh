#!/bin/bash
set -e

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "Starting Kafka and Zookeeper..."
docker compose up -d

echo "Waiting for Kafka to be ready..."
until docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; do
    echo "  Kafka not ready yet — retrying in 3s..."
    sleep 3
done
echo "Kafka is ready!"

echo ""
echo "==========================================================="
echo " Dashboard ready. Open in your browser:"
echo "   Live Dashboard   ->  http://<your-ec2-ip>:8000"
echo "   Control Center   ->  http://<your-ec2-ip>:8000/master"
echo ""
echo " Use the START / STOP buttons on /master to control"
echo " the synthetic data stream."
echo ""
echo " Press Ctrl+C to shut everything down."
echo "==========================================================="
echo ""

cleanup() {
    echo "Stopping services..."
    docker compose down
    echo "All services stopped."
}
trap cleanup INT TERM

uvicorn src.reporting_api:app --host 0.0.0.0 --port 8000 --reload
