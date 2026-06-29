$env:PYTHONPATH = "src;$env:PYTHONPATH"
$uvicorn = ".\venv\Scripts\uvicorn.exe"

Write-Host "Starting Kafka and Zookeeper..."
docker compose up -d

Write-Host "Waiting 15 seconds for Kafka to be ready..."
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "==========================================================="
Write-Host " Dashboard ready. Open in your browser:"
Write-Host "   Live Dashboard   ->  http://127.0.0.1:8000"
Write-Host "   Control Center   ->  http://127.0.0.1:8000/master"
Write-Host ""
Write-Host " Use the START / STOP buttons on /master to control"
Write-Host " the synthetic data stream."
Write-Host ""
Write-Host " Press Ctrl+C to shut everything down."
Write-Host "==========================================================="
Write-Host ""

try {
    & $uvicorn src.reporting_api:app --host 0.0.0.0 --port 8000 --reload
} finally {
    Write-Host ""
    Write-Host "Stopping Kafka..."
    docker compose down
    Write-Host "All services stopped."
}
