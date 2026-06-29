$env:PYTHONPATH = "src;$env:PYTHONPATH"
$python  = ".\venv\Scripts\python.exe"
$uvicorn = ".\venv\Scripts\uvicorn.exe"

Write-Host "Starting Kafka and Zookeeper..."
docker compose up -d

Write-Host "Waiting 15 seconds for Kafka to be ready..."
Start-Sleep -Seconds 15

Write-Host "Starting background services (Producer, Preprocessing, Decision Engine)..."
$jobs = @(
    Start-Job -Name "Producer"        -ScriptBlock { param($p) & $p src/producer.py        } -ArgumentList (Resolve-Path $python)
    Start-Job -Name "Preprocessing"   -ScriptBlock { param($p) & $p src/preprocessing.py   } -ArgumentList (Resolve-Path $python)
    Start-Job -Name "DecisionEngine"  -ScriptBlock { param($p) & $p src/decision_engine.py } -ArgumentList (Resolve-Path $python)
)

Write-Host ""
Write-Host "==========================================================="
Write-Host " Pipeline is Running! Open in your browser:"
Write-Host "   Live Dashboard   ->  http://127.0.0.1:8000"
Write-Host "   Training Metrics ->  http://127.0.0.1:8000/metrics"
Write-Host ""
Write-Host " Press Ctrl+C to stop everything."
Write-Host "==========================================================="
Write-Host ""

try {
    & $uvicorn src.reporting_api:app --host 0.0.0.0 --port 8000 --reload
} finally {
    Write-Host ""
    Write-Host "Shutting down background services..."
    $jobs | Stop-Job
    $jobs | Remove-Job
    Write-Host "Stopping Kafka..."
    docker compose down
    Write-Host "All services stopped."
}
