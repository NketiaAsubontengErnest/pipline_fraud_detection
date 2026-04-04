Write-Host "Starting Kafka and Zookeeper..."
docker-compose up -d

Write-Host "Waiting 15 seconds for Kafka..."
Start-Sleep -Seconds 15

Write-Host "Starting Producer Service..."
Start-Process powershell -ArgumentList "-NoExit -Command .\venv\Scripts\python.exe src\producer.py"

Write-Host "Starting Preprocessing Service..."
Start-Process powershell -ArgumentList "-NoExit -Command .\venv\Scripts\python.exe src\preprocessing.py"

Write-Host "Starting Decision Engine Service..."
Start-Process powershell -ArgumentList "-NoExit -Command .\venv\Scripts\python.exe src\decision_engine.py"

Write-Host "Starting Reporting API / Dashboard..."
Start-Process powershell -ArgumentList "-NoExit -Command .\venv\Scripts\uvicorn.exe src.reporting_api:app --host 0.0.0.0 --port 8000 --reload"

Write-Host "==========================================================="
Write-Host "Pipeline is Running! Access the dashboards in your browser:"
Write-Host " - Live Transaction Monitor:  http://127.0.0.1:8000"
Write-Host " - Model Training Metrics:    http://127.0.0.1:8000/metrics"
Write-Host "==========================================================="
