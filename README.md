# pipeline_fraud_detection

A real-time fraud detection system that streams financial transactions through Apache Kafka, scores them with a 3-model ML ensemble (Random Forest, XGBoost, LightGBM), and displays results on a live web dashboard.

---

## Architecture

```text
data.csv -> producer.py -> [Kafka: raw-transactions]
                                |
                       preprocessing.py  (feature extraction)
                                |
                       [Kafka: processed-transactions]
                                |
                       decision_engine.py  (majority vote: >=2/3 = fraud)
                                |
                       [Kafka: scored-transactions]
                                |
                       reporting_api.py  (FastAPI dashboard on port 8000)
                                |
                       Data/transactions.db  (SQLite history)
```

---

## Requirements

- Python 3.9+
- Docker (for Kafka + Zookeeper)

---

## First-Time Setup

### Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running the Pipeline

> **Docker Desktop must be running** before executing the start script — it launches Kafka and ZooKeeper automatically.

### Linux

```bash
chmod +x start.sh
./start.sh
```

### Mac

```bash
chmod +x start.sh
./start.sh
```

### Windows (PowerShell) — Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

Both scripts:

1. Start Kafka and Zookeeper via Docker Compose
2. Wait 15 seconds for Kafka to be ready
3. Launch Producer, Preprocessing, Decision Engine, and Reporting API in separate processes

---

## Dashboard Pages

| URL                      | Description                     |
| -------------------------| ------------------------------- |
| `http://127.0.0.1:8000/` | Live transaction monitor (auto-refreshes every 1.5s) |
| `http://127.0.0.1:8000/simulate` | Manually submit a transaction for prediction |
| `http://127.0.0.1:8000/history` | Full SQLite transaction log |
| `http://127.0.0.1:8000/metrics` | Offline training metrics and charts |
| `http://127.0.0.1:8000/api-docs` | REST API reference with code samples |
| `http://127.0.0.1:8000/master` | Control center -- trigger full pipeline restart |

---

## Training Models

Train on current dataset only:

**Linux / Mac:**

```bash
PYTHONPATH=src python src/analyze_metrics.py
```

**Windows:**

```powershell
$env:PYTHONPATH="src"; python src\analyze_metrics.py
```

Full restart — merge live SQLite predictions into data.csv, then retrain:

**Linux / Mac:**

```bash
PYTHONPATH=src python src/analyze_metrics.py --absorb-live
```

**Windows:**

```powershell
$env:PYTHONPATH="src"; python src\analyze_metrics.py --absorb-live
```

Models are saved to `Models/` and metrics to `metrics.json`.

---

## Features Used (16 total)

| Feature | Description |
| --- | --- |
| Amount | Raw transaction amount |
| MerchantID | Merchant identifier (1–1000) |
| TransactionType | Encoded: purchase=1, refund=0 |
| Location | City encoded as integer (10 cities) |
| Hour | Hour of day (0–23) |
| DayOfWeek | Day of week (0–6) |
| AmountLog | Log-transformed amount |
| IsNightHour | 1 if Hour between 22:00–05:00 |
| IsWeekend | 1 if Saturday or Sunday |
| IsHighRiskMerchant | 1 if MerchantID ≤ 30 |
| NightHighAmount | IsNightHour × (Amount > 1000) |
| RefundHighAmount | IsRefund × (Amount > 500) |
| RiskyMerchantAmount | IsHighRiskMerchant × Amount |
| WeekendNight | IsWeekend × IsNightHour |
| HighRiskRefund | IsHighRiskMerchant × IsRefund |
| AmountBin | Binned amount category |

---

## Model Performance

Dataset: 100,000 transactions — 91,498 legitimate / 8,502 fraud (≈11:1 ratio).
Test set: 15,000 transactions (13,725 legit / 1,275 fraud).

| Model | Precision | Recall | F1 | AUC-ROC |
| --- | --- | --- | --- | --- |
| XGBoost **(Winner)** | 0.6069 | 0.8016 | 0.6908 | 0.9503 |
| LightGBM | 0.6529 | 0.7302 | 0.6894 | 0.9493 |
| Random Forest | 0.5937 | 0.8275 | 0.6913 | 0.9449 |

Fraud decision uses **majority vote** — at least 2 out of 3 models must flag a transaction as fraud.

---

## Running Tests

```bash
PYTHONPATH=src pytest tests/
```

---

## Project Structure

```text
pipline_fraud_detection/
├── Data/
│   ├── data.csv               # Training dataset (100k transactions)
│   └── transactions.db        # SQLite live prediction history
├── Models/                    # Trained model .pkl files
├── static/                    # Generated chart images
├── src/
│   ├── features.py            # Shared feature extraction logic
│   ├── producer.py            # Kafka producer (streams data.csv)
│   ├── preprocessing.py       # Kafka consumer/producer (feature transform)
│   ├── decision_engine.py     # Kafka consumer (ML scoring + voting)
│   ├── reporting_api.py       # FastAPI dashboard + REST API
│   ├── train_models.py        # Simple standalone model trainer
│   └── analyze_metrics.py     # Full training pipeline with metrics + charts
├── tests/
│   └── test_pipeline.py
├── docker-compose.yml         # Kafka + Zookeeper
├── requirements.txt
├── start.sh                   # Linux/Mac startup script
└── run.ps1                    # Windows PowerShell startup script
```
