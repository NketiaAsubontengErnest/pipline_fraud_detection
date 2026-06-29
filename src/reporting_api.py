import json
import asyncio
import logging
import os
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel, field_validator

import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

KAFKA_BROKER     = os.getenv('KAFKA_BROKER', 'localhost:9092')
DB_PATH          = os.getenv('DB_PATH', 'Data/transactions.db')
RESTART_API_KEY  = os.getenv('RESTART_API_KEY', '')
RESTART_COOLDOWN = int(os.getenv('RESTART_COOLDOWN', '300'))

RAW_TOPIC    = 'raw-transactions'
SCORED_TOPIC = 'scored-transactions'
MAX_HISTORY  = 100

last_restart_time = 0
_db_lock          = threading.Lock()
_stats_lock       = threading.Lock()
_kafka_producer   = None

# Training progress state
_training_state = {"running": False, "percent": 0, "message": "Idle", "error": None}
_training_lock  = threading.Lock()

# Map log substrings → percentage milestones (in order)
_PROGRESS_MILESTONES = [
    ("Loading Data",                          5),
    ("Extracting features",                  10),
    ("Applying Hybrid Resampling",           20),
    ("Training 3 models",                    28),
    ("Training Random Forest",               35),
    ("Training Extreme Gradient Boosting",   52),
    ("Training Light Gradient Boosting",     68),
    ("Training meta-learner",                80),
    ("Best Individual Model",                87),
    ("Archived",                             90),
    ("Saved Models",                         93),
    ("Saved static",                         96),
    ("All metrics and charts generated",    100),
]

def _run_training_tracked(cmd_args):
    """Run training subprocess, parse its log output to update _training_state."""
    import sys
    with _training_lock:
        _training_state.update({"running": True, "percent": 0, "message": "Starting...", "error": None})
    try:
        proc = subprocess.Popen(
            [sys.executable] + cmd_args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env={**os.environ, "PYTHONPATH": "src"},
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                logger.info("[training] %s", line)
            for keyword, pct in _PROGRESS_MILESTONES:
                if keyword.lower() in line.lower():
                    with _training_lock:
                        if pct > _training_state["percent"]:
                            _training_state["percent"] = pct
                            _training_state["message"] = line
                    break
        proc.wait()
        with _training_lock:
            if proc.returncode == 0:
                _training_state.update({"running": False, "percent": 100, "message": "Training complete!"})
            else:
                _training_state.update({"running": False, "percent": 0, "message": "Training failed.", "error": "Non-zero exit code"})
    except Exception as e:
        with _training_lock:
            _training_state.update({"running": False, "percent": 0, "message": "Error", "error": str(e)})

def _get_producer():
    """Lazy singleton Kafka producer for publishing live API transactions."""
    global _kafka_producer
    if _kafka_producer is None:
        try:
            _kafka_producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except Exception as e:
            logger.warning("Could not connect Kafka producer: %s", e)
    return _kafka_producer

def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS live_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_id TEXT,
                    tx_date TEXT,
                    amount REAL,
                    merchant_id INTEGER,
                    tx_type TEXT,
                    location TEXT,
                    rf_vote INTEGER,
                    xgb_vote INTEGER,
                    lgb_vote INTEGER,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

def _run_training():
    """Run analyze_metrics.py in a subprocess (non-blocking background thread)."""
    import sys
    logger.info("Auto-training triggered...")
    result = subprocess.run(
        [sys.executable, "src/analyze_metrics.py"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    if result.returncode == 0:
        logger.info("Auto-training completed successfully.")
    else:
        logger.error("Auto-training failed:\n%s", result.stderr)

def _watch_data_csv():
    """Watch data.csv for changes and retrain automatically when it is modified."""
    path = os.path.abspath("Data/data.csv")
    last_mtime = os.path.getmtime(path) if os.path.exists(path) else None
    logger.info("Watching %s for changes...", path)
    while True:
        time.sleep(30)
        try:
            mtime = os.path.getmtime(path)
            if last_mtime is not None and mtime != last_mtime:
                logger.info("data.csv changed — triggering auto-retrain...")
                _run_training()
            last_mtime = mtime
        except FileNotFoundError:
            pass

def _daemon(target, *args):
    """Start target as a daemon thread — killed automatically on process exit."""
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-recreate metrics.json if it is missing or empty
    if not os.path.exists("metrics.json") or os.path.getsize("metrics.json") == 0:
        logger.info("metrics.json not found — running initial training...")
        _daemon(_run_training_tracked, ["src/analyze_metrics.py"])

    if not os.getenv('DISABLE_KAFKA_CONSUMER'):
        _daemon(consume_scored_events)

    if not os.getenv('DISABLE_DATA_WATCHER'):
        _daemon(_watch_data_csv)

    yield

app = FastAPI(title="Fraud Detection Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for our charts
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory store for the latest transactions
transactions_store = deque(maxlen=MAX_HISTORY)

def _load_stats_from_db():
    """Seed in-memory counters from SQLite so stats survive server restarts."""
    try:
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT COUNT(*), SUM(CASE WHEN status='ALERT' THEN 1 ELSE 0 END) FROM live_transactions")
            row = c.fetchone()
            conn.close()
        total  = row[0] or 0
        alerts = row[1] or 0
        return {"total": total, "legit": total - alerts, "alerts": alerts}
    except Exception:
        return {"total": 0, "legit": 0, "alerts": 0}

stats_store = _load_stats_from_db()

def consume_scored_events():
    """Consume scored transactions from Kafka, update dashboard, and persist to SQLite.
    Reconnects automatically with exponential backoff on failure."""
    consumer = None
    retry_delay = 5
    consecutive_failures = 0
    while True:
        try:
            logger.info("Connecting Kafka consumer to '%s'...", SCORED_TOPIC)
            consumer = KafkaConsumer(
                SCORED_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            logger.info("Kafka consumer connected.")
            retry_delay = 5
            consecutive_failures = 0
            for message in consumer:
                tx = message.value
                transactions_store.appendleft(tx)

                with _stats_lock:
                    stats_store["total"] += 1
                    if tx.get("status") == "ALERT":
                        stats_store["alerts"] += 1
                    else:
                        stats_store["legit"] += 1

                raw = tx.get('raw', {})
                try:
                    with _db_lock:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            '''INSERT INTO live_transactions
                               (tx_id, tx_date, amount, merchant_id, tx_type, location,
                                rf_vote, xgb_vote, lgb_vote, status)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (
                                tx.get('tx_id', ''),
                                raw.get('TransactionDate', ''),
                                raw.get('Amount', 0),
                                raw.get('MerchantID', 0),
                                raw.get('TransactionType', ''),
                                raw.get('Location', ''),
                                tx.get('rf_pred', 0),
                                tx.get('xgb_pred', 0),
                                tx.get('lgb_pred', 0),
                                tx.get('status', 'LEGIT'),
                            )
                        )
                        conn.commit()
                        conn.close()
                except Exception as db_err:
                    logger.error("Error saving Kafka tx to SQLite: %s", db_err)

        except Exception as e:
            consecutive_failures += 1
            log_fn = logger.error if consecutive_failures == 1 else logger.warning
            log_fn("Kafka unavailable (%s). Retrying in %ds...", e, retry_delay)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    pass
                consumer = None

@app.get("/api/stats")
def get_stats():
    return stats_store

@app.get("/health")
def health_check():
    """Returns ok/degraded status for models, database, and Kafka producer."""
    health: dict = {"status": "ok", "models": "unknown", "database": "unknown", "kafka": "unknown"}
    try:
        try:
            from decision_engine import _models, _load_models
        except ImportError:
            from src.decision_engine import _models, _load_models
        _load_models()
        health["models"] = "ok" if len(_models) == 3 else "degraded"
    except Exception as e:
        health["models"] = f"error: {e}"
        health["status"] = "degraded"
    try:
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("SELECT 1")
            conn.close()
        health["database"] = "ok"
    except Exception as e:
        health["database"] = f"error: {e}"
        health["status"] = "degraded"
    producer = _get_producer()
    health["kafka"] = "ok" if producer is not None else "unavailable"
    return health

@app.get("/api/training-progress")
def get_training_progress():
    """Return current training progress state."""
    with _training_lock:
        return dict(_training_state)

@app.get("/api/dataset-stats")
def get_dataset_stats():
    """Read Data/data.csv directly and return live row counts."""
    try:
        import csv
        total = legit = fraud = 0
        with open("Data/data.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row.get("IsFraud", "0").strip() == "1":
                    fraud += 1
                else:
                    legit += 1
        return {
            "total": total,
            "legit": legit,
            "fraud": fraud,
            "legit_pct": round(legit / total * 100, 2) if total else 0,
            "fraud_pct": round(fraud / total * 100, 2) if total else 0,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/dist-stats")
def get_dist_stats():
    """Return before/after class distribution counts from the latest metrics.json."""
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
        dist = data.get("dist", {})
        after = dist.get("after", [0, 0])
        return {
            "after_legit": after[0],
            "after_fraud": after[1],
            "after_total": after[0] + after[1],
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/transactions")
def get_recent_transactions():
    return list(transactions_store)

@app.post("/api/run-analysis")
def run_analysis():
    """Retrain models on current data.csv — runs in background, track via /api/training-progress."""
    with _training_lock:
        if _training_state["running"]:
            return {"status": "already_running", "detail": "Training is already in progress."}
    t = threading.Thread(target=_run_training_tracked, args=(["src/analyze_metrics.py"],), daemon=True)
    t.start()
    return {"status": "started"}

@app.post("/api/system-restart")
def system_restart(request: Request):
    """
    Full pipeline restart with a configurable cooldown:
    1. Merges live SQLite traffic into data.csv (active learning).
    2. Re-applies K-Means SMOTE-ENN balancing.
    3. Re-trains all 3 models.
    Protected by an optional API key (set RESTART_API_KEY env var to enable).
    """
    global last_restart_time

    if RESTART_API_KEY:
        provided = request.headers.get("X-API-Key", "")
        if provided != RESTART_API_KEY:
            logger.warning("Unauthorized restart attempt from %s", request.client.host if request.client else "unknown")
            return JSONResponse(
                status_code=401,
                content={"status": "error", "detail": "Unauthorized: invalid or missing X-API-Key header."}
            )

    current_time = time.time()
    time_passed  = current_time - last_restart_time

    if time_passed < RESTART_COOLDOWN:
        remaining = int(RESTART_COOLDOWN - time_passed)
        return {"status": "error", "detail": f"Cooldown Active. Please wait {remaining} seconds before restarting again."}

    try:
        import sys
        last_restart_time = current_time
        logger.info("Starting full pipeline restart (absorb-live + retrain)...")
        result = subprocess.run(
            [sys.executable, "src/analyze_metrics.py", "--absorb-live"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        logger.info("Full restart completed.")
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        logger.error("system-restart failed: %s", e)
        return {"status": "error", "detail": str(e)}

@app.post("/api/hard-restart")
def hard_restart(request: Request):
    """
    Clean-slate restart: delete all .pkl model files and metrics.json, clear the
    in-memory model cache, then retrain from scratch.  Track progress via
    /api/training-progress.  Protected by the same API key and cooldown as
    /api/system-restart.
    """
    global last_restart_time

    if RESTART_API_KEY:
        provided = request.headers.get("X-API-Key", "")
        if provided != RESTART_API_KEY:
            logger.warning("Unauthorized hard-restart attempt from %s",
                           request.client.host if request.client else "unknown")
            return JSONResponse(
                status_code=401,
                content={"status": "error",
                         "detail": "Unauthorized: invalid or missing X-API-Key header."}
            )

    with _training_lock:
        if _training_state["running"]:
            return {"status": "already_running",
                    "detail": "Training is already in progress."}

    current_time = time.time()
    time_passed  = current_time - last_restart_time
    if time_passed < RESTART_COOLDOWN:
        remaining = int(RESTART_COOLDOWN - time_passed)
        return {"status": "error",
                "detail": f"Cooldown Active. Please wait {remaining} seconds before restarting again."}

    last_restart_time = current_time

    # Delete every trained model file and the metrics cache
    _pkl_files = [
        'Models/Random_Forest_model.pkl',
        'Models/Extreme_Gradient_Boosting_model.pkl',
        'Models/Light_Gradient_Boosting_Machine_model.pkl',
        'Models/meta_learner.pkl',
    ]
    deleted = []
    for path in _pkl_files:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
    if os.path.exists('metrics.json'):
        os.remove('metrics.json')
        deleted.append('metrics.json')

    # Delete generated static chart images so stale charts don't linger
    static_dir = 'static'
    if os.path.isdir(static_dir):
        for fname in os.listdir(static_dir):
            if fname.endswith('.png'):
                fpath = os.path.join(static_dir, fname)
                os.remove(fpath)
                deleted.append(fpath)

    # Clear in-memory model cache so decision engine reloads after training
    try:
        try:
            from decision_engine import _models, _model_mtimes
        except ImportError:
            from src.decision_engine import _models, _model_mtimes
        _models.clear()
        _model_mtimes.clear()
    except Exception as e:
        logger.warning("Could not clear model cache: %s", e)

    logger.info("Hard restart: deleted %d file(s): %s", len(deleted), deleted)

    t = threading.Thread(
        target=_run_training_tracked, args=(["src/analyze_metrics.py", "--absorb-live"],), daemon=True
    )
    t.start()

    return {"status": "started", "deleted": deleted}

# ---------------------------------------------------------------------------
# Pipeline process management (producer / preprocessing / decision_engine)
# ---------------------------------------------------------------------------
_PIPELINE_SCRIPTS = {
    'producer':        'src/producer.py',
    'preprocessing':   'src/preprocessing.py',
    'decision_engine': 'src/decision_engine.py',
}
_pipeline_procs: dict = {}
_pipeline_procs_lock = threading.Lock()


def _proc_alive(proc) -> bool:
    return proc is not None and proc.poll() is None


@app.get("/api/pipeline/status")
def pipeline_status():
    with _pipeline_procs_lock:
        return {
            name: "running" if _proc_alive(_pipeline_procs.get(name)) else "stopped"
            for name in _PIPELINE_SCRIPTS
        }


@app.post("/api/pipeline/start")
def pipeline_start():
    import sys
    started = []
    already = []
    with _pipeline_procs_lock:
        for name, script in _PIPELINE_SCRIPTS.items():
            if _proc_alive(_pipeline_procs.get(name)):
                already.append(name)
                continue
            proc = subprocess.Popen(
                [sys.executable, script],
                env={**os.environ, "PYTHONPATH": "src"},
            )
            _pipeline_procs[name] = proc
            started.append(name)
            logger.info("Pipeline: started %s (pid %d)", name, proc.pid)
    return {"status": "ok", "started": started, "already_running": already}


@app.post("/api/pipeline/stop")
def pipeline_stop():
    stopped = []
    with _pipeline_procs_lock:
        for name in list(_PIPELINE_SCRIPTS):
            proc = _pipeline_procs.pop(name, None)
            if _proc_alive(proc):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                stopped.append(name)
                logger.info("Pipeline: stopped %s", name)
    return {"status": "ok", "stopped": stopped}


try:
    from features import process_single_transaction
except ImportError:
    from src.features import process_single_transaction

class TransactionPayload(BaseModel):
    TransactionID: str
    TransactionDate: str
    Amount: float
    MerchantID: int
    TransactionType: str
    Location: str

    @field_validator('Amount')
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Amount must be greater than 0')
        return v

    @field_validator('TransactionType')
    @classmethod
    def transaction_type_must_be_valid(cls, v: str) -> str:
        if v not in ('purchase', 'refund'):
            raise ValueError("TransactionType must be 'purchase' or 'refund'")
        return v


@app.post("/api/predict")
def predict_transaction(payload: TransactionPayload):
    """
    Smartphone / Live API path (matches architecture diagram).
    1. Scores the transaction directly for an immediate HTTP response.
    2. Publishes to Kafka raw-transactions so it flows through the full
       pipeline (Preprocessing -> Decision Engine -> Vault/SQLite -> Active Learning).
    If Kafka is unavailable, falls back to a direct SQLite write so no data is lost.
    """
    try:
        import time as _time
        data_dict = payload.model_dump()

        # 1. Extract feature vector
        feature_vector = process_single_transaction(data_dict)
        arr = [feature_vector]

        # 2. Score via the shared decision engine (uses per-model optimal thresholds from metrics.json)
        try:
            from decision_engine import decide
        except ImportError:
            from src.decision_engine import decide

        result = decide(feature_vector)

        predictions = {
            "Random_Forest":                  result['rf'],
            "Extreme_Gradient_Boosting":      result['xgb'],
            "Light_Gradient_Boosting_Machine":result['lgb'],
        }
        status = result['status']

        resp = {
            "status": status,
            "predictions_breakdown": predictions,
            "probabilities": {
                "Random_Forest":                   round(result['rf_prob'],  4),
                "Extreme_Gradient_Boosting":       round(result['xgb_prob'], 4),
                "Light_Gradient_Boosting_Machine": round(result['lgb_prob'], 4),
            },
            "feature_vector_used": feature_vector,
        }

        # 4. Publish to Kafka raw-transactions so the transaction travels the full
        #    pipeline and is persisted to SQLite by consume_scored_events()
        kafka_tx = {
            'tx_id':           data_dict['TransactionID'],
            'timestamp':       _time.time(),
            'TransactionDate': data_dict['TransactionDate'],
            'Amount':          data_dict['Amount'],
            'MerchantID':      data_dict['MerchantID'],
            'TransactionType': data_dict['TransactionType'],
            'Location':        data_dict['Location'],
        }
        producer = _get_producer()
        kafka_ok = False
        if producer:
            try:
                producer.send(RAW_TOPIC, kafka_tx)
                producer.flush()
                kafka_ok = True
            except Exception as ke:
                logger.warning("Kafka publish failed: %s", ke)

        if not kafka_ok:
            # Fallback: Kafka unavailable — write directly so no data is lost
            try:
                with _db_lock:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute(
                        '''INSERT INTO live_transactions
                           (tx_id, tx_date, amount, merchant_id, tx_type, location,
                            rf_vote, xgb_vote, lgb_vote, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            data_dict['TransactionID'], data_dict['TransactionDate'],
                            data_dict['Amount'], data_dict['MerchantID'],
                            data_dict['TransactionType'], data_dict['Location'],
                            predictions.get("Random_Forest", 0),
                            predictions.get("Extreme_Gradient_Boosting", 0),
                            predictions.get("Light_Gradient_Boosting_Machine", 0),
                            status,
                        )
                    )
                    conn.commit()
                    conn.close()
                tx_doc = {
                    "timestamp": int(_time.time()),
                    "tx_id": data_dict['TransactionID'],
                    "rf_pred": predictions.get("Random_Forest", 0),
                    "xgb_pred": predictions.get("Extreme_Gradient_Boosting", 0),
                    "lgb_pred": predictions.get("Light_Gradient_Boosting_Machine", 0),
                    "status": status,
                }
                transactions_store.appendleft(tx_doc)
                with _stats_lock:
                    stats_store["total"] += 1
                    if status == "ALERT":
                        stats_store["alerts"] += 1
                    else:
                        stats_store["legit"] += 1
            except Exception as db_err:
                logger.error("Fallback SQLite write failed: %s", db_err)

        return resp

    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/saved-transactions")
def get_saved_transactions(page: int = 1, limit: int = 50):
    """Retrieve paginated transactions from SQLite. Use ?page=N&limit=M."""
    try:
        offset = (max(page, 1) - 1) * limit
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(
                "SELECT * FROM live_transactions ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = c.fetchall()
            conn.close()
        return [dict(ix) for ix in rows]
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/api/clear-transactions")
def clear_transactions():
    """Delete all rows from live_transactions, reset in-memory store and counters."""
    try:
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM live_transactions")
            conn.commit()
            conn.close()
        transactions_store.clear()
        with _stats_lock:
            stats_store["total"]  = 0
            stats_store["legit"]  = 0
            stats_store["alerts"] = 0
        logger.info("All live transaction data cleared.")
        return {"status": "ok"}
    except Exception as e:
        logger.error("clear-transactions failed: %s", e)
        return {"status": "error", "detail": str(e)}

@app.get("/metrics", response_class=HTMLResponse)
def get_metrics_page():
    import time
    v = int(time.time())

    if not os.path.exists("metrics.json") or os.path.getsize("metrics.json") == 0:
        return HTMLResponse(content="""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
        <title>Training in Progress</title>
        <style>
        *{box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:#f4f6f9;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}
        .box{background:white;padding:48px 64px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.12);text-align:center;width:520px;max-width:95vw;}
        h2{color:#1a237e;margin:16px 0 6px;font-size:1.5em;}
        .subtitle{color:#666;margin:0 0 28px;font-size:0.95em;}
        .spin-wrap{font-size:2.4em;animation:spin 1.5s linear infinite;display:inline-block;margin-bottom:4px;}
        @keyframes spin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
        .pct-label{font-size:2.8em;font-weight:bold;color:#1a237e;margin:10px 0 8px;letter-spacing:1px;}
        .bar-track{background:#e0e0e0;border-radius:8px;height:20px;overflow:hidden;margin:0 0 14px;}
        .bar-fill{height:100%;width:0%;background:linear-gradient(90deg,#1a237e,#39aa73);border-radius:8px;transition:width 0.6s ease;}
        .step-msg{color:#555;font-size:0.88em;min-height:20px;margin-bottom:20px;word-break:break-word;}
        .hint{color:#bbb;font-size:0.78em;margin-top:16px;}
        </style>
        </head>
        <body>
        <div class="box" id="box">
            <div class="spin-wrap" id="spin-icon">&#9696;</div>
            <h2 id="title">Training in Progress</h2>
            <p class="subtitle" id="sub">Models are being trained automatically. Please wait.</p>
            <div class="pct-label" id="pct-label">0%</div>
            <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
            <div class="step-msg" id="step-msg">Starting...</div>
            <p class="hint">Updates every second &mdash; page reloads automatically when done.</p>
        </div>
        <script>
        function poll(){
            fetch('/api/training-progress')
            .then(r=>r.json())
            .then(s=>{
                const p=Math.min(s.percent||0,100);
                document.getElementById('bar').style.width=p+'%';
                document.getElementById('pct-label').textContent=p+'%';
                document.getElementById('step-msg').textContent=s.message||'';
                if(!s.running && p>=100){
                    document.getElementById('bar').style.background='linear-gradient(90deg,#2e7d32,#39aa73)';
                    document.getElementById('pct-label').style.color='#2e7d32';
                    document.getElementById('spin-icon').textContent='✓';
                    document.getElementById('spin-icon').style.cssText='font-size:2.4em;color:#39aa73;';
                    document.getElementById('title').textContent='Training Complete!';
                    document.getElementById('title').style.color='#2e7d32';
                    document.getElementById('sub').textContent='Reloading metrics page…';
                    document.getElementById('step-msg').textContent='All models trained successfully.';
                    setTimeout(()=>window.location.reload(true),1500);
                } else if(!s.running && s.error){
                    document.getElementById('bar').style.background='#d32f2f';
                    document.getElementById('title').textContent='Training Failed';
                    document.getElementById('title').style.color='#c62828';
                    document.getElementById('step-msg').textContent='Error: '+(s.error||'Unknown error');
                } else {
                    setTimeout(poll,1000);
                }
            })
            .catch(()=>setTimeout(poll,2000));
        }
        poll();
        </script>
        </body></html>
        """, status_code=202)

    precision, recall, f1, auc = "0.000", "0.000", "0.000", "0.000"
    data_str = "{}"
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
            data_str = json.dumps(data)
            precision = data.get("precision", "0.000")
            recall    = data.get("recall",    "0.000")
            f1        = data.get("f1_score",  "0.000")
            auc       = data.get("auc_roc",   "0.000")
    except Exception:
        pass
        
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Training Metrics & Data</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f6f9; color: #333; }
            .header { background-color: #1a237e; color: white; padding: 20px; text-align: center; }
            .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
            .section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .img-container { text-align: center; }
            .img-container img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }
            .metrics-table { font-size: 1.2rem; }
            .btn { background: white; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; margin-top: 15px; display: inline-block; border: 1px solid white; transition: 0.3s; }
            .btn:hover { background: transparent; color: white; }
            .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; text-align: center; margin-top: 20px; }
            .metric-item { background: #fafafa; padding: 20px; border-radius: 8px; border: 1px solid #eee; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .metric-item img { width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
            .metric-item p { font-size: 1.1em; margin-top: 10px; }
            .download-btn { display: inline-block; margin-top: 10px; padding: 6px 12px; font-size: 0.9em; background-color: #e8eaf6; color: #1a237e; text-decoration: none; border-radius: 4px; font-weight: bold; transition: 0.2s; }
            .download-btn:hover { background-color: #c5cae9; }
            @media (max-width: 768px) {
                .header { padding: 15px 10px; }
                .header h1 { font-size: 1.3em; }
                .header p { font-size: 0.85em; }
                .header > div { display: flex !important; flex-wrap: wrap; justify-content: center; gap: 5px; }
                .btn { padding: 6px 10px !important; font-size: 0.78em; margin: 2px !important; }
                .container { padding: 0 12px; }
                .metrics-grid { grid-template-columns: 1fr; }
                .section { padding: 14px; }
                #run-btn { width: 100%; box-sizing: border-box; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Training Metrics</h1>
            <p>Analysis of Real Data Balancing and Offline Evaluation</p>
            <div style="margin-top: 15px;">
                <a href="/" class="btn" style="margin: 0 5px;">Live Dashboard</a>
                <a href="/simulate" class="btn" style="margin: 0 5px;">Simulate Transaction</a>
                <a href="/history" class="btn" style="margin: 0 5px;">Transaction History</a>
                <a href="/metrics" class="btn" style="margin: 0 5px; background: transparent; color: white; border-color: white;">Offline Training Metrics</a>
                <a href="/api-docs" class="btn" style="margin: 0 5px;">API Reference</a>
                <a href="/master" class="btn" style="margin: 0 5px; background: #333; color: white; border: none;">Control Center</a>
            </div>
        </div>
        <div class="container">
            <div style="display: flex; flex-direction: column; align-items: flex-end; margin-bottom: 20px; gap: 10px;">
                <button id="run-btn" onclick="runAnalysis()" style="background: #e8eaf6; padding: 10px 15px; border-radius: 4px; color: #1a237e; font-weight: bold; border: none; cursor: pointer;">
                    &#9654; Run Active Loop &amp; Train Model
                </button>
                <div id="progress-wrap" style="display:none; width: 100%; max-width: 500px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span id="progress-msg" style="font-size:0.85em; color:#555;">Starting...</span>
                        <span id="progress-pct" style="font-size:0.85em; font-weight:bold; color:#1a237e;">0%</span>
                    </div>
                    <div style="background:#e0e0e0; border-radius:6px; height:14px; overflow:hidden;">
                        <div id="progress-bar" style="height:100%; width:0%; background:linear-gradient(90deg,#1a237e,#39aa73); border-radius:6px; transition:width 0.5s ease;"></div>
                    </div>
                </div>
            </div>

            <div class="section" id="overview-section" style="display: none;">
                <h2 style="text-align:center; margin-bottom: 20px;">Dataset Overview (Original Data)</h2>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; text-align: center; gap: 20px;">
                    <div style="background: #e8eaf6; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; border: 1px solid #c5cae9;">
                        <h3 style="margin: 0; color: #1a237e;">Total Transactions</h3>
                        <p id="overview-total" style="font-size: 2em; font-weight: bold; margin: 10px 0 0; color: #333;">Loading...</p>
                    </div>
                    <div style="background: #e8f5e9; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; border: 1px solid #c8e6c9;">
                        <h3 style="margin: 0; color: #2e7d32;">Total Legitimate</h3>
                        <p id="overview-legit" style="font-size: 2em; font-weight: bold; margin: 10px 0 0; color: #333;">Loading...</p>
                    </div>
                    <div style="background: #ffebee; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; border: 1px solid #ffcdd2;">
                        <h3 style="margin: 0; color: #c62828;">Total Fraud</h3>
                        <p id="overview-fraud" style="font-size: 2em; font-weight: bold; margin: 10px 0 0; color: #333;">Loading...</p>
                    </div>
                </div>
            </div>

            <div class="section" id="df-section" style="display: none;">
                <h2 style="text-align:center; margin-bottom: 5px;">Data Structure / CSV Preview</h2>
                <p style="text-align:center; margin-top: 0; color: #666;" id="df-shape"></p>
                
                <h3 style="margin-top: 30px; margin-bottom: 10px; color: #1a237e;">df.head()</h3>
                <div style="overflow-x: auto; border: 1px solid #ddd; border-radius: 4px;">
                    <table id="df-table" style="width: 100%; border-collapse: collapse; font-size: 0.9em; text-align: left; white-space: nowrap;">
                        <thead>
                            <tr id="df-headers" style="background-color: #1a237e; color: white;"></tr>
                        </thead>
                        <tbody id="df-body"></tbody>
                    </table>
                </div>

                <h3 style="margin-top: 30px; margin-bottom: 10px; color: #39aa73;">df.describe()</h3>
                <div style="overflow-x: auto; border: 1px solid #ddd; border-radius: 4px;">
                    <table id="describe-table" style="width: 100%; border-collapse: collapse; font-size: 0.9em; text-align: left; white-space: nowrap;">
                        <thead>
                            <tr id="describe-headers" style="background-color: #39aa73; color: white;"></tr>
                        </thead>
                        <tbody id="describe-body"></tbody>
                    </table>
                </div>
            </div>

            <div class="section">
                <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
                    <div style="flex: 1; min-width: 300px; max-width: 500px; position: relative; height: 350px;">
                        <canvas id="distBeforeChart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('distBeforeChart', 'dist_before.png'); return false;" class="download-btn">&darr; Download Before</a>
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 300px; max-width: 500px; position: relative; height: 350px;">
                        <canvas id="distAfterChart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('distAfterChart', 'dist_after.png'); return false;" class="download-btn">&darr; Download After</a>
                        </div>
                    </div>
                </div>
            </div>

            <div class="section" id="comparison-section" style="display: none;">
                <h2 style="text-align:center; margin-bottom: 5px;">Model Performance Comparison</h2>
                <h3 style="text-align:center; margin-top: 5px; color: #39aa73;" id="best-model-text"></h3>
                <div style="overflow-x: auto; margin-top: 20px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 1.1em; text-align: center; white-space: nowrap;">
                        <thead>
                            <tr style="background-color: #2a707e; color: white;">
                                <th style="padding: 12px; border: 1px solid #ddd;">Algorithm</th>
                                <th style="padding: 12px; border: 1px solid #ddd;">Precision</th>
                                <th style="padding: 12px; border: 1px solid #ddd;">Recall</th>
                                <th style="padding: 12px; border: 1px solid #ddd;">F1-Score</th>
                                <th style="padding: 12px; border: 1px solid #ddd;">AUC-ROC</th>
                            </tr>
                        </thead>
                        <tbody id="comparison-body"></tbody>
                    </table>
                </div>
            </div>

            <div class="section">
                <div class="img-container" style="max-width: 800px; margin: auto; position: relative; height: 400px;">
                    <canvas id="metricsChart"></canvas>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <a href="#" onclick="downloadCanvas('metricsChart', 'evaluation_metrics.png'); return false;" class="download-btn">&darr; Download Metrics Bar Chart</a>
                </div>
            </div>

            <div class="section">
                <h2 style="text-align:center; margin-bottom: 20px;">Model Performance Curves</h2>
                <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
                    <div style="flex: 1; min-width: 280px; max-width: 380px; position: relative; height: 320px;">
                        <canvas id="recallChart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('recallChart', 'recall_curve.png'); return false;" class="download-btn">&darr; Download Recall Curve</a>
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 280px; max-width: 380px; position: relative; height: 320px;">
                        <canvas id="f1Chart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('f1Chart', 'f1_score.png'); return false;" class="download-btn">&darr; Download F1-score Curve</a>
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 280px; max-width: 380px; position: relative; height: 320px;">
                        <canvas id="rocChart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('rocChart', 'roc_curve.png'); return false;" class="download-btn">&darr; Download AUC-ROC Curve</a>
                        </div>
                    </div>
                </div>
            </div>

            <div class="section" id="cm-section" style="display:none;">
                <h2 style="text-align:center;">Confusion Matrices</h2>
                <div id="cm-container" style="display:flex; flex-wrap:wrap; gap:30px; justify-content:center; margin-top:20px;"></div>
            </div>
            
            <div style="text-align: center; color: #666; margin-top: 40px; margin-bottom: 20px; font-size: 0.9em; padding-top: 15px; border-top: 1px solid #ccc;">
                &copy; 2026 Real-Time Fraud Detection System. All metrics generated dynamically.
            </div>
        </div>
        
        <script>
        let _pollTimer = null;

        function runAnalysis() {
            const btn  = document.getElementById('run-btn');
            const wrap = document.getElementById('progress-wrap');
            const bar  = document.getElementById('progress-bar');
            const msg  = document.getElementById('progress-msg');
            const pct  = document.getElementById('progress-pct');

            btn.disabled = true;
            btn.style.opacity = '0.6';
            btn.textContent = 'Training...';
            wrap.style.display = 'block';
            bar.style.width = '0%';
            msg.textContent = 'Starting...';
            pct.textContent = '0%';

            fetch('/api/run-analysis', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'already_running') {
                    msg.textContent = 'Already running — tracking existing progress...';
                } else if (data.status !== 'started') {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    btn.innerHTML = '&#9654; Run Active Loop &amp; Train Model';
                    wrap.style.display = 'none';
                    alert('Failed to start training: ' + (data.detail || JSON.stringify(data)));
                    return;
                }
                _pollProgress();
            })
            .catch(err => {
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.innerHTML = '&#9654; Run Active Loop &amp; Train Model';
                wrap.style.display = 'none';
                alert('Request failed: ' + err);
            });
        }

        function _pollProgress() {
            if (_pollTimer) clearInterval(_pollTimer);
            _pollTimer = setInterval(() => {
                fetch('/api/training-progress')
                .then(r => r.json())
                .then(state => {
                    const btn  = document.getElementById('run-btn');
                    const wrap = document.getElementById('progress-wrap');
                    const bar  = document.getElementById('progress-bar');
                    const msg  = document.getElementById('progress-msg');
                    const pct  = document.getElementById('progress-pct');

                    const p = Math.min(state.percent || 0, 100);
                    bar.style.width = p + '%';
                    pct.textContent = p + '%';
                    msg.textContent = state.message || '';

                    if (!state.running && p >= 100) {
                        clearInterval(_pollTimer);
                        bar.style.background = 'linear-gradient(90deg,#2e7d32,#39aa73)';
                        msg.textContent = 'Training complete! Reloading...';
                        btn.innerHTML = '&#10004; Done';
                        setTimeout(() => window.location.reload(true), 1200);
                    } else if (!state.running && state.error) {
                        clearInterval(_pollTimer);
                        bar.style.background = '#d32f2f';
                        msg.textContent = 'Error: ' + state.error;
                        btn.disabled = false;
                        btn.style.opacity = '1';
                        btn.innerHTML = '&#9654; Run Active Loop &amp; Train Model';
                    }
                })
                .catch(() => {});
            }, 1000);
        }

        // Resume polling if training was already running when page loaded
        fetch('/api/training-progress').then(r => r.json()).then(state => {
            if (state.running) {
                document.getElementById('progress-wrap').style.display = 'block';
                document.getElementById('run-btn').disabled = true;
                document.getElementById('run-btn').style.opacity = '0.6';
                document.getElementById('run-btn').textContent = 'Training...';
                _pollProgress();
            }
        }).catch(() => {});

        const pluginBg = {
            id: 'customCanvasBackgroundColor',
            beforeDraw: (chart, args, options) => {
                const {ctx} = chart;
                ctx.save();
                ctx.globalCompositeOperation = 'destination-over';
                ctx.fillStyle = options.color || '#ffffff';
                ctx.fillRect(0, 0, chart.width, chart.height);
                ctx.restore();
            }
        };

        const pluginDataLabels = {
            id: 'dataLabels',
            afterDatasetsDraw(chart) {
                const { ctx } = chart;
                ctx.save();
                ctx.font = 'bold 15px sans-serif';
                ctx.fillStyle = '#111';
                ctx.textAlign = 'center';
                chart.data.datasets.forEach((dataset, i) => {
                    const meta = chart.getDatasetMeta(i);
                    meta.data.forEach((bar, index) => {
                        if (meta.type === 'bar') {
                            const data = dataset.data[index];
                            if (data !== undefined && data !== null) {
                                ctx.fillText(data, bar.x, bar.y - 12);
                            }
                        }
                    });
                });
                ctx.restore();
            }
        };

        const pluginFooter = {
            id: 'customFooter',
            afterDraw(chart, args, options) {
                if (options.text) {
                    const { ctx } = chart;
                    ctx.save();
                    ctx.font = 'italic 12px sans-serif';
                    ctx.fillStyle = '#666';
                    ctx.textAlign = 'center';
                    ctx.fillText(options.text, chart.width / 2, chart.height - 8);
                    ctx.restore();
                }
            }
        };

        Chart.register(pluginBg, pluginDataLabels, pluginFooter);

        function downloadCanvas(canvasId, filename) {
            const canvas = document.getElementById(canvasId);
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png', 1.0);
            link.click();
        }

        // Render Dynamic Charts
        const metricsData = {JSON_DATA};

        if(metricsData.curves && metricsData.dist) {
            
            // Dataset Overview cards + Before Balance chart are both populated
            // from /api/dataset-stats which reads data.csv live (see chart section below)
            document.getElementById('overview-section').style.display = 'block';

            if(metricsData.df) {
                document.getElementById('df-shape').innerText = `Dataset Shape: ${metricsData.df.shape[0].toLocaleString()} records × ${metricsData.df.shape[1]} features`;
                
                // --- Head Table ---
                const headerTr = document.getElementById('df-headers');
                metricsData.df.columns.forEach(col => {
                    const th = document.createElement('th');
                    th.innerText = col;
                    th.style.padding = '10px';
                    th.style.border = '1px solid #ddd';
                    headerTr.appendChild(th);
                });
                
                const body = document.getElementById('df-body');
                metricsData.df.head.forEach((row, i) => {
                    const tr = document.createElement('tr');
                    tr.style.backgroundColor = (i % 2 === 0) ? '#f9f9f9' : '#fff';
                    metricsData.df.columns.forEach(col => {
                        const td = document.createElement('td');
                        td.innerText = (typeof row[col] === 'number') ? row[col].toFixed(4).replace(/\\.?0+$/, '') : row[col];
                        td.style.padding = '8px 10px';
                        td.style.border = '1px solid #ddd';
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });

                // --- Describe Table ---
                if (metricsData.df.describe && metricsData.df.describe.length > 0) {
                    const dHeaderTr = document.getElementById('describe-headers');
                    const dCols = Object.keys(metricsData.df.describe[0]);
                    
                    dCols.forEach(col => {
                        const th = document.createElement('th');
                        th.innerText = col === 'index' ? 'Statistic' : col;
                        th.style.padding = '10px';
                        th.style.border = '1px solid #ddd';
                        dHeaderTr.appendChild(th);
                    });

                    const dBody = document.getElementById('describe-body');
                    metricsData.df.describe.forEach((row, i) => {
                        const tr = document.createElement('tr');
                        tr.style.backgroundColor = (i % 2 === 0) ? '#f9f9f9' : '#fff';
                        dCols.forEach(col => {
                            const td = document.createElement('td');
                            if (col === 'index') {
                                td.innerText = row[col];
                                td.style.fontWeight = 'bold';
                                td.style.backgroundColor = '#e8f5e9'; // Accent context
                            } else {
                                td.innerText = (typeof row[col] === 'number') ? row[col].toFixed(4).replace(/\\.?0+$/, '') : row[col];
                            }
                            td.style.padding = '8px 10px';
                            td.style.border = '1px solid #ddd';
                            tr.appendChild(td);
                        });
                        dBody.appendChild(tr);
                    });
                }

                document.getElementById('df-section').style.display = 'block';
            }

            if(metricsData.model_comparison && metricsData.best_model) {
                document.getElementById('best-model-text').innerText = `Best Model: ${metricsData.best_model}`;
                const tbody = document.getElementById('comparison-body');
                
                metricsData.model_comparison.forEach((item, index) => {
                    const tr = document.createElement('tr');
                    // Highlight best model row mathematically
                    const isBest = item.name === metricsData.best_model;
                    tr.style.backgroundColor = isBest ? '#e8f5e9' : (index % 2 === 0 ? '#f9f9f9' : '#fff');
                    tr.style.fontWeight = isBest ? 'bold' : 'normal';

                    // Alg name
                    const tdName = document.createElement('td');
                    tdName.innerText = item.name + (isBest ? ' (Winner)' : '');
                    tdName.style.padding = '10px';
                    tdName.style.border = '1px solid #ddd';
                    tr.appendChild(tdName);

                    // Scores
                    ['precision', 'recall', 'f1_score', 'auc_roc'].forEach(metric => {
                        const td = document.createElement('td');
                        td.innerText = parseFloat(item[metric]).toFixed(4);
                        td.style.padding = '10px';
                        td.style.border = '1px solid #ddd';
                        // if best, make text dark green
                        if (isBest) td.style.color = '#1b5e20';
                        tr.appendChild(td);
                    });

                    tbody.appendChild(tr);
                });
                document.getElementById('comparison-section').style.display = 'block';

                // Render confusion matrices — academic Canvas style (Blues colormap, row-normalised)
                const cmContainer = document.getElementById('cm-container');
                if (cmContainer) {
                    // Blues colormap: #f7fbff (0) → #08306b (1)
                    function blueCss(t) {
                        const r = Math.round(247 + (8   - 247) * t);
                        const g = Math.round(251 + (48  - 251) * t);
                        const b = Math.round(255 + (107 - 255) * t);
                        return [r, g, b];
                    }
                    function blueFill(ctx, t) {
                        const [r,g,b] = blueCss(t);
                        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                    }
                    function cellTextColor(t) { return t > 0.45 ? 'white' : '#111'; }

                    function drawCM(canvas, item) {
                        const [[TN, FP], [FN, TP]] = item.cm;
                        const isBest     = item.name === metricsData.best_model;
                        const W = canvas.width, H = canvas.height;
                        const ctx = canvas.getContext('2d');

                        // White background
                        ctx.clearRect(0, 0, W, H);
                        ctx.fillStyle = '#ffffff';
                        ctx.fillRect(0, 0, W, H);

                        // Layout margins
                        const mTop = 72, mLeft = 92, mRight = 68, mBottom = 72;
                        const gW = W - mLeft - mRight;
                        const gH = H - mTop - mBottom;
                        const cW = gW / 2, cH = gH / 2;

                        const legitTotal = TN + FP, fraudTotal = FN + TP;
                        const total      = TN + FP + FN + TP;
                        const cells      = [[TN, FP], [FN, TP]];
                        const rowTotals  = [legitTotal, fraudTotal];
                        const classLabels = ['Legit (0)', 'Fraud (1)'];

                        // ── Title ──────────────────────────────────────────
                        ctx.textAlign = 'center';
                        ctx.font = 'bold 15px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = '#111';
                        ctx.fillText('Confusion Matrix', W / 2, 20);

                        ctx.font = '12px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = isBest ? '#1b5e20' : '#444';
                        ctx.fillText(item.name + (isBest ? '  ★ Best Model' : ''), W / 2, 38);

                        ctx.font = '10.5px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = '#888';
                        const aucTxt = 'AUC-ROC: ' + parseFloat(item.auc_roc || 0).toFixed(4)
                                     + '   Threshold: ' + (item.threshold !== undefined ? item.threshold.toFixed(2) : 'N/A')
                                     + '   Accuracy: ' + (total > 0 ? ((TN+TP)/total*100).toFixed(2) : '—') + '%';
                        ctx.fillText(aucTxt, W / 2, 56);

                        // ── Cells ──────────────────────────────────────────
                        for (let row = 0; row < 2; row++) {
                            for (let col = 0; col < 2; col++) {
                                const val  = cells[row][col];
                                const rTot = rowTotals[row];
                                const t    = rTot > 0 ? val / rTot : 0;
                                const x    = mLeft + col * cW;
                                const y    = mTop  + row * cH;

                                // Cell fill
                                blueFill(ctx, t);
                                ctx.fillRect(x, y, cW, cH);

                                // White grid lines
                                ctx.strokeStyle = '#ffffff';
                                ctx.lineWidth = 2;
                                ctx.strokeRect(x, y, cW, cH);

                                const tc = cellTextColor(t);
                                ctx.textAlign = 'center';

                                // Raw count (large)
                                ctx.font = 'bold 20px "Helvetica Neue", Arial, sans-serif';
                                ctx.fillStyle = tc;
                                ctx.fillText(val.toLocaleString(), x + cW / 2, y + cH / 2 - 4);

                                // Row-normalised %
                                ctx.font = '12px "Helvetica Neue", Arial, sans-serif';
                                ctx.fillStyle = tc;
                                ctx.globalAlpha = 0.85;
                                ctx.fillText(rTot > 0 ? (t * 100).toFixed(1) + '%' : '—',
                                             x + cW / 2, y + cH / 2 + 18);
                                ctx.globalAlpha = 1.0;
                            }
                        }

                        // Outer border
                        ctx.strokeStyle = '#aaa';
                        ctx.lineWidth = 1;
                        ctx.strokeRect(mLeft, mTop, gW, gH);

                        // ── X-axis (Predicted Label) ────────────────────────
                        ctx.font = '12px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = '#333';
                        ctx.textAlign = 'center';
                        classLabels.forEach((lbl, i) => {
                            ctx.fillText(lbl, mLeft + i * cW + cW / 2, mTop + gH + 18);
                        });
                        ctx.font = 'bold 13px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = '#111';
                        ctx.fillText('Predicted Label', mLeft + gW / 2, mTop + gH + 42);

                        // ── Y-axis (Actual Label) ───────────────────────────
                        ctx.font = '12px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = '#333';
                        classLabels.forEach((lbl, i) => {
                            ctx.save();
                            ctx.translate(mLeft - 36, mTop + i * cH + cH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.textAlign = 'center';
                            ctx.fillText(lbl, 0, 0);
                            ctx.restore();
                        });
                        ctx.save();
                        ctx.translate(16, mTop + gH / 2);
                        ctx.rotate(-Math.PI / 2);
                        ctx.textAlign = 'center';
                        ctx.font = 'bold 13px "Helvetica Neue", Arial, sans-serif';
                        ctx.fillStyle = '#111';
                        ctx.fillText('Actual Label', 0, 0);
                        ctx.restore();

                        // ── Colorbar ────────────────────────────────────────
                        const cbX = W - mRight + 14, cbY = mTop, cbW = 12, cbH = gH;
                        const grad = ctx.createLinearGradient(0, cbY + cbH, 0, cbY);
                        [0, 0.25, 0.5, 0.75, 1].forEach(s => {
                            const [r,g,b] = blueCss(s);
                            grad.addColorStop(s, 'rgb(' + r + ',' + g + ',' + b + ')');
                        });
                        ctx.fillStyle = grad;
                        ctx.fillRect(cbX, cbY, cbW, cbH);
                        ctx.strokeStyle = '#bbb';
                        ctx.lineWidth = 0.5;
                        ctx.strokeRect(cbX, cbY, cbW, cbH);

                        ctx.font = '9.5px Arial';
                        ctx.fillStyle = '#555';
                        ctx.textAlign = 'left';
                        ctx.fillText('100%', cbX + cbW + 3, cbY + 9);
                        ctx.fillText('75%',  cbX + cbW + 3, cbY + cbH * 0.25 + 4);
                        ctx.fillText('50%',  cbX + cbW + 3, cbY + cbH * 0.5  + 4);
                        ctx.fillText('25%',  cbX + cbW + 3, cbY + cbH * 0.75 + 4);
                        ctx.fillText('0%',   cbX + cbW + 3, cbY + cbH);

                        // Colorbar label
                        ctx.save();
                        ctx.translate(W - 8, cbY + cbH / 2);
                        ctx.rotate(Math.PI / 2);
                        ctx.textAlign = 'center';
                        ctx.font = '9px Arial';
                        ctx.fillStyle = '#777';
                        ctx.fillText('Row-normalised', 0, 0);
                        ctx.restore();
                    }

                    metricsData.model_comparison.forEach(item => {
                        if (!item.cm) return;

                        const wrap = document.createElement('div');
                        wrap.style.cssText = 'display:flex; flex-direction:column; align-items:center; gap:10px;';

                        const canvas = document.createElement('canvas');
                        canvas.width  = 440;
                        canvas.height = 390;
                        canvas.style.cssText = 'border:1px solid #ddd; border-radius:4px; background:white; max-width:100%;';
                        wrap.appendChild(canvas);

                        // Download button
                        const dlBtn = document.createElement('button');
                        dlBtn.innerHTML = '&#8595; Download PNG';
                        dlBtn.style.cssText = 'padding:7px 20px; background:#0B1B3D; color:white; border:none; border-radius:5px; cursor:pointer; font-size:0.84em; font-family:inherit; letter-spacing:0.3px;';
                        dlBtn.onmouseover = () => { dlBtn.style.background = '#1a3a6b'; };
                        dlBtn.onmouseout  = () => { dlBtn.style.background = '#0B1B3D'; };
                        dlBtn.onclick = () => {
                            const a = document.createElement('a');
                            a.download = 'confusion_matrix_' + item.name.replace(/\\s+/g, '_') + '.png';
                            a.href = canvas.toDataURL('image/png', 1.0);
                            a.click();
                        };
                        wrap.appendChild(dlBtn);

                        cmContainer.appendChild(wrap);
                        drawCM(canvas, item);
                    });
                    document.getElementById('cm-section').style.display = 'block';
                }
            }

            Chart.defaults.plugins.customCanvasBackgroundColor = { color: 'white' };
            const footerOpt = { text: "Generated by Real-Time Fraud Detection System" };

            // Distribution Graphic Before — data read live from data.csv via API
            fetch('/api/dataset-stats')
                .then(r => r.json())
                .then(ds => {
                    if (ds.error) return;

                    // Update Dataset Overview cards
                    document.getElementById('overview-total').innerText = ds.total.toLocaleString();
                    document.getElementById('overview-legit').innerText = ds.legit.toLocaleString() + ' (' + ds.legit_pct + '%)';
                    document.getElementById('overview-fraud').innerText = ds.fraud.toLocaleString() + ' (' + ds.fraud_pct + '%)';

                    const beforeData = [ds.legit, ds.fraud];
                    new Chart(document.getElementById('distBeforeChart'), {
                        type: 'bar',
                        data: {
                            labels: ['0', '1'],
                            datasets: [{ label: 'Count', data: beforeData, backgroundColor: ['#0B1B3D', '#FF8C00'], barPercentage: 0.6 }]
                        },
                        options: {
                            layout: { padding: { top: 45, bottom: 25 } },
                            responsive: true, maintainAspectRatio: false,
                            plugins: { title: {display: true, text: 'Class Distribution Before Balance', font: {size: 15}}, legend: {display:false}, customFooter: footerOpt },
                            scales: { y: { suggestedMax: Math.max(...beforeData) * 1.20, title: {display: true, text:'Count'} }, x: { title: {display: true, text:'IsFraud'} } }
                        }
                    });
                })
                .catch(() => {});

            // Distribution Graphic After — fetched live from metrics.json via API
            fetch('/api/dist-stats')
                .then(r => r.json())
                .then(ds => {
                    if (ds.error) return;
                    const afterData = [ds.after_legit, ds.after_fraud];
                    new Chart(document.getElementById('distAfterChart'), {
                        type: 'bar',
                        data: {
                            labels: ['0', '1'],
                            datasets: [{ label: 'Count', data: afterData, backgroundColor: ['#0B1B3D', '#FF8C00'], barPercentage: 0.6 }]
                        },
                        options: {
                            layout: { padding: { top: 45, bottom: 25 } },
                            responsive: true, maintainAspectRatio: false,
                            plugins: { title: {display: true, text: 'Class Distribution After Balance (Under & Over)', font: {size: 15}}, legend: {display:false}, customFooter: footerOpt },
                            scales: { y: { suggestedMax: Math.max(...afterData) * 1.20, title: {display: true, text:'Count'} }, x: { title: {display: true, text:'IsFraud'} } }
                        }
                    });
                })
                .catch(() => {});

            // Model Evaluation Metrics
            new Chart(document.getElementById('metricsChart'), {
                type: 'bar',
                data: { 
                    labels: ['Precision', 'Recall', 'F1-score', 'AUC-ROC'], 
                    datasets: [{ 
                        label: 'Score', 
                        data: [parseFloat(metricsData.precision), parseFloat(metricsData.recall), parseFloat(metricsData.f1_score), parseFloat(metricsData.auc_roc)], 
                        backgroundColor: ['#444873', '#2a707e', '#39aa73', '#85c35b'],
                        barPercentage: 0.6
                    }] 
                },
                options: { 
                    layout: { padding: { top: 35, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false, 
                    plugins: { title: {display: true, text: 'Model Evaluation Metrics', font: {size: 16}}, legend: {display: false}, customFooter: footerOpt },
                    scales: { y: { min: 0, max: 1.15, title: {display: true, text:'Score'} } }
                }
            });

            // Recall Curve — linear x-axis so points are at their true threshold positions
            new Chart(document.getElementById('recallChart'), {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Recall',
                        data: metricsData.curves.thresholds.map((t, i) => ({x: t, y: metricsData.curves.recalls[i]})),
                        borderColor: '#d62728', fill: false, tension: 0, borderWidth: 2, pointRadius: 0, hitRadius: 10,
                    }]
                },
                options: {
                    layout: { padding: { top: 10, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: {display: true, text: 'Recall Curve', font: {size: 15}}, legend: {display:false}, customFooter: footerOpt },
                    scales: {
                        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Decision Threshold' } },
                        y: { min: 0, max: 1.05, title: { display: true, text: 'Recall' } }
                    }
                }
            });

            // F1-score Curve — linear x-axis so points are at their true threshold positions
            new Chart(document.getElementById('f1Chart'), {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'F1-score',
                        data: metricsData.curves.thresholds.map((t, i) => ({x: t, y: metricsData.curves.f1_scores[i]})),
                        borderColor: '#ff7f0e', fill: false, tension: 0, borderWidth: 2, pointRadius: 0, hitRadius: 10,
                    }]
                },
                options: {
                    layout: { padding: { top: 10, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: {display: true, text: 'F1-score Curve', font: {size: 15}}, legend: {display:false}, customFooter: footerOpt },
                    scales: {
                        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Decision Threshold' } },
                        y: { min: 0, max: 1.05, title: { display: true, text: 'F1-score' } }
                    }
                }
            });

            // AUC-ROC Curve — linear x-axis so FPR is at its true numeric position
            new Chart(document.getElementById('rocChart'), {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'AUC = ' + metricsData.auc_roc,
                            data: metricsData.curves.fpr.map((f, i) => ({x: f, y: metricsData.curves.tpr[i]})),
                            borderColor: '#1f77b4', fill: false, tension: 0, borderWidth: 2, pointRadius: 0, hitRadius: 10,
                        },
                        {
                            label: 'Random Guess',
                            data: [{x: 0, y: 0}, {x: 1, y: 1}],
                            borderColor: 'navy', borderDash: [5, 5], fill: false, tension: 0, borderWidth: 2, pointRadius: 0,
                        }
                    ]
                },
                options: {
                    layout: { padding: { top: 10, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: {display: true, text: 'AUC-ROC Curve', font: {size: 15}}, legend: {position: 'bottom'}, customFooter: footerOpt },
                    scales: {
                        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate' } },
                        y: { min: 0, max: 1.05, title: { display: true, text: 'True Positive Rate' } }
                    }
                }
            });

        }
        </script>
    </body>
    </html>
    """
    
    html_content = html_content.replace("{V}", str(v))
    html_content = html_content.replace("{PRECISION}", precision)
    html_content = html_content.replace("{RECALL}", recall)
    html_content = html_content.replace("{F1_SCORE}", f1)
    html_content = html_content.replace("{AUC_ROC}", auc)
    html_content = html_content.replace("{JSON_DATA}", data_str)

    return html_content

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fraud Detection Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f6f9; color: #333; }
            .header { background-color: #1a237e; color: white; padding: 20px; text-align: center; }
            .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
            .stats { display: flex; justify-content: space-between; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; margin: 0 10px; text-align: center; }
            .stat-card:first-child { margin-left: 0; }
            .stat-card:last-child { margin-right: 0; }
            .stat-value { font-size: 2em; font-weight: bold; margin-top: 10px; }
            .val-alert { color: #d32f2f; }
            .val-legit { color: #388e3c; }
            .btn { background: white; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; margin-top: 15px; display: inline-block; border: 1px solid white; transition: 0.3s; }
            .btn:hover { background: transparent; color: white; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #e8eaf6; font-weight: bold; }
            .badge { padding: 5px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }
            .badge-LEGIT { background-color: #c8e6c9; color: #2e7d32; }
            .badge-ALERT { background-color: #ffcdd2; color: #c62828; }
            .votes { font-family: monospace; }
            @media (max-width: 768px) {
                .header { padding: 15px 10px; }
                .header h1 { font-size: 1.3em; }
                .header p { font-size: 0.85em; }
                .header > div { display: flex !important; flex-wrap: wrap; justify-content: center; gap: 5px; }
                .btn { padding: 6px 10px !important; font-size: 0.78em; margin: 2px !important; }
                .stats { flex-direction: column; }
                .stat-card { margin: 5px 0 !important; min-width: unset !important; }
                .stat-value { font-size: 1.6em; }
                .container { padding: 0 12px; }
                th, td { padding: 8px 10px; font-size: 0.85em; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Real-Time Fraud Detection</h1>
            <p>Live Transaction Monitoring System</p>
            <div style="margin-top: 15px;">
                <a href="/" class="btn" style="margin: 0 5px; background: transparent; color: white; border-color: white;">Live Dashboard</a>
                <a href="/simulate" class="btn" style="margin: 0 5px;">Simulate Transaction</a>
                <a href="/history" class="btn" style="margin: 0 5px;">Transaction History</a>
                <a href="/metrics" class="btn" style="margin: 0 5px;">Offline Training Metrics</a>
                <a href="/api-docs" class="btn" style="margin: 0 5px;">API Reference</a>
                <a href="/master" class="btn" style="margin: 0 5px; background: #333; color: white; border: none;">Control Center</a>
            </div>
        </div>
        <div class="container">
            <div class="stats">
                <div class="stat-card">
                    <div>Total Monitored</div>
                    <div class="stat-value" id="stat-total">0</div>
                </div>
                <div class="stat-card">
                    <div>Legit Transactions</div>
                    <div class="stat-value val-legit" id="stat-legit">0</div>
                </div>
                <div class="stat-card">
                    <div>Fraud Alerts</div>
                    <div class="stat-value val-alert" id="stat-alerts">0</div>
                </div>
            </div>

            </div>
            
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:8px;">
                <h3 style="margin:0;">Recent Transactions (Live Data)</h3>
                <button id="clear-btn" onclick="clearTransactions()" style="background:#d32f2f;color:white;border:none;padding:8px 18px;border-radius:6px;font-weight:bold;font-size:0.85em;cursor:pointer;">Clear All Data</button>
            </div>
            <div style="overflow-x:auto;">
            <table id="tx-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Transaction ID</th>
                        <th>RF Vote</th>
                        <th>XGB Vote</th>
                        <th>LGB Vote</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Populated by JS -->
                </tbody>
            </table>
            </div>
        </div>

        <script>

            function updateStats() {
                fetch('/api/stats')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('stat-total').innerText = data.total;
                        document.getElementById('stat-legit').innerText = data.legit;
                        document.getElementById('stat-alerts').innerText = data.alerts;
                    });
            }

            function updateTable() {
                fetch('/api/transactions')
                    .then(r => r.json())
                    .then(data => {
                        const tbody = document.querySelector('#tx-table tbody');
                        tbody.innerHTML = '';
                        data.forEach(tx => {
                            const date = new Date(tx.timestamp * 1000).toLocaleTimeString();
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${date}</td>
                                <td>${tx.tx_id.substring(0, 8)}...</td>
                                <td class="votes">${tx.rf_pred}</td>
                                <td class="votes">${tx.xgb_pred}</td>
                                <td class="votes">${tx.lgb_pred}</td>
                                <td><span class="badge badge-${tx.status}">${tx.status}</span></td>
                            `;
                            tbody.appendChild(tr);
                        });
                    });
            }

            // Poll every 1.5 seconds
            setInterval(() => {
                updateStats();
                updateTable();
            }, 1500);
            
            // Initial load
            updateStats();
            updateTable();

            function clearTransactions() {
                if (!confirm("Clear all live transaction data? This will delete every record from the database and reset all counters.")) return;
                var btn = document.getElementById('clear-btn');
                btn.disabled = true;
                btn.textContent = 'Clearing...';
                fetch('/api/clear-transactions', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    btn.disabled = false;
                    btn.textContent = 'Clear All Data';
                    if (data.status === 'ok') {
                        updateStats();
                        updateTable();
                    } else {
                        alert('Error: ' + (data.detail || 'Unknown error'));
                    }
                })
                .catch(function(err) {
                    btn.disabled = false;
                    btn.textContent = 'Clear All Data';
                    alert('Error: ' + err);
                });
            }

            function fullSystemRestart() {
                if(!confirm("This will merge all live data into the master CSV, re-apply Hybrid Resampling (K-Means SMOTE-ENN), and completely re-train all models. Continue?")) return;
                
                const btn = document.querySelector('button[onclick="fullSystemRestart()"]');
                const originalText = btn.innerHTML;
                btn.disabled = true;
                btn.innerHTML = "Processing Full Restart...";
                btn.style.opacity = '0.7';

                fetch('/api/system-restart', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if(data.status === 'success') {
                        alert("System Re-trained Successfully! Models are now updated.");
                        window.location.reload();
                    } else {
                        alert("Restart Failed: " + data.detail);
                        btn.disabled = false;
                        btn.innerHTML = originalText;
                        btn.style.opacity = '1';
                    }
                })
                .catch(err => {
                    alert("Error: " + err);
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                    btn.style.opacity = '1';
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/simulate", response_class=HTMLResponse)
def get_simulate_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simulate Transaction</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f6f9; color: #333; }
            .header { background-color: #1a237e; color: white; padding: 20px; text-align: center; }
            .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
            .btn { background: white; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; margin-top: 15px; display: inline-block; border: 1px solid white; transition: 0.3s; }
            .btn:hover { background: transparent; color: white; }
            .badge { padding: 5px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; text-transform: uppercase; display: inline-block; text-align: center;}
            .badge-LEGIT { background-color: #c8e6c9; color: #2e7d32; }
            .badge-ALERT { background-color: #ffcdd2; color: #c62828; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 40px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; font-size: 0.9em; }
            th { background-color: #e8eaf6; font-weight: bold; color: #1a237e; }
            @media (max-width: 768px) {
                .header { padding: 15px 10px; }
                .header h1 { font-size: 1.3em; }
                .header p { font-size: 0.85em; }
                .header > div { display: flex !important; flex-wrap: wrap; justify-content: center; gap: 5px; }
                .btn { padding: 6px 10px !important; font-size: 0.78em; margin: 2px !important; }
                .container { padding: 0 12px; }
                th, td { padding: 8px 8px; font-size: 0.82em; }
                #simulate-form { flex-direction: column; }
                #simulate-form > div { min-width: unset !important; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Simulate Live Transaction</h1>
            <p>Test the AI Ensemble Live Payload Integration</p>
            <div style="margin-top: 15px;">
                <a href="/" class="btn" style="margin: 0 5px;">Live Dashboard</a>
                <a href="/simulate" class="btn" style="margin: 0 5px; background: transparent; color: white; border-color: white;">Simulate Transaction</a>
                <a href="/history" class="btn" style="margin: 0 5px;">Transaction History</a>
                <a href="/metrics" class="btn" style="margin: 0 5px;">Offline Training Metrics</a>
                <a href="/api-docs" class="btn" style="margin: 0 5px;">API Reference</a>
                <a href="/master" class="btn" style="margin: 0 5px; background: #333; color: white; border: none;">Control Center</a>
            </div>
        </div>
        <div class="container">
            <div style="background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px;">
                <h3 style="margin-top: 0; color: #1a237e;">Run A Prediction Payload</h3>
                <p style="color: #666; font-size: 0.9em;">Test the trained Ensemble (Random Forest, XGBoost, LightGBM) natively using the predict API endpoint.</p>
                <form id="simulate-form" style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px;">
                    <div style="flex: 1; min-width: 200px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold; font-size: 0.9em;">Amount ($)</label>
                        <input type="number" id="sim-amount" step="0.01" value="1596.79" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold; font-size: 0.9em;">Merchant ID</label>
                        <input type="number" id="sim-merchant" value="675" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold; font-size: 0.9em;">Type</label>
                        <select id="sim-type" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                            <option value="purchase">Purchase</option>
                            <option value="refund" selected>Refund</option>
                        </select>
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold; font-size: 0.9em;">Location</label>
                        <select id="sim-location" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                            <option value="Houston" selected>Houston</option>
                            <option value="Dallas">Dallas</option>
                            <option value="New York">New York</option>
                            <option value="Los Angeles">Los Angeles</option>
                            <option value="San Jose">San Jose</option>
                        </select>
                    </div>
                    <div style="width: 100%;">
                        <button type="submit" style="background: #2a707e; color: white; border: none; padding: 10px 20px; font-weight: bold; border-radius: 4px; cursor: pointer;">Run Prediction</button>
                    </div>
                </form>
                <div id="sim-result" style="display: none; margin-top: 15px; padding: 15px; border-radius: 4px; background: #e8eaf6; font-family: monospace;"></div>
            </div>
            
            <h3 style="color: #1a237e;">Simulated API Traffic History (SQLite Db)</h3>
            <div style="overflow-x:auto;">
            <table id="saved-tx-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Time</th>
                        <th>Transaction ID</th>
                        <th>Amount</th>
                        <th>RF</th>
                        <th>XGB</th>
                        <th>LGBM</th>
                        <th>Final Status</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Populated by JS -->
                </tbody>
            </table>
            </div>
        </div>
        
        <script>
            // Handle Simulation Form
            document.getElementById('simulate-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const resDiv = document.getElementById('sim-result');
                resDiv.style.display = 'block';
                resDiv.style.backgroundColor = '#e8eaf6';
                resDiv.style.color = '#333';
                resDiv.innerText = 'Evaluating Models Objectively...';

                // Construct payload match API schema
                const payload = {
                    TransactionID: "SIM_" + Math.floor(Math.random() * 1000000),
                    TransactionDate: new Date().toISOString().replace('T', ' ').substring(0, 23),
                    Amount: parseFloat(document.getElementById('sim-amount').value),
                    MerchantID: parseInt(document.getElementById('sim-merchant').value),
                    TransactionType: document.getElementById('sim-type').value,
                    Location: document.getElementById('sim-location').value
                };

                fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                })
                .then(r => r.json())
                .then(data => {
                    if(data.status === 'ERROR' || data.detail) {
                        resDiv.style.backgroundColor = '#ffcdd2';
                        resDiv.style.color = '#c62828';
                        resDiv.innerText = JSON.stringify(data, null, 2);
                    } else {
                        if(data.status === 'ALERT') {
                            resDiv.style.backgroundColor = '#ffebee';
                            resDiv.style.color = '#c62828';
                        } else {
                            resDiv.style.backgroundColor = '#e8f5e9';
                            resDiv.style.color = '#2e7d32';
                        }
                        resDiv.innerHTML = `<strong>Ensemble Result: ${data.status}</strong><br><br>
                        <strong>Engine Mapping Used:</strong> ${JSON.stringify(data.feature_vector_used)}<br>
                        <strong>Logic Breakdown:</strong><br>
                        RF: ${data.predictions_breakdown.Random_Forest === 1 ? 'FRAUD' : 'LEGIT'}<br>
                        XGB: ${data.predictions_breakdown.Extreme_Gradient_Boosting === 1 ? 'FRAUD' : 'LEGIT'}<br>
                        LGBM: ${data.predictions_breakdown.Light_Gradient_Boosting_Machine === 1 ? 'FRAUD' : 'LEGIT'}
                        `;
                        
                        updateSavedTable();
                    }
                })
                .catch(err => {
                    resDiv.style.backgroundColor = '#ffcdd2';
                    resDiv.style.color = '#c62828';
                    resDiv.innerText = "Fetch Error: " + err;
                });
            });

            function updateSavedTable() {
                fetch('/api/saved-transactions')
                    .then(r => r.json())
                    .then(data => {
                        const tbody = document.querySelector('#saved-tx-table tbody');
                        tbody.innerHTML = '';
                        if (!Array.isArray(data)) return;
                        
                        data.forEach(tx => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${tx.id}</td>
                                <td>${tx.timestamp}</td>
                                <td>${tx.tx_id}</td>
                                <td>$${tx.amount.toFixed(2)}</td>
                                <td style="font-family: monospace;">${tx.rf_vote}</td>
                                <td style="font-family: monospace;">${tx.xgb_vote}</td>
                                <td style="font-family: monospace;">${tx.lgb_vote}</td>
                                <td><span class="badge badge-${tx.status}">${tx.status}</span></td>
                            `;
                            tbody.appendChild(tr);
                        });
                    });
            }
            
            // Invoke table load securely on cold boots
            updateSavedTable();

            function fullSystemRestart() {
                if(!confirm("This will merge all live data into the master CSV, re-apply Hybrid Resampling (K-Means SMOTE-ENN), and completely re-train all models. Continue?")) return;
                
                const btn = document.querySelector('button[onclick="fullSystemRestart()"]');
                const originalText = btn.innerHTML;
                btn.disabled = true;
                btn.innerHTML = "Processing Full Restart...";
                btn.style.opacity = '0.7';

                fetch('/api/system-restart', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if(data.status === 'success') {
                        alert("System Re-trained Successfully! Models are now updated.");
                        window.location.reload();
                    } else {
                        alert("Restart Failed: " + data.detail);
                        btn.disabled = false;
                        btn.innerHTML = originalText;
                        btn.style.opacity = '1';
                    }
                })
                .catch(err => {
                    alert("Error: " + err);
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                    btn.style.opacity = '1';
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/history", response_class=HTMLResponse)
def get_history_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transaction History</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f6f9; color: #333; }
            .header { background-color: #1a237e; color: white; padding: 20px; text-align: center; }
            .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
            .btn { background: white; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; margin-top: 15px; display: inline-block; border: 1px solid white; transition: 0.3s; }
            .btn:hover { background: transparent; color: white; }
            .badge { padding: 5px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; text-transform: uppercase; }
            .badge-LEGIT { background-color: #c8e6c9; color: #2e7d32; }
            .badge-ALERT { background-color: #ffcdd2; color: #c62828; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 40px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #e8eaf6; font-weight: bold; color: #1a237e; }
            @media (max-width: 768px) {
                .header { padding: 15px 10px; }
                .header h1 { font-size: 1.3em; }
                .header p { font-size: 0.85em; }
                .header > div { display: flex !important; flex-wrap: wrap; justify-content: center; gap: 5px; }
                .btn { padding: 6px 10px !important; font-size: 0.78em; margin: 2px !important; }
                .container { padding: 0 12px; }
                th, td { padding: 8px 8px; font-size: 0.82em; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Transaction History</h1>
            <p>Complete Log of Scored Live Traffic (SQLite Database)</p>
            <div style="margin-top: 15px;">
                <a href="/" class="btn" style="margin: 0 5px;">Live Dashboard</a>
                <a href="/simulate" class="btn" style="margin: 0 5px;">Simulate Transaction</a>
                <a href="/history" class="btn" style="margin: 0 5px; background: transparent; color: white; border-color: white;">Transaction History</a>
                <a href="/metrics" class="btn" style="margin: 0 5px;">Offline Training Metrics</a>
                <a href="/api-docs" class="btn" style="margin: 0 5px;">API Reference</a>
                <a href="/master" class="btn" style="margin: 0 5px; background: #333; color: white; border: none;">Control Center</a>
            </div>
        </div>
        <div class="container">
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <p>This page displays all transactions saved in the local <strong>transactions.db</strong>. These records are merged back into the model during the next Active Loop training run.</p>
            </div>
            <div style="overflow-x:auto;">
            <table id="history-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>TX ID</th>
                        <th>Amount</th>
                        <th>Merchant</th>
                        <th>Type</th>
                        <th>Location</th>
                        <th>RF</th>
                        <th>XGB</th>
                        <th>LGBM</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Populated by JS -->
                </tbody>
            </table>
            </div>
        </div>
        <script>
            function loadHistory() {
                fetch('/api/saved-transactions')
                    .then(r => r.json())
                    .then(data => {
                        const tbody = document.querySelector('#history-table tbody');
                        tbody.innerHTML = '';
                        if (!Array.isArray(data)) return;
                        data.forEach(tx => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${tx.id}</td>
                                <td>${tx.timestamp}</td>
                                <td>${tx.tx_id}</td>
                                <td>$${tx.amount.toFixed(2)}</td>
                                <td>${tx.merchant_id}</td>
                                <td>${tx.tx_type}</td>
                                <td>${tx.location}</td>
                                <td style="font-family: monospace;">${tx.rf_vote}</td>
                                <td style="font-family: monospace;">${tx.xgb_vote}</td>
                                <td style="font-family: monospace;">${tx.lgb_vote}</td>
                                <td><span class="badge badge-${tx.status}">${tx.status}</span></td>
                            `;
                            tbody.appendChild(tr);
                        });
                    });
            }
            loadHistory();

            function fullSystemRestart() {
                if(!confirm("This will merge all live data into the master CSV, re-apply Hybrid Resampling (K-Means SMOTE-ENN), and completely re-train all models. Continue?")) return;
                
                const btn = document.querySelector('button[onclick="fullSystemRestart()"]');
                const originalText = btn.innerHTML;
                btn.disabled = true;
                btn.innerHTML = "Processing Full Restart...";
                btn.style.opacity = '0.7';

                fetch('/api/system-restart', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if(data.status === 'success') {
                        alert("System Re-trained Successfully! Models are now updated.");
                        window.location.reload();
                    } else {
                        alert("Restart Failed: " + data.detail);
                        btn.disabled = false;
                        btn.innerHTML = originalText;
                        btn.style.opacity = '1';
                    }
                })
                .catch(err => {
                    alert("Error: " + err);
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                    btn.style.opacity = '1';
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api-docs", response_class=HTMLResponse)
def get_api_docs_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API Reference Docs</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f6f9; color: #333; }
            .header { background-color: #1a237e; color: white; padding: 20px; text-align: center; }
            .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
            .btn { background: white; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; margin-top: 15px; display: inline-block; border: 1px solid white; transition: 0.3s; }
            .btn:hover { background: transparent; color: white; }
            pre { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; overflow-x: auto; }
            .endpoint { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 5px solid #2a707e; }
            .method { background: #2a707e; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; font-size: 0.9em; }
            .url { font-weight: bold; font-family: monospace; font-size: 1.1em; }
            .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; overflow-x: auto; }
            .tab-btn { background: none; border: none; padding: 10px 20px; cursor: pointer; font-size: 1em; font-weight: bold; color: #666; transition: 0.3s; outline: none; border-bottom: 2px solid transparent; margin-bottom: -2px; white-space: nowrap; }
            .tab-btn:hover { color: #1a237e; }
            .tab-btn.active { color: #1a237e; border-bottom: 2px solid #1a237e; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            @media (max-width: 768px) {
                .header { padding: 15px 10px; }
                .header h1 { font-size: 1.3em; }
                .header p { font-size: 0.85em; }
                .header > div { display: flex !important; flex-wrap: wrap; justify-content: center; gap: 5px; }
                .btn { padding: 6px 10px !important; font-size: 0.78em; margin: 2px !important; }
                .container { padding: 0 12px; }
                .endpoint { padding: 15px; }
                pre { font-size: 0.78em; }
                .tabs { gap: 0; }
                .tab-btn { padding: 8px 12px; font-size: 0.85em; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>API Reference Docs</h1>
            <p>HTTP Endpoints & JSON Schema Integration Map</p>
            <div style="margin-top: 15px;">
                <a href="/" class="btn" style="margin: 0 5px;">Live Dashboard</a>
                <a href="/simulate" class="btn" style="margin: 0 5px;">Simulate Transaction</a>
                <a href="/history" class="btn" style="margin: 0 5px;">Transaction History</a>
                <a href="/metrics" class="btn" style="margin: 0 5px;">Offline Training Metrics</a>
                <a href="/api-docs" class="btn" style="margin: 0 5px; background: transparent; color: white; border-color: white;">API Reference</a>
                <a href="/master" class="btn" style="margin: 0 5px; background: #333; color: white; border: none;">Control Center</a>
            </div>
        </div>
        <div class="container">
            <h2 style="margin-top: 40px; color: #1a237e;">Core HTTP Endpoints</h2>
            <p style="margin-bottom: 20px;">Use the following REST endpoints to interface with the Data Science Risk pipeline externally. You can also view the interactive <a href="/docs" style="font-weight: bold; color: #1f77b4; text-decoration: none;">Interactive Swagger OpenAPI UI &rarr;</a></p>
            
            <div class="endpoint">
                <div><span class="method">POST</span><span class="url">/api/predict</span></div>
                <p>Accepts a single transaction payload, vectors its independent string components into numeric mapped attributes natively, and fires its values simultaneously into the cached Random Forest, XGBoost, and LightGBM models. If any model scores a 1, it safely alerts for Fraud organically.</p>
                <strong>Request Body Map (JSON)</strong>
                <pre>
{
    "TransactionID": "STR_ABCD123",
    "TransactionDate": "YYYY-MM-DD HH:MM:SS.000",
    "Amount": 1596.79,
    "MerchantID": 675,
    "TransactionType": "purchase" | "refund",
    "Location": "Houston"
}</pre>
                <strong>Response Map (JSON)</strong>
                <pre>
{
    "status": "LEGIT" | "ALERT",
    "predictions_breakdown": {
        "Random_Forest": 0,
        "Extreme_Gradient_Boosting": 0,
        "Light_Gradient_Boosting_Machine": 0
    },
    "probabilities": {
        "Random_Forest": 0.0312,
        "Extreme_Gradient_Boosting": 0.0521,
        "Light_Gradient_Boosting_Machine": 0.0843
    },
    "feature_vector_used": [1596.79, 675, 1, 8, 19, 1]
}</pre>
            </div>
            
            <div class="endpoint">
                <div><span class="method" style="background:#444873;">GET</span><span class="url">/api/stats</span></div>
                <p>Returns the cumulative rolling metrics tracking total system throughput sizes and categorical classifications analyzed over Kafka/Live traffic engines.</p>
                <pre>{"total": 0, "legit": 0, "alerts": 0}</pre>
            </div>
            
            <div class="endpoint">
                <div><span class="method" style="background:#444873;">GET</span><span class="url">/api/transactions</span></div>
                <p>Returns the last 100 rolling transactions scored objectively in the history deque store array list.</p>
            </div>
            
            <div class="endpoint" style="border-left-color: #ff7f0e;">
                <div><span class="method" style="background:#ff7f0e;">POST</span><span class="url">/api/run-analysis</span></div>
                <p>Programmatically triggers an entire automated `analyze_metrics.py` python script run on the host. Triggers Hybrid Resampling via K-Means SMOTE-ENN under/over data sampling explicitly over latest 196k row database block, trains all 3 ML models explicitly, evaluates test-holdout set natively, and overwrites the JSON payload file.</p>
            </div>

            <h2 style="margin-top: 40px; color: #1a237e;">Integration Code Scripts</h2>
            <p style="margin-bottom: 20px;">Connect any frontend logic straight to the Fraud Engine using these native payloads natively!</p>
            
            <div class="endpoint">
                <div class="tabs">
                    <button class="tab-btn active" onclick="openTab('python-tab', this)">Python (Requests)</button>
                    <button class="tab-btn" onclick="openTab('node-tab', this)">Node.js (Fetch API)</button>
                    <button class="tab-btn" onclick="openTab('php-tab', this)">PHP (cURL)</button>
                </div>

                <div id="python-tab" class="tab-content active">
                    <pre>
import requests
import json

url = "http://127.0.0.1:8000/api/predict"

payload = json.dumps({
  "TransactionID": "STR_ABCD123",
  "TransactionDate": "2024-03-05 19:41:36.000",
  "Amount": 1596.79,
  "MerchantID": 675,
  "TransactionType": "purchase",
  "Location": "Houston"
})
headers = { 'Content-Type': 'application/json' }

response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)

# Expected Output JSON:
# {
#   "status": "LEGIT",
#   "predictions_breakdown": {
#     "Random_Forest": 0,
#     "Extreme_Gradient_Boosting": 0,
#     "Light_Gradient_Boosting_Machine": 0
#   },
#   "probabilities": {"Random_Forest": 0.03, "Extreme_Gradient_Boosting": 0.05, "Light_Gradient_Boosting_Machine": 0.08},
#   "feature_vector_used": [1596.79, 675, 1, 8, 19, 1]
# }</pre>
                </div>

                <div id="node-tab" class="tab-content">
                    <pre>
const url = "http://127.0.0.1:8000/api/predict";

const payload = {
  TransactionID: "STR_ABCD123",
  TransactionDate: "2024-03-05 19:41:36.000",
  Amount: 1596.79,
  MerchantID: 675,
  TransactionType: "purchase",
  Location: "Houston"
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => console.log(data));

// Expected Output JSON:
// {
//   "status": "LEGIT",
//   "predictions_breakdown": {
//     "Random_Forest": 0,
//     "Extreme_Gradient_Boosting": 0,
//     "Light_Gradient_Boosting_Machine": 0
//   },
//   "feature_vector_used": [1596.79, 675, 1, 8, 19, 1]
// }</pre>
                </div>

                <div id="php-tab" class="tab-content">
                    <pre>
&lt;?php
$curl = curl_init();
curl_setopt_array($curl, array(
  CURLOPT_URL => 'http://127.0.0.1:8000/api/predict',
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_CUSTOMREQUEST => 'POST',
  CURLOPT_POSTFIELDS =>'{
    "TransactionID": "STR_ABCD123",
    "TransactionDate": "2024-03-05 19:41:36.000",
    "Amount": 1596.79,
    "MerchantID": 675,
    "TransactionType": "purchase",
    "Location": "Houston"
}',
  CURLOPT_HTTPHEADER => array('Content-Type: application/json'),
));

$response = curl_exec($curl);
curl_close($curl);
echo $response;
?&gt;
// Expected Output JSON:
// {
//   "status": "LEGIT",
//   "predictions_breakdown": {
//     "Random_Forest": 0,
//     "Extreme_Gradient_Boosting": 0,
//     "Light_Gradient_Boosting_Machine": 0
//   },
//   "feature_vector_used": [1596.79, 675, 1, 8, 19, 1]
// }</pre>
                </div>
            </div>
        </div>
        
        <script>
            function openTab(tabId, btn) {
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(c => c.classList.remove('active'));
                
                const buttons = document.querySelectorAll('.tab-btn');
                buttons.forEach(b => b.classList.remove('active'));
                
                document.getElementById(tabId).classList.add('active');
                btn.classList.add('active');
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/master", response_class=HTMLResponse)
def get_master_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Master Control Panel</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f6f9; color: #333; display: flex; flex-direction: column; min-height: 100vh; }
            .header { background-color: #1a237e; color: white; padding: 20px; text-align: center; }
            .container { max-width: 800px; margin: 60px auto; padding: 40px; background: white; border-radius: 16px; border: 1px solid #e0e0e0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); text-align: center; }
            .btn { background: white; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; margin-top: 15px; display: inline-block; border: 1px solid white; transition: 0.3s; }
            .btn:hover { background: transparent; color: white; }
            .btn-master { background: #ef4444; color: white; padding: 20px 40px; border-radius: 50%; width: 220px; height: 220px; font-size: 1.25rem; font-weight: 800; text-transform: uppercase; border: 8px solid #7f1d1d; cursor: pointer; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); display: flex; align-items: center; justify-content: center; margin: 0 auto; position: relative; overflow: hidden; }
            .btn-master:hover:not(:disabled) { transform: scale(1.05); background: #f87171; box-shadow: 0 0 40px rgba(239, 68, 68, 0.6); }
            .btn-master:active:not(:disabled) { transform: scale(0.95); }
            .btn-master:disabled { background: #9e9e9e; border-color: #757575; cursor: not-allowed; opacity: 0.8; }
            .status-text { margin-top: 30px; font-size: 1.1rem; color: #666; font-weight: 500; min-height: 1.5em; }
            .cooldown-text { margin-top: 15px; font-size: 0.9rem; font-weight: bold; opacity: 0; transition: opacity 0.3s; }
            .master-decoration { font-size: 3rem; margin-bottom: 20px; color: #ef4444; }
            .description { color: #666; margin-bottom: 40px; font-size: 1rem; line-height: 1.6; }
            @media (max-width: 768px) {
                .header { padding: 15px 10px; }
                .header h1 { font-size: 1.3em; }
                .header p { font-size: 0.85em; }
                .header > div { display: flex !important; flex-wrap: wrap; justify-content: center; gap: 5px; }
                .btn { padding: 6px 10px !important; font-size: 0.78em; margin: 2px !important; }
                .container { padding: 25px 18px; margin: 30px auto; }
            }
            @media (max-width: 480px) {
                .btn-master { width: 170px; height: 170px; font-size: 1rem; border-width: 6px; }
                .master-decoration { font-size: 2rem; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Engine Master Control</h1>
            <p>Full Pipeline Orchestration</p>
            <div style="margin-top: 15px;">
                <a href="/" class="btn" style="margin: 0 5px;">Live Dashboard</a>
                <a href="/simulate" class="btn" style="margin: 0 5px;">Simulate Transaction</a>
                <a href="/history" class="btn" style="margin: 0 5px;">Transaction History</a>
                <a href="/metrics" class="btn" style="margin: 0 5px;">Offline Training Metrics</a>
                <a href="/api-docs" class="btn" style="margin: 0 5px;">API Reference</a>
                <a href="/master" class="btn" style="margin: 0 5px; background: transparent; color: white; border-color: white;">Control Center</a>
            </div>
        </div>
        
        <div style="max-width: 800px; margin: 60px auto; padding: 0 20px;">

        <!-- ── Pipeline Control ── -->
        <div style="background:white;padding:28px 35px;border-radius:14px;border:1px solid #e0e0e0;box-shadow:0 2px 10px rgba(0,0,0,0.08);margin-bottom:24px;text-align:center;">
            <h3 style="margin:0 0 6px;color:#1a237e;font-size:0.85rem;letter-spacing:2px;text-transform:uppercase;font-weight:700;">Live Pipeline Control</h3>
            <p style="margin:0 0 22px;color:#888;font-size:0.82rem;">Start or stop the streaming services independently of the model training cycle.</p>

            <div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin-bottom:22px;">
                <div id="badge-producer"        style="padding:8px 18px;border-radius:20px;font-size:0.82rem;font-weight:600;background:#f5f5f5;color:#888;border:1px solid #ddd;">● Producer</div>
                <div id="badge-preprocessing"   style="padding:8px 18px;border-radius:20px;font-size:0.82rem;font-weight:600;background:#f5f5f5;color:#888;border:1px solid #ddd;">● Preprocessing</div>
                <div id="badge-decision_engine" style="padding:8px 18px;border-radius:20px;font-size:0.82rem;font-weight:600;background:#f5f5f5;color:#888;border:1px solid #ddd;">● Decision Engine</div>
            </div>

            <div style="display:flex;gap:14px;justify-content:center;">
                <button id="pipeline-start-btn" onclick="pipelineStart()" style="background:#1b5e20;color:white;border:none;padding:12px 32px;border-radius:8px;font-weight:700;font-size:0.95rem;cursor:pointer;letter-spacing:1px;">&#9654; START</button>
                <button id="pipeline-stop-btn"  onclick="pipelineStop()"  style="background:#b71c1c;color:white;border:none;padding:12px 32px;border-radius:8px;font-weight:700;font-size:0.95rem;cursor:pointer;letter-spacing:1px;">&#9632; STOP</button>
            </div>
        </div>

        <div class="container">
            <div class="master-decoration">☢</div>
            <h2 style="margin-bottom: 10px; color: #333;">Full Pipeline Orchestration</h2>
            <p class="description">Consolidate live traffic data, re-apply Hybrid Balancing (K-Means SMOTE-ENN), and rebuild all predictive models. This action is restricted by a 5-minute cooldown.</p>

            <button id="master-btn" class="btn-master" onclick="triggerMasterRestart()">
                <span id="btn-label">INITIATE<br>RESTART</span>
            </button>

            <div id="status-display" class="status-text">SYSTEM STANDBY</div>
            <div id="cooldown-display" class="cooldown-text"></div>
        </div>

        <script>
            var isRunning     = false;
            var progressTimer = null;
            var cooldownTimer = null;

            function pctLabel(p, sub) {
                return '<span style="font-size:2.4rem;font-weight:900;line-height:1;">' + p + '%</span>'
                     + '<span style="display:block;font-size:0.65rem;letter-spacing:3px;margin-top:6px;">' + (sub||'TRAINING') + '</span>';
            }

            function _pollProgress() {
                if (progressTimer) clearInterval(progressTimer);
                progressTimer = setInterval(function () {
                    fetch('/api/training-progress')
                    .then(function(r) { return r.json(); })
                    .then(function(state) {
                        var btn    = document.getElementById('master-btn');
                        var label  = document.getElementById('btn-label');
                        var status = document.getElementById('status-display');

                        var p = Math.min(state.percent || 0, 100);
                        label.innerHTML = pctLabel(p);

                        if (!state.running && p >= 100) {
                            clearInterval(progressTimer);
                            progressTimer = null;
                            label.innerHTML    = pctLabel(100, 'COMPLETE');
                            status.innerText   = 'ENGINE REBUILD COMPLETE';
                            status.style.color = '#10b981';
                            setTimeout(function () {
                                alert('SUCCESS: All model files deleted and rebuilt from scratch.');
                                location.reload();
                            }, 900);
                        } else if (!state.running && state.error) {
                            clearInterval(progressTimer);
                            progressTimer      = null;
                            label.innerHTML    = 'INITIATE<br>RESTART';
                            status.innerText   = 'TRAINING FAILED';
                            status.style.color = '#f43f5e';
                            alert('ERROR: ' + state.error);
                            btn.disabled = false;
                            isRunning    = false;
                        }
                    })
                    .catch(function() {});
                }, 1000);
            }

            // ── Pipeline Control ──────────────────────────────────────────
            var _pipelineStatusInterval = null;

            function _applyBadge(id, state) {
                var el = document.getElementById(id);
                if (!el) return;
                var label = id.replace('badge-', '').replace('_', ' ');
                label = label.charAt(0).toUpperCase() + label.slice(1);
                if (state === 'running') {
                    el.style.background = '#e8f5e9';
                    el.style.color      = '#1b5e20';
                    el.style.border     = '1px solid #a5d6a7';
                    el.textContent      = '● ' + label;
                } else {
                    el.style.background = '#ffebee';
                    el.style.color      = '#b71c1c';
                    el.style.border     = '1px solid #ef9a9a';
                    el.textContent      = '○ ' + label;
                }
            }

            function refreshPipelineStatus() {
                fetch('/api/pipeline/status')
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    _applyBadge('badge-producer',        d.producer);
                    _applyBadge('badge-preprocessing',   d.preprocessing);
                    _applyBadge('badge-decision_engine', d.decision_engine);
                })
                .catch(function() {});
            }

            function pipelineStart() {
                var btn = document.getElementById('pipeline-start-btn');
                btn.disabled = true;
                btn.textContent = 'Starting...';
                fetch('/api/pipeline/start', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function() {
                    btn.disabled = false;
                    btn.innerHTML = '&#9654; START';
                    refreshPipelineStatus();
                })
                .catch(function() {
                    btn.disabled = false;
                    btn.innerHTML = '&#9654; START';
                });
            }

            function pipelineStop() {
                var btn = document.getElementById('pipeline-stop-btn');
                btn.disabled = true;
                btn.textContent = 'Stopping...';
                fetch('/api/pipeline/stop', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function() {
                    btn.disabled = false;
                    btn.innerHTML = '&#9632; STOP';
                    refreshPipelineStatus();
                })
                .catch(function() {
                    btn.disabled = false;
                    btn.innerHTML = '&#9632; STOP';
                });
            }

            // Initial status check + poll every 3 seconds
            refreshPipelineStatus();
            _pipelineStatusInterval = setInterval(refreshPipelineStatus, 3000);
            // ─────────────────────────────────────────────────────────────

            function triggerMasterRestart() {
                if (isRunning) return;
                if (!confirm("COMMAND CONFIRMATION: This will DELETE all trained model files (.pkl) and metrics.json, then retrain all models from scratch. Continue?")) return;

                var btn      = document.getElementById('master-btn');
                var label    = document.getElementById('btn-label');
                var status   = document.getElementById('status-display');
                var cooldown = document.getElementById('cooldown-display');

                isRunning    = true;
                btn.disabled = true;
                cooldown.style.opacity = '0';
                if (cooldownTimer) { clearInterval(cooldownTimer); cooldownTimer = null; }

                label.innerHTML    = pctLabel(0);
                status.innerText   = 'DELETING MODELS & RETRAINING FROM SCRATCH...';
                status.style.color = '#ef4444';

                fetch('/api/hard-restart', { method: 'POST' })
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.status === 'started' || data.status === 'already_running') {
                            if (data.status === 'already_running') {
                                status.innerText = 'TRAINING ALREADY RUNNING — TRACKING PROGRESS...';
                            }
                            _pollProgress();
                        } else if (data.detail && data.detail.toLowerCase().includes('cooldown')) {
                            label.innerHTML    = 'INITIATE<br>RESTART';
                            status.innerText   = 'COMMAND REJECTED';
                            status.style.color = '#f43f5e';

                            var m         = data.detail.match(/(\\d+)\\s+second/i);
                            var remaining = m ? parseInt(m[1]) : 300;

                            cooldown.innerText     = 'COOLDOWN ACTIVE. PLEASE WAIT ' + remaining + ' SECONDS.';
                            cooldown.style.color   = '#c62828';
                            cooldown.style.opacity = '1';

                            cooldownTimer = setInterval(function () {
                                remaining -= 1;
                                if (remaining <= 0) {
                                    clearInterval(cooldownTimer);
                                    cooldownTimer        = null;
                                    cooldown.innerText   = 'COOLDOWN CLEARED. READY TO RESTART.';
                                    cooldown.style.color = '#10b981';
                                    btn.disabled         = false;
                                    isRunning            = false;
                                } else {
                                    cooldown.innerText = 'COOLDOWN ACTIVE. PLEASE WAIT ' + remaining + ' SECONDS.';
                                }
                            }, 1000);
                        } else {
                            label.innerHTML    = 'INITIATE<br>RESTART';
                            status.innerText   = 'COMMAND REJECTED';
                            status.style.color = '#f43f5e';
                            alert('ERROR: ' + (data.detail || 'Unknown error during build'));
                            btn.disabled = false;
                            isRunning    = false;
                        }
                    })
                    .catch(function() {
                        label.innerHTML    = 'INITIATE<br>RESTART';
                        status.innerText   = 'INTERFACE ERROR';
                        status.style.color = '#f43f5e';
                        alert('COMMAND FAILURE: Interface timed out or server unavailable.');
                        btn.disabled = false;
                        isRunning    = false;
                    });
            }
        </script>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
