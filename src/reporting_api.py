import json
import asyncio
import subprocess
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer
from pydantic import BaseModel

import sqlite3

KAFKA_BROKER = 'localhost:9092'
SCORED_TOPIC = 'scored-transactions'
MAX_HISTORY = 100

import os
def init_db():
    if not os.path.exists('Data'):
        os.makedirs('Data')
    conn = sqlite3.connect("Data/transactions.db")
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

app = FastAPI(title="Fraud Detection Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for our charts
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory store for the latest transactions
transactions_store = deque(maxlen=MAX_HISTORY)
stats_store = {
    "total": 0,
    "legit": 0,
    "alerts": 0
}

def consume_scored_events():
    """Background task to consume scored transactions from Kafka."""
    try:
        consumer = KafkaConsumer(
            SCORED_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        print("API Consumer connected to Kafka!")
        for message in consumer:
            tx = message.value
            transactions_store.appendleft(tx)
            
            stats_store["total"] += 1
            if tx.get("status") == "ALERT":
                stats_store["alerts"] += 1
            else:
                stats_store["legit"] += 1
                
    except Exception as e:
        print(f"Error in background consumer: {e}")

@app.on_event("startup")
async def startup_event():
    # Run the Kafka consumer in a background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, consume_scored_events)

@app.get("/api/stats")
def get_stats():
    return stats_store

@app.get("/api/transactions")
def get_recent_transactions():
    return list(transactions_store)

@app.post("/api/run-analysis")
def run_analysis():
    """Endpoint triggered by UI to run analyze_metrics.py"""
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ["venv\\Scripts\\python.exe", "src\\analyze_metrics.py"], 
            capture_output=True, 
            text=True
        )
        
        # After creating images natively, move them to static directory via PS copy to handle windows paths
        subprocess.run(
            ["powershell.exe", "-Command", "Move-Item -Path '*.png' -Destination 'static\\' -Force"]
        )
        
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

import os
import joblib
from typing import Dict, Any

from features import process_single_transaction

class TransactionPayload(BaseModel):
    TransactionID: str
    TransactionDate: str
    Amount: float
    MerchantID: int
    TransactionType: str
    Location: str
    
@app.post("/api/predict")
def predict_transaction(payload: TransactionPayload):
    """
    Accepts a single transaction JSON.
    Loads precisely cached AI Models on demand & performs live scoring.
    """
    try:
        data_dict = payload.dict()
        
        # 1. Map raw json into correct numerical feature vector map
        feature_vector = process_single_transaction(data_dict)
        
        # Structure as 2D array expected by scikit
        arr = [feature_vector]
        
        # 2. Lazy load the highly-optimized cached trained models from Models directory
        models = {}
        for algo in ["Random_Forest", "Extreme_Gradient_Boosting", "Light_Gradient_Boosting_Machine"]:
            path = f"Models/{algo}_model.pkl"
            if os.path.exists(path):
                models[algo] = joblib.load(path)
        
        if not models:
            return {"status": "error", "detail": "No machine learning models have been trained and cached. Please visit /metrics to evaluate Offline processing."}
            
        # 3. Predict dynamically against the parsed features!
        predictions = {}
        status = "LEGIT"
        
        for name, model in models.items():
            pred = int(model.predict(arr)[0])
            predictions[name] = pred
            # Simple ensemble check logic: if ANY model thinks it is fraud, raise an ALERT!
            if pred == 1:
                status = "ALERT"
                
        # 4. Formulate cleanly formatted analytical response output!
        resp = {
            "status": status,
            "predictions_breakdown": predictions,
            "feature_vector_used": feature_vector
        }
        
        # Save to SQLite table for live tracking
        try:
            conn = sqlite3.connect("Data/transactions.db")
            c = conn.cursor()
            c.execute('''INSERT INTO live_transactions 
                         (tx_id, tx_date, amount, merchant_id, tx_type, location, rf_vote, xgb_vote, lgb_vote, status)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (data_dict['TransactionID'], data_dict['TransactionDate'], data_dict['Amount'],
                       data_dict['MerchantID'], data_dict['TransactionType'], data_dict['Location'],
                       predictions.get("Random_Forest", 0), predictions.get("Extreme_Gradient_Boosting", 0),
                       predictions.get("Light_Gradient_Boosting_Machine", 0), status))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"Error saving live to DB: {db_err}")
            
        import time
        tx_doc = {
            "timestamp": int(time.time()),
            "tx_id": data_dict['TransactionID'],
            "rf_pred": predictions.get("Random_Forest", 0),
            "xgb_pred": predictions.get("Extreme_Gradient_Boosting", 0),
            "lgb_pred": predictions.get("Light_Gradient_Boosting_Machine", 0),
            "status": status
        }
        transactions_store.appendleft(tx_doc)
        
        stats_store["total"] += 1
        if status == "ALERT":
            stats_store["alerts"] += 1
        else:
            stats_store["legit"] += 1
            
        return resp
        
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/saved-transactions")
def get_saved_transactions():
    """Retrieve the recent transactions recorded successfully in the local SQLite db array."""
    try:
        conn = sqlite3.connect("Data/transactions.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM live_transactions ORDER BY id DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        return [dict(ix) for ix in rows]
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/metrics", response_class=HTMLResponse)
def get_metrics_page():
    import time
    v = int(time.time())
    
    precision, recall, f1, auc = "0.000", "0.000", "0.000", "0.000"
    data_str = "{}"
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
            data_str = json.dumps(data)
            precision = data.get("precision", "0.000")
            recall = data.get("recall", "0.000")
            f1 = data.get("f1_score", "0.000")
            auc = data.get("auc_roc", "0.000")
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
            </div>
        </div>
        <div class="container">
            <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 20px;">
                <button id="run-btn" onclick="runAnalysis()" style="background: #e8eaf6; padding: 10px 15px; border-radius: 4px; color: #1a237e; text-decoration: none; font-weight: bold; border: none; cursor: pointer; display: flex; align-items: center;">
                    <span id="run-text">&#9654; Run Active Loop & Train Model</span>
                    <span id="run-spinner" style="display: none; margin-left: 10px;">...</span>
                </button>
            </div>

            <div class="section" id="overview-section" style="display: none;">
                <h2 style="text-align:center; margin-bottom: 20px;">Dataset Overview (Original Data)</h2>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; text-align: center; gap: 20px;">
                    <div style="background: #e8eaf6; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; border: 1px solid #c5cae9;">
                        <h3 style="margin: 0; color: #1a237e;">Total Transactions</h3>
                        <p id="overview-total" style="font-size: 2em; font-weight: bold; margin: 10px 0 0; color: #333;"></p>
                    </div>
                    <div style="background: #e8f5e9; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; border: 1px solid #c8e6c9;">
                        <h3 style="margin: 0; color: #2e7d32;">Total Legitimate</h3>
                        <p id="overview-legit" style="font-size: 2em; font-weight: bold; margin: 10px 0 0; color: #333;"></p>
                    </div>
                    <div style="background: #ffebee; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; border: 1px solid #ffcdd2;">
                        <h3 style="margin: 0; color: #c62828;">Total Fraud</h3>
                        <p id="overview-fraud" style="font-size: 2em; font-weight: bold; margin: 10px 0 0; color: #333;"></p>
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
                <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
                    <div style="flex: 1; min-width: 300px; max-width: 550px; position: relative; height: 350px;">
                        <canvas id="f1Chart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('f1Chart', 'f1_score.png'); return false;" class="download-btn">&darr; Download F1-score Curve</a>
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 300px; max-width: 550px; position: relative; height: 350px;">
                        <canvas id="rocChart"></canvas>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="#" onclick="downloadCanvas('rocChart', 'roc_curve.png'); return false;" class="download-btn">&darr; Download ROC Curve</a>
                        </div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2 style="text-align:center;">Confusion Matrix (Random Forest)</h2>
                <div class="img-container" style="max-width: 500px; margin: auto;">
                    <img src="/static/confusion_matrix.png?v={V}" alt="Confusion Matrix" style="width: 100%; border-radius: 6px;">
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <a href="/static/confusion_matrix.png" download="confusion_matrix.png" class="download-btn">&darr; Download Confusion Matrix</a>
                </div>
            </div>
            
            <div style="text-align: center; color: #666; margin-top: 40px; margin-bottom: 20px; font-size: 0.9em; padding-top: 15px; border-top: 1px solid #ccc;">
                &copy; 2026 Real-Time Fraud Detection System. All metrics generated dynamically.
            </div>
        </div>
        
        <script>
        function runAnalysis() {
            const btn = document.getElementById('run-btn');
            const text = document.getElementById('run-text');
            const spinner = document.getElementById('run-spinner');
            
            // UI Loading state
            btn.disabled = true;
            btn.style.opacity = '0.7';
            text.innerText = "Training Model... (Please wait a few seconds)";
            spinner.style.display = "inline-block";

            // Trigger the internal training script
            fetch('/api/run-analysis', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.style.opacity = '1';
                text.innerHTML = "&#10004; Training Complete!";
                spinner.style.display = "none";
                
                if (data.status === 'success') {
                    // Log the python console output inside the browser!
                    console.log(data.output);
                    // Refresh the page immediately to load the brand new updated graph PNGs
                    setTimeout(() => window.location.reload(true), 1000);
                } else {
                    alert("Error running analysis: " + data.detail);
                }
            })
            .catch(err => {
                alert("Request failed: " + err);
                btn.disabled = false;
                btn.style.opacity = '1';
                text.innerHTML = "&#9654; Run 70/15/15 Split & Train Model";
                spinner.style.display = "none";
            });
        }

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
            
            // Render Stats Overview Block
            const legitCt = metricsData.dist.before[0];
            const fraudCt = metricsData.dist.before[1];
            const totalCt = legitCt + fraudCt;
            const legitPct = ((legitCt / totalCt) * 100).toFixed(2);
            const fraudPct = ((fraudCt / totalCt) * 100).toFixed(2);
            
            document.getElementById('overview-total').innerText = totalCt.toLocaleString();
            document.getElementById('overview-legit').innerText = `${legitCt.toLocaleString()} (${legitPct}%)`;
            document.getElementById('overview-fraud').innerText = `${fraudCt.toLocaleString()} (${fraudPct}%)`;
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
                        td.innerText = (typeof row[col] === 'number') ? row[col].toFixed(4).replace(/\.?0+$/, '') : row[col];
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
                                td.innerText = (typeof row[col] === 'number') ? row[col].toFixed(4).replace(/\.?0+$/, '') : row[col];
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
                    tdName.innerText = item.name + (isBest ? ' (Winner) ' + String.fromCodePoint(0x1F451) : '');
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
            }

            Chart.defaults.plugins.customCanvasBackgroundColor = { color: 'white' };
            const footerOpt = { text: "Generated by Real-Time Fraud Detection System" };

            // Distribution Graphic Before
            new Chart(document.getElementById('distBeforeChart'), {
                type: 'bar',
                data: {
                    labels: ['0', '1'],
                    datasets: [{ label: 'Count', data: metricsData.dist.before, backgroundColor: ['#0B1B3D', '#FF8C00'], barPercentage: 0.6 }]
                },
                options: { 
                    layout: { padding: { top: 45, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: {display: true, text: 'Class Distribution Before Balance', font: {size: 15}}, legend: {display:false}, customFooter: footerOpt },
                    scales: { y: { suggestedMax: Math.max(...metricsData.dist.before) * 1.20, title: {display: true, text:'Count'} }, x: { title: {display: true, text:'IsFraud'} } }
                }
            });

            // Distribution Graphic After
            new Chart(document.getElementById('distAfterChart'), {
                type: 'bar',
                data: {
                    labels: ['0', '1'],
                    datasets: [{ label: 'Count', data: metricsData.dist.after, backgroundColor: ['#0B1B3D', '#FF8C00'], barPercentage: 0.6 }]
                },
                options: { 
                    layout: { padding: { top: 45, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: {display: true, text: 'Class Distribution After Balance (Under & Over)', font: {size: 15}}, legend: {display:false}, customFooter: footerOpt },
                    scales: { y: { suggestedMax: Math.max(...metricsData.dist.after) * 1.20, title: {display: true, text:'Count'} }, x: { title: {display: true, text:'IsFraud'} } }
                }
            });

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

            // F1 Curve
            new Chart(document.getElementById('f1Chart'), {
                type: 'line',
                data: { labels: metricsData.curves.thresholds.map(t => t.toFixed(2)), datasets: [{ label: 'F1-score', data: metricsData.curves.f1_scores, borderColor: '#ff7f0e', fill: false, tension: 0.1, borderWidth: 2 }] },
                options: { 
                    layout: { padding: { top: 10, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false, elements: { point: { radius: 0, hitRadius: 10 } }, 
                    plugins: { title: {display: true, text: 'F1-score Curve', font: {size: 16}}, legend: {display:false}, customFooter: footerOpt },
                    scales: { y: { min: 0, max: 1.05, title: {display: true, text:'F1-score'} }, x: { title: { display: true, text: 'Decision Threshold' } } } 
                }
            });

            // Large ROC Curve
            new Chart(document.getElementById('rocChart'), {
                type: 'line',
                data: { 
                    labels: metricsData.curves.fpr.map(f => f.toFixed(2)), 
                    datasets: [
                        { label: 'AUC = ' + metricsData.auc_roc, data: metricsData.curves.tpr, borderColor: '#1f77b4', fill: false, tension: 0.1, borderWidth: 2 }, 
                        { label: 'Random Guess', data: metricsData.curves.fpr, borderColor: 'navy', borderDash:[5,5], fill: false, borderWidth: 2 }
                    ] 
                },
                options: { 
                    layout: { padding: { top: 10, bottom: 25 } },
                    responsive: true, maintainAspectRatio: false, elements: { point: { radius: 0, hitRadius: 10 } }, 
                    plugins: { title: {display: true, text: 'AUC-ROC Curve', font: {size: 16}}, legend: {position: 'bottom'}, customFooter: footerOpt },
                    scales: { y: { min: 0, title: {display: true, text:'True Positive Rate'} }, x: { title: { display: true, text: 'False Positive Rate' } } } 
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
            
            <h3>Recent Transactions (Live Data)</h3>
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
            </div>
        </div>
        <div class="container">
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <p>This page displays all transactions saved in the local <strong>transactions.db</strong>. These records are merged back into the model during the next Active Loop training run.</p>
            </div>
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
                <p>Programmatically triggers an entire automated `analyze_metrics.py` python script run on the host. Triggers SMOTETomek under/over data sampling explicitly over latest 196k row database block, trains all 3 ML models explicitly, evaluates test-holdout set natively, and overwrites the JSON payload file.</p>
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

if __name__ == "__main__":
    import uvicorn
    # Optional: run directly without uvicorn command
    uvicorn.run(app, host="0.0.0.0", port=8000)
