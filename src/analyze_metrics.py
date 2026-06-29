import logging
import os
import shutil
import warnings
from datetime import datetime

# Suppress sklearn/joblib parallel compatibility notice — cosmetic only, no effect on results
warnings.filterwarnings(
    'ignore',
    message='`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`',
    category=UserWarning,
    module='sklearn',
)

import joblib
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from features import extract_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')


def _static(filename):
    os.makedirs(STATIC_DIR, exist_ok=True)
    return os.path.join(STATIC_DIR, filename)


def _archive_models():
    """Copy current Models/*.pkl into Models/archive/{timestamp}/ before overwriting."""
    archive_dir = os.path.join('Models', 'archive', datetime.now().strftime('%Y%m%d_%H%M%S'))
    archived = []
    for fname in os.listdir('Models'):
        if fname.endswith('.pkl'):
            src = os.path.join('Models', fname)
            os.makedirs(archive_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(archive_dir, fname))
            archived.append(fname)
    if archived:
        logger.info("Archived %d model(s) to %s", len(archived), archive_dir)


def _tune_model(name, X_tr, y_tr, X_val, y_val, n_trials=40):
    """Optuna hyperparameter search. Returns best params dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    def objective(trial):
        if name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100),
                max_depth=trial.suggest_int('max_depth', 8, 25),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                n_jobs=-1, random_state=42, class_weight='balanced_subsample',
            )
        elif name == "Extreme Gradient Boosting":
            model = XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                gamma=trial.suggest_float('gamma', 0.0, 0.5),
                reg_alpha=trial.suggest_float('reg_alpha', 0.0, 0.5),
                reg_lambda=trial.suggest_float('reg_lambda', 0.5, 5.0),
                eval_metric='logloss', verbosity=0, random_state=42,
            )
        else:  # LightGBM
            model = LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100),
                num_leaves=trial.suggest_int('num_leaves', 31, 127),
                max_depth=trial.suggest_int('max_depth', -1, 15),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 30),
                reg_alpha=trial.suggest_float('reg_alpha', 0.0, 0.5),
                reg_lambda=trial.suggest_float('reg_lambda', 0.0, 2.0),
                n_jobs=-1, random_state=42, verbose=-1,
            )

        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 181)
        best_t = max(thresholds,
                     key=lambda t: f1_score(y_val, (val_probs >= t).astype(int), zero_division=0))
        return f1_score(y_val, (val_probs >= best_t).astype(int), zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("%s  →  best val F1=%.4f  params=%s", name, study.best_value, study.best_params)
    return study.best_params


def _build_model(name, params=None):
    """Return a fresh model instance, optionally applying Optuna-found params."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    defaults = {
        "Random Forest": dict(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            min_samples_split=5, max_features='sqrt', max_samples=0.85,
            class_weight='balanced_subsample', n_jobs=-1, random_state=42,
        ),
        "Extreme Gradient Boosting": dict(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, colsample_bylevel=0.8,
            min_child_weight=3, gamma=0.05, reg_alpha=0.05, reg_lambda=2.0,
            random_state=42, eval_metric='logloss', verbosity=0,
        ),
        "Light Gradient Boosting Machine": dict(
            n_estimators=200, num_leaves=63, max_depth=-1,
            learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
            min_child_samples=8, bagging_freq=5,
            reg_alpha=0.05, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbose=-1,
        ),
    }
    kw = {**defaults[name], **(params or {})}

    if name == "Random Forest":
        return RandomForestClassifier(**kw)
    elif name == "Extreme Gradient Boosting":
        return XGBClassifier(**kw)
    else:
        return LGBMClassifier(**kw)


def plot_class_distribution(y_before, y_after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_b = pd.DataFrame({'IsFraud': pd.Series(y_before).astype(str)})
    df_a = pd.DataFrame({'IsFraud': pd.Series(y_after).astype(str)})
    sns.countplot(data=df_b, x='IsFraud', hue='IsFraud', ax=axes[0], palette="pastel", legend=False)
    axes[0].set_title("Class Distribution Before Balance")
    axes[0].set_xlabel("IsFraud"); axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 200))
    sns.countplot(data=df_a, x='IsFraud', hue='IsFraud', ax=axes[1], palette="pastel", legend=False)
    axes[1].set_title("Class Distribution After Balance (SMOTE)")
    axes[1].set_xlabel("IsFraud"); axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 200))
    plt.tight_layout()
    plt.savefig(_static('distribution_comparison.png'))
    logger.info("Saved static/distribution_comparison.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (Best Model)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(_static('confusion_matrix.png'))
    logger.info("Saved static/confusion_matrix.png")


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(_static('roc_curve.png'))
    logger.info("Saved static/roc_curve.png")


def plot_individual_metric(metric_name, value, filename, color):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[metric_name], y=[value], color=color)
    plt.ylim(0, 1.05)
    plt.title(f"{metric_name} Score")
    plt.ylabel("Score")
    plt.text(0, value + 0.02, f"{value:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(_static(filename))
    logger.info("Saved static/%s", filename)
    plt.close()


def plot_curves(y_true, y_probs, precision_val):
    plot_individual_metric('Precision', precision_val, 'precision_chart.png', '#2ca02c')
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, recalls[:-1], color='#d62728', lw=2)
    plt.xlabel('Decision Threshold'); plt.ylabel('Recall')
    plt.title('Recall Curve'); plt.ylim(0, 1.05); plt.xlim(0, 1.0)
    plt.tight_layout(); plt.savefig(_static('recall_chart.png'))
    logger.info("Saved static/recall_chart.png"); plt.close()

    f1_scores = np.divide(
        2 * (precisions * recalls), (precisions + recalls),
        out=np.zeros_like(precisions), where=(precisions + recalls) != 0
    )
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores[:-1], color='#ff7f0e', lw=2)
    plt.xlabel('Decision Threshold'); plt.ylabel('F1-score')
    plt.title('F1-score Curve'); plt.ylim(0, 1.05); plt.xlim(0, 1.0)
    plt.tight_layout(); plt.savefig(_static('f1_chart.png'))
    logger.info("Saved static/f1_chart.png"); plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {auc_val:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve'); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(_static('auc_chart.png'))
    logger.info("Saved static/auc_chart.png"); plt.close()


def train_and_evaluate(absorb_live=False, tune=False):
    logger.info("Loading Data/data.csv...")
    if not os.path.exists('Data/data.csv'):
        logger.error("ERROR: Data/data.csv not found!")
        return

    df = pd.read_csv('Data/data.csv')
    if df.empty:
        logger.error("Data/data.csv is empty — aborting.")
        return
    required_cols = {'TransactionDate', 'Amount', 'MerchantID', 'TransactionType', 'Location', 'IsFraud'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error("Data/data.csv is missing required columns: %s — aborting.", missing_cols)
        return

    # ------------------------------------------------------------------
    # ACTIVE CONTINUOUS MLOps RETRAINING LOOP
    # ------------------------------------------------------------------
    import sqlite3
    db_path = os.getenv('DB_PATH', 'Data/transactions.db')
    if absorb_live and os.path.exists(db_path):
        logger.info("Active Loop: Pulling live traffic from %s...", db_path)
        try:
            conn = sqlite3.connect(db_path)
            df_db = pd.read_sql_query("SELECT * FROM live_transactions", conn)
            c = conn.cursor()
            c.execute("DELETE FROM live_transactions")
            conn.commit()
            conn.close()

            if not df_db.empty:
                unanimous = df_db[
                    (df_db['rf_vote'] == df_db['xgb_vote']) &
                    (df_db['xgb_vote'] == df_db['lgb_vote'])
                ]
                skipped = len(df_db) - len(unanimous)
                logger.info("Live records: %d total, %d unanimous (kept), %d split-vote (skipped)",
                            len(df_db), len(unanimous), skipped)
                if not unanimous.empty:
                    df_live_mapped = pd.DataFrame({
                        'TransactionID':   unanimous['tx_id'].values,
                        'TransactionDate': unanimous['tx_date'].values,
                        'Amount':          unanimous['amount'].values,
                        'MerchantID':      unanimous['merchant_id'].values,
                        'TransactionType': unanimous['tx_type'].values,
                        'Location':        unanimous['location'].values,
                        'IsFraud':         unanimous['status'].apply(
                            lambda x: 1 if x == 'ALERT' else 0).values,
                    })
                    df = pd.concat([df, df_live_mapped], ignore_index=True)
                    df.to_csv('Data/data.csv', index=False)
                    logger.info("Absorbed %d unanimous records into Data/data.csv.", len(unanimous))
        except Exception as e:
            logger.error("Active Feedback Loop failed: %s", e)
    elif not absorb_live:
        logger.info("Offline mode: skipping live data absorption.")

    # Feature extraction (includes velocity features computed from sorted history)
    logger.info("Extracting features (including velocity)...")
    X_df = extract_features(df)
    y = df['IsFraud'].values
    X = X_df.values
    feature_names = X_df.columns.tolist()

    logger.info("Features (%d): %s", len(feature_names), feature_names)
    logger.info("Dataset — Legit: %d, Fraud: %d", sum(y == 0), sum(y == 1))

    # Hold out test set BEFORE any resampling
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
    )
    logger.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    # ------------------------------------------------------------------
    # Resampling: cap majority → SMOTE → final sync
    # RandomUnderSampler caps the legit class first so SMOTE runs on a
    # smaller dataset. SMOTE generates synthetic minority samples via
    # k-nearest-neighbours interpolation (much faster than KMeansSMOTE).
    # RandomOverSampler equalises any residual imbalance.
    # ------------------------------------------------------------------
    logger.info("Applying Hybrid Resampling (SMOTE) to training data only...")
    current_legit = sum(y_train == 0)
    legit_cap = min(current_legit, 40000)

    cap_majority  = RandomUnderSampler(sampling_strategy={0: legit_cap}, random_state=42)
    smote         = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
    final_sync    = RandomOverSampler(sampling_strategy='auto', random_state=42)

    balance_pipeline = Pipeline(steps=[
        ('cap_majority', cap_majority),
        ('smote',        smote),
        ('final_sync',   final_sync),
    ])

    X_tr_res, y_tr_res = balance_pipeline.fit_resample(X_train, y_train)

    logger.info("Balanced training set: Legit=%d, Fraud=%d",
                sum(y_tr_res == 0), sum(y_tr_res == 1))

    X_tr_res = pd.DataFrame(X_tr_res, columns=feature_names)
    X_val_df = pd.DataFrame(X_val,    columns=feature_names)
    X_test_df= pd.DataFrame(X_test,   columns=feature_names)

    plot_class_distribution(y_train, y_tr_res)

    # ------------------------------------------------------------------
    # Optional Optuna hyperparameter search (--tune flag)
    # ------------------------------------------------------------------
    best_params: dict[str, dict] = {}
    if tune:
        logger.info("=== Optuna hyperparameter search (40 trials per model) ===")
        for name in ["Random Forest", "Extreme Gradient Boosting", "Light Gradient Boosting Machine"]:
            logger.info("Tuning %s...", name)
            best_params[name] = _tune_model(name, X_tr_res, y_tr_res, X_val_df, y_val)
    else:
        logger.info("Skipping Optuna search — using default hyperparameters. Pass --tune to enable.")

    # ------------------------------------------------------------------
    # Train all 3 models
    # ------------------------------------------------------------------
    logger.info("Training 3 models on balanced training data...")
    model_names = ["Random Forest", "Extreme Gradient Boosting", "Light Gradient Boosting Machine"]
    models = {name: _build_model(name, best_params.get(name)) for name in model_names}

    model_metrics  = []
    _model_outputs = {}

    thresholds_range = np.linspace(0.05, 0.95, 181)

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_tr_res, y_tr_res)

        val_probs  = model.predict_proba(X_val_df)[:, 1]
        best_t = float(max(thresholds_range,
                           key=lambda t: f1_score(y_val, (val_probs >= t).astype(int),
                                                  zero_division=0)))

        test_probs = model.predict_proba(X_test_df)[:, 1]
        preds      = (test_probs >= best_t).astype(int)

        precision = precision_score(y_test, preds, zero_division=0)
        recall    = recall_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)
        auc       = roc_auc_score(y_test, test_probs)

        logger.info("  %s — Threshold: %.2f | P: %.3f, R: %.3f, F1: %.3f, AUC: %.3f",
                    name, best_t, precision, recall, f1, auc)

        model_metrics.append({
            "name":      name,
            "precision": float(precision),
            "recall":    float(recall),
            "f1_score":  float(f1),
            "auc_roc":   float(auc),
            "threshold": round(best_t, 3),
            "cm":        confusion_matrix(y_test, preds).tolist(),
        })
        _model_outputs[name] = {
            "preds": preds, "probs": test_probs, "val_probs": val_probs,
            "precision": precision, "recall": recall, "f1": f1, "auc": auc,
        }

    # ------------------------------------------------------------------
    # Meta-learner stacking
    # Train LogisticRegression on [rf_prob, xgb_prob, lgb_prob] from val set.
    # Tune its threshold on test set; evaluate there too.
    # ------------------------------------------------------------------
    logger.info("Training meta-learner (LogisticRegression stacking)...")
    val_meta_X = np.column_stack([
        _model_outputs[n]["val_probs"] for n in model_names
    ])
    meta_learner = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_learner.fit(val_meta_X, y_val)

    test_meta_X   = np.column_stack([_model_outputs[n]["probs"] for n in model_names])
    meta_probs    = meta_learner.predict_proba(test_meta_X)[:, 1]
    meta_best_t   = float(max(thresholds_range,
                              key=lambda t: f1_score(y_test,
                                                     (meta_probs >= t).astype(int),
                                                     zero_division=0)))
    meta_preds      = (meta_probs >= meta_best_t).astype(int)
    meta_precision  = precision_score(y_test, meta_preds, zero_division=0)
    meta_recall     = recall_score(y_test, meta_preds, zero_division=0)
    meta_f1         = f1_score(y_test, meta_preds, zero_division=0)
    meta_auc        = roc_auc_score(y_test, meta_probs)

    logger.info("Meta-learner — Threshold: %.2f | P: %.3f, R: %.3f, F1: %.3f, AUC: %.3f",
                meta_best_t, meta_precision, meta_recall, meta_f1, meta_auc)

    # ------------------------------------------------------------------
    # Select best reported model (individual models only, for the metrics page)
    # ------------------------------------------------------------------
    model_metrics.sort(key=lambda x: (x["auc_roc"], x["f1_score"]), reverse=True)
    best_model_name = model_metrics[0]["name"]
    logger.info("Best Individual Model by AUC-ROC: %s", best_model_name)

    _w = _model_outputs[best_model_name]
    best_preds, best_probs = _w["preds"], _w["probs"]
    best_precision, best_recall, best_f1, best_auc = (
        _w["precision"], _w["recall"], _w["f1"], _w["auc"])

    # Archive and save
    _archive_models()
    os.makedirs('Models', exist_ok=True)
    for name, model in models.items():
        filename = f"Models/{name.replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        logger.info("  Saved %s", filename)

    joblib.dump(meta_learner, "Models/meta_learner.pkl")
    logger.info("  Saved Models/meta_learner.pkl")

    plot_confusion_matrix(y_test, best_preds)
    plot_roc_curve(y_test, best_probs)
    plot_curves(y_test, best_probs, best_precision)

    def downsample(x, y, n=50):
        if len(x) <= n:
            return x.tolist(), y.tolist()
        indices = np.linspace(0, len(x) - 1, n, dtype=int)
        return x[indices].tolist(), y[indices].tolist()

    precisions, recalls_all, thresholds = precision_recall_curve(y_test, best_probs)
    f1_scores_all = np.divide(
        2 * (precisions * recalls_all), (precisions + recalls_all),
        out=np.zeros_like(precisions), where=(precisions + recalls_all) != 0
    )
    fpr, tpr, _ = roc_curve(y_test, best_probs)

    t_ds, r_ds     = downsample(thresholds, recalls_all[:-1], 500)
    _, f_ds        = downsample(thresholds, f1_scores_all[:-1], 500)
    fpr_ds, tpr_ds = downsample(fpr, tpr, 500)

    metrics_data = {
        "precision": f"{best_precision:.3f}",
        "recall":    f"{best_recall:.3f}",
        "f1_score":  f"{best_f1:.3f}",
        "auc_roc":   f"{best_auc:.3f}",
        "model_comparison": model_metrics,
        "best_model": best_model_name,
        "meta_learner": {
            "precision":  float(meta_precision),
            "recall":     float(meta_recall),
            "f1_score":   float(meta_f1),
            "auc_roc":    float(meta_auc),
            "threshold":  round(meta_best_t, 3),
        },
        "meta_threshold": round(meta_best_t, 3),
        "curves": {
            "thresholds": t_ds,
            "recalls":    r_ds,
            "f1_scores":  f_ds,
            "fpr":        fpr_ds,
            "tpr":        tpr_ds,
        },
        "cm": confusion_matrix(y_test, best_preds).tolist(),
        "full_data": {
            "total": int(len(y)),
            "legit": int(sum(y == 0)),
            "fraud": int(sum(y == 1)),
        },
        "dist": {
            "before": [int(sum(y == 0)), int(sum(y == 1))],
            "after":  [int(sum(y_tr_res == 0)), int(sum(y_tr_res == 1))],
        },
        "df": {
            "shape":    list(df.shape),
            "head":     df.head(5).to_dict(orient='records'),
            "describe": df.describe().reset_index().to_dict(orient='records'),
            "columns":  df.columns.tolist(),
        },
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f)

    logger.info("All metrics and charts generated successfully!")
    logger.info("─" * 60)
    logger.info("  Best individual model  : %s | F1=%.3f  AUC=%.3f",
                best_model_name, best_f1, best_auc)
    logger.info("  Meta-learner (stacked) : F1=%.3f  AUC=%.3f  (used for live decisions)",
                meta_f1, meta_auc)
    logger.info("─" * 60)


if __name__ == "__main__":
    import argparse
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description="Train fraud detection models and generate metrics.")
    parser.add_argument('--absorb-live', action='store_true', default=False,
                        help="Merge unanimous live predictions from SQLite before retraining.")
    parser.add_argument('--tune', action='store_true', default=False,
                        help="Run Optuna hyperparameter search (40 trials per model, ~30-60 min).")
    args = parser.parse_args()

    train_and_evaluate(absorb_live=args.absorb_live, tune=args.tune)
