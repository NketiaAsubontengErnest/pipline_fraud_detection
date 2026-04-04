import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
    precision_recall_curve
)
from imblearn.combine import SMOTETomek

from features import extract_features

def plot_class_distribution(y_before, y_after):
    """Plot the before & after class distribution side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before SMOTE
    sns.countplot(x=y_before, ax=axes[0], palette="pastel")
    axes[0].set_title("Class Distribution Before Balance")
    axes[0].set_xlabel("IsFraud")
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 500))
        
    # After SMOTE
    sns.countplot(x=y_after, ax=axes[1], palette="pastel")
    axes[1].set_title("Class Distribution After Balance (SMOTE)")
    axes[1].set_xlabel("IsFraud")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 500))
        
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    print("Saved 'distribution_comparison.png'")
    # Note: If running on a headless server, don't use plt.show() here.

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (Random Forest)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('confusion_matrix.png')
    print("Saved 'confusion_matrix.png'")

def plot_roc_curve(y_true, y_probs):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("Saved 'roc_curve.png'")

def plot_individual_metric(metric_name, value, filename, color):
    """Plot a single metric as its own individual bar chart."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[metric_name], y=[value], color=color)
    plt.ylim(0, 1.05)
    plt.title(f"{metric_name} Score")
    plt.ylabel("Score")
    
    # Add text label above the bar
    plt.text(0, value + 0.02, f"{value:.3f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved '{filename}'")
    plt.close() # Close to prevent overlapping in the next plot

def plot_curves(y_true, y_probs, precision_val):
    """Plot Precision as bar, and Recall, F1, AUC as curves."""
    # 1. Precision (Bar chart as requested to remain or just because not specified)
    plot_individual_metric('Precision', precision_val, 'precision_chart.png', '#2ca02c')

    # Calculate curves
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # 2. Recall Curve
    plt.figure(figsize=(8, 6))
    # precision_recall_curve returns thresholds len N, and recalls len N+1
    plt.plot(thresholds, recalls[:-1], color='#d62728', lw=2)
    plt.xlabel('Decision Threshold')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig('recall_chart.png')
    print("Saved 'recall_chart.png'")
    plt.close()

    # 3. F1-score Curve
    # calculate F1 for each threshold
    f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), out=np.zeros_like(precisions), where=(precisions + recalls)!=0)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores[:-1], color='#ff7f0e', lw=2)
    plt.xlabel('Decision Threshold')
    plt.ylabel('F1-score')
    plt.title('F1-score Curve')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig('f1_chart.png')
    print("Saved 'f1_chart.png'")
    plt.close()

    # 4. AUC-ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {auc_val:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('auc_chart.png')
    print("Saved 'auc_chart.png'")
    plt.close()

    
def train_and_evaluate():
    print("Loading Data/data.csv...")
    if not os.path.exists('Data/data.csv'):
        print("ERROR: Data/data.csv not found!")
        return
        
    df = pd.read_csv('Data/data.csv')
    
    # --------------------------------------------------------------------------------------------------
    # ACTIVE CONTINUOUS MLOPS RETRAINING LOOP:
    # --------------------------------------------------------------------------------------------------
    # Dynamically query the physical SQLite history to retrieve all natively scored live data!
    # Convert 'LEGIT/ALERT' determinations back into ground truth '0/1' binaries respectively.
    # Append dynamically to batch dataset & overwrite data.csv permanently so models organically
    # 'remember' concept drifting vectors organically indefinitely!
    import sqlite3
    if os.path.exists('Data/transactions.db'):
        print("Evaluating Active Loop: Pulling live traffic histories from Data/transactions.db...")
        try:
            conn = sqlite3.connect('Data/transactions.db')
            df_db = pd.read_sql_query("SELECT * FROM live_transactions", conn)
            
            # Wipe the temporary staging db natively after it gets exported so it doesn't duplicate loops safely!
            c = conn.cursor()
            c.execute("DELETE FROM live_transactions")
            conn.commit()
            conn.close()
            
            if not df_db.empty:
                print(f"Absorbing {len(df_db)} uniquely generated live predictions into master AI dataset!")
                df_live_mapped = pd.DataFrame()
                df_live_mapped['TransactionID'] = df_db['tx_id']
                df_live_mapped['TransactionDate'] = df_db['tx_date']
                df_live_mapped['Amount'] = df_db['amount']
                df_live_mapped['MerchantID'] = df_db['merchant_id']
                df_live_mapped['TransactionType'] = df_db['tx_type']
                df_live_mapped['Location'] = df_db['location']
                df_live_mapped['IsFraud'] = df_db['status'].apply(lambda x: 1 if x == 'ALERT' else 0)
                
                df = pd.concat([df, df_live_mapped], ignore_index=True)
                
                # Commit directly back to physical CSV file to memorialize the retraining permanently
                df.to_csv('Data/data.csv', index=False)
                print("Memory updated successfully explicitly onto Data/data.csv.")
        except Exception as e:
            print(f"Active Feedback Loop Failed organically: {e}")
    
    # Extract Features
    X_df = extract_features(df)
    y = df['IsFraud'].values
    X = X_df.values
    feature_names = X_df.columns.tolist()
    
    print(f"\nExtracted Features: {feature_names}")
    
    print(f"\nOriginal Entire Dataset - Legit: {sum(y == 0)}, Fraud: {sum(y == 1)}")
    
    # 1. APPLY BALANCING (Under and Over-sampling) BEFORE SPLITTING
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    
    print("Applying Pipeline (Under and Over-sampling) to balance the entire data...")
    over = SMOTE(sampling_strategy={1: 50000}, random_state=42)
    under = RandomUnderSampler(sampling_strategy={0: 50000}, random_state=42)
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    
    print(f"Resampled Entire Dataset - Legit: {sum(y_resampled == 0)}, Fraud: {sum(y_resampled == 1)}")
    
    # Plot Distribution Before/After
    plot_class_distribution(y, y_resampled)

    # Split into Train (70%), Validation (15%), Test (15%) AFTER balancing
    # First split into Train+Val (85%) and Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.15, random_state=42, stratify=y_resampled)
    # Then split Train+Val into Train (70/85) and Val (15/85)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp)
    
    print(f"\n--- DATA SPLITS ---")
    print(f"Total dataset after hybrid balancing: {len(X_resampled)} samples")
    print(f"Training set    : {len(X_train)} samples (70%)")
    print(f"Validation set  : {len(X_val)} samples (15%)")
    print(f"Testing set     : {len(X_test)} samples (15%)")
    
    # 2. TRAIN MODELS
    print("\nTraining 3 Models on Balanced Data...")
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extreme Gradient Boosting": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        "Light Gradient Boosting Machine": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }
    
    model_metrics = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Get probabilities for positive class (if model supports it)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = preds  # fallback
            
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        print(f"  {name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        model_metrics.append({
            "name": name,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "cm": confusion_matrix(y_test, preds).tolist()
        })
        
        # We will use Random Forest for the large curves as baseline/best initially
        if name == "Random Forest":
            best_preds = preds
            best_probs = probs
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_auc = auc

    print("\nEvaluating on Test Set (15% Holdout) with best model (Random Forest)...")
    
    import joblib
    
    # Sort models by F1-score to determine the actual best
    model_metrics.sort(key=lambda x: x["f1_score"], reverse=True)
    best_model_name = model_metrics[0]["name"]
    print(f"\nBest Model by F1-Score: {best_model_name}")

    # Serialize models to risk scoring endpoints can load them
    print("Caching trained models in Models/ directory for API evaluation...")
    if not os.path.exists('Models'):
        os.makedirs('Models')
        
    for name, model in models.items():
        filename = f"Models/{name.replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)

    # For original visualization, we keep using Random Forest's values as requested previously
    # Or actually, we can export the model_metrics list to json.
    
    # e) Plot Confusion Matrix & ROC & Curves (using best default config for plot images)
    plot_confusion_matrix(y_test, best_preds)
    plot_roc_curve(y_test, best_probs)
    plot_curves(y_test, best_probs, best_precision)
    
    # Save metrics to JSON file
    # helper for downsampling curves for the browser
    def downsample(x, y, n=50):
        if len(x) <= n:
            return x.tolist(), y.tolist()
        indices = np.linspace(0, len(x)-1, n, dtype=int)
        return x[indices].tolist(), y[indices].tolist()

    precisions, recalls_all, thresholds = precision_recall_curve(y_test, best_probs)
    f1_scores_all = np.divide(2 * (precisions * recalls_all), (precisions + recalls_all), out=np.zeros_like(precisions), where=(precisions + recalls_all)!=0)
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    cm = confusion_matrix(y_test, best_preds)
    
    t_ds, r_ds = downsample(thresholds, recalls_all[:-1], 50)
    _, f_ds = downsample(thresholds, f1_scores_all[:-1], 50)
    fpr_ds, tpr_ds = downsample(fpr, tpr, 50)

    import json
    metrics_data = {
        "precision": f"{best_precision:.3f}",
        "recall": f"{best_recall:.3f}",
        "f1_score": f"{best_f1:.3f}",
        "auc_roc": f"{best_auc:.3f}",
        "model_comparison": model_metrics,
        "best_model": best_model_name,
        "curves": {
            "thresholds": t_ds,
            "recalls": r_ds,
            "f1_scores": f_ds,
            "fpr": fpr_ds,
            "tpr": tpr_ds
        },
        "cm": cm.tolist(),
        "dist": {
            "before": [int(sum(y == 0)), int(sum(y == 1))],
            "after": [int(sum(y_resampled == 0)), int(sum(y_resampled == 1))]
        },
        "df": {
            "shape": list(df.shape),
            "head": df.head(5).to_dict(orient='records'),
            "describe": df.describe().reset_index().to_dict(orient='records'),
            "columns": df.columns.tolist()
        }
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f)
        
    print("\nAll metrics and graphs have been generated!")

if __name__ == "__main__":
    import os
    # Ensure matplotlib plots are created headless instead of attempting to spawn GUI windows
    import matplotlib
    matplotlib.use('Agg')
    
    train_and_evaluate()
