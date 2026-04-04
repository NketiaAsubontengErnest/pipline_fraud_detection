import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

from features import extract_features

# Set seed for reproducibility
np.random.seed(42)

def train_and_save_models():
    # 1. Load Real Data
    print("Loading Data/data.csv...")
    if not os.path.exists('Data/data.csv'):
        print("ERROR: Data/data.csv not found!")
        return
        
    df = pd.read_csv('Data/data.csv')
    
    # 2. Extract Features using the shared logic
    print("Extracting features...")
    X_df = extract_features(df)
    y = df['IsFraud'].values
    
    # Convert features to numpy array for training
    X = X_df.values
    feature_names = X_df.columns.tolist()
    
    print(f"Features used: {feature_names}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Models
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print("RF Accuracy:", accuracy_score(y_test, rf_preds))
    
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    print("XGB Accuracy:", accuracy_score(y_test, xgb_preds))
    
    print("\nTraining LightGBM...")
    lgb_model = LGBMClassifier(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    print("LGB Accuracy:", accuracy_score(y_test, lgb_preds))
    
    # 4. Save Models
    os.makedirs('Models', exist_ok=True)
    
    print("\nSaving models to 'Models/' directory...")
    with open('Models/Random_Forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    with open('Models/Extreme_Gradient_Boosting_model.pkl', 'wb') as f:
        # Saving as pkl for consistency with other scripts
        pickle.dump(xgb_model, f)
        
    with open('Models/Light_Gradient_Boosting_Machine_model.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
        
    print("Models saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
