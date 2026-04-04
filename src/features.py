import pandas as pd
from datetime import datetime

# Shared mappings
TYPES = {'refund': 0, 'purchase': 1}
# Creating an arbitrary mapping for testing. 
LOCATIONS = {
    'San Antonio': 0, 'Dallas': 1, 'New York': 2, 'Philadelphia': 3,
    'Phoenix': 4, 'Chicago': 5, 'San Jose': 6, 'San Diego': 7,
    'Houston': 8, 'Los Angeles': 9
}

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw DataFrame containing data.csv records into
    the numerical feature array expected by the ML Models.
    """
    df_feat = df.copy()
    
    # 1. Parse dates using pandas
    df_feat['TransactionDate'] = pd.to_datetime(df_feat['TransactionDate'])
    df_feat['Hour'] = df_feat['TransactionDate'].dt.hour
    df_feat['DayOfWeek'] = df_feat['TransactionDate'].dt.dayofweek
    
    # 2. Map Categoricals
    df_feat['TransactionType'] = df_feat['TransactionType'].map(TYPES).fillna(1)
    df_feat['Location'] = df_feat['Location'].map(LOCATIONS).fillna(0)
    
    # 3. Select final columns in exact order
    feature_cols = ['Amount', 'MerchantID', 'TransactionType', 'Location', 'Hour', 'DayOfWeek']
    
    return df_feat[feature_cols]

def process_single_transaction(raw_dict: dict) -> list:
    """
    Used by the Kafka prepocessing stream.
    raw_dict maps to exactly a single row of data.csv
    """
    # Quick string date parsing
    dt = datetime.strptime(raw_dict['TransactionDate'], "%Y-%m-%d %H:%M:%S.%f")
    
    hour = dt.hour
    dayofweek = dt.weekday()
    
    # Map
    t_type = TYPES.get(raw_dict.get('TransactionType', 'purchase'), 1)
    loc = LOCATIONS.get(raw_dict.get('Location', ''), 0)
    
    # Must match the order in feature_cols above!
    features = [
        float(raw_dict['Amount']),
        int(raw_dict['MerchantID']),
        t_type,
        loc,
        hour,
        dayofweek
    ]
    return features
