import json
import time
import pickle
import numpy as np
from xgboost import XGBClassifier
from kafka import KafkaConsumer, KafkaProducer

KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC = 'processed-transactions'
OUTPUT_TOPIC = 'scored-transactions'

# Load Models
print("Loading Models...")
try:
    with open('Models/Random_Forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
        
    with open('Models/Extreme_Gradient_Boosting_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open('Models/Light_Gradient_Boosting_Machine_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
        
    print("All models loaded successfully from Models/!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def decide(features):
    """
    Run features through all 3 models and apply voting.
    """
    # Reshape features to 2D array [1, num_features]
    X = np.array(features).reshape(1, -1)
    
    rf_pred = int(rf_model.predict(X)[0])
    xgb_pred = int(pdb_model.predict(X)[0])
    lgb_pred = int(lgb_model.predict(X)[0])
    
    # Majority vote
    total_votes = rf_pred + xgb_pred + lgb_pred
    is_fraud = total_votes >= 2
    
    return {
        'rf': rf_pred,
        'xgb': xgb_pred,
        'lgb': lgb_pred,
        'is_fraud': bool(is_fraud),
        'status': 'ALERT' if is_fraud else 'LEGIT'
    }

def start_decision_engine():
    print("Starting Decision Engine...")
    
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print(f"Listening to '{INPUT_TOPIC}' and publishing to '{OUTPUT_TOPIC}'...")
    
    try:
        for message in consumer:
            tx = message.value
            
            # Predict
            decision = decide(tx['features'])
            
            scored_tx = {
                'tx_id': tx['tx_id'],
                'timestamp': tx['timestamp'],
                'rf_pred': decision['rf'],
                'xgb_pred': decision['xgb'],
                'lgb_pred': decision['lgb'],
                'is_fraud': decision['is_fraud'],
                'status': decision['status']
            }
            
            # Publish
            producer.send(OUTPUT_TOPIC, scored_tx)
            
            # Print Alert
            color = '\033[91m' if decision['status'] == 'ALERT' else '\033[92m'
            reset = '\033[0m'
            print(f"{color}[{decision['status']}] TX: {tx['tx_id']} | Votes: RF={decision['rf']}, XGB={decision['xgb']}, LGB={decision['lgb']}{reset}")
            
    except KeyboardInterrupt:
        print("\nStopping decision engine.")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    print("Waiting 10s for Kafka to start...")
    time.sleep(10)
    start_decision_engine()
