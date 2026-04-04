import os
import sys
import json
import uuid
import time
from kafka import KafkaConsumer, KafkaProducer

# Import the shared feature logic
from features import process_single_transaction

KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC = 'raw-transactions'
OUTPUT_TOPIC = 'processed-transactions'

def process_transaction(raw_tx):
    """
    Extract the features exactly as expected by the ML models using the shared extraction module.
    """
    
    # Run the raw JSON dictionary through our processor
    features = process_single_transaction(raw_tx)
    
    processed_tx = {
        'tx_id': raw_tx.get('tx_id', str(uuid.uuid4())),
        'features': features,
        'timestamp': raw_tx.get('timestamp', time.time())
    }
    return processed_tx

def start_preprocessing():
    print(f"Starting Preprocessing Service...")
    
    # Setup Consumer
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    # Setup Producer
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print(f"Listening to '{INPUT_TOPIC}' and publishing to '{OUTPUT_TOPIC}'...")
    
    try:
        for message in consumer:
            raw_tx = message.value
            
            # Preprocess
            try:
                processed_tx = process_transaction(raw_tx)
                
                # Publish
                producer.send(OUTPUT_TOPIC, processed_tx)
                print(f"Processed TX: {processed_tx['tx_id']}")
            except Exception as e:
                 print(f"Failed to process transaction: {e}")
            
    except KeyboardInterrupt:
        print("\nStopping preprocessing service.")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    print("Waiting 10s for Kafka to start...")
    time.sleep(10)
    start_preprocessing()
