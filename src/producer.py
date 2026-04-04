import os
import json
import time
import random
import uuid
import csv
from kafka import KafkaProducer

KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'raw-transactions'

def get_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer
    except Exception as e:
        print(f"Failed to connect to Kafka at {KAFKA_BROKER}: {e}")
        return None

def stream_transactions():
    producer = get_producer()
    if not producer:
        print("Exiting producer...")
        return

    csv_file = 'Data/data.csv'
    if not os.path.exists(csv_file):
         print(f"ERROR: {csv_file} not found!")
         return

    print(f"Started producing transactions from {csv_file} to topic '{TOPIC_NAME}'...")
    
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # We can add a fake UUID so the UI has a consistent tx identifier if needed, 
                # or just use the original TransactionID.
                tx_id = row.get('TransactionID', str(uuid.uuid4()))
                
                # We send the entire CSV row exactly as a dictionary
                # excluding the 'IsFraud' label if we want to simulate a pure inference string,
                # though it doesn't matter since preprocessing extracts only the correct features.
                tx = {
                    'tx_id': str(tx_id),
                    'timestamp': time.time(),
                    **row
                }
                
                producer.send(TOPIC_NAME, tx)
                print(f"Produced TX: {tx['tx_id']} | Amount: ${tx['Amount']}")
                
                # Simulate real-time delay between transactions
                time.sleep(random.uniform(0.1, 1.0))
                
    except KeyboardInterrupt:
        print("\nStopping producer.")
    finally:
        producer.close()

if __name__ == "__main__":
    # Wait for Kafka to be ready
    print("Waiting 10s for Kafka to start...")
    time.sleep(10)
    stream_transactions()
