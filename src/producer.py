import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
TOPIC_NAME   = 'raw-transactions'

LOCATIONS = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
]
# Merchant IDs 1-30 are flagged high-risk in the training data
HIGH_RISK_MERCHANTS = list(range(1, 31))
NORMAL_MERCHANTS    = list(range(31, 1001))


def _generate_transaction():
    """
    Generate one realistic transaction.
    Fraud probability follows the same rules used to build the training set:
      - High amount (>4000) + late night (11 pm – 4 am)  → +0.65
      - Refund + high amount (>3500)                      → +0.45
      - High-risk merchant (ID 1-30) + amount >2500       → +0.50
      - Weekend + late night (10 pm-1 am) + amount >3500  → +0.30
    Base noise rate for non-suspicious rows               →  0.4 %
    """
    now     = datetime.now()
    hour    = now.hour
    weekday = now.weekday()

    amount      = round(random.uniform(1.0, 5000.0), 2)
    tx_type     = random.choices(['purchase', 'refund'], weights=[70, 30])[0]
    location    = random.choice(LOCATIONS)

    # Bias merchant selection: ~3 % chance of a high-risk merchant
    if random.random() < 0.03:
        merchant_id = random.choice(HIGH_RISK_MERCHANTS)
    else:
        merchant_id = random.choice(NORMAL_MERCHANTS)

    fp = 0.0
    if amount > 4000 and hour in {23, 0, 1, 2, 3, 4}:
        fp += 0.65
    if tx_type == 'refund' and amount > 3500:
        fp += 0.45
    if merchant_id <= 30 and amount > 2500:
        fp += 0.50
    if weekday >= 5 and hour in {22, 23, 0, 1} and amount > 3500:
        fp += 0.30

    fp = min(fp, 0.95) if fp > 0 else 0.004
    is_fraud = int(random.random() < fp)

    return {
        'tx_id':           str(uuid.uuid4()),
        'timestamp':       time.time(),
        'TransactionID':   str(random.randint(100001, 999999)),
        'TransactionDate': now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        'Amount':          amount,
        'MerchantID':      merchant_id,
        'TransactionType': tx_type,
        'Location':        location,
        'IsFraud':         is_fraud,  # ground-truth label; not sent to models
    }


def stream_transactions():
    tx_count = fraud_count = 0
    producer = None
    while True:
        try:
            logger.info("Connecting to Kafka at %s...", KAFKA_BROKER)
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            )
            logger.info("Connected. Streaming to '%s' (Ctrl-C to stop)...", TOPIC_NAME)

            while True:
                tx = _generate_transaction()
                producer.send(TOPIC_NAME, tx)

                tx_count    += 1
                fraud_count += tx['IsFraud']
                label = 'FRAUD' if tx['IsFraud'] else 'LEGIT'
                logger.info(
                    "[%s] TX#%d | $%.2f | Merchant %d | %s | %s | %s  (fraud rate: %.1f%%)",
                    label, tx_count, tx['Amount'], tx['MerchantID'],
                    tx['TransactionType'], tx['Location'], tx['TransactionDate'][11:19],
                    fraud_count / tx_count * 100,
                )

                time.sleep(random.choices(
                    [0.3, 0.8, 2.0, 5.0],
                    weights=[60, 25, 10, 5]
                )[0])

        except KeyboardInterrupt:
            logger.info("Stopped. Sent %d transactions (%d fraud).", tx_count, fraud_count)
            break
        except Exception as e:
            logger.error("Producer error: %s. Reconnecting in 5s...", e)
            time.sleep(5)
        finally:
            if producer is not None:
                try:
                    producer.close()
                except Exception:
                    pass
                producer = None


if __name__ == "__main__":
    logger.info("Waiting 10s for Kafka to start...")
    time.sleep(10)
    stream_transactions()
