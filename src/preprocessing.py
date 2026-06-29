import json
import logging
import os
import time
import threading
import uuid
from collections import deque
from kafka import KafkaConsumer, KafkaProducer

from features import process_single_transaction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
INPUT_TOPIC  = 'raw-transactions'
OUTPUT_TOPIC = 'processed-transactions'


class VelocityTracker:
    """Per-merchant sliding-window tracker for live velocity feature computation.

    Each transaction records (unix_timestamp, amount) in a per-merchant deque.
    Stats are computed over the PRIOR history (before the current tx is added),
    matching the look-back semantics used in extract_features() at training time.
    """

    def __init__(self, window_1h: int = 3600, window_24h: int = 86400):
        self._history: dict[int, deque] = {}
        self._lock = threading.Lock()
        self._window_1h  = window_1h
        self._window_24h = window_24h

    def get_stats(self, merchant_id: int, amount: float, timestamp: float) -> dict:
        """Return velocity stats for a transaction and record it in history."""
        with self._lock:
            if merchant_id not in self._history:
                self._history[merchant_id] = deque()

            hist = self._history[merchant_id]

            # Drop entries older than 24h
            cutoff_24h = timestamp - self._window_24h
            while hist and hist[0][0] < cutoff_24h:
                hist.popleft()

            # Count prior transactions inside each window
            cutoff_1h = timestamp - self._window_1h
            v1h  = sum(1 for ts, _ in hist if ts >= cutoff_1h)
            v24h = len(hist)

            # Amount z-score vs merchant's running history (prior txns only)
            if hist:
                past_amounts = [a for _, a in hist]
                mean_a = sum(past_amounts) / len(past_amounts)
                var_a  = sum((a - mean_a) ** 2 for a in past_amounts) / len(past_amounts)
                std_a  = max(var_a ** 0.5, 1e-6)
                avz    = max(-5.0, min(5.0, (amount - mean_a) / std_a))
            else:
                avz = 0.0

            # Record this transaction in history
            hist.append((timestamp, amount))

        return {
            'tx_velocity_1h':  v1h,
            'tx_velocity_24h': v24h,
            'amount_vs_avg':   avz,
        }


_velocity_tracker = VelocityTracker()


def process_transaction(raw_tx: dict) -> dict:
    timestamp   = raw_tx.get('timestamp', time.time())
    merchant_id = int(raw_tx.get('MerchantID', 0))
    amount      = float(raw_tx.get('Amount', 0))

    velocity_stats = _velocity_tracker.get_stats(merchant_id, amount, timestamp)
    features = process_single_transaction(raw_tx, velocity_stats=velocity_stats)

    return {
        'tx_id':     raw_tx.get('tx_id', str(uuid.uuid4())),
        'features':  features,
        'timestamp': timestamp,
        'raw': {
            'TransactionDate': raw_tx.get('TransactionDate', ''),
            'Amount':          amount,
            'MerchantID':      merchant_id,
            'TransactionType': raw_tx.get('TransactionType', ''),
            'Location':        raw_tx.get('Location', ''),
        }
    }


def start_preprocessing():
    logger.info("Starting Preprocessing Service...")
    consumer = producer = None

    while True:
        try:
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
            logger.info("Listening to '%s' → publishing to '%s'...", INPUT_TOPIC, OUTPUT_TOPIC)

            for message in consumer:
                raw_tx = message.value
                try:
                    processed_tx = process_transaction(raw_tx)
                    producer.send(OUTPUT_TOPIC, processed_tx)
                    logger.info("Processed TX: %s", processed_tx['tx_id'])
                except Exception as e:
                    logger.error("Failed to process transaction %s: %s",
                                 raw_tx.get('tx_id', '?'), e)

        except KeyboardInterrupt:
            logger.info("Stopping preprocessing service.")
            break
        except Exception as e:
            logger.error("Preprocessing error: %s. Reconnecting in 5s...", e)
            time.sleep(5)
        finally:
            for obj in (consumer, producer):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
            consumer = producer = None


if __name__ == "__main__":
    logger.info("Waiting 10s for Kafka to start...")
    time.sleep(10)
    start_preprocessing()
