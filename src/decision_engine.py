import json
import logging
import os
import time
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
INPUT_TOPIC  = 'processed-transactions'
OUTPUT_TOPIC = 'scored-transactions'

_MODEL_PATHS = {
    'rf':  'Models/Random_Forest_model.pkl',
    'xgb': 'Models/Extreme_Gradient_Boosting_model.pkl',
    'lgb': 'Models/Light_Gradient_Boosting_Machine_model.pkl',
}
META_LEARNER_PATH = 'Models/meta_learner.pkl'
METRICS_PATH      = 'metrics.json'

_NAME_TO_KEY = {
    'Random Forest':                   'rf',
    'Extreme Gradient Boosting':       'xgb',
    'Light Gradient Boosting Machine': 'lgb',
}
_DEFAULT_THRESHOLDS = {'rf': 0.10, 'xgb': 0.10, 'lgb': 0.10}
_DEFAULT_META_THRESHOLD = 0.50

_models        = {}
_model_mtimes  = {}
_thresholds    = {}
_metrics_mtime = None
_meta_learner  = None
_meta_threshold = _DEFAULT_META_THRESHOLD
_meta_mtime    = None


def _files_changed():
    for key, path in _MODEL_PATHS.items():
        try:
            mtime = os.path.getmtime(path)
            if _model_mtimes.get(key) != mtime:
                return True
        except OSError:
            return True
    return False


def _load_thresholds():
    global _metrics_mtime, _meta_threshold
    try:
        mtime = os.path.getmtime(METRICS_PATH)
        if _thresholds and _metrics_mtime == mtime:
            return
        with open(METRICS_PATH) as f:
            data = json.load(f)
        for m in data.get('model_comparison', []):
            key = _NAME_TO_KEY.get(m['name'])
            if key and 'threshold' in m:
                _thresholds[key] = m['threshold']
        if 'meta_threshold' in data:
            _meta_threshold = data['meta_threshold']
        _metrics_mtime = mtime
        logger.info("Loaded thresholds: %s  meta=%.2f", _thresholds, _meta_threshold)
    except Exception as e:
        logger.warning("Could not load thresholds (%s); using defaults.", e)
        _thresholds.update(_DEFAULT_THRESHOLDS)


def _load_meta_learner():
    global _meta_learner, _meta_mtime
    if not os.path.exists(META_LEARNER_PATH):
        return
    try:
        mtime = os.path.getmtime(META_LEARNER_PATH)
        if _meta_learner is not None and _meta_mtime == mtime:
            return
        _meta_learner = joblib.load(META_LEARNER_PATH)
        _meta_mtime   = mtime
        logger.info("Meta-learner loaded from %s", META_LEARNER_PATH)
    except Exception as e:
        logger.warning("Could not load meta-learner: %s", e)


def _load_models():
    if _models and not _files_changed():
        return
    if _models:
        logger.info("Model files updated — hot-reloading...")
    else:
        logger.info("Loading models...")
    try:
        for key, path in _MODEL_PATHS.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            _models[key] = joblib.load(path)
            _model_mtimes[key] = os.path.getmtime(path)
        _thresholds.clear()
        _load_thresholds()
        _load_meta_learner()
        logger.info("All models loaded from Models/!")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")


def decide(features):
    """
    Score a single transaction.

    Decision hierarchy:
      1. If meta-learner (stacking) is available → use its probability + meta_threshold.
      2. Fallback: majority vote across the 3 base models (≥2/3 = fraud).

    Returns votes, raw probabilities, meta probability (if available), and final status.
    """
    _load_models()

    try:
        from features import FEATURE_COLS
        X = pd.DataFrame([features], columns=FEATURE_COLS)
    except Exception:
        X = np.array(features).reshape(1, -1)

    t = _thresholds if _thresholds else _DEFAULT_THRESHOLDS

    rf_prob  = float(_models['rf'].predict_proba(X)[0][1])
    xgb_prob = float(_models['xgb'].predict_proba(X)[0][1])
    lgb_prob = float(_models['lgb'].predict_proba(X)[0][1])

    rf_pred  = int(rf_prob  >= t.get('rf',  _DEFAULT_THRESHOLDS['rf']))
    xgb_pred = int(xgb_prob >= t.get('xgb', _DEFAULT_THRESHOLDS['xgb']))
    lgb_pred = int(lgb_prob >= t.get('lgb', _DEFAULT_THRESHOLDS['lgb']))

    # Meta-learner decision (preferred path)
    meta_prob = None
    if _meta_learner is not None:
        meta_X    = np.array([[rf_prob, xgb_prob, lgb_prob]])
        meta_prob = float(_meta_learner.predict_proba(meta_X)[0][1])
        is_fraud  = meta_prob >= _meta_threshold
    else:
        # Fallback: majority vote
        is_fraud = (rf_pred + xgb_pred + lgb_pred) >= 2

    result = {
        'rf':        rf_pred,
        'xgb':       xgb_pred,
        'lgb':       lgb_pred,
        'rf_prob':   round(rf_prob,  4),
        'xgb_prob':  round(xgb_prob, 4),
        'lgb_prob':  round(lgb_prob, 4),
        'is_fraud':  bool(is_fraud),
        'status':    'ALERT' if is_fraud else 'LEGIT',
    }
    if meta_prob is not None:
        result['meta_prob'] = round(meta_prob, 4)
    return result


def start_decision_engine():
    _load_models()
    logger.info("Starting Decision Engine...")
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
                tx       = message.value
                decision = decide(tx['features'])

                scored_tx = {
                    'tx_id':     tx['tx_id'],
                    'timestamp': tx['timestamp'],
                    'rf_pred':   decision['rf'],
                    'xgb_pred':  decision['xgb'],
                    'lgb_pred':  decision['lgb'],
                    'rf_prob':   decision['rf_prob'],
                    'xgb_prob':  decision['xgb_prob'],
                    'lgb_prob':  decision['lgb_prob'],
                    'is_fraud':  decision['is_fraud'],
                    'status':    decision['status'],
                    'raw':       tx.get('raw', {}),
                }
                if 'meta_prob' in decision:
                    scored_tx['meta_prob'] = decision['meta_prob']

                producer.send(OUTPUT_TOPIC, scored_tx)

                level = logging.WARNING if decision['status'] == 'ALERT' else logging.INFO
                logger.log(
                    level,
                    "[%s] TX: %s | Votes RF=%d XGB=%d LGB=%d | Probs RF=%.3f XGB=%.3f LGB=%.3f%s",
                    decision['status'], tx['tx_id'],
                    decision['rf'], decision['xgb'], decision['lgb'],
                    decision['rf_prob'], decision['xgb_prob'], decision['lgb_prob'],
                    f" | Meta={decision['meta_prob']:.3f}" if 'meta_prob' in decision else "",
                )

        except KeyboardInterrupt:
            logger.info("Stopping decision engine.")
            break
        except Exception as e:
            logger.error("Decision engine error: %s. Reconnecting in 5s...", e)
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
    start_decision_engine()
