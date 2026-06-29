import os
import sys
import pytest
import unittest.mock as mock

sys.path.insert(0, os.path.abspath('src'))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_models_exist():
    assert os.path.exists('Models/Random_Forest_model.pkl'), "Random Forest model not found"
    assert os.path.exists('Models/Extreme_Gradient_Boosting_model.pkl'), "XGBoost model not found"
    assert os.path.exists('Models/Light_Gradient_Boosting_Machine_model.pkl'), "LightGBM model not found"


def test_feature_extraction_shape_and_values():
    from features import process_single_transaction

    sample = {
        'TransactionDate': '2024-03-05 19:41:36.000000',
        'Amount': '1596.79',
        'MerchantID': '675',
        'TransactionType': 'purchase',
        'Location': 'Houston',
    }
    features = process_single_transaction(sample)
    assert len(features) == 19, f"Expected 19 features, got {len(features)}"
    assert features[0] == 1596.79   # Amount
    assert features[1] == 675       # MerchantID


def test_feature_extraction_unknown_location_does_not_crash():
    """Unknown locations should map to 0 with a warning, not raise an exception."""
    from features import process_single_transaction

    sample = {
        'TransactionDate': '2024-03-05 10:00:00',
        'Amount': '500.00',
        'MerchantID': '200',
        'TransactionType': 'purchase',
        'Location': 'UnknownCity',
    }
    features = process_single_transaction(sample)
    assert len(features) == 19
    assert features[3] == 0  # Location defaults to 0 (San Antonio encoding)


# ---------------------------------------------------------------------------
# Decision engine — output structure
# ---------------------------------------------------------------------------

def test_decide_returns_all_expected_keys():
    """decide() must return votes, fraud probabilities, and status."""
    from decision_engine import decide
    from features import process_single_transaction

    raw = {
        'TransactionDate': '2024-03-05 11:30:00',
        'Amount': '15.50',
        'MerchantID': '444',
        'TransactionType': 'purchase',
        'Location': 'Houston',
    }
    result = decide(process_single_transaction(raw))

    for key in ('rf', 'xgb', 'lgb', 'rf_prob', 'xgb_prob', 'lgb_prob', 'status', 'is_fraud'):
        assert key in result, f"Missing key '{key}' in decide() output"

    assert result['status'] in ('LEGIT', 'ALERT')
    assert isinstance(result['is_fraud'], bool)

    # Probabilities must be valid
    for prob_key in ('rf_prob', 'xgb_prob', 'lgb_prob'):
        assert 0.0 <= result[prob_key] <= 1.0, f"{prob_key} out of range: {result[prob_key]}"

    # Majority-vote logic must be consistent with the returned status
    total_votes = result['rf'] + result['xgb'] + result['lgb']
    assert result['is_fraud'] == (total_votes >= 2), "is_fraud inconsistent with majority vote"


# ---------------------------------------------------------------------------
# Decision engine — fraud detection accuracy
# ---------------------------------------------------------------------------

def test_low_risk_transaction_classified_legit():
    """Small daytime purchase from a non-risky merchant should be LEGIT."""
    from decision_engine import decide
    from features import process_single_transaction

    low_risk = {
        'TransactionDate': '2024-03-05 11:30:00',
        'Amount': '15.50',
        'MerchantID': '444',
        'TransactionType': 'purchase',
        'Location': 'Houston',
    }
    result = decide(process_single_transaction(low_risk))
    assert result['status'] == 'LEGIT', (
        f"Low-risk transaction incorrectly flagged ALERT "
        f"(RF={result['rf_prob']:.3f}, XGB={result['xgb_prob']:.3f}, LGB={result['lgb_prob']:.3f})"
    )


def test_high_risk_transaction_classified_alert():
    """High amount + 2am + high-risk merchant + refund should be ALERT."""
    from decision_engine import decide
    from features import process_single_transaction

    high_risk = {
        'TransactionDate': '2024-03-09 02:15:00',
        'Amount': '4800.00',
        'MerchantID': '5',           # High-risk (ID <= 30)
        'TransactionType': 'refund', # Refund adds signal
        'Location': 'Houston',
    }
    result = decide(process_single_transaction(high_risk))
    assert result['status'] == 'ALERT', (
        f"High-risk transaction not flagged as ALERT "
        f"(RF={result['rf_prob']:.3f}, XGB={result['xgb_prob']:.3f}, LGB={result['lgb_prob']:.3f})"
    )


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def api_client():
    """Create a TestClient with Kafka consumer disabled."""
    import importlib
    from fastapi.testclient import TestClient

    os.environ['DISABLE_KAFKA_CONSUMER'] = '1'
    import reporting_api
    importlib.reload(reporting_api)

    with TestClient(reporting_api.app) as client:
        yield client

    del os.environ['DISABLE_KAFKA_CONSUMER']


def test_api_predict_returns_status_and_probabilities(api_client):
    payload = {
        "TransactionID":   "TEST_LEGIT_001",
        "TransactionDate": "2024-03-05 11:30:00",
        "Amount":          15.50,
        "MerchantID":      444,
        "TransactionType": "purchase",
        "Location":        "Houston",
    }
    resp = api_client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert 'status' in data
    assert data['status'] in ('LEGIT', 'ALERT')
    assert 'predictions_breakdown' in data
    assert 'probabilities' in data

    for model_key in ('Random_Forest', 'Extreme_Gradient_Boosting', 'Light_Gradient_Boosting_Machine'):
        assert model_key in data['predictions_breakdown'], f"Missing {model_key} in predictions_breakdown"
        assert model_key in data['probabilities'],         f"Missing {model_key} in probabilities"
        assert 0.0 <= data['probabilities'][model_key] <= 1.0, f"{model_key} probability out of range"


def test_api_predict_high_risk_returns_alert(api_client):
    payload = {
        "TransactionID":   "TEST_FRAUD_001",
        "TransactionDate": "2024-03-09 02:15:00",
        "Amount":          4800.00,
        "MerchantID":      5,
        "TransactionType": "refund",
        "Location":        "Houston",
    }
    resp = api_client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()['status'] == 'ALERT'


def test_api_stats_returns_counters(api_client):
    resp = api_client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    for key in ('total', 'legit', 'alerts'):
        assert key in data, f"Missing key '{key}' in /api/stats response"


def test_api_saved_transactions_pagination(api_client):
    """Pagination params must be accepted and limit the result set."""
    resp = api_client.get("/api/saved-transactions?page=1&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) <= 5


# ---------------------------------------------------------------------------
# Input validation edge cases
# ---------------------------------------------------------------------------

def test_api_predict_rejects_negative_amount(api_client):
    """Negative Amount must be rejected with a 422 validation error."""
    payload = {
        "TransactionID":   "TEST_NEG_AMT",
        "TransactionDate": "2024-03-05 11:30:00",
        "Amount":          -50.00,
        "MerchantID":      100,
        "TransactionType": "purchase",
        "Location":        "Houston",
    }
    resp = api_client.post("/api/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for negative amount, got {resp.status_code}"


def test_api_predict_rejects_zero_amount(api_client):
    """Zero Amount must be rejected with a 422 validation error."""
    payload = {
        "TransactionID":   "TEST_ZERO_AMT",
        "TransactionDate": "2024-03-05 11:30:00",
        "Amount":          0.0,
        "MerchantID":      100,
        "TransactionType": "purchase",
        "Location":        "Houston",
    }
    resp = api_client.post("/api/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for zero amount, got {resp.status_code}"


def test_api_predict_rejects_invalid_transaction_type(api_client):
    """TransactionType values other than 'purchase'/'refund' must be rejected."""
    payload = {
        "TransactionID":   "TEST_BAD_TYPE",
        "TransactionDate": "2024-03-05 11:30:00",
        "Amount":          100.00,
        "MerchantID":      100,
        "TransactionType": "withdrawal",
        "Location":        "Houston",
    }
    resp = api_client.post("/api/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for invalid TransactionType, got {resp.status_code}"


def test_api_health_returns_ok_structure(api_client):
    """/health endpoint must return status, models, database, and kafka keys."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("status", "models", "database", "kafka"):
        assert key in data, f"Missing key '{key}' in /health response"


# ---------------------------------------------------------------------------
# Feature extraction edge cases
# ---------------------------------------------------------------------------

def test_feature_extraction_raises_on_missing_amount():
    """process_single_transaction must raise ValueError when Amount is absent."""
    from features import process_single_transaction
    import pytest as _pytest

    sample = {
        'TransactionDate': '2024-03-05 10:00:00',
        'MerchantID': '200',
        'TransactionType': 'purchase',
        'Location': 'Houston',
        # Amount intentionally omitted
    }
    with _pytest.raises(ValueError, match="Amount"):
        process_single_transaction(sample)


def test_feature_extraction_raises_on_empty_amount():
    """process_single_transaction must raise ValueError when Amount is empty string."""
    from features import process_single_transaction
    import pytest as _pytest

    sample = {
        'TransactionDate': '2024-03-05 10:00:00',
        'Amount': '',
        'MerchantID': '200',
        'TransactionType': 'purchase',
        'Location': 'Houston',
    }
    with _pytest.raises(ValueError, match="Amount"):
        process_single_transaction(sample)


# ---------------------------------------------------------------------------
# Model file checks
# ---------------------------------------------------------------------------

def test_decision_engine_raises_on_missing_model(tmp_path, monkeypatch):
    """_load_models() must raise RuntimeError with a clear message when a .pkl is absent."""
    import importlib
    import decision_engine as de

    bad_paths = {
        'rf':  str(tmp_path / 'rf_missing.pkl'),
        'xgb': str(tmp_path / 'xgb_missing.pkl'),
        'lgb': str(tmp_path / 'lgb_missing.pkl'),
    }
    monkeypatch.setattr(de, '_MODEL_PATHS', bad_paths)
    de._models.clear()
    de._model_mtimes.clear()

    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="Model file not found"):
        de._load_models()
