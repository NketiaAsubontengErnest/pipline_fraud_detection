import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

TYPES = {'refund': 0, 'purchase': 1}
LOCATIONS = {
    'San Antonio': 0, 'Dallas': 1, 'New York': 2, 'Philadelphia': 3,
    'Phoenix': 4, 'Chicago': 5, 'San Jose': 6, 'San Diego': 7,
    'Houston': 8, 'Los Angeles': 9
}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4}

# Canonical feature order — must stay consistent between training and inference
FEATURE_COLS = [
    'Amount', 'MerchantID', 'TransactionType', 'Location', 'Hour', 'DayOfWeek',
    'AmountLog', 'IsNightHour', 'IsWeekend', 'IsHighRiskMerchant',
    'NightHighAmount', 'RefundHighAmount', 'RiskyMerchantAmount',
    'WeekendNight', 'HighRiskRefund', 'AmountBin',
    # Behavioral / velocity features (computed from transaction history)
    'TxVelocity_1h', 'TxVelocity_24h', 'AmountVsAvg',
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw DataFrame rows into the numerical feature matrix for training.
    Includes velocity features computed from per-merchant transaction history."""
    d = df.copy()

    d['TransactionDate'] = pd.to_datetime(d['TransactionDate'], format='mixed')
    d['Hour']      = d['TransactionDate'].dt.hour
    d['DayOfWeek'] = d['TransactionDate'].dt.dayofweek

    d['TransactionType'] = d['TransactionType'].map(TYPES).fillna(1).astype(int)
    d['Location']        = d['Location'].map(LOCATIONS).fillna(0).astype(int)

    d['AmountLog']           = np.log1p(d['Amount'])
    d['IsNightHour']         = d['Hour'].isin(NIGHT_HOURS).astype(int)
    d['IsWeekend']           = (d['DayOfWeek'] >= 5).astype(int)
    d['IsHighRiskMerchant']  = (d['MerchantID'] <= 30).astype(int)

    d['NightHighAmount']     = d['IsNightHour'] * d['AmountLog']
    d['RefundHighAmount']    = (1 - d['TransactionType']) * d['AmountLog']
    d['RiskyMerchantAmount'] = d['IsHighRiskMerchant'] * d['AmountLog']
    d['WeekendNight']        = d['IsWeekend'] * d['IsNightHour']
    d['HighRiskRefund']      = d['IsHighRiskMerchant'] * (1 - d['TransactionType'])
    d['AmountBin']           = pd.cut(d['Amount'], bins=[0, 1250, 2500, 3750, 5001],
                                      labels=[0, 1, 2, 3]).astype(int)

    # --- Velocity features (per-merchant rolling windows) ---
    # Each transaction sees only PRIOR transactions (shift(1)) to avoid leakage.
    vel_parts = []
    for merchant_id, grp in d.groupby('MerchantID', sort=False):
        g = grp[['Amount', 'TransactionDate']].copy()
        g = g.set_index('TransactionDate').sort_index()

        shifted = g['Amount'].shift(1)
        v1h  = shifted.rolling('1h',  min_periods=0).count().fillna(0).astype(int)
        v24h = shifted.rolling('24h', min_periods=0).count().fillna(0).astype(int)

        exp_mean = shifted.expanding().mean()
        exp_std  = shifted.expanding().std().fillna(0) + 1e-6
        avz = ((g['Amount'] - exp_mean) / exp_std).fillna(0.0).clip(-5, 5)

        vel_parts.append(pd.DataFrame(
            {'TxVelocity_1h': v1h.values,
             'TxVelocity_24h': v24h.values,
             'AmountVsAvg': avz.values},
            index=grp.index,
        ))

    vel_df = pd.concat(vel_parts).sort_index()
    d['TxVelocity_1h']  = vel_df['TxVelocity_1h']
    d['TxVelocity_24h'] = vel_df['TxVelocity_24h']
    d['AmountVsAvg']    = vel_df['AmountVsAvg']

    return d[FEATURE_COLS]


def process_single_transaction(raw_dict: dict, velocity_stats: dict | None = None) -> list:
    """Used by the Kafka preprocessing stream and the REST API.
    Returns features in FEATURE_COLS order.
    velocity_stats: optional dict with keys tx_velocity_1h, tx_velocity_24h, amount_vs_avg.
    When None (e.g. direct API call with no history), neutral defaults are used.
    """
    tx_date = raw_dict.get('TransactionDate', '2024-01-01 12:00:00.000000')
    try:
        dt = datetime.strptime(tx_date, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        dt = datetime.strptime(tx_date, '%Y-%m-%d %H:%M:%S')

    hour      = dt.hour
    dayofweek = dt.weekday()

    t_type = TYPES.get(raw_dict.get('TransactionType', 'purchase'), 1)

    loc_str = raw_dict.get('Location', '')
    if loc_str and loc_str not in LOCATIONS:
        logger.warning("Unknown location '%s' — defaulting to 0 (San Antonio encoding)", loc_str)
    loc = LOCATIONS.get(loc_str, 0)

    raw_amount = raw_dict.get('Amount')
    if raw_amount is None or raw_amount == '':
        raise ValueError("Missing required field: Amount")
    amount      = float(raw_amount)
    merchant_id = int(raw_dict.get('MerchantID', 0))

    amount_log            = np.log1p(amount)
    is_night              = 1 if hour in NIGHT_HOURS else 0
    is_weekend            = 1 if dayofweek >= 5 else 0
    is_high_risk_merchant = 1 if merchant_id <= 30 else 0

    if amount <= 1250:   amount_bin = 0
    elif amount <= 2500: amount_bin = 1
    elif amount <= 3750: amount_bin = 2
    else:                amount_bin = 3

    # Velocity: use provided stats or neutral defaults
    if velocity_stats is not None:
        v1h  = int(velocity_stats.get('tx_velocity_1h', 1))
        v24h = int(velocity_stats.get('tx_velocity_24h', 1))
        avz  = float(velocity_stats.get('amount_vs_avg', 0.0))
    else:
        v1h, v24h, avz = 1, 1, 0.0

    return [
        amount,
        merchant_id,
        t_type,
        loc,
        hour,
        dayofweek,
        amount_log,
        is_night,
        is_weekend,
        is_high_risk_merchant,
        is_night * amount_log,                 # NightHighAmount
        (1 - t_type) * amount_log,             # RefundHighAmount
        is_high_risk_merchant * amount_log,    # RiskyMerchantAmount
        is_weekend * is_night,                 # WeekendNight
        is_high_risk_merchant * (1 - t_type), # HighRiskRefund
        amount_bin,                            # AmountBin
        v1h,                                   # TxVelocity_1h
        v24h,                                  # TxVelocity_24h
        avz,                                   # AmountVsAvg
    ]
