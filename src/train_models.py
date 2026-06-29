"""
Quick training entry-point.

This module is a thin wrapper around analyze_metrics.train_and_evaluate().
All real training logic — hyperparameters, K-Means SMOTE-ENN resampling,
threshold optimisation, and model serialisation — lives there.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_metrics import train_and_evaluate

if __name__ == "__main__":
    absorb = "--absorb-live" in sys.argv
    train_and_evaluate(absorb_live=absorb)
