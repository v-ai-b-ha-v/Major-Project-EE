"""
Power Quality Disturbance Inference Engine
==========================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Runner: predict.py

This script loads the dedicated Config-L compatibility model
(xgboost_model_L_all_features.json) and scaler
(scaler_config_L_all_features.joblib) to perform inference using all 66 features.

Usage:
------
1. Run demo across multiple disturbances:
   python predict.py

2. Predict on a specific disturbance class from dataset:
   python predict.py --class Sag
   python predict.py --class Flicker_with_Harmonics

3. Predict on a custom CSV file:
   python predict.py --csv data/raw/XPQRS/Transient.csv --row 0
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.signal_processing import extract_all_extended_features_single_signal
from src.data_loader import CLASS_NAMES

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "xgboost_model_L_all_features.json")
SCALER_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "scaler_config_L_all_features.joblib")


def load_inference_artifacts():
    """Loads the trained XGBoost model and fitted StandardScaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run 'python main.py' first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}. Run 'python main.py' first.")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_waveform(signal, model=None, scaler=None, fs=5000):
    """
    Perform inference on one 100-sample waveform using the Config-L model.

    Returns predicted class, confidence, top-3 candidates, latency breakdown,
    and the extracted feature dictionary.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size != 100:
        raise ValueError(
            f"Expected exactly 100 samples per waveform; received {signal.size}."
        )
    if not np.isfinite(signal).all():
        raise ValueError("Waveform contains NaN or infinite values.")
    if not np.isclose(float(fs), 5000.0):
        raise ValueError(
            f"Model was trained for 5000 Hz sampling; received fs={fs} Hz."
        )

    if model is None or scaler is None:
        model, scaler = load_inference_artifacts()

    start_time = time.perf_counter()

    # 1. Extract all 66 features in the same family order used by the dataset.
    feat_start = time.perf_counter()
    features_dict = extract_all_extended_features_single_signal(signal, fs=fs)
    feat_time_ms = (time.perf_counter() - feat_start) * 1000.0
    feature_vector = np.asarray([list(features_dict.values())], dtype=np.float64)

    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is not None and feature_vector.shape[1] != expected_features:
        raise ValueError(
            "Feature-schema mismatch: extracted "
            f"{feature_vector.shape[1]} features, but Config-L scaler expects "
            f"{expected_features}."
        )

    # 2. Apply the Config-L training scaler.
    feature_vector_scaled = scaler.transform(feature_vector)

    # 3. XGBoost inference.
    inf_start = time.perf_counter()
    probabilities = model.predict_proba(feature_vector_scaled)[0]
    inf_time_ms = (time.perf_counter() - inf_start) * 1000.0

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} class probabilities; "
            f"expected {len(CLASS_NAMES)}."
        )

    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [(CLASS_NAMES[idx], float(probabilities[idx])) for idx in top3_indices]
    best_idx = int(top3_indices[0])

    return {
        "predicted_class": CLASS_NAMES[best_idx],
        "confidence": float(probabilities[best_idx]),
        "top3": top3,
        "total_latency_ms": total_time_ms,
        "feature_latency_ms": feat_time_ms,
        "inference_latency_ms": inf_time_ms,
        "features": features_dict,
    }

def main():
    parser = argparse.ArgumentParser(description="Real-Time PQD Inference Engine")
    parser.add_argument("--class_name", "--class", type=str, default=None,
                        help="Class name to sample from data/raw/XPQRS/ (e.g. Sag, Swell, Flicker)")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV file")
    parser.add_argument("--row", type=int, default=0, help="Row index in CSV file (default: 0)")
    args = parser.parse_args()

    print("=" * 80)
    print("      POWER QUALITY DISTURBANCE (PQD) REAL-TIME INFERENCE ENGINE        ")
    print("=" * 80)
    print(f"Loading trained XGBoost model from: {MODEL_PATH}")
    print(f"Loading fitted StandardScaler from: {SCALER_PATH}")

    model, scaler = load_inference_artifacts()
    print("Artifacts loaded successfully!\n")

    if args.csv is not None:
        csv_path = args.csv
        df = pd.read_csv(csv_path, header=None)
        sig = df.iloc[args.row].values
        ground_truth = os.path.splitext(os.path.basename(csv_path))[0]
        test_samples = [(ground_truth, sig)]
    elif args.class_name is not None:
        target_csv = os.path.join(PROJECT_ROOT, "data", "raw", "XPQRS", f"{args.class_name}.csv")
        if not os.path.exists(target_csv):
            print(f"Error: {target_csv} does not exist.")
            sys.exit(1)
        df = pd.read_csv(target_csv, header=None)
        sig = df.iloc[args.row].values
        test_samples = [(args.class_name, sig)]
    else:
        # Default demo: test a diverse set of 6 disturbance classes
        demo_classes = ["Pure_Sinusoidal", "Sag", "Swell", "Harmonics", "Flicker_with_Sag", "Harmonics_with_Swell"]
        test_samples = []
        for cls in demo_classes:
            csv_file = os.path.join(PROJECT_ROOT, "data", "raw", "XPQRS", f"{cls}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, header=None)
                # Sample row 42 for a test signal
                test_samples.append((cls, df.iloc[42].values))

    print(f"{'True Label':<26} | {'Predicted Label':<26} | {'Confidence':<10} | {'Latency':<8} | {'Status'}")
    print("-" * 88)

    for true_label, waveform in test_samples:
        result = predict_waveform(waveform, model=model, scaler=scaler)
        pred = result["predicted_class"]
        conf = result["confidence"] * 100.0
        lat = result["total_latency_ms"]
        match = "MATCH [OK]" if pred == true_label else "MISMATCH"

        print(f"{true_label:<26} | {pred:<26} | {conf:>7.2f}%   | {lat:>5.2f} ms | {match}")

    print("-" * 88)
    print("Inference completed. Reported latency is a local measurement; validate repeated-run latency on the target hardware before making real-time claims.")


if __name__ == "__main__":
    main()
