"""
Runner Script: Step 9 Noise Robustness & AWGN Stress Testing
===========================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Runner: run_noise_robustness.py

Evaluates model resilience under IEEE Std 1159 grid noise:
- SNR Levels: Clean (inf), 40 dB, 30 dB, and 20 dB
- Evaluates individual methods vs. multi-domain combined models
- Produces SNR degradation curves and noisy confusion matrices
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, get_signal_metadata
from src.classification import run_comparative_experiments
from src.visualization import plot_confusion_matrix
from src.noise_robustness import (
    evaluate_noise_robustness,
    plot_noise_waveforms,
    plot_snr_accuracy_curves,
)


def main():
    total_start = time.time()
    print("=" * 80)
    print("  STEP 9: NOISE ROBUSTNESS & AWGN STRESS TESTING PIPELINE  ")
    print("=" * 80)

    # 1. Load Dataset
    print("\n>>> [STEP 1] Loading Raw Signal Dataset...")
    X, y_str, y_idx = load_dataset(source_format="csv")
    meta = get_signal_metadata()
    fs = meta["sampling_rate_hz"]

    figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(figures_dir, exist_ok=True)

    # 2. Load Extended Features Matrix
    extended_csv = os.path.join(processed_dir, "extended_features.csv")
    if not os.path.exists(extended_csv):
        print(f"Error: {extended_csv} not found. Run main.py first!")
        sys.exit(1)

    print(f"\n>>> [STEP 2] Loading Extended Feature Matrix from '{extended_csv}'...")
    df_features = pd.read_csv(extended_csv)
    print(f"Loaded matrix: {df_features.shape}")

    # 3. Fit Models & Scalers on Clean Training Partition
    print("\n>>> [STEP 3] Fitting Models & Scalers on Clean Training Partition...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from src.classification import get_feature_groups, define_configurations, build_xgb_classifier

    y = df_features["class_index"].values
    indices = np.arange(len(df_features))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, stratify=y, random_state=42)
    y_train = y[train_idx]

    unique_indices = np.unique(y)
    class_names = [
        df_features[df_features["class_index"] == idx]["class_label"].iloc[0]
        for idx in unique_indices
    ]

    clean_results_csv = os.path.join(PROJECT_ROOT, "outputs", "comparative_model_results.csv")
    if os.path.exists(clean_results_csv):
        clean_results_df = pd.read_csv(clean_results_csv)
    else:
        clean_eval = run_comparative_experiments(
            df_features,
            test_size=0.20,
            random_state=42,
            cv_folds=5,
            save_dir=os.path.join(PROJECT_ROOT, "outputs"),
        )
        clean_results_df = clean_eval["comparison_df"]

    f_groups = get_feature_groups(df_features)
    configs = define_configurations(f_groups)

    trained_models = {}
    scalers = {}
    for cfg_id, cfg in configs.items():
        cols = cfg["cols"]
        X_train_raw = df_features[cols].values[train_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        scalers[cfg_id] = scaler

        clf = build_xgb_classifier(random_state=42)
        clf.fit(X_train_scaled, y_train)
        trained_models[cfg_id] = clf

    print(f"Fitted all 12 XGBoost models and Scalers on training partition ({len(train_idx)} samples).")

    # 4. Run Noise Robustness Benchmark
    print("\n>>> [STEP 4] Executing Noise Robustness Benchmark (40 dB, 30 dB, 20 dB SNR)...")
    noise_eval = evaluate_noise_robustness(
        X=X,
        y_str=y_str,
        y_idx=y_idx,
        trained_models=trained_models,
        scalers=scalers,
        train_idx=train_idx,
        test_idx=test_idx,
        clean_results_df=clean_results_df,
        snr_levels=(40, 30, 20),
        fs=fs,
        save_dir=os.path.join(PROJECT_ROOT, "outputs"),
    )

    noise_df = noise_eval["noise_df"]
    cms_noisy = noise_eval["confusion_matrices_noisy"]

    # 5. Plot 20 dB Confusion Matrix for Best Model (Config L)
    if "L_20dB" in cms_noisy:
        cm_20 = cms_noisy["L_20dB"]
        acc_20 = noise_df[(noise_df["config_id"] == "L") & (noise_df["snr_level"] == 20)]["accuracy"].values[0]
        plot_confusion_matrix(
            cm_20,
            class_names,
            title=f"Normalized Confusion Matrix: Configuration L under Severe Noise (20 dB SNR) (Test Acc: {acc_20*100:.2f}%)",
            save_path=os.path.join(figures_dir, "confusion_matrix_noisy_20dB.png"),
        )

    # 6. Print Summary Pivot Table
    print("\n" + "=" * 90)
    print("  SUMMARY: TEST ACCURACY ACROSS NOISE LEVELS (AWGN SNR)  ")
    print("=" * 90)
    pivot_acc = noise_df.pivot(index="config_label", columns="snr_level", values="accuracy") * 100.0
    pivot_acc = pivot_acc[["Clean", 40, 30, 20]]
    pivot_acc.columns = ["Clean (Inf)", "40 dB SNR", "30 dB SNR", "20 dB SNR"]
    pivot_acc["Degradation (Clean -> 20dB)"] = pivot_acc["Clean (Inf)"] - pivot_acc["20 dB SNR"]
    print(pivot_acc.to_string())
    print("=" * 90)

    total_time = time.time() - total_start
    print(f"\nSTEP 9 NOISE ROBUSTNESS BENCHMARK COMPLETED IN {total_time:.2f} SECONDS!")


if __name__ == "__main__":
    main()
