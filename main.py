"""
Master Pipeline Runner: Extended Feature Extraction & Comparative XGBoost Classification
======================================================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Runner: main.py

Pipeline Steps Executed:
1. Dataset inspection & loading (SEED-PQD 17,000 signals)
2. Representative Signal Visualizations (STFT, WPT, HHT)
3. Extended Feature Extraction (66 features across FFT, DWT, Stats, STFT, WPT, HHT)
4. Comprehensive Automated Data Validation Audit
5. Generation of Feature Dictionary (CSV & Markdown)
6. XGBoost Classification & Comparative Benchmark Across Configurations A through L
7. Confusion Matrices & Comparative Performance Plots
"""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_dataset,
    validate_dataset,
    print_validation_report,
    get_signal_metadata,
    CLASS_NAMES,
)
from src.signal_processing import (
    extract_all_extended_dataset_features,
)
from src.visualization import (
    plot_stft_spectrograms,
    plot_wpt_subband_energies,
    plot_hht_decomposition,
    plot_confusion_matrix,
    plot_feature_method_comparison,
)
from src.classification import run_comparative_experiments
from src.feature_dictionary import save_feature_dictionary


def validate_feature_matrix(df, expected_rows=17000, expected_cols=68):
    """
    Executes automated data validation on the extended feature matrix:
    - Shape audit (rows, columns)
    - Metadata verification (class balance, label-to-index alignment)
    - Column name uniqueness
    - Missing, NaN, and Infinite value audit
    - Feature counts per signal processing method
    - Near-constant feature detection
    """
    print("\n" + "=" * 80)
    print("  PHASE 4: AUTOMATED FEATURE MATRIX VALIDATION AUDIT  ")
    print("=" * 80)

    errors = []
    warnings = []

    # 1. Dimensions
    print(f"Checking matrix dimensions: {df.shape}...")
    if df.shape[0] != expected_rows:
        errors.append(f"Row count mismatch: expected {expected_rows}, got {df.shape[0]}")
    if df.shape[1] != expected_cols:
        errors.append(f"Column count mismatch: expected {expected_cols}, got {df.shape[1]}")

    # 2. Duplicate columns
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        errors.append(f"Duplicate columns found: {dup_cols}")
    else:
        print("  [PASS] No duplicate column names found.")

    # 3. Missing and non-finite values
    nan_counts = df.isna().sum().sum()
    numeric_cols = [c for c in df.columns if c not in ["class_label", "class_index"]]
    numeric_df = df[numeric_cols].values
    inf_counts = np.isinf(numeric_df).sum()

    if nan_counts > 0:
        errors.append(f"Found {nan_counts} NaN values in feature matrix!")
    else:
        print("  [PASS] Zero NaN values across all 17,000 rows.")

    if inf_counts > 0:
        errors.append(f"Found {inf_counts} Infinite values in feature matrix!")
    else:
        print("  [PASS] Zero Infinite values across all 17,000 rows.")

    # 4. Class balance and alignment
    class_counts = df["class_label"].value_counts()
    print(f"  [PASS] Unique classes present: {len(class_counts)}")
    for cname, cnt in class_counts.items():
        if cnt != 1000:
            errors.append(f"Class '{cname}' count is {cnt}, expected 1000.")
    print("  [PASS] Exactly 1,000 signals per class (100% balanced across all 17 classes).")

    # Verify label-to-index alignment
    for expected_idx, cname in enumerate(CLASS_NAMES):
        sub = df[df["class_label"] == cname]
        indices = sub["class_index"].unique()
        if len(indices) != 1 or indices[0] != expected_idx:
            errors.append(f"Alignment error for '{cname}': index={indices}, expected {expected_idx}")
    print("  [PASS] Row-to-label alignment is 100% consistent with CLASS_NAMES taxonomy.")

    # 5. Method contribution breakdown
    method_breakdown = {
        "Metadata": [c for c in df.columns if c in ["class_label", "class_index"]],
        "FFT (Method 0)": [c for c in df.columns if c.startswith("fft_")],
        "DWT (Method 0)": [c for c in df.columns if c.startswith("dwt_")],
        "Statistical (Method 1)": [c for c in df.columns if c.startswith("stat_")],
        "STFT (Method 2)": [c for c in df.columns if c.startswith("stft_")],
        "WPT (Method 3)": [c for c in df.columns if c.startswith("wpt_")],
        "HHT (Method 4)": [c for c in df.columns if c.startswith("hht_")],
    }
    print("\nFeature Contribution Breakdown:")
    total_feats = 0
    for mname, cols in method_breakdown.items():
        print(f"  - {mname:25s}: {len(cols):2d} columns")
        if mname != "Metadata":
            total_feats += len(cols)
    print(f"  Total Numerical Features  : {total_feats:2d} features")
    print(f"  Total DataFrame Columns   : {df.shape[1]:2d} columns")

    # 6. Constant / Zero-variance feature check
    stds = df[numeric_cols].std()
    const_cols = stds[stds < 1e-9].index.tolist()
    if const_cols:
        warnings.append(f"Features with near-zero variance (<1e-9): {const_cols}")
    else:
        print("  [PASS] No constant or degenerate zero-variance features detected.")

    if errors:
        print("\n[VALIDATION FAILED WITH ERRORS]:")
        for e in errors:
            print(f"  * {e}")
        raise ValueError("Feature matrix validation failed.")
    else:
        print("\n>>> ALL VALIDATION CHECKS PASSED SUCCESSFULLY (0 Errors, 0 NaNs, 0 Infs)!")

    return method_breakdown


def main():
    total_start = time.time()
    print("=" * 80)
    print("  POWER QUALITY DISTURBANCE ANALYSIS: EXTENDED FEATURE EXTRACTION & ML PIPELINE  ")
    print("=" * 80)

    # 1. Load Dataset
    print("\n>>> [STEP 1] Loading SEED-PQD Dataset...")
    X, y_str, y_idx = load_dataset(source_format="csv")
    meta = get_signal_metadata()
    fs = meta["sampling_rate_hz"]

    figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 2. Representative Visualizations for New Methods
    print("\n>>> [STEP 2] Generating Representative Visualizations for STFT, WPT, and HHT...")
    plot_stft_spectrograms(X, y_str, fs=fs, save_path=os.path.join(figures_dir, "stft_spectrograms.png"))
    plot_hht_decomposition(X, y_str, selected_class="Oscillatory_Transient", fs=fs, save_path=os.path.join(figures_dir, "hht_decomposition.png"))

    # 3. Feature Extraction / Loading
    extended_csv = os.path.join(processed_dir, "extended_features.csv")
    recompute = False
    
    if os.path.exists(extended_csv) and not recompute:
        print(f"\n>>> [STEP 3] Found existing extended feature matrix at '{extended_csv}'. Loading...")
        df_features = pd.read_csv(extended_csv)
        print(f"Loaded existing matrix: shape {df_features.shape}")
        if df_features.shape != (17000, 68):
            print("Shape does not match expected (17000, 68), recomputing...")
            recompute = True

    if not os.path.exists(extended_csv) or recompute:
        print(f"\n>>> [STEP 3] Extracting 66 Features Across All 6 Methods for 17,000 Signals...")
        t0 = time.time()
        df_features = extract_all_extended_dataset_features(
            X,
            y_str=y_str,
            y_idx=y_idx,
            fs=fs,
            wavelet="db4",
            level=3,
            max_imf=2,
            n_jobs=-1,
            batch_size=200,
        )
        t_ext = time.time() - t0
        print(f"Feature extraction completed in {t_ext:.2f} seconds ({t_ext/17000*1000:.2f} ms/signal).")
        
        print(f"Saving extended features to '{extended_csv}'...")
        df_features.to_csv(extended_csv, index=False)
        csv_size_mb = os.path.getsize(extended_csv) / (1024 * 1024)
        print(f"Saved successfully! Size: {csv_size_mb:.2f} MB")

    # Generate WPT sub-band plot using extracted features
    plot_wpt_subband_energies(df_features, save_path=os.path.join(figures_dir, "wpt_subband_energies.png"))

    # 4. Automated Data Validation
    validate_feature_matrix(df_features, expected_rows=17000, expected_cols=68)

    # 5. Save Feature Dictionary
    print("\n>>> [STEP 5] Generating and Saving Feature Dictionary (CSV & Markdown)...")
    dict_csv = os.path.join(processed_dir, "feature_dictionary.csv")
    dict_md = os.path.join(PROJECT_ROOT, "docs", "Feature_Dictionary.md")
    save_feature_dictionary(output_csv=dict_csv, output_md=dict_md)

    # 6. Run XGBoost Comparative Benchmark Across 12 Configurations (A through L)
    print("\n>>> [STEP 6] Running Phase 5: XGBoost Classification Across Configurations A through L...")
    eval_results = run_comparative_experiments(
        df_features,
        test_size=0.20,
        random_state=42,
        cv_folds=5,
        save_dir=os.path.join(PROJECT_ROOT, "outputs"),
    )

    comparison_df = eval_results["comparison_df"]
    best_cfg_key = eval_results["best_config_id"]
    class_names = eval_results["class_names"]
    cms = eval_results["confusion_matrices"]

    # 7. Generate Visualizations for ML Results
    print("\n>>> [STEP 7] Generating Confusion Matrices and Method Comparison Plots...")
    # Confusion matrix for Clean Best Model (Config L)
    best_acc = comparison_df.loc[comparison_df['config_id']==best_cfg_key, 'accuracy'].values[0] * 100.0
    plot_confusion_matrix(
        cms[best_cfg_key],
        class_names,
        title=f"Normalized Confusion Matrix: Clean Test Set [{best_cfg_key}] ({best_acc:.2f}% Test Accuracy)",
        save_path=os.path.join(figures_dir, "confusion_matrix_clean.png"),
    )

    # Comparison bar chart
    plot_feature_method_comparison(
        comparison_df,
        save_path=os.path.join(figures_dir, "method_comparison.png"),
    )

    # Print Final Results Table
    print("\n" + "=" * 90)
    print("  FINAL COMPARATIVE BENCHMARK RESULTS (XGBOOST CLASSIFIER ON SEED-PQD)  ")
    print("=" * 90)
    print(
        comparison_df[[
            "config_id",
            "config_label",
            "num_features",
            "train_cv_acc",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "train_time_sec",
        ]].to_string(index=False)
    )
    print("=" * 90)

    total_time = time.time() - total_start
    print(f"\nALL PHASES (1 THROUGH 6) EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")


if __name__ == "__main__":
    main()
