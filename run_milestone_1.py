"""
Milestone 1 Runner: Power Quality Disturbance Signal Processing
================================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Script: run_milestone_1.py (Project Root Runner)

Tasks Executed:
1. Dataset inspection and verification
2. Preprocessing & numerical validation
3. Waveform and class distribution plotting -> outputs/figures/
4. FFT and DWT signal transforms & spectral plots -> outputs/figures/
5. Feature extraction across all 17,000 signals -> data/processed/extracted_features.csv
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_dataset,
    validate_dataset,
    print_validation_report,
    get_signal_metadata,
)
from src.signal_processing import extract_all_dataset_features
from src.visualization import (
    plot_class_distribution,
    plot_representative_waveforms,
    plot_fft_magnitude_spectra,
    plot_dwt_decomposition,
    plot_dwt_feature_comparison,
)


def main():
    total_start = time.time()
    print("=" * 72)
    print("  POWER QUALITY DISTURBANCE ANALYSIS: MILESTONE 1 PIPELINE  ")
    print("=" * 72)

    # 1. Load Dataset
    print("\n>>> [TASK 1 & 3] Loading SEED-PQD Dataset...")
    X, y_str, y_idx = load_dataset(source_format="csv")

    meta = get_signal_metadata()
    print(f"Sampling frequency   : {meta['sampling_rate_hz']} Hz")
    print(f"Fundamental frequency: {meta['fundamental_freq_hz']} Hz")
    print(f"Signal cycle duration: {meta['signal_duration_ms']:.1f} ms ({meta['samples_per_signal']} samples)")

    # 2. Validate Data
    validation_summary = validate_dataset(X, y_str, y_idx)
    print_validation_report(validation_summary)

    # 3. Visualizations (Saved to outputs/figures/)
    figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(">>> [TASK 2] Generating Dataset Analysis and Waveform Plots...")
    plot_class_distribution(y_str, save_path=os.path.join(figures_dir, "class_distribution.png"))
    plot_representative_waveforms(X, y_str, fs=meta["sampling_rate_hz"], save_path=os.path.join(figures_dir, "time_domain_waveforms.png"))

    # 4. Signal Processing (FFT & DWT)
    print("\n>>> [TASK 4] Computing FFT and DWT Signal Processing Transforms...")
    plot_fft_magnitude_spectra(X, y_str, fs=meta["sampling_rate_hz"], save_path=os.path.join(figures_dir, "fft_magnitude_spectra.png"))
    plot_dwt_decomposition(X, y_str, selected_class="Oscillatory_Transient", wavelet="db4", level=3, save_path=os.path.join(figures_dir, "dwt_decomposition.png"))

    # 5. Feature Extraction across 17,000 Signals
    print("\n>>> [TASK 4] Extracting FFT and DWT Features for all 17,000 Signals...")
    t0 = time.time()
    features_df = extract_all_dataset_features(
        X,
        y_str=y_str,
        y_idx=y_idx,
        fs=meta["sampling_rate_hz"],
        wavelet="db4",
        level=3,
    )
    t_extract = time.time() - t0
    print(f"Feature extraction completed in {t_extract:.2f} seconds.")

    plot_dwt_feature_comparison(features_df, save_path=os.path.join(figures_dir, "dwt_features.png"))

    # 6. Save Extracted Features to data/processed/
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_csv = os.path.join(processed_dir, "extracted_features.csv")
    
    print(f"\n>>> Saving extracted features to '{output_csv}'...")
    features_df.to_csv(output_csv, index=False)
    csv_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"Saved successfully! (Size: {csv_size_mb:.2f} MB, Rows: {len(features_df)}, Columns: {features_df.shape[1]})")

    # Summary
    fft_cols = [c for c in features_df.columns if c.startswith("fft_")]
    dwt_cols = [c for c in features_df.columns if c.startswith("dwt_")]
    print("\n--- Features Summary ---")
    print(f"Total FFT features ({len(fft_cols)}): {fft_cols}")
    print(f"Total DWT features ({len(dwt_cols)}): {dwt_cols}")

    total_time = time.time() - total_start
    print("\n" + "=" * 72)
    print(f"  MILESTONE 1 PIPELINE EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!  ")
    print("=" * 72)


if __name__ == "__main__":
    main()
