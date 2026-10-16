"""
Noise Robustness and Additive White Gaussian Noise (AWGN) Evaluation Module
==========================================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Module: src/noise_robustness.py

Implements physical grid noise corruption according to IEEE Std 1159:
- Signal-to-Noise Ratios: Clean (inf), 40 dB, 30 dB, and 20 dB SNR.
- Evaluates feature degradation and model robustness under noisy field conditions.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.signal_processing import extract_all_extended_dataset_features
from src.classification import get_feature_groups, define_configurations


def add_awgn_to_signal(signal, snr_db, rng=None):
    """
    Injects Additive White Gaussian Noise (AWGN) into a 1D discrete signal
    for a specified Signal-to-Noise Ratio (SNR in decibels).

    Formula:
        P_signal = (1/N) * sum(x[n]^2)
        P_noise  = P_signal / (10^(SNR_dB / 10))
        noise    ~ N(0, sqrt(P_noise))
        x_noisy  = x + noise
    """
    if rng is None:
        rng = np.random.default_rng()

    p_signal = np.mean(signal ** 2)
    if p_signal < 1e-12:
        return signal.copy()

    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=signal.shape)
    return signal + noise


def add_awgn_to_matrix(X, snr_db, random_state=42):
    """
    Applies AWGN to an entire 2D matrix of signals (shape: N_signals x 100).
    """
    rng = np.random.default_rng(random_state)
    X_noisy = np.empty_like(X)
    for i in range(X.shape[0]):
        X_noisy[i] = add_awgn_to_signal(X[i], snr_db, rng=rng)
    return X_noisy


def plot_noise_waveforms(X, y_str, selected_class="Oscillatory_Transient", snr_levels=(40, 30, 20), save_path=None):
    """
    Visualizes a representative disturbance waveform under Clean, 40 dB, 30 dB, and 20 dB SNR.
    """
    if save_path is None:
        save_path = os.path.join("outputs", "figures", "noise_waveform_comparison.png")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    match_idx = np.where(y_str == selected_class)[0]
    sig = X[match_idx[0]] if len(match_idx) > 0 else np.zeros(100)
    t_ms = np.linspace(0, 20.0, len(sig), endpoint=False)

    rng = np.random.default_rng(42)
    noisy_signals = [sig] + [add_awgn_to_signal(sig, snr, rng=rng) for snr in snr_levels]
    titles = ["Clean Signal (No Noise, SNR = Inf)"] + [f"Corrupted with AWGN @ {snr} dB SNR" for snr in snr_levels]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), dpi=300, sharex=True, sharey=True)

    for idx, (ax, s_noisy, title, col) in enumerate(zip(axes, noisy_signals, titles, colors)):
        ax.plot(t_ms, s_noisy, color=col, lw=1.3, alpha=0.9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Amplitude (p.u.)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (ms)", fontsize=10, fontweight="bold")
    fig.suptitle(
        f"Additive White Gaussian Noise (AWGN) Degradation on {selected_class.replace('_', ' ')}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved noise waveform comparison plot to: {save_path}")


def plot_snr_accuracy_curves(noise_df, save_path=None):
    """
    Plots Test Accuracy vs. SNR across individual signal processing methods
    and combined multi-domain configurations.
    """
    if save_path is None:
        save_path = os.path.join("outputs", "figures", "snr_accuracy_curves.png")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Key configurations to plot
    key_configs = [
        ("A", "A. FFT only", "#1f77b4", "o-"),
        ("B", "B. DWT only", "#2ca02c", "s-"),
        ("C", "C. Statistical only", "#8c564b", "^-"),
        ("D", "D. STFT only", "#9467bd", "d-"),
        ("E", "E. WPT only", "#e377c2", "v-"),
        ("F", "F. HHT only", "#7f7f7f", "p-"),
        ("G", "G. FFT + DWT", "#ff7f0e", "x--"),
        ("H", "H. FFT + DWT + Stats", "#bcbd22", "*--"),
        ("K", "K. FFT + DWT + HHT", "#17becf", "+--"),
        ("L", "L. All Features Combined", "#d62728", "D-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    snr_labels = ["20 dB", "30 dB", "40 dB", "Clean (Inf)"]
    snr_order = [20, 30, 40, "Clean"]

    for cfg_id, cfg_label, col, mark in key_configs:
        sub = noise_df[noise_df["config_id"] == cfg_id]
        if sub.empty:
            continue
        
        # Sort by snr_order
        sub_dict = dict(zip(sub["snr_level"], sub["accuracy"] * 100.0))
        sub_f1_dict = dict(zip(sub["snr_level"], sub["macro_f1"] * 100.0))
        acc_vals = [sub_dict.get(k, np.nan) for k in snr_order]
        f1_vals = [sub_f1_dict.get(k, np.nan) for k in snr_order]

        lw = 2.2 if cfg_id in ["L", "B", "G"] else 1.4
        ax1.plot(range(len(snr_labels)), acc_vals, mark, color=col, label=cfg_label, lw=lw, markersize=6)
        ax2.plot(range(len(snr_labels)), f1_vals, mark, color=col, label=cfg_label, lw=lw, markersize=6)

    for ax, title, ylabel in [
        (ax1, "Classification Accuracy vs. Noise Level (SNR)", "Test Accuracy (%)"),
        (ax2, "Macro F1-Score vs. Noise Level (SNR)", "Macro F1-Score (%)"),
    ]:
        ax.set_xticks(range(len(snr_labels)))
        ax.set_xticklabels(snr_labels, fontsize=10, fontweight="bold")
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Noise Robustness Benchmark: Model Resilience Under Additive White Gaussian Noise (AWGN)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved SNR accuracy curve plots to: {save_path}")


def evaluate_noise_robustness(
    X,
    y_str,
    y_idx,
    trained_models,
    scalers,
    train_idx,
    test_idx,
    clean_results_df,
    snr_levels=(40, 30, 20),
    fs=5000,
    save_dir="outputs",
):
    """
    Runs the complete noise robustness benchmark:
    For each SNR level:
    1. Injects AWGN into the test set signals X[test_idx].
    2. Extracts 66 features from the noisy test signals.
    3. Transforms using training-fitted StandardScaler.
    4. Evaluates all 12 trained XGBoost models.
    5. Saves full comparative table and plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Plot representative waveforms under noise
    plot_noise_waveforms(
        X,
        y_str,
        selected_class="Oscillatory_Transient",
        snr_levels=snr_levels,
        save_path=os.path.join(figures_dir, "noise_waveform_comparison.png"),
    )

    X_test_clean = X[test_idx]
    y_test = y_idx[test_idx]
    y_test_str = y_str[test_idx]

    all_noise_records = []

    # Include clean baseline results first
    for _, row in clean_results_df.iterrows():
        all_noise_records.append({
            "snr_level": "Clean",
            "snr_db": np.inf,
            "config_id": row["config_id"],
            "config_label": row["config_label"],
            "num_features": row["num_features"],
            "accuracy": row["accuracy"],
            "macro_precision": row["macro_precision"],
            "macro_recall": row["macro_recall"],
            "macro_f1": row["macro_f1"],
        })

    confusion_matrices_noisy = {}

    for snr in snr_levels:
        print(f"\n================================================================================")
        print(f"  EVALUATING NOISE ROBUSTNESS AT {snr} dB SNR (AWGN)  ")
        print(f"================================================================================")

        # Corrupt test partition
        t0 = time.time()
        X_test_noisy = add_awgn_to_matrix(X_test_clean, snr_db=snr, random_state=42)

        # Extract features for noisy test signals
        print(f"Extracting features for {len(X_test_noisy)} noisy test signals...")
        df_noisy_test = extract_all_extended_dataset_features(
            X_test_noisy,
            y_str=y_test_str,
            y_idx=y_test,
            fs=fs,
            wavelet="db4",
            level=3,
            max_imf=2,
            n_jobs=4,
            batch_size=100,
        )
        t_ext = time.time() - t0
        print(f"Noisy feature extraction completed in {t_ext:.2f} s ({t_ext/len(X_test_noisy)*1000:.2f} ms/signal).")

        f_groups = get_feature_groups(df_noisy_test)
        configs = define_configurations(f_groups)

        for cfg_id, cfg in configs.items():
            cols = cfg["cols"]
            label = cfg["label"]
            clf = trained_models[cfg_id]
            scaler = scalers[cfg_id]

            # Scale using training scaler
            X_test_feat = df_noisy_test[cols].values
            X_test_scaled = scaler.transform(X_test_feat)

            y_pred = clf.predict(X_test_scaled)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

            all_noise_records.append({
                "snr_level": snr,
                "snr_db": float(snr),
                "config_id": cfg_id,
                "config_label": label,
                "num_features": len(cols),
                "accuracy": acc,
                "macro_precision": prec,
                "macro_recall": rec,
                "macro_f1": f1,
            })

            if cfg_id == "L" and snr == 20:
                confusion_matrices_noisy["L_20dB"] = confusion_matrix(y_test, y_pred)

            print(f"  [{cfg_id}] {label:26s} | Acc: {acc*100:5.2f}% | F1: {f1*100:5.2f}%")

    noise_df = pd.DataFrame(all_noise_records)
    results_csv = os.path.join(save_dir, "noise_robustness_results.csv")
    noise_df.to_csv(results_csv, index=False)
    print(f"\nSaved noise robustness results table to: {results_csv}")

    # Plot SNR curves
    plot_snr_accuracy_curves(noise_df, save_path=os.path.join(figures_dir, "snr_accuracy_curves.png"))

    return {
        "noise_df": noise_df,
        "confusion_matrices_noisy": confusion_matrices_noisy,
    }
