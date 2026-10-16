"""
Visualization Module for Power Quality Disturbance Analysis
============================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Module: src/visualization.py

This module provides plotting functions for:
1. Class distribution across all 17 PQD classes.
2. Representative time-domain waveforms.
3. FFT magnitude spectra with physical frequency axis (0 to 2500 Hz).
4. Multi-level DWT decomposition showing approximation and detail bands.
5. Wavelet sub-band energy feature distributions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.signal_processing import compute_fft, compute_dwt, compute_stft, compute_wpt, compute_hht

# Modern clean aesthetic styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

DEFAULT_FIGURES_DIR = os.path.join("outputs", "figures")


def _resolve_save_path(save_path, default_filename):
    """
    Ensures the target folder exists and creates a safe file path.
    """
    if save_path is None:
        save_path = os.path.join(DEFAULT_FIGURES_DIR, default_filename)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    return save_path


def plot_class_distribution(y_str, save_path=None):
    """
    Plots the class distribution across all disturbance classes.
    """
    save_path = _resolve_save_path(save_path, "class_distribution.png")
    unique_classes, counts = np.unique(y_str, return_counts=True)
    
    sorted_indices = np.argsort(unique_classes)
    sorted_classes = unique_classes[sorted_indices]
    sorted_counts = counts[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bars = ax.bar(
        range(len(sorted_classes)),
        sorted_counts,
        color="#1f77b4",
        edgecolor="#0d47a1",
        alpha=0.85,
        width=0.6,
    )

    # Highlight normal condition in green
    for i, cname in enumerate(sorted_classes):
        if cname == "Pure_Sinusoidal":
            bars[i].set_color("#2ca02c")
            bars[i].set_edgecolor("#1b5e20")

    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels(sorted_classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Signals", fontsize=11, fontweight="bold")
    ax.set_title(
        f"SEED-PQD Class Distribution (Total Signals: {len(y_str)}, Classes: {len(sorted_classes)})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, max(sorted_counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved class distribution plot to: {save_path}")


def plot_representative_waveforms(
    X, y_str, selected_classes=None, fs=5000, save_path=None
):
    """
    Plots representative time-domain waveforms for selected disturbance classes,
    including normal pure sinusoidal operation.
    """
    save_path = _resolve_save_path(save_path, "time_domain_waveforms.png")
    if selected_classes is None:
        selected_classes = [
            "Pure_Sinusoidal",
            "Sag",
            "Swell",
            "Interruption",
            "Transient",
            "Oscillatory_Transient",
            "Harmonics",
            "Flicker",
            "Notch",
        ]

    N = X.shape[1]
    time_ms = np.arange(N) * (1000.0 / fs)

    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9), dpi=300, sharex=True, sharey=False)
    axes = axes.flatten()

    for idx, cname in enumerate(selected_classes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        class_indices = np.where(y_str == cname)[0]
        if len(class_indices) == 0:
            continue

        sample_idx = class_indices[0]
        signal = X[sample_idx]

        color = "#2ca02c" if cname == "Pure_Sinusoidal" else "#d62728" if "Transient" in cname or "Notch" in cname else "#1f77b4"

        ax.plot(time_ms, signal, color=color, linewidth=1.5, label=f"Sample #{sample_idx}")
        ax.set_title(cname.replace("_", " "), fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(0, 20.0)

        if idx % cols == 0:
            ax.set_ylabel("Voltage (p.u.)", fontsize=9)
        if idx >= (rows - 1) * cols:
            ax.set_xlabel("Time (ms)", fontsize=9)

    plt.suptitle(
        "Representative Time-Domain Waveforms (1 Power Cycle = 20 ms @ 50 Hz, fs = 5 kHz)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved representative waveforms plot to: {save_path}")


def plot_fft_magnitude_spectra(
    X, y_str, selected_classes=None, fs=5000, save_path=None
):
    """
    Plots the one-sided FFT magnitude spectra for representative classes.
    """
    save_path = _resolve_save_path(save_path, "fft_magnitude_spectra.png")
    if selected_classes is None:
        selected_classes = [
            "Pure_Sinusoidal",
            "Harmonics",
            "Sag",
            "Swell",
            "Oscillatory_Transient",
            "Notch",
        ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300, sharex=True)
    axes = axes.flatten()

    for idx, cname in enumerate(selected_classes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        class_indices = np.where(y_str == cname)[0]
        if len(class_indices) == 0:
            continue

        signal = X[class_indices[0]]
        freqs, mag = compute_fft(signal, fs=fs)

        color = "#2ca02c" if cname == "Pure_Sinusoidal" else "#ff7f0e" if "Harmonics" in cname else "#1f77b4"
        
        ax.plot(freqs, mag, color=color, linewidth=1.3)
        ax.scatter(freqs[1], mag[1], color="red", s=25, zorder=4, label="Fundamental (50 Hz)")

        ax.set_title(f"FFT Spectrum: {cname.replace('_', ' ')}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 2500)
        ax.grid(True, linestyle=":", alpha=0.6)

        if idx % 3 == 0:
            ax.set_ylabel("Magnitude (p.u.)", fontsize=9)
        if idx >= 3:
            ax.set_xlabel("Frequency (Hz)", fontsize=9)

        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "FFT Magnitude Spectra (fs = 5000 Hz, Nyquist = 2500 Hz, Delta_f = 50 Hz)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved FFT magnitude spectra plot to: {save_path}")


def plot_dwt_decomposition(
    X, y_str, selected_class="Oscillatory_Transient", wavelet="db4", level=3, save_path=None
):
    """
    Visualizes multi-level DWT decomposition for a representative disturbance signal.
    """
    save_path = _resolve_save_path(save_path, "dwt_decomposition.png")
    class_indices = np.where(y_str == selected_class)[0]
    if len(class_indices) == 0:
        class_indices = [0]
    sample_idx = class_indices[0]
    signal = X[sample_idx]

    coeffs, band_names = compute_dwt(signal, wavelet=wavelet, level=level)

    fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(10, 8), dpi=300, sharex=False)

    time_ms = np.arange(len(signal)) * (1000.0 / 5000.0)
    axes[0].plot(time_ms, signal, color="#1f77b4", linewidth=1.4)
    axes[0].set_title(f"Original Signal: {selected_class.replace('_', ' ')} (N=100, fs=5 kHz)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Voltage (p.u.)", fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    band_descriptions = [
        "cA3: Approx Band [0 - 312.5 Hz] (Fundamental & Low Harmonics)",
        "cD3: Detail Band [312.5 - 625 Hz] (Harmonic Components)",
        "cD2: Detail Band [625 - 1250 Hz] (Oscillatory Frequencies)",
        "cD1: Detail Band [1250 - 2500 Hz] (High-Frequency Transients & Notches)",
    ]

    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for i, (name, coeff, desc, c) in enumerate(zip(band_names, coeffs, band_descriptions, colors)):
        ax = axes[i + 1]
        ax.plot(coeff, color=c, marker="o", markersize=3, linewidth=1.0)
        ax.set_title(f"{name} ({desc}) | {len(coeff)} coeffs", fontsize=9, fontweight="bold")
        ax.set_ylabel("Coeff Mag", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Wavelet Coefficient Index", fontsize=9)
    plt.suptitle(
        f"Multi-Level DWT Decomposition using Wavelet '{wavelet}' (Level={level})",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved DWT decomposition plot to: {save_path}")


def plot_dwt_feature_comparison(features_df, save_path=None):
    """
    Plots the mean wavelet energy distribution across sub-bands (cA3, cD3, cD2, cD1).
    """
    save_path = _resolve_save_path(save_path, "dwt_features.png")
    classes_to_compare = [
        "Pure_Sinusoidal",
        "Sag",
        "Swell",
        "Interruption",
        "Harmonics",
        "Transient",
        "Oscillatory_Transient",
        "Notch",
    ]

    available = [c for c in classes_to_compare if c in features_df["class_label"].values]
    
    bands = ["cA3", "cD3", "cD2", "cD1"]
    band_labels = ["cA3 (0-312.5 Hz)", "cD3 (312.5-625 Hz)", "cD2 (625-1250 Hz)", "cD1 (1250-2500 Hz)"]
    
    mean_energies = []
    for cname in available:
        subset = features_df[features_df["class_label"] == cname]
        band_vals = [subset[f"dwt_{b}_energy"].mean() for b in bands]
        mean_energies.append(band_vals)

    mean_energies = np.array(mean_energies)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(available))
    width = 0.2

    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for i in range(len(bands)):
        ax.bar(
            x + (i - 1.5) * width,
            mean_energies[:, i],
            width,
            label=band_labels[i],
            color=colors[i],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in available], fontsize=9)
    ax.set_ylabel("Mean Sub-band Energy (Sum of c^2)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Comparison of DWT Sub-band Energy Features across Disturbance Classes (db4, Level 3)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(title="Wavelet Sub-bands", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    print(f"Saved DWT feature comparison plot to: {save_path}")


def plot_stft_spectrograms(X, y_str, fs=5000, save_path=None):
    """
    Plots STFT time-frequency spectrograms for 4 representative power quality disturbances:
    1. Pure Sinusoidal (baseline)
    2. Sag (amplitude sag)
    3. Harmonics (harmonic distortion)
    4. Oscillatory Transient (high-frequency oscillation)
    """
    save_path = _resolve_save_path(save_path, "stft_spectrograms.png")
    rep_classes = ["Pure_Sinusoidal", "Sag", "Harmonics", "Oscillatory_Transient"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.flatten()

    for idx, cname in enumerate(rep_classes):
        match_idx = np.where(y_str == cname)[0]
        sig = X[match_idx[0]] if len(match_idx) > 0 else np.zeros(100)
        f, t, Zxx = compute_stft(sig, fs=fs, nperseg=32, noverlap=16, nfft=64)
        S = np.abs(Zxx) ** 2

        ax = axes[idx]
        pcm = ax.pcolormesh(
            t * 1000.0,
            f,
            S,
            shading="gouraud",
            cmap="viridis",
        )
        ax.set_title(f"Class: {cname.replace('_', ' ')}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.set_ylim(0, 2500)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Power Density", fontsize=8)

    fig.suptitle(
        "Short-Time Fourier Transform (STFT) Spectrograms Across PQ Disturbances (fs = 5 kHz)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved STFT spectrograms plot to: {save_path}")


def plot_wpt_subband_energies(df_features, save_path=None):
    """
    Plots Wavelet Packet Transform (WPT) relative energy distribution across all 8 terminal
    sub-bands for key representative classes.
    """
    save_path = _resolve_save_path(save_path, "wpt_subband_energies.png")
    wpt_rel_cols = [f"wpt_band{i}_rel_energy" for i in range(8)]
    band_names = [f"B{i}\n[{i*312.5:.0f}-{(i+1)*312.5:.0f}Hz]" for i in range(8)]

    rep_classes = ["Pure_Sinusoidal", "Sag", "Harmonics", "Oscillatory_Transient", "Notch", "Flicker"]
    available = [c for c in rep_classes if c in df_features["class_label"].values]

    mean_profiles = []
    for c in available:
        sub = df_features[df_features["class_label"] == c][wpt_rel_cols]
        mean_profiles.append(sub.mean().values)
    mean_profiles = np.array(mean_profiles)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    x = np.arange(8)
    n_classes = len(available)
    width = 0.8 / n_classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, c in enumerate(available):
        offset = (i - (n_classes - 1) / 2) * width
        ax.bar(
            x + offset,
            mean_profiles[i],
            width,
            label=c.replace("_", " "),
            color=colors[i],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=8)
    ax.set_ylabel("Mean Relative Energy (Normalized)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Wavelet Packet Transform (WPT Level 3, db4): Terminal Sub-Band Energy Distributions",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(title="Disturbance Class", fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved WPT sub-band energy plot to: {save_path}")


def plot_hht_decomposition(X, y_str, selected_class="Oscillatory_Transient", fs=5000, save_path=None):
    """
    Plots the Empirical Mode Decomposition (EMD) and Hilbert instantaneous amplitude profile
    for a representative disturbance signal.
    """
    save_path = _resolve_save_path(save_path, "hht_decomposition.png")
    match_idx = np.where(y_str == selected_class)[0]
    sig = X[match_idx[0]] if len(match_idx) > 0 else np.zeros(100)
    t_ms = np.linspace(0, 20.0, len(sig), endpoint=False)

    imfs = compute_hht(sig, fs=fs, max_imf=2)
    num_imfs = imfs.shape[0] if (imfs is not None and imfs.ndim == 2) else 0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=300, sharex=True)

    # 1. Raw Signal
    axes[0].plot(t_ms, sig, color="#1f77b4", lw=1.6)
    axes[0].set_title(f"Original Signal: {selected_class.replace('_', ' ')}", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude (p.u.)", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. IMF 1 (High frequency transient mode)
    if num_imfs >= 1:
        axes[1].plot(t_ms, imfs[0], color="#d62728", lw=1.4)
        axes[1].set_title("IMF 1 (High-Frequency / Transient Mode)", fontsize=10, fontweight="bold")
    else:
        axes[1].plot(t_ms, np.zeros_like(t_ms), color="#d62728")
        axes[1].set_title("IMF 1 (Not Active)", fontsize=10)
    axes[1].set_ylabel("IMF 1 (p.u.)", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. IMF 2 (Dominant fundamental mode)
    if num_imfs >= 2:
        axes[2].plot(t_ms, imfs[1], color="#2ca02c", lw=1.4)
        axes[2].set_title("IMF 2 (Fundamental 50 Hz Component)", fontsize=10, fontweight="bold")
    elif num_imfs == 1:
        axes[2].plot(t_ms, imfs[0], color="#2ca02c", lw=1.4)
        axes[2].set_title("IMF 1 as Dominant Fundamental Mode", fontsize=10, fontweight="bold")
    else:
        axes[2].plot(t_ms, np.zeros_like(t_ms), color="#2ca02c")
        axes[2].set_title("IMF 2 (Not Active)", fontsize=10)
    axes[2].set_ylabel("IMF 2 (p.u.)", fontsize=9)
    axes[2].set_xlabel("Time (ms)", fontsize=10, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Hilbert-Huang Transform (HHT): Empirical Mode Decomposition of {selected_class.replace('_', ' ')}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved HHT decomposition plot to: {save_path}")


def plot_confusion_matrix(cm, class_names, title="XGBoost Confusion Matrix", save_path=None):
    """
    Plots a normalized confusion matrix heatmap for 17 PQD classes.
    """
    save_path = _resolve_save_path(save_path, "confusion_matrix.png")
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Accuracy", fontsize=10, fontweight="bold")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=[c.replace("_", " ") for c in class_names],
        yticklabels=[c.replace("_", " ") for c in class_names],
        title=title,
        ylabel="True Class Label",
        xlabel="Predicted Class Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Format numbers inside cells
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            if val >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def plot_feature_method_comparison(comparison_df, save_path=None):
    """
    Plots comparison bar charts across all evaluated feature configurations (A through L).
    """
    save_path = _resolve_save_path(save_path, "method_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    configs = comparison_df["config_label"].values
    acc = comparison_df["accuracy"].values * 100.0
    f1 = comparison_df["macro_f1"].values * 100.0
    features_count = comparison_df["num_features"].values

    x = np.arange(len(configs))

    # Subplot 1: Accuracy & Macro-F1
    bars1 = ax1.bar(x - 0.2, acc, width=0.38, label="Test Accuracy (%)", color="#1f77b4", alpha=0.85)
    bars2 = ax1.bar(x + 0.2, f1, width=0.38, label="Macro F1-Score (%)", color="#ff7f0e", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(min(acc.min(), f1.min()) - 5, 102)
    ax1.set_title("Classification Accuracy & Macro F1 Across Feature Sets", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Feature Dimensionality vs Performance
    ax2.bar(x, features_count, color="#2ca02c", alpha=0.85, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Number of Input Features", fontsize=10, fontweight="bold")
    ax2.set_title("Feature Dimensionality Per Configuration", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(features_count):
        ax2.text(i, v + 1, str(v), ha="center", fontsize=8, fontweight="bold")

    fig.suptitle(
        "Comparative Analysis of Signal Processing Techniques for PQD Classification (XGBoost)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved feature method comparison plot to: {save_path}")
