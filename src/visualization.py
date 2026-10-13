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
import matplotlib.pyplot as plt
from src.signal_processing import compute_fft, compute_dwt

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

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved DWT feature comparison plot to: {save_path}")
