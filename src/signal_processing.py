"""
Signal Processing Module: FFT and DWT Implementation
=====================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Module: src/signal_processing.py

THEORETICAL FOUNDATION & NOTE ON FFT VS. DFT:
---------------------------------------------
Mathematically, the Discrete Fourier Transform (DFT) transforms a finite sequence
of discrete-time samples x[n] (n = 0, ..., N-1) into discrete frequency-domain
coefficients X[k] according to the definition:

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N),   k = 0, ..., N-1

Computing the DFT directly requires O(N^2) complex multiplication and addition
operations. The Fast Fourier Transform (FFT), pioneered by Cooley and Tukey (1965),
is an algorithm for efficiently computing the exact mathematical DFT in O(N log N)
operations.

IMPORTANT: The FFT is an efficient ALGORITHM for computing the DFT. They are not
fundamentally independent mathematical transforms. In this codebase, NumPy/SciPy's
FFT routines are used as the standard algorithm to compute the discrete Fourier
spectrum.
"""

import numpy as np
import pandas as pd
import scipy.fft as sfft
import pywt


# ==============================================================================
# SECTION A: FAST FOURIER TRANSFORM (FFT / DFT)
# ==============================================================================

def compute_fft(signal, fs=5000):
    """
    Computes the one-sided Discrete Fourier Transform (DFT) of a 1D real signal
    using the Fast Fourier Transform (FFT) algorithm.

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    fs : float, default 5000.0
        Sampling frequency in Hertz (Hz).

    Returns:
    --------
    freqs : np.ndarray, shape (N//2 + 1,)
        Physical frequency axis values in Hertz (0 to Nyquist, fs/2).
    magnitude : np.ndarray, shape (N//2 + 1,)
        Normalized one-sided amplitude spectrum.
        Scaled so that a pure sinusoidal component of peak amplitude A
        produces an amplitude of A at its frequency bin.
    """
    N = len(signal)
    
    # Compute real FFT (efficient DFT for real-valued sequences)
    X = sfft.rfft(signal)
    
    # Frequency axis: freqs[k] = k * (fs / N)
    freqs = sfft.rfftfreq(N, d=1.0 / fs)
    
    # One-sided amplitude scaling:
    magnitude = np.abs(X) / N
    if N % 2 == 0:
        magnitude[1:-1] *= 2.0
    else:
        magnitude[1:] *= 2.0
        
    return freqs, magnitude


def extract_fft_features(signal, fs=5000, fund_freq=50.0):
    """
    Extracts physically meaningful frequency-domain features from the FFT spectrum:
    1. Fundamental frequency magnitude (at 50 Hz).
    2. Total Harmonic Distortion (THD) - ratio of harmonic RMS to fundamental.
    3. Spectral energy - total energy distributed across frequency bins.
    4. Spectral centroid - center-of-mass frequency of the spectrum (in Hz).
    5. Spectral entropy - measure of spectral peakiness vs. spread of energy.
    6. Dominant frequency - frequency corresponding to maximum spectral peak.
    7. Dominant peak magnitude - amplitude of the dominant spectral peak.

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    fs : float, default 5000.0
        Sampling frequency in Hz.
    fund_freq : float, default 50.0
        Nominal power fundamental frequency in Hz.

    Returns:
    --------
    features : dict
        Extracted FFT features with descriptive keys.
    """
    freqs, mag = compute_fft(signal, fs=fs)
    N = len(signal)
    freq_resolution = fs / N  # For fs=5000 and N=100, delta_f = 50 Hz

    # Identify fundamental frequency index (closest bin to 50 Hz)
    fund_idx = int(round(fund_freq / freq_resolution))
    fund_mag = float(mag[fund_idx]) if fund_idx < len(mag) else 0.0

    # Total Harmonic Distortion (THD)
    harmonic_indices = [
        k for k in range(2 * fund_idx, len(mag), fund_idx) if k < len(mag)
    ]
    if harmonic_indices:
        harmonic_energy = np.sum(mag[harmonic_indices] ** 2)
        thd = float(np.sqrt(harmonic_energy) / (fund_mag + 1e-6))
    else:
        thd = 0.0

    # Spectral Energy (sum of squared magnitude bins)
    spectral_energy = float(np.sum(mag ** 2))

    # Spectral Centroid (Center of mass of frequency spectrum)
    total_mag = np.sum(mag)
    if total_mag > 1e-12:
        spectral_centroid = float(np.sum(freqs * mag) / total_mag)
    else:
        spectral_centroid = 0.0

    # Spectral Entropy (Entropy of normalized power distribution)
    power_dist = (mag ** 2) / (spectral_energy + 1e-12)
    nonzero_p = power_dist[power_dist > 1e-12]
    spectral_entropy = float(-np.sum(nonzero_p * np.log(nonzero_p)))

    # Dominant Peak Frequency and Magnitude
    dom_idx = int(np.argmax(mag))
    dominant_freq = float(freqs[dom_idx])
    dominant_mag = float(mag[dom_idx])

    return {
        "fft_fund_mag": fund_mag,
        "fft_thd": thd,
        "fft_spec_energy": spectral_energy,
        "fft_spec_centroid": spectral_centroid,
        "fft_spec_entropy": spectral_entropy,
        "fft_dominant_freq": dominant_freq,
        "fft_dominant_mag": dominant_mag,
    }


# ==============================================================================
# SECTION B: DISCRETE WAVELET TRANSFORM (DWT)
# ==============================================================================

def compute_dwt(signal, wavelet="db4", level=3):
    """
    Computes the Discrete Wavelet Transform (DWT) decomposition of a 1D signal
    using PyWavelets.

    Decomposition Scheme (for N = 100 samples, fs = 5000 Hz, wavelet='db4', level=3):
    - Approximation cA3 : Frequency band  0.0  to 312.5 Hz (contains fundamental 50 Hz & 3rd, 5th harmonics)
    - Detail cD3        : Frequency band 312.5 to 625.0 Hz (contains higher-order harmonics)
    - Detail cD2        : Frequency band 625.0 to 1250.0 Hz (contains mid-frequency oscillatory transients)
    - Detail cD1        : Frequency band 1250.0 to 2500.0 Hz (contains high-frequency transients & notches)

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    wavelet : str, default 'db4'
        Wavelet family and order (Daubechies 4 is standard in PQ disturbance literature).
    level : int, default 3
        Decomposition level chosen based on signal length (N=100) to prevent boundary distortion.

    Returns:
    --------
    coeffs : list of np.ndarray
        List [cA_level, cD_level, cD_{level-1}, ..., cD_1]
    band_names : list of str
        Descriptive names of each sub-band.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    band_names = [f"cA{level}"] + [f"cD{i}" for i in range(level, 0, -1)]
    return coeffs, band_names


def extract_dwt_features(signal, wavelet="db4", level=3):
    """
    Extracts statistical and energy wavelet features across all decomposition sub-bands:
    For each sub-band (cA3, cD3, cD2, cD1):
    1. Sub-band Energy (sum of squared coefficients).
    2. Sub-band Standard Deviation (dispersion of wavelet coefficients).
    3. Relative Sub-band Energy (ratio of band energy to total wavelet energy).
    4. Mean Absolute Value (MAV).
    5. Maximum Absolute Value (peak disturbance magnitude in the band).

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    wavelet : str, default 'db4'
        Wavelet family and order.
    level : int, default 3
        Decomposition level.

    Returns:
    --------
    features : dict
        Extracted DWT features with descriptive keys.
    """
    coeffs, band_names = compute_dwt(signal, wavelet=wavelet, level=level)
    
    # Calculate energy per band
    band_energies = [float(np.sum(c ** 2)) for c in coeffs]
    total_energy = float(sum(band_energies)) + 1e-12

    features = {}
    for name, c, energy in zip(band_names, coeffs, band_energies):
        features[f"dwt_{name}_energy"] = energy
        features[f"dwt_{name}_std"] = float(np.std(c))
        features[f"dwt_{name}_rel_energy"] = float(energy / total_energy)
        features[f"dwt_{name}_mav"] = float(np.mean(np.abs(c)))
        features[f"dwt_{name}_max"] = float(np.max(np.abs(c)))

    return features


# ==============================================================================
# SECTION C: COMBINED FEATURE EXTRACTION PIPELINE
# ==============================================================================

def extract_features_single_signal(signal, fs=5000, wavelet="db4", level=3):
    """
    Extracts all FFT and DWT features for a single 1D signal.
    """
    fft_feats = extract_fft_features(signal, fs=fs)
    dwt_feats = extract_dwt_features(signal, wavelet=wavelet, level=level)
    combined = {}
    combined.update(fft_feats)
    combined.update(dwt_feats)
    return combined


def extract_all_dataset_features(X, y_str=None, y_idx=None, fs=5000, wavelet="db4", level=3):
    """
    Performs batch extraction of FFT and DWT features across all signals in dataset matrix X.

    Parameters:
    -----------
    X : np.ndarray, shape (N_signals, 100)
        Matrix of preprocessed signal rows.
    y_str : np.ndarray or list of str, optional
        Class label names corresponding to each row.
    y_idx : np.ndarray or list of int, optional
        Integer class indices corresponding to each row.
    fs : float, default 5000
        Sampling frequency in Hz.
    wavelet : str, default 'db4'
        Wavelet family.
    level : int, default 3
        DWT decomposition level.

    Returns:
    --------
    df_features : pd.DataFrame
        Structured DataFrame containing all extracted features and labels.
    """
    num_signals = X.shape[0]
    print(f"Extracting FFT and DWT features for {num_signals} signals...")

    feature_rows = []
    for i in range(num_signals):
        feats = extract_features_single_signal(X[i], fs=fs, wavelet=wavelet, level=level)
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)

    if y_idx is not None:
        df_features.insert(0, "class_index", y_idx)
    if y_str is not None:
        df_features.insert(0, "class_label", y_str)

    print(f"Feature extraction complete! DataFrame shape: {df_features.shape}")
    return df_features
