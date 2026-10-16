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
import scipy.signal as ssignal
import scipy.stats as sstats
import pywt
from PyEMD import EMD
from joblib import Parallel, delayed


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


# ==============================================================================
# SECTION D: STATISTICAL TIME-DOMAIN FEATURES (METHOD 1)
# ==============================================================================

def extract_statistical_features(signal):
    """
    Extracts 7 statistical time-domain features directly from a 1D discrete signal:
    1. stat_rms: Root Mean Square amplitude: sqrt(1/N * sum(x[n]^2)).
    2. stat_std: Sample standard deviation (degree of dispersion around the mean).
    3. stat_var: Signal variance (square of standard deviation).
    4. stat_skewness: Fisher-Pearson standardized 3rd moment (asymmetry of amplitude distribution).
    5. stat_kurtosis: Fisher excess kurtosis (degree of peakedness/heavy-tails relative to normal=0).
    6. stat_crest_factor: Ratio of absolute peak amplitude to RMS (max|x| / (RMS + eps)).
    7. stat_peak_to_peak: Dynamic range excursion (max(x) - min(x)).

    Edge cases (constant signals, zero voltage, division-by-zero) are handled gracefully
    with zero-fallbacks and non-finite value protection.

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.

    Returns:
    --------
    features : dict
        7 statistical time-domain features.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    var_val = float(np.var(signal))
    rms_val = float(np.sqrt(np.mean(signal ** 2)))

    # Skewness: 3rd standardized moment
    if std_val > 1e-12:
        skew_val = float(np.mean(((signal - mean_val) / std_val) ** 3))
    else:
        skew_val = 0.0

    # Kurtosis: Fisher excess kurtosis (normal distribution = 0.0)
    if std_val > 1e-12:
        kurt_val = float(np.mean(((signal - mean_val) / std_val) ** 4) - 3.0)
    else:
        kurt_val = 0.0

    # Crest factor: peak amplitude / RMS
    if rms_val > 1e-12:
        crest_val = float(np.max(np.abs(signal)) / rms_val)
    else:
        crest_val = 0.0

    peak_to_peak = float(np.max(signal) - np.min(signal))

    return {
        "stat_rms": rms_val,
        "stat_std": std_val,
        "stat_var": var_val,
        "stat_skewness": skew_val,
        "stat_kurtosis": kurt_val,
        "stat_crest_factor": crest_val,
        "stat_peak_to_peak": peak_to_peak,
    }


# ==============================================================================
# SECTION E: SHORT-TIME FOURIER TRANSFORM (STFT) (METHOD 2)
# ==============================================================================

def compute_stft(signal, fs=5000, nperseg=32, noverlap=16, nfft=64):
    """
    Computes the Short-Time Fourier Transform (STFT) of a 1D signal.

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array (N=100 samples).
    fs : float, default 5000.0
        Sampling frequency in Hz.
    nperseg : int, default 32
        Window segment length in samples (32 samples = 6.4 ms, approx 1/3 of a 20 ms cycle).
    noverlap : int, default 16
        Segment overlap in samples (50% overlap, hop size = 3.2 ms).
    nfft : int, default 64
        FFT size (zero-padded to 64 points, yielding 33 frequency bins from 0 to 2500 Hz).

    Returns:
    --------
    f : np.ndarray, shape (nfft//2 + 1,)
        Array of sample frequencies (0 to 2500 Hz).
    t : np.ndarray
        Array of segment times (in seconds).
    Zxx : np.ndarray, complex
        STFT of signal.
    """
    f, t, Zxx = ssignal.stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )
    return f, t, Zxx


def extract_stft_features(signal, fs=5000, nperseg=32, noverlap=16, nfft=64):
    """
    Extracts 8 compact, physically grounded time-frequency features from the STFT spectrogram:
    1. stft_mean_energy: Mean energy density across the time-frequency plane.
    2. stft_max_energy: Peak time-frequency energy concentration.
    3. stft_std_energy: Standard deviation of spectrogram power (energy dispersion).
    4. stft_energy_low_band: Energy in [0, 300 Hz] (fundamental & 3rd/5th harmonics).
    5. stft_energy_mid_band: Energy in (300, 1200 Hz] (interharmonics & oscillatory transients).
    6. stft_energy_high_band: Energy in (1200, 2500 Hz] (high-frequency notches & impulsive transients).
    7. stft_rel_energy_low: Ratio of low-band power to total spectrogram power.
    8. stft_temporal_variability: Average across frequency bins of temporal standard deviation.

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    fs : float, default 5000.0
        Sampling frequency in Hz.

    Returns:
    --------
    features : dict
        8 STFT time-frequency features.
    """
    f, t, Zxx = compute_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    S = np.abs(Zxx) ** 2  # Spectrogram power density
    total_energy = float(np.sum(S)) + 1e-12

    # Physical frequency masks
    low_mask = (f >= 0.0) & (f <= 300.0)
    mid_mask = (f > 300.0) & (f <= 1200.0)
    high_mask = (f > 1200.0) & (f <= (fs / 2.0))

    e_low = float(np.sum(S[low_mask, :]))
    e_mid = float(np.sum(S[mid_mask, :]))
    e_high = float(np.sum(S[high_mask, :]))

    # Temporal variability across frequency channels
    temporal_std = np.std(S, axis=1)  # shape (num_freq_bins,)
    temporal_var = float(np.mean(temporal_std))

    return {
        "stft_mean_energy": float(np.mean(S)),
        "stft_max_energy": float(np.max(S)),
        "stft_std_energy": float(np.std(S)),
        "stft_energy_low_band": e_low,
        "stft_energy_mid_band": e_mid,
        "stft_energy_high_band": e_high,
        "stft_rel_energy_low": float(e_low / total_energy),
        "stft_temporal_variability": temporal_var,
    }


# ==============================================================================
# SECTION F: WAVELET PACKET TRANSFORM (WPT) (METHOD 3)
# ==============================================================================

def compute_wpt(signal, wavelet="db4", level=3):
    """
    Computes the full balanced binary Wavelet Packet Transform (WPT) decomposition.

    Unlike standard DWT (which decomposes only the approximation sub-band at each level),
    WPT symmetrically decomposes both approximation and detail sub-bands, producing
    2^level = 8 terminal nodes at level 3. This provides uniform frequency resolution
    across the entire spectrum (each node covers 2500 / 8 = 312.5 Hz bandwidth).

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    wavelet : str, default 'db4'
        Daubechies 4 wavelet (matching DWT for exact consistency).
    level : int, default 3
        Decomposition depth (yields 2^3 = 8 terminal sub-bands).

    Returns:
    --------
    coeffs : list of np.ndarray
        Terminal node coefficient arrays ordered by ascending natural frequency.
    node_paths : list of str
        Terminal node paths (e.g., 'aaa', 'aad', ..., 'daa').
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="symmetric", maxlevel=level)
    # Natural ascending frequency ordering using Gray-code frequency sort
    nodes = [node.path for node in wp.get_level(level, order="freq")]
    coeffs = [wp[node].data for node in nodes]
    return coeffs, nodes


def extract_wpt_features(signal, wavelet="db4", level=3):
    """
    Extracts 16 Wavelet Packet Transform features from 8 terminal sub-bands:
    - 8 Sub-band Energies: wpt_band0_energy to wpt_band7_energy
      (sum of squared coefficients per terminal frequency band).
    - 8 Relative Sub-band Energies: wpt_band0_rel_energy to wpt_band7_rel_energy
      (sub-band energy normalized by total WPT tree energy).

    Terminal Band Frequency Boundaries (fs = 5000 Hz, Nyquist = 2500 Hz):
    - Band 0: [   0.0,  312.5 Hz] (Contains fundamental 50 Hz & 3rd, 5th harmonics)
    - Band 1: [ 312.5,  625.0 Hz] (7th, 9th, 11th harmonics)
    - Band 2: [ 625.0,  937.5 Hz] (Mid-frequency resonance)
    - Band 3: [ 937.5, 1250.0 Hz] (High-order harmonics)
    - Band 4: [1250.0, 1562.5 Hz] (Oscillatory transient components)
    - Band 5: [1562.5, 1875.0 Hz] (High-frequency transient oscillations)
    - Band 6: [1875.0, 2187.5 Hz] (High-frequency disturbance components)
    - Band 7: [2187.5, 2500.0 Hz] (Fast notches and steep edge transients)

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.

    Returns:
    --------
    features : dict
        16 WPT sub-band energy features.
    """
    coeffs, _ = compute_wpt(signal, wavelet=wavelet, level=level)
    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total_energy = float(sum(energies)) + 1e-12

    features = {}
    for i, e in enumerate(energies):
        features[f"wpt_band{i}_energy"] = e
        features[f"wpt_band{i}_rel_energy"] = float(e / total_energy)
    return features


# ==============================================================================
# SECTION G: HILBERT-HUANG TRANSFORM (HHT) (METHOD 4)
# ==============================================================================

def compute_hht(signal, fs=5000, max_imf=2):
    """
    Computes Empirical Mode Decomposition (EMD) and extracts Intrinsic Mode Functions (IMFs).

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    fs : float, default 5000.0
        Sampling frequency in Hz.
    max_imf : int, default 2
        Maximum number of primary IMFs to extract (IMF 1 = highest frequency/transient,
        IMF 2 = fundamental/intermediate frequency).

    Returns:
    --------
    imfs : np.ndarray, shape (num_imfs, N)
        Extracted IMFs.
    """
    emd = EMD()
    try:
        imfs = emd(signal, max_imf=max_imf)
    except Exception:
        imfs = np.zeros((0, len(signal)), dtype=np.float64)
    return imfs


def extract_hht_features(signal, fs=5000, max_imf=2):
    """
    Extracts 8 Hilbert-Huang Transform (HHT) features across the 2 primary IMFs:
    For each IMF k (k = 1, 2):
    1. hht_imf{k}_mean_amp: Mean instantaneous amplitude.
    2. hht_imf{k}_std_amp: Standard deviation of instantaneous amplitude.
    3. hht_imf{k}_mean_freq: Mean instantaneous frequency (in Hz, bounded to [0, fs/2]).
    4. hht_imf{k}_energy: Energy of the IMF (sum of squared values).

    Robust Handling:
    - If EMD extracts fewer than max_imf modes (e.g. Pure Sinusoidal which only has 1 IMF),
      remaining modes are padded with zeros to ensure a consistent, non-NaN schema.
    - Instantaneous frequency is computed via numerical differentiation of unwrapped analytic phase:
        f_inst[n] = (fs / (2*pi)) * (theta[n] - theta[n-1])
      and bounded to the physical Nyquist band [0, fs/2] to suppress numerical boundary differentiation spikes.

    Parameters:
    -----------
    signal : np.ndarray, shape (N,)
        1D discrete time-domain signal array.
    fs : float, default 5000.0
        Sampling frequency in Hz.
    max_imf : int, default 2
        Number of IMFs in feature schema.

    Returns:
    --------
    features : dict
        8 HHT instantaneous amplitude, frequency, and energy features.
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    emd = EMD()

    try:
        imfs = emd(signal, max_imf=max_imf)
    except Exception:
        imfs = np.zeros((0, N), dtype=np.float64)

    num_imfs = imfs.shape[0] if (imfs is not None and imfs.ndim == 2) else 0

    features = {}
    for k in range(1, max_imf + 1):
        idx = k - 1
        if idx < num_imfs and np.sum(imfs[idx] ** 2) > 1e-12:
            c = imfs[idx]
            energy = float(np.sum(c ** 2))
            try:
                analytic = ssignal.hilbert(c)
                amp = np.abs(analytic)
                phase = np.unwrap(np.angle(analytic))
                inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
                inst_freq = np.clip(inst_freq, 0.0, fs / 2.0)
                mean_amp = float(np.mean(amp))
                std_amp = float(np.std(amp))
                mean_freq = float(np.mean(inst_freq))
            except Exception:
                mean_amp, std_amp, mean_freq = 0.0, 0.0, 0.0
        else:
            mean_amp, std_amp, mean_freq, energy = 0.0, 0.0, 0.0, 0.0

        features[f"hht_imf{k}_mean_amp"] = mean_amp
        features[f"hht_imf{k}_std_amp"] = std_amp
        features[f"hht_imf{k}_mean_freq"] = mean_freq
        features[f"hht_imf{k}_energy"] = energy

    return features


# ==============================================================================
# SECTION H: COMPREHENSIVE COMBINED PIPELINE (ALL 6 FEATURE FAMILIES)
# ==============================================================================

def extract_all_extended_features_single_signal(
    signal, fs=5000, fund_freq=50.0, wavelet="db4", level=3, max_imf=2
):
    """
    Extracts all 66 features from all 6 signal processing methods for a single 1D signal:
    1. FFT features (7)
    2. DWT features (20)
    3. Statistical time-domain features (7)
    4. STFT time-frequency features (8)
    5. WPT sub-band energy features (16)
    6. HHT instantaneous Hilbert features (8)
    Total numerical features = 7 + 20 + 7 + 8 + 16 + 8 = 66 features.
    """
    features = {}
    # 1. FFT
    features.update(extract_fft_features(signal, fs=fs, fund_freq=fund_freq))
    # 2. DWT
    features.update(extract_dwt_features(signal, wavelet=wavelet, level=level))
    # 3. Statistical
    features.update(extract_statistical_features(signal))
    # 4. STFT
    features.update(extract_stft_features(signal, fs=fs))
    # 5. WPT
    features.update(extract_wpt_features(signal, wavelet=wavelet, level=level))
    # 6. HHT
    features.update(extract_hht_features(signal, fs=fs, max_imf=max_imf))
    return features


def _extract_worker_task(x_row, fs, fund_freq, wavelet, level, max_imf):
    return extract_all_extended_features_single_signal(
        x_row,
        fs=fs,
        fund_freq=fund_freq,
        wavelet=wavelet,
        level=level,
        max_imf=max_imf,
    )


def extract_all_extended_dataset_features(
    X,
    y_str=None,
    y_idx=None,
    fs=5000,
    fund_freq=50.0,
    wavelet="db4",
    level=3,
    max_imf=2,
    n_jobs=4,
    batch_size=200,
):
    """
    Performs batch extraction across all signals in dataset matrix X for all 6 methods.
    Employs joblib for parallel execution with progress tracking.

    Returns:
    --------
    df_features : pd.DataFrame
        DataFrame of shape (N_signals, 68) with 2 metadata columns + 66 features.
    """
    num_signals = X.shape[0]
    print(f"Extracting 66 features across 6 methods for {num_signals} signals...")
    print(f"Parallel execution enabled (n_jobs={n_jobs})...")

    try:
        feature_rows = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=1)(
            delayed(_extract_worker_task)(
                X[i],
                fs,
                fund_freq,
                wavelet,
                level,
                max_imf,
            )
            for i in range(num_signals)
        )
    except Exception as e:
        print(f"Parallel extraction encountered {e}; falling back to sequential extraction...")
        feature_rows = []
        for i in range(num_signals):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{num_signals} signals...")
            feature_rows.append(
                _extract_worker_task(
                    X[i],
                    fs,
                    fund_freq,
                    wavelet,
                    level,
                    max_imf,
                )
            )

    df_features = pd.DataFrame(feature_rows)

    if y_idx is not None:
        df_features.insert(0, "class_index", y_idx)
    if y_str is not None:
        df_features.insert(0, "class_label", y_str)

    print(f"Extraction complete! Matrix shape: {df_features.shape}")
    return df_features
