"""
Feature Dictionary Generator
============================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Module: src/feature_dictionary.py

Defines the mathematical description, physical domain, and units for all 66 features.
"""

import os
import pandas as pd


def generate_feature_dictionary():
    """
    Returns a structured DataFrame documenting all 66 extracted features + 2 metadata columns.
    """
    entries = [
        # Metadata
        {"column_name": "class_label", "family": "Metadata", "domain": "Labels", "units": "String", "description": "Human-readable disturbance class name (e.g., Pure_Sinusoidal, Sag, Swell)"},
        {"column_name": "class_index", "family": "Metadata", "domain": "Labels", "units": "Integer [0-16]", "description": "Zero-indexed integer label for ML classification"},

        # 1. FFT Features (7)
        {"column_name": "fft_fund_mag", "family": "FFT", "domain": "Frequency", "units": "p.u.", "description": "Normalized magnitude of fundamental power frequency bin at 50 Hz"},
        {"column_name": "fft_thd", "family": "FFT", "domain": "Frequency", "units": "Ratio (dimensionless)", "description": "Total Harmonic Distortion: sqrt(sum(harmonics^2)) / fundamental_magnitude"},
        {"column_name": "fft_spec_energy", "family": "FFT", "domain": "Frequency", "units": "p.u.^2", "description": "Total discrete spectral energy: sum of squared magnitude bins"},
        {"column_name": "fft_spec_centroid", "family": "FFT", "domain": "Frequency", "units": "Hz", "description": "Spectral center-of-mass frequency: sum(f * mag) / sum(mag)"},
        {"column_name": "fft_spec_entropy", "family": "FFT", "domain": "Frequency", "units": "nats", "description": "Spectral entropy measuring peakedness vs. dispersion of power distribution"},
        {"column_name": "fft_dominant_freq", "family": "FFT", "domain": "Frequency", "units": "Hz", "description": "Frequency corresponding to the maximum spectral magnitude peak"},
        {"column_name": "fft_dominant_mag", "family": "FFT", "domain": "Frequency", "units": "p.u.", "description": "Magnitude of the dominant spectral peak"},
    ]

    # 2. DWT Features (20)
    dwt_bands = [
        ("cA3", "Approximation band [0.0 - 312.5 Hz] containing 50 Hz fundamental and 3rd/5th harmonics"),
        ("cD3", "Detail band [312.5 - 625.0 Hz] containing mid-order harmonics"),
        ("cD2", "Detail band [625.0 - 1250.0 Hz] containing oscillatory transients"),
        ("cD1", "Detail band [1250.0 - 2500.0 Hz] containing high-frequency notches and steep edges"),
    ]
    for bname, bdesc in dwt_bands:
        entries.extend([
            {"column_name": f"dwt_{bname}_energy", "family": "DWT", "domain": "Time-Scale", "units": "p.u.^2", "description": f"Sub-band energy (sum of squared coefficients) for {bdesc}"},
            {"column_name": f"dwt_{bname}_std", "family": "DWT", "domain": "Time-Scale", "units": "p.u.", "description": f"Standard deviation of wavelet coefficients for {bdesc}"},
            {"column_name": f"dwt_{bname}_rel_energy", "family": "DWT", "domain": "Time-Scale", "units": "Ratio [0-1]", "description": f"Relative energy ratio: {bname} energy divided by total wavelet energy"},
            {"column_name": f"dwt_{bname}_mav", "family": "DWT", "domain": "Time-Scale", "units": "p.u.", "description": f"Mean Absolute Value of wavelet coefficients for {bdesc}"},
            {"column_name": f"dwt_{bname}_max", "family": "DWT", "domain": "Time-Scale", "units": "p.u.", "description": f"Maximum absolute coefficient amplitude in {bname}"},
        ])

    # 3. Statistical Time-Domain Features (7)
    entries.extend([
        {"column_name": "stat_rms", "family": "Statistical", "domain": "Time", "units": "p.u.", "description": "Root Mean Square amplitude: sqrt(1/N * sum(x[n]^2))"},
        {"column_name": "stat_std", "family": "Statistical", "domain": "Time", "units": "p.u.", "description": "Sample standard deviation (signal spread around mean)"},
        {"column_name": "stat_var", "family": "Statistical", "domain": "Time", "units": "p.u.^2", "description": "Signal variance (second central moment)"},
        {"column_name": "stat_skewness", "family": "Statistical", "domain": "Time", "units": "Dimensionless", "description": "Fisher-Pearson standardized 3rd moment measuring waveform asymmetry"},
        {"column_name": "stat_kurtosis", "family": "Statistical", "domain": "Time", "units": "Dimensionless", "description": "Fisher excess kurtosis measuring heavy-tailedness (normal distribution = 0.0)"},
        {"column_name": "stat_crest_factor", "family": "Statistical", "domain": "Time", "units": "Ratio", "description": "Ratio of absolute peak amplitude to RMS: max|x| / RMS"},
        {"column_name": "stat_peak_to_peak", "family": "Statistical", "domain": "Time", "units": "p.u.", "description": "Dynamic excursion range: max(x) - min(x)"},
    ])

    # 4. STFT Features (8)
    entries.extend([
        {"column_name": "stft_mean_energy", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2 / Hz", "description": "Mean spectrogram power density across time-frequency bins"},
        {"column_name": "stft_max_energy", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2 / Hz", "description": "Maximum spectrogram power density (peak energy concentration)"},
        {"column_name": "stft_std_energy", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2 / Hz", "description": "Standard deviation of spectrogram power density"},
        {"column_name": "stft_energy_low_band", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2", "description": "Integrated energy in fundamental low band [0.0 - 300.0 Hz]"},
        {"column_name": "stft_energy_mid_band", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2", "description": "Integrated energy in intermediate harmonic/transient band (300.0 - 1200.0 Hz]"},
        {"column_name": "stft_energy_high_band", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2", "description": "Integrated energy in high-frequency band (1200.0 - 2500.0 Hz]"},
        {"column_name": "stft_rel_energy_low", "family": "STFT", "domain": "Time-Frequency", "units": "Ratio [0-1]", "description": "Ratio of low-band power to total spectrogram power"},
        {"column_name": "stft_temporal_variability", "family": "STFT", "domain": "Time-Frequency", "units": "p.u.^2", "description": "Average across frequency bins of standard deviation across time slices"},
    ])

    # 5. WPT Features (16)
    wpt_subbands = [
        ("band0", "0.0 - 312.5 Hz", "Fundamental 50 Hz and low-order 3rd/5th harmonics"),
        ("band1", "312.5 - 625.0 Hz", "7th, 9th, 11th harmonic components"),
        ("band2", "625.0 - 937.5 Hz", "Mid-frequency resonance oscillations"),
        ("band3", "937.5 - 1250.0 Hz", "Higher-order harmonic distortions"),
        ("band4", "1250.0 - 1562.5 Hz", "Lower oscillatory transient band"),
        ("band5", "1562.5 - 1875.0 Hz", "Mid oscillatory transient band"),
        ("band6", "1875.0 - 2187.5 Hz", "High oscillatory transient band"),
        ("band7", "2187.5 - 2500.0 Hz", "Fast notches and steep edge transients"),
    ]
    for bkey, freq_range, bdesc in wpt_subbands:
        entries.append({
            "column_name": f"wpt_{bkey}_energy",
            "family": "WPT",
            "domain": "Wavelet Packet",
            "units": "p.u.^2",
            "description": f"Sub-band energy (sum of squared coefficients) in {freq_range} ({bdesc})",
        })
        entries.append({
            "column_name": f"wpt_{bkey}_rel_energy",
            "family": "WPT",
            "domain": "Wavelet Packet",
            "units": "Ratio [0-1]",
            "description": f"Relative energy ratio: {bkey} energy divided by total WPT tree energy",
        })

    # 6. HHT Features (8)
    hht_modes = [
        ("imf1", "Intrinsic Mode Function 1 (high-frequency, fast transient mode)"),
        ("imf2", "Intrinsic Mode Function 2 (dominant fundamental 50 Hz intermediate mode)"),
    ]
    for mkey, mdesc in hht_modes:
        entries.extend([
            {"column_name": f"hht_{mkey}_mean_amp", "family": "HHT", "domain": "Hilbert-Huang", "units": "p.u.", "description": f"Mean instantaneous amplitude of {mdesc}"},
            {"column_name": f"hht_{mkey}_std_amp", "family": "HHT", "domain": "Hilbert-Huang", "units": "p.u.", "description": f"Standard deviation of instantaneous amplitude of {mdesc}"},
            {"column_name": f"hht_{mkey}_mean_freq", "family": "HHT", "domain": "Hilbert-Huang", "units": "Hz", "description": f"Mean instantaneous frequency of {mdesc} (bounded to [0, fs/2])"},
            {"column_name": f"hht_{mkey}_energy", "family": "HHT", "domain": "Hilbert-Huang", "units": "p.u.^2", "description": f"Discrete energy (sum of squared values) of {mdesc}"},
        ])

    df = pd.DataFrame(entries)
    return df


def save_feature_dictionary(output_csv="data/processed/feature_dictionary.csv", output_md="docs/feature_dictionary.md"):
    """
    Saves the feature dictionary in both CSV and GitHub Flavored Markdown formats.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)

    df = generate_feature_dictionary()
    df.to_csv(output_csv, index=False)
    print(f"Saved feature dictionary CSV to: {output_csv}")

    # Generate Markdown documentation
    md_lines = [
        "# Power Quality Disturbance Feature Dictionary",
        "",
        "**National Institute of Technology (NIT) Jalandhar | Department of Electrical Engineering**  ",
        "**Project:** Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification  ",
        "",
        "This dictionary documents all **66 extracted features** across **6 signal processing families** plus **2 metadata columns** (Total: 68 columns in `extended_features.csv`).",
        "",
        "## Summary by Feature Family",
        "",
        "| Family | Physical Domain | Feature Count | Target Phenomenon |",
        "| :--- | :--- | :---: | :--- |",
        "| **Metadata** | Labels & Index | 2 | Class label string and integer class index |",
        "| **FFT** | Frequency (Fourier) | 7 | Steady-state harmonics, THD, spectral centroid |",
        "| **DWT** | Time-Scale (Wavelet) | 20 | Multi-resolution sub-band energies & statistics |",
        "| **Statistical** | Time Domain | 7 | Amplitude spread, RMS, crest factor, skewness, kurtosis |",
        "| **STFT** | Time-Frequency | 8 | Temporal energy progression across frequency bands |",
        "| **WPT** | Wavelet Packet | 16 | Uniform 312.5 Hz sub-band energies & relative energies |",
        "| **HHT** | Hilbert-Huang (EMD) | 8 | Instantaneous amplitude and frequency of IMFs |",
        "| **TOTAL** | **Multi-Domain** | **68 (66 features + 2 metadata)** | **Comprehensive PQ Disturbance Representation** |",
        "",
        "---",
        "",
        "## Complete Feature Specification",
        "",
        "| # | Column Name | Family | Domain | Units | Description |",
        "| :---: | :--- | :---: | :---: | :---: | :--- |",
    ]

    for idx, row in df.iterrows():
        md_lines.append(
            f"| {idx+1} | `{row['column_name']}` | {row['family']} | {row['domain']} | {row['units']} | {row['description']} |"
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved feature dictionary Markdown to: {output_md}")
