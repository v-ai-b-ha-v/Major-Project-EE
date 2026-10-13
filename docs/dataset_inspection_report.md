# SEED-PQD Dataset Inspection & Signal Verification Report

**Project Title:** Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification  
**Milestone:** Tasks 1–4: Acquisition, Inspection, Preprocessing, and FFT/DWT Signal Processing  
**Dataset Source:** SEED-PQD v1 ([Zenodo Record 11843312](https://zenodo.org/records/11843312))  
**Verified On:** October 13, 2026  

---

## 1. Executive Summary

This report documents the verified physical characteristics, directory structure, data formats, and data integrity of the **SEED-PQD** (Synthetic Electrical Energy Disturbance - Power Quality Disturbance) benchmark dataset. All attributes reported herein were empirically verified by loading and inspecting the raw files.

---

## 2. Dataset Acquisition & File System Structure

### 2.1 File Organization
The dataset distribution contains two primary files from Zenodo:
1. `SEED_PQD_v1.zip` (Compressed archive, 22,195,803 bytes ~ 22.2 MB)
2. `Details.txt` (Metadata specification file, 891 bytes)

When extracted, the archive uncompresses into an internal subfolder named `XPQRS/` containing:
- **1 MATLAB Matrix File:** `5Kfs_1Cycle_50f_1000Sam_1A.mat` (11,362,502 bytes ~ 11.36 MB)
- **17 Class CSV Files:** One per disturbance category (approx. 1.78 MB to 1.89 MB each)
- **1 Copy of Details.txt:** (891 bytes)

### 2.2 Available File Formats
1. **CSV Format (`.csv`):**
   - 17 separate CSV files named after their respective disturbance classes.
   - Matrix dimension per file: **1,000 rows × 100 columns**.
   - No column headers. Values are comma-separated floating-point numbers representing discrete signal voltages.
2. **MATLAB Format (`.mat`):**
   - Formatted as MATLAB v7 file.
   - Contains a single 3D array key `'Out'` of dimension **1000 × 100 × 17** (`Signals per Class × Samples per Signal × Disturbance Classes`).
   - Slices `Out[:, :, i]` match the 17 CSV files to double-precision floating-point accuracy ($|\text{CSV} - \text{MAT}| \le 5.11 \times 10^{-15}$).

---

## 3. Physical Signal Characteristics

From `Details.txt` and empirical waveform verification:

| Parameter | Value | Engineering Significance |
| :--- | :--- | :--- |
| **Nominal Power Frequency ($f_0$)** | **50 Hz** | Standard AC power frequency in Europe, Asia, and India. |
| **Sampling Frequency ($f_s$)** | **5,000 Hz (5 kHz)** | Satisfies Nyquist criterion ($f_s \gg 2 f_0$); Nyquist limit = 2,500 Hz. |
| **Sampling Period ($\Delta t$)** | **0.2 ms ($200\ \mu\text{s}$)** | Time elapsed between consecutive discrete samples. |
| **Samples per Signal ($N$)** | **100 samples** | Discrete sequence length. |
| **Duration of Signal ($T$)** | **20 ms (0.02 s)** | Exactly **1 complete AC cycle** at 50 Hz ($T = 1 / 50\text{ s} = 20\text{ ms}$). |
| **Frequency Resolution ($\Delta f$)** | **50 Hz** | $\Delta f = f_s / N = 5000 / 100 = 50\text{ Hz}$. Bin 1 is the 50 Hz fundamental. |
| **Nominal Signal Amplitude** | **-1.0 to +1.0 p.u.** | Pure sinusoidal voltage is normalized to unity peak (RMS = 0.7071 p.u.). |
| **Global Signal Amplitude Range** | **-2.4601 to +2.4857 p.u.** | Observed during Swell and Oscillatory Transient events. |

---

## 4. Disturbance Classes and Label Taxonomy

The dataset contains **17 distinct classes** divided into baseline, single, and compound power quality disturbance phenomena:

| Class Index | Class Label | Type | Electrical Phenomenon Description |
| :---: | :--- | :--- | :--- |
| **0** | `Pure_Sinusoidal` | Baseline | Undisturbed 50 Hz sinusoidal voltage ($V_{\text{peak}} = 1.0$, $\text{RMS} = 0.707$). |
| **1** | `Sag` | Single | Short-duration drop in RMS voltage between 0.1 and 0.9 p.u. |
| **2** | `Swell` | Single | Short-duration increase in RMS voltage above 1.1 p.u. (up to ~1.8 p.u.). |
| **3** | `Interruption` | Single | Severe supply reduction where voltage drops below 0.1 p.u. |
| **4** | `Transient` | Single | Sudden high-frequency unidirectional voltage impulse / spike. |
| **5** | `Oscillatory_Transient` | Single | High-frequency oscillatory ringing superimposed on the voltage waveform. |
| **6** | `Harmonics` | Single | Steady-state waveform distortion containing integer multiples of 50 Hz. |
| **7** | `Harmonics_with_Sag` | Compound | Harmonics superimposed on an RMS voltage reduction. |
| **8** | `Harmonics_with_Swell` | Compound | Harmonics superimposed on an elevated RMS voltage condition. |
| **9** | `Flicker` | Single | Low-frequency amplitude modulation causing perceptible lighting fluctuations. |
| **10** | `Flicker_with_Sag` | Compound | Amplitude modulation coupled with a voltage sag. |
| **11** | `Flicker_with_Swell` | Compound | Amplitude modulation coupled with a voltage swell. |
| **12** | `Sag_with_Oscillatory_Transient` | Compound | Oscillatory transient event occurring during a voltage sag. |
| **13** | `Swell_with_Oscillatory_Transient` | Compound | Oscillatory transient event occurring during a voltage swell. |
| **14** | `Sag_with_Harmonics` | Compound | Voltage sag accompanied by harmonic distortion. |
| **15** | `Swell_with_Harmonics` | Compound | Voltage swell accompanied by harmonic distortion. |
| **16** | `Notch` | Single | Periodic or commutation disturbances causing sharp downward voltage notches. |

---

## 5. Data Integrity & Quality Checks

1. **Missing Values (`NaN`):**  
   - Checked across all 17 CSV files ($17 \times 1,000 \times 100 = 1,700,000$ points) and MATLAB matrix: **0 missing values found**.
2. **Infinite Values (`Inf` / `-Inf`):**  
   - Checked across all samples: **0 infinite values found**.
3. **Class Balance:**  
   - Every single class contains exactly **1,000 signals**.
   - Total dataset size: **17,000 signals**.
   - Class proportion: **5.882% per class** (perfectly balanced dataset; no class re-weighting or synthetic oversampling required).
4. **Data Types:**  
   - Loaded and cast to IEEE 754 64-bit floating point (`np.float64`) for numerical stability.

---

## 6. Preprocessing Methodology & Assumptions

1. **Data Preservation:**  
   The raw dataset is already clean and normalized in per-unit (p.u.) form. No aggressive denoising or artificial filtering is applied, as filtering would unintentionally attenuate high-frequency transients and notches.
2. **Matrix Structuring:**  
   The signals are assembled into a feature matrix $\mathbf{X} \in \mathbb{R}^{17000 \times 100}$ and target label arrays $\mathbf{y}_{\text{str}} \in \{\text{class names}\}$ and $\mathbf{y}_{\text{idx}} \in \{0, \dots, 16\}$.
3. **Assumptions:**  
   - Each signal row captures exactly one single-cycle window (20 ms at 50 Hz).
   - The sampling frequency $f_s = 5000\text{ Hz}$ is uniform across all signals.
   - Class index 0 (`Pure_Sinusoidal`) represents the ideal reference condition.

---

## 7. Signal Processing Strategy

### 7.1 Fast Fourier Transform (FFT) / DFT
- **Theoretical Formulation:** The Fast Fourier Transform (FFT) is an efficient algorithm to compute the Discrete Fourier Transform (DFT). It reduces computational complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$. They are mathematical identities, not distinct transforms.
- **Physical Grid:** With $N = 100$ and $f_s = 5000\text{ Hz}$, the one-sided spectrum contains 51 bins spanning $0\text{ Hz}$ to $2500\text{ Hz}$ in steps of $\Delta f = 50\text{ Hz}$.
- **Extracted Features:**
  - Fundamental magnitude (50 Hz peak)
  - Total Harmonic Distortion (THD)
  - Spectral Energy, Spectral Centroid, Spectral Entropy
  - Dominant frequency and peak magnitude

### 7.2 Discrete Wavelet Transform (DWT)
- **Multi-Resolution Analysis:** Uses Daubechies 4 (`db4`) at Level 3.
  - Approximation $\mathbf{cA_3}$ (0 – 312.5 Hz): captures fundamental voltage and low harmonics (sensitive to Sag, Swell, Interruption).
  - Detail $\mathbf{cD_3}$ (312.5 – 625 Hz): captures harmonic frequencies.
  - Detail $\mathbf{cD_2}$ (625 – 1250 Hz): captures mid-frequency oscillatory transients.
  - Detail $\mathbf{cD_1}$ (1250 – 2500 Hz): captures rapid impulse transients and notches.
- **Extracted Features:** Sub-band energies, standard deviations, relative energy ratios, and mean absolute values across each band (20 features total).

---

## 8. Verified Artifact Checklist

- [x] `dataset_inspection_report.md` (This document)
- [x] `requirements.txt` (Dependencies: NumPy, Pandas, SciPy, Matplotlib, PyWavelets)
- [x] `data_loader.py` (Modular loading, shape validation, sanity checks)
- [x] `signal_processing.py` (FFT & DWT implementation and feature extractors)
- [x] `visualization.py` (Waveform and spectral plotting functions)
- [x] `main_milestone.py` (End-to-end execution script)
- [x] `class_distribution.png` (Class balance visualization)
- [x] `time_domain_waveforms.png` (Waveforms of Pure Sinusoid and 8 disturbance classes)
- [x] `fft_magnitude_spectra.png` (One-sided FFT spectra comparing disturbances to normal)
- [x] `dwt_decomposition.png` (Multi-level DWT decomposition plot for transient disturbance)
- [x] `dwt_features.png` (Sub-band energy feature distribution comparison)
- [x] `extracted_features.csv` (17,000 rows × 29 columns of extracted FFT/DWT features and labels)
- [x] `README.md` (Execution guide)
