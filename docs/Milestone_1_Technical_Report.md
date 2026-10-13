# Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification

**Department of Electrical Engineering**  
**National Institute of Technology (NIT) Jalandhar**  
*Major Project — Milestone 1 Comprehensive Technical Report*  
**Date:** October 13, 2026  
**Project Workspace:** `ML Code/`  

---

## Executive Summary

Power quality disturbances (PQDs) such as voltage sags, swells, harmonics, and oscillatory transients cause severe operational degradation in electrical power grids, sensitive industrial automation, and microgrids. Developing an automated, robust machine-learning classification system requires extracting physically discriminative features from non-stationary voltage signals. 

This technical report documents the complete implementation, verification, and empirical findings for **Milestone 1 (Tasks 1 to 4)** of the undergraduate major project. In this milestone:
1. The benchmark **SEED-PQD v1** dataset was acquired, inspected, and validated across all 17 disturbance classes ($17,000$ signals).
2. Data structures, numerical types, and sampling parameters ($f_s = 5000\text{ Hz}$, $f_0 = 50\text{ Hz}$, $N = 100$ samples) were verified.
3. Modular Python software was built for reproducible data loading, sanity checking, and preprocessing without artificial distortions.
4. Two core signal processing techniques—**Discrete Fourier Transform (computed via FFT)** and **Multi-Resolution Discrete Wavelet Transform (DWT)**—were mathematically formulated, implemented, and visualized.
5. A comprehensive feature dataset containing **27 extracted signal processing features** across all $17,000$ signals was constructed and saved to [`extracted_features.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/extracted_features.csv).

---

## 1. Introduction and Problem Definition

### 1.1 Background & Engineering Motivation
Modern power networks increasingly interface with non-linear power electronic loads (rectifiers, inverters, variable-frequency drives) and intermittent renewable energy sources (solar PV, wind generators). These interfaces produce a complex spectrum of disturbances governed by **IEEE Std 1159-2019** (Recommended Practice for Monitoring Electric Power Quality). Accurate classification of PQDs is essential for automated power quality monitoring, root-cause diagnostics, and rapid grid protection.

### 1.2 Objective of the Comparative Analysis
While conventional techniques rely on simple spectral analysis, transient events and compound disturbances present distinct non-stationary characteristics that require multi-domain feature analysis:
- **Frequency-Domain (FFT):** Excels at steady-state harmonic analysis, but lacks time localization.
- **Time-Scale Domain (DWT):** Excels at isolating rapid transients and localized singularities across dyadic frequency octaves.
- **Time-Frequency Domain (STFT):** Tracks spectral energy evolution across moving time windows (Phase 2).
- **Time-Domain Statistical Moments:** Captures amplitude deviations and waveform shape factors (Phase 2).

This milestone establishes the foundational signal processing, validation, and feature generation pipeline.

---

## 2. Task 1: Dataset Acquisition & Physical Signal Inspection

### 2.1 Dataset Source & Repository Structure
The project utilizes the benchmark **SEED-PQD v1** (Synthetic Electrical Energy Disturbance - Power Quality Disturbance) dataset hosted on Zenodo:
- **DOI/URL:** [https://zenodo.org/records/11843312](https://zenodo.org/records/11843312)
- **Archive File:** `SEED_PQD_v1.zip` ($22,195,803$ bytes $\approx 22.2\text{ MB}$)
- **Specification Metadata:** `Details.txt` ($891$ bytes)

When extracted, the archive uncompresses to the internal folder `XPQRS/` containing:
1. `5Kfs_1Cycle_50f_1000Sam_1A.mat`: MATLAB v7 file containing a 3D matrix `'Out'` of dimension $1000 \times 100 \times 17$.
2. **17 Class CSV Files:** Distinct comma-delimited files named after each disturbance category, each sized $1000 \times 100$.

### 2.2 Physical Signal Characteristics
Through parsing `Details.txt` and verifying the data arrays, the physical parameters were confirmed:

| Parameter | Documented & Verified Value | Physical Power System Meaning |
| :--- | :--- | :--- |
| **Fundamental Frequency ($f_0$)** | **50.0 Hz** | Nominal power grid operating frequency (Indian / European grid standard). |
| **Sampling Frequency ($f_s$)** | **5,000 Hz (5 kHz)** | Sample collection rate satisfying the Nyquist-Shannon criterion ($f_s \gg 2f_0$). |
| **Sampling Period ($\Delta t$)** | **0.2 ms ($200\ \mu\text{s}$)** | Discrete time interval between consecutive samples ($\Delta t = 1/f_s$). |
| **Samples per Signal ($N$)** | **100 samples** | Discrete sequence length per recorded window. |
| **Observation Window ($T$)** | **20.0 ms (0.02 s)** | Exactly **1 complete fundamental cycle** at 50 Hz ($T = 1/f_0 = 100 \times 0.2\text{ ms}$). |
| **Frequency Resolution ($\Delta f$)** | **50.0 Hz** | $\Delta f = f_s / N = 5000 / 100 = 50\text{ Hz}$. Bin 1 corresponds to $50\text{ Hz}$. |
| **Nyquist Frequency ($f_{\text{Nyq}}$)** | **2,500 Hz (2.5 kHz)** | Upper physical limit of observable spectral components ($f_s / 2$). |
| **Nominal Voltage Amplitude** | **-1.0 to +1.0 p.u.** | Pure sinusoidal voltage normalized to unity peak ($\text{RMS} = 1/\sqrt{2} \approx 0.7071$). |
| **Observed Global Amplitude** | **[-2.4601, +2.4857] p.u.** | Dynamic range during compound swell and oscillatory transients. |

### 2.3 Taxonomy of the 17 Disturbance Classes
The dataset provides a balanced distribution across 17 distinct power quality classes:

| Index | Disturbance Class | Type | Electrical Phenomenon & Standard Definition |
| :---: | :--- | :--- | :--- |
| **0** | `Pure_Sinusoidal` | Baseline | Undisturbed 50 Hz sine wave ($V_{\text{peak}} = 1.0\text{ p.u.}$, $\text{RMS} = 0.7071\text{ p.u.}$). |
| **1** | `Sag` | Single | Short-duration drop in RMS voltage between $0.1$ and $0.9\text{ p.u.}$ (e.g., motor starting, faults). |
| **2** | `Swell` | Single | Short-duration rise in RMS voltage between $1.1$ and $1.8\text{ p.u.}$ (e.g., sudden load disconnection). |
| **3** | `Interruption` | Single | Severe supply reduction with RMS voltage $< 0.1\text{ p.u.}$ (e.g., breaker trip). |
| **4** | `Transient` | Single | Unidirectional impulsive voltage surge of sub-cycle duration (e.g., lightning strikes). |
| **5** | `Oscillatory_Transient` | Single | Rapid high-frequency oscillatory voltage ringing (e.g., capacitor bank switching). |
| **6** | `Harmonics` | Single | Steady-state non-sinusoidal distortion composed of integer multiples of $50\text{ Hz}$. |
| **7** | `Harmonics_with_Sag` | Compound | Superposition of harmonic distortion on an RMS voltage reduction. |
| **8** | `Harmonics_with_Swell` | Compound | Superposition of harmonic distortion on an elevated RMS voltage condition. |
| **9** | `Flicker` | Single | Low-frequency periodic envelope modulation ($f_m \approx 5 - 25\text{ Hz}$) causing visual lamp flicker. |
| **10** | `Flicker_with_Sag` | Compound | Low-frequency modulation occurring simultaneously with an RMS voltage drop. |
| **11** | `Flicker_with_Swell` | Compound | Low-frequency modulation occurring simultaneously with an RMS voltage swell. |
| **12** | `Sag_with_Oscillatory_Transient` | Compound | High-frequency oscillatory ringing occurring during a voltage sag event. |
| **13** | `Swell_with_Oscillatory_Transient`| Compound | High-frequency oscillatory ringing occurring during a voltage swell event. |
| **14** | `Sag_with_Harmonics` | Compound | Voltage sag condition accompanied by high-order harmonic injection. |
| **15** | `Swell_with_Harmonics` | Compound | Voltage swell condition accompanied by high-order harmonic injection. |
| **16** | `Notch` | Single | Periodic sub-cycle voltage drops caused by commutation overlap in phase-controlled converters. |

### 2.4 Data Integrity Audit
A verification script was executed across every sample in both CSV and MAT formats:
- **Missing (`NaN`) Values:** **0** across all $1,700,000$ points ($17 \times 1000 \times 100$).
- **Infinite (`Inf`) Values:** **0**.
- **Class Balance:** Exactly $1,000$ signals per class ($5.882\%$ uniform representation across all 17 classes).
- **Format Equivalence:** Cross-validation between the 17 CSV files and the MATLAB 3D tensor verified that:
  $$\max_{i,j,k} |\mathbf{X}_{\text{CSV}}[j, k] - \mathbf{X}_{\text{MAT}}[j, k, i]| \le 5.11 \times 10^{-15}$$
  confirming that both distributions are identical to IEEE double-precision limits.

![Class Distribution](class_distribution.png)

---

## 3. Task 2: Dataset Analysis & Waveform Characterization

### 3.1 Time-Domain Behavior
Representative single-cycle waveforms were extracted and plotted over the physical $0 - 20\text{ ms}$ time window:

![Representative Waveforms](time_domain_waveforms.png)

### 3.2 Electrical Analysis of Waveforms
1. **Pure Sinusoidal:** Demonstrates a textbook sine wave starting at $0\text{ ms}$, peaking at $+1.0\text{ p.u.}$ ($5\text{ ms}$), passing zero at $10\text{ ms}$, reaching minimum $-1.0\text{ p.u.}$ ($15\text{ ms}$), and returning to zero at $20\text{ ms}$. RMS value is $0.7071\text{ p.u.}$.
2. **Voltage Sag:** Nominal during the first half-cycle ($0 - 10\text{ ms}$), followed by a sharp transition and reduction in amplitude during the negative half-cycle, lowering the overall RMS value ($0.625\text{ p.u.}$).
3. **Voltage Swell:** The waveform exceeds nominal limits, peaking beyond $-1.75\text{ p.u.}$, increasing RMS voltage to $1.007\text{ p.u.}$.
4. **Interruption:** The waveform exhibits a rapid collapse toward near-zero potential, yielding an RMS voltage of only $0.150\text{ p.u.}$.
5. **Impulsive Transient:** A sudden, steep voltage excursion occurring over a narrow time window ($< 1\text{ ms}$).
6. **Oscillatory Transient:** High-frequency damped oscillations superimposed near the zero-crossing region ($15 - 20\text{ ms}$).
7. **Harmonics:** Severe non-sinusoidal distortion with prominent 3rd ($150\text{ Hz}$) and 5th ($250\text{ Hz}$) harmonic rippling superimposed on the fundamental waveform.
8. **Flicker:** Observable amplitude modulation of the fundamental envelope.
9. **Notch:** Sharp, periodic commutation notches showing sudden sub-cycle downward voltage dips.

---

## 4. Task 3: Data Loading & Preprocessing Architecture

### 4.1 Modular Code Structure
The loading and preprocessing routines were encapsulated in [`data_loader.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/data_loader.py):
- `get_signal_metadata()`: Provides sampling parameters, time vector, and label lists.
- `locate_dataset_dir()`: Automatically detects dataset location across workspace directories and extracts the ZIP archive if needed.
- `load_dataset()`: Parses and structures the signals into feature matrix $\mathbf{X} \in \mathbb{R}^{17000 \times 100}$ and label vectors $\mathbf{y}_{\text{str}} \in \text{str}^{17000}$, $\mathbf{y}_{\text{idx}} \in \mathbb{Z}^{17000}$.
- `validate_dataset()`: Validates array shapes, finite numbers, data types, and class balance.

### 4.2 Preprocessing Methodology & Assumptions
1. **Minimalist Preprocessing Principle:**  
   In power quality classification, aggressive pre-filtering or smoothing can attenuate high-frequency transients and notches. Because the SEED-PQD benchmark is pre-calibrated in per-unit (p.u.) scale, the raw physical waveforms are preserved without artificial filtering.
2. **Data Casting:**  
   All signal columns were converted to `np.float64` to prevent type mismatches during frequency and wavelet transformations.
3. **Structuring:**  
   The dataset was structured into a clean NumPy array $\mathbf{X}$ and paired label vectors ready for downstream feature extraction.

---

## 5. Task 4: Signal Processing Implementation & Mathematics

### 5.1 Section A: Discrete Fourier Transform (FFT)

#### 5.1.1 Mathematical Formulation
The Discrete Fourier Transform (DFT) converts a finite sequence of discrete-time samples $x[n]$ ($n = 0, \dots, N-1$) into discrete frequency coefficients $X[k]$:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$

Computing the DFT directly requires $\mathcal{O}(N^2)$ complex operations. The **Fast Fourier Transform (FFT)** is the standard algorithm developed by Cooley and Tukey (1965) that computes the exact mathematical DFT in $\mathcal{O}(N \log N)$ operations. **FFT is the computational algorithm that evaluates the DFT, not an independent mathematical transform.**

#### 5.1.2 Physical Frequency Grid
For real-valued signals, Hermitian symmetry implies that negative frequencies are redundant. The one-sided spectrum contains $N/2 + 1 = 51$ bins:

$$f_k = k \cdot \frac{f_s}{N} = k \cdot \frac{5000\text{ Hz}}{100} = k \cdot 50\text{ Hz}, \quad k = 0, 1, \dots, 50$$

- $k = 0 \implies 0\text{ Hz}$ (DC component)
- $k = 1 \implies 50\text{ Hz}$ (Fundamental power frequency component $f_0$)
- $k = 2 \implies 100\text{ Hz}$ (2nd harmonic)
- $k = 3 \implies 150\text{ Hz}$ (3rd harmonic)
- $k = 50 \implies 2500\text{ Hz}$ (Nyquist limit $f_{\text{Nyq}}$)

Amplitude scaling is applied to reconstruct peak physical amplitudes:
$$A[k] = \begin{cases} \frac{|X[k]|}{N}, & k = 0 \text{ and } k = N/2 \\ \frac{2 \cdot |X[k]|}{N}, & 1 \le k < N/2 \end{cases}$$

#### 5.1.3 Extracted FFT Features (7 Features)
1. **Fundamental Magnitude ($V_1$):** Peak voltage at $50\text{ Hz}$ ($k=1$).
2. **Total Harmonic Distortion (THD):** Ratio of RMS harmonic content to fundamental:
   $$\text{THD} = \frac{\sqrt{\sum_{h=2}^{50} A[h]^2}}{A[1] + \epsilon}$$
3. **Spectral Energy ($E_{\text{spec}}$):** Total energy across all frequency bins:
   $$E_{\text{spec}} = \sum_{k=0}^{50} A[k]^2$$
4. **Spectral Centroid ($f_c$):** Spectral center-of-mass (in Hz):
   $$f_c = \frac{\sum_{k=0}^{50} f_k \cdot A[k]}{\sum_{k=0}^{50} A[k] + \epsilon}$$
5. **Spectral Entropy ($H_{\text{spec}}$):** Measure of spectral flatness versus peakiness:
   $$P[k] = \frac{A[k]^2}{E_{\text{spec}} + \epsilon}, \quad H_{\text{spec}} = -\sum_{k} P[k] \ln(P[k] + \epsilon)$$
6. **Dominant Frequency ($f_{\text{dom}}$):** Frequency of the largest spectral component:
   $$f_{\text{dom}} = f_{\arg\max_k A[k]}$$
7. **Dominant Magnitude ($A_{\text{dom}}$):** Amplitude of the largest spectral peak ($\max_k A[k]$).

![FFT Magnitude Spectra](fft_magnitude_spectra.png)

#### 5.1.4 Observations from FFT Spectra
- **Pure Sinusoid:** A single discrete impulse of magnitude $1.0\text{ p.u.}$ at $50\text{ Hz}$; zero across all other bins ($\text{THD} \approx 0.0$, $f_c = 50.0\text{ Hz}$).
- **Harmonics:** Strong fundamental peak at $50\text{ Hz}$ accompanied by distinct harmonic spikes at $150\text{ Hz}$ (3rd) and $250\text{ Hz}$ (5th), yielding $\text{THD} = 0.194$ and $f_c = 94.8\text{ Hz}$.
- **Sag & Swell:** The fundamental peak shifts proportionately ($0.867\text{ p.u.}$ for Sag; $1.374\text{ p.u.}$ for Swell).
- **Oscillatory Transient:** A wide spectral hump between $600\text{ Hz}$ and $1000\text{ Hz}$ reflecting high-frequency ringing.
- **Notch:** Broad, periodic harmonic leakage across the entire spectrum up to $2500\text{ Hz}$.

---

### 5.2 Section B: Discrete Wavelet Transform (DWT)

#### 5.2.1 Multi-Resolution Analysis (MRA) Theory
While the Fourier transform averages spectral content across the entire 20 ms window, the Discrete Wavelet Transform decomposes a signal into localized approximation and detail coefficients via iterative dyadic filtering:

$$x[n] = \sum_{k} cA_J[k] \phi_{J,k}[n] + \sum_{j=1}^{J} \sum_{k} cD_j[k] \psi_{j,k}[n]$$

where $\phi(t)$ is the scaling function, $\psi(t)$ is the mother wavelet, $cA_j$ are approximation coefficients (low-pass), and $cD_j$ are detail coefficients (high-pass).

#### 5.2.2 Wavelet & Decomposition Level Selection
- **Mother Wavelet:** Daubechies 4 (`db4`) was chosen due to its compact support, 4 vanishing moments, and proven sensitivity to power system transients.
- **Decomposition Level:** Level 3 was selected. For $N = 100$ samples, PyWavelets confirms that $J=3$ prevents boundary distortion artifacts while matching power system frequency bands.

$$\text{Sampling Frequency } f_s = 5000\text{ Hz} \implies \text{Nyquist Band } [0, 2500\text{ Hz}]$$

| Sub-Band | Filter Type | Frequency Band (Hz) | Coefficient Length | Electrical Phenomenon Isolated |
| :---: | :---: | :---: | :---: | :--- |
| $\mathbf{cA_3}$ | Approximation Level 3 | **$0.0 - 312.5\text{ Hz}$** | 18 | Fundamental 50 Hz, Sag, Swell, Interruption, lower harmonics |
| $\mathbf{cD_3}$ | Detail Level 3 | **$312.5 - 625.0\text{ Hz}$** | 18 | Intermediate harmonics (7th, 9th, 11th) |
| $\mathbf{cD_2}$ | Detail Level 2 | **$625.0 - 1250.0\text{ Hz}$** | 30 | Medium-frequency oscillatory transients |
| $\mathbf{cD_1}$ | Detail Level 1 | **$1250.0 - 2500.0\text{ Hz}$** | 53 | Fast impulsive transients, notches, switching spikes |

![DWT Decomposition](dwt_decomposition.png)

#### 5.2.3 Extracted Wavelet Features (20 Features)
For each sub-band $b \in \{\mathbf{cA_3}, \mathbf{cD_3}, \mathbf{cD_2}, \mathbf{cD_1}\}$:
1. **Sub-Band Energy ($E_b$):**
   $$E_b = \sum_{i=1}^{M_b} c_{b,i}^2$$
2. **Sub-Band Standard Deviation ($\sigma_b$):**
   $$\sigma_b = \sqrt{\frac{1}{M_b} \sum_{i=1}^{M_b} (c_{b,i} - \mu_b)^2}$$
3. **Relative Energy Ratio ($P_b$):**
   $$P_b = \frac{E_b}{\sum_{\text{all bands}} E_k}$$
4. **Mean Absolute Value ($\text{MAV}_b$):**
   $$\text{MAV}_b = \frac{1}{M_b} \sum_{i=1}^{M_b} |c_{b,i}|$$
5. **Maximum Absolute Amplitude ($|c_{b}|_{\max}$):**
   $$|c_b|_{\max} = \max_{i} |c_{b,i}|$$

![DWT Feature Comparison](dwt_features.png)

#### 5.2.4 Observations from DWT Features
- **Time Localization:** In [dwt_decomposition.png](dwt_decomposition.png), the oscillatory transient occurring between $15\text{ ms}$ and $20\text{ ms}$ produces zero response in $cD_2$ and $cD_1$ during $0 - 15\text{ ms}$, followed by sharp, localized oscillations right where the disturbance occurs.
- **Energy Discrimination:** As depicted in [dwt_features.png](dwt_features.png), Swell exhibits the highest mean $cA_3$ energy ($84.2$), Sag exhibits reduced energy ($29.4$), and Interruption drops to $20.3$. High-frequency events (Oscillatory Transient and Notch) produce significant energy in the detail bands ($cD_2, cD_1$) where pure sinusoidal energy is negligible ($< 0.0001$).

---

## 6. Consolidated Feature Dataset: `extracted_features.csv`

The batch feature extraction pipeline processed all $17,000$ signals in **$2.07\text{ seconds}$** ($121.7\ \mu\text{s}$ per signal). The resulting dataset was validated and saved as [`extracted_features.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/extracted_features.csv):

- **Dataset Dimensions:** **$17,000$ rows $\times$ $29$ columns**
- **File Size on Disk:** **$8.71\text{ MB}$**
- **Integrity:** Exactly 0 missing values, 0 infinite values.

### Feature Dictionary (29 Columns)

```text
[Labels - 2 Columns]
1.  class_label         : Name of the disturbance class (string)
2.  class_index         : Numeric class identifier (integer 0 to 16)

[FFT Features - 7 Columns]
3.  fft_fund_mag        : Fundamental component magnitude at 50 Hz
4.  fft_thd             : Total Harmonic Distortion
5.  fft_spec_energy     : Total spectral energy across all frequency bins
6.  fft_spec_centroid   : Spectral center-of-mass frequency (Hz)
7.  fft_spec_entropy    : Spectral entropy of normalized power distribution
8.  fft_dominant_freq   : Frequency corresponding to the highest spectral peak (Hz)
9.  fft_dominant_mag    : Magnitude of the highest spectral peak

[DWT Features - 20 Columns across 4 Sub-Bands: cA3, cD3, cD2, cD1]
10. dwt_cA3_energy      : Energy of Approximation Band 3 (0 - 312.5 Hz)
11. dwt_cA3_std         : Standard deviation of Approximation Band 3
12. dwt_cA3_rel_energy  : Relative energy ratio of Approximation Band 3
13. dwt_cA3_mav         : Mean absolute value of Approximation Band 3
14. dwt_cA3_max         : Maximum absolute coefficient of Approximation Band 3
15. dwt_cD3_energy      : Energy of Detail Band 3 (312.5 - 625.0 Hz)
16. dwt_cD3_std         : Standard deviation of Detail Band 3
17. dwt_cD3_rel_energy  : Relative energy ratio of Detail Band 3
18. dwt_cD3_mav         : Mean absolute value of Detail Band 3
19. dwt_cD3_max         : Maximum absolute coefficient of Detail Band 3
20. dwt_cD2_energy      : Energy of Detail Band 2 (625.0 - 1250.0 Hz)
21. dwt_cD2_std         : Standard deviation of Detail Band 2
22. dwt_cD2_rel_energy  : Relative energy ratio of Detail Band 2
23. dwt_cD2_mav         : Mean absolute value of Detail Band 2
24. dwt_cD2_max         : Maximum absolute coefficient of Detail Band 2
25. dwt_cD1_energy      : Energy of Detail Band 1 (1250.0 - 2500.0 Hz)
26. dwt_cD1_std         : Standard deviation of Detail Band 1
27. dwt_cD1_rel_energy  : Relative energy ratio of Detail Band 1
28. dwt_cD1_mav         : Mean absolute value of Detail Band 1
29. dwt_cD1_max         : Maximum absolute coefficient of Detail Band 1
```

---

## 7. Software Architecture & Verification

The codebase was developed under a modular, object-oriented design in the project directory:

```text
ML Code/
├── dataset/
│   ├── SEED_PQD_v1.zip              # Raw Zenodo archive
│   ├── Details.txt                  # Physical specification
│   └── extracted/XPQRS/             # Extracted CSV and MAT files
├── data_loader.py                   # Data ingestion, dimension checks, sanity audit
├── signal_processing.py             # FFT and DWT mathematical algorithms
├── visualization.py                 # Matplotlib waveform and spectrum generators
├── main_milestone.py                # End-to-end execution pipeline
├── requirements.txt                 # Environment dependencies
├── dataset_inspection_report.md     # Inspection summary document
├── README.md                        # Execution documentation
└── extracted_features.csv           # Final extracted feature matrix (17,000 x 29)
```

### Verification & Performance
The entire pipeline was executed via [`main_milestone.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/main_milestone.py):
- **Total Pipeline Execution Time:** **$5.69\text{ seconds}$**
- **FFT & DWT Feature Extraction Time:** **$2.07\text{ seconds}$** across all $17,000$ signals
- **Test Status:** Passed with zero runtime errors, zero missing values, and validated plots.

---

## 8. Milestone Conclusion & Transition to Phase 2

All requirements for Milestone 1 (Tasks 1 through 4) have been completed:
- [x] Real dataset acquired, unpacked, and verified (Task 1).
- [x] Class labels, signal characteristics, and dimensions verified (Task 1).
- [x] Waveforms and class distributions analyzed and plotted (Task 2).
- [x] Modular data loader and preprocessor implemented with strict sanity checks (Task 3).
- [x] FFT and DWT signal transforms and feature extractors implemented and verified (Task 4).
- [x] Consolidated feature dataset exported to CSV (Task 4).

### Immediate Next Steps (Phase 2):
1. **Task 4 (Extended): Additional Signal Processing:**
   - Implement **Time-Domain Statistical Features** (RMS, Crest Factor, Shape Factor, Skewness, Kurtosis).
   - Implement **Short-Time Fourier Transform (STFT)** spectrogram features.
2. **Task 5: Feature Dataset Preparation:**
   - Partition feature sets ($\mathbf{X}_{\text{Time}}$, $\mathbf{X}_{\text{FFT}}$, $\mathbf{X}_{\text{DWT}}$, $\mathbf{X}_{\text{STFT}}$, and $\mathbf{X}_{\text{Combined}}$).
   - Apply `StandardScaler` normalization and stratified 80/20 train/test splitting to prevent data leakage.
3. **Task 6 & 7: Machine Learning Implementation:**
   - Train and benchmark Random Forest and SVM classifiers across each domain.
