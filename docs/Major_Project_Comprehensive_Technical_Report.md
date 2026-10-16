# Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification

**Department of Electrical Engineering**  
**National Institute of Technology (NIT) Jalandhar**  
*B.Tech Major Project Final Comprehensive Technical Report*  
**Academic Year:** 2026 | **Date:** October 2026  
**Authors:** Major Project Team, Department of Electrical Engineering  

---

## Abstract

Power quality disturbances (PQDs)—such as voltage sags, swells, interruptions, harmonics, flickers, notches, and oscillatory transients—pose critical operational and economic challenges to modern power systems, microgrids, and sensitive industrial automation. The proliferation of power-electronic converter interfaces, renewable energy generators (photovoltaic and wind), and non-linear loads has increased both the frequency and complexity of multi-component disturbances. 

This research presents a rigorous, empirical comparative evaluation of six prominent signal processing feature extraction methodologies applied to the benchmark **SEED-PQD v1** dataset (17 classes, 17,000 signals, sampled at $5\text{ kHz}$ over 1 complete $50\text{ Hz}$ cycle). The six investigated techniques span multiple mathematical domains:
1. **Discrete Fourier Transform via FFT** (Frequency domain; 7 features)
2. **Discrete Wavelet Transform via DWT** (Time-scale dyadic multi-resolution analysis; 20 features)
3. **Statistical Time-Domain Analysis** (Time-domain probability moments; 7 features)
4. **Short-Time Fourier Transform via STFT** (Time-frequency energy distribution; 8 features)
5. **Wavelet Packet Transform via WPT** (Uniform full-tree sub-band analysis; 16 features)
6. **Hilbert-Huang Transform via HHT** (Non-linear, non-stationary empirical mode decomposition; 8 features)

A combined 66-dimensional feature dataset was extracted and audited against strict numerical and data-leakage standards. A multi-class **XGBoost** classifier was systematically trained and evaluated across 12 distinct feature configurations (A through L) under 5-fold cross-validation. Furthermore, all configurations were subjected to noise-stress testing under Additive White Gaussian Noise (AWGN) at $40\text{ dB}$, $30\text{ dB}$, and $20\text{ dB}$ Signal-to-Noise Ratios (SNR) pursuant to **IEEE Std 1159-2019**.

Key experimental findings:
- **Under Clean Conditions**: Multi-domain fusion (Configuration L: all 66 features) achieved the global maximum classification performance with **91.09% Test Accuracy** and **91.10% Macro F1-score**, achieving 100% precision and recall on pure sinusoidal and steady-state harmonic waveforms. Single-method performance was led by DWT (82.03%), followed by STFT (71.44%) and Statistical Features (71.32%).
- **Under Severe Noise Conditions ($20\text{ dB}$ SNR)**: STFT features exhibited superior noise resilience (**38.18% accuracy**, only 33.26% total degradation from clean), whereas HHT suffered severe mode-mixing degradation ($6.41\%$).
- The results establish concrete engineering design trade-offs between classification accuracy, noise resilience, and computational complexity for edge smart grid sensors.

---

## 1. Introduction and Problem Definition

### 1.1 Engineering Background & Motivation
In electrical power distribution networks, power quality (PQ) defines the degree to which voltage, frequency, and waveform conform to established sinusoidal standards. Non-linear loads, arc furnaces, variable-frequency motor drives, switching power supplies, and grid-tied inverters continuously generate transient, oscillatory, and harmonic distortions.

Under **IEEE Std 1159-2019** (*IEEE Recommended Practice for Monitoring Electric Power Quality*), disturbances are categorized based on their spectral content, duration, and magnitude:
- **Instantaneous / Momentary Variations**: Voltage Sags, Swells, and Interruptions.
- **Transients**: Impulsive (unidirectional) and Oscillatory (bidirectional high-frequency resonance).
- **Steady-State Waveform Distortions**: Harmonics, Interharmonics, Notching, and Noise.
- **Voltage Fluctuations**: Low-frequency envelope modulation (Flicker).

Failure to classify and mitigate PQDs promptly leads to transformer overheating, insulation failure, false tripping of protection relays, programmable logic controller (PLC) crashes, and substantial economic losses in semiconductor fabrication and automated assembly lines.

### 1.2 Objective of the Comparative Study
While classical monitoring approaches rely on standalone Fourier transforms, transient and compound disturbances are intrinsically non-stationary and localized in time. A single transform domain cannot optimally capture all disturbance classes. This project systematically compares six signal processing representations to answer three core questions:
1. *Which individual transform domain provides the highest baseline classification accuracy across 17 PQD classes?*
2. *Does fusing time, frequency, time-scale, and instantaneous phase features yield synergistic gains over single-domain features?*
3. *How robust is each signal processing technique when exposed to measurement noise (AWGN) at realistic grid SNR levels ($40\text{ dB}$, $30\text{ dB}$, and $20\text{ dB}$)?*

---

## 2. Dataset Architecture & Physical Signal Inspection

### 2.1 Benchmark Dataset Source (SEED-PQD v1)
The investigation utilizes the verified **SEED-PQD v1** (Synthetic Electrical Energy Disturbance - Power Quality Disturbance) benchmark dataset hosted on Zenodo:
- **Zenodo DOI**: [10.5281/zenodo.11843312](https://zenodo.org/records/11843312)
- **Total Signals**: $17,000$ discrete recordings
- **Number of Classes**: 17 distinct disturbance categories ($1,000$ signals per class, 100% balanced)
- **Signal Representation**: 100 discrete floating-point samples per recorded cycle

### 2.2 Physical Grid Parameters

| Parameter | Mathematical Symbol | Value | Physical Significance |
| :--- | :---: | :---: | :--- |
| **Fundamental Frequency** | $f_0$ | **$50.0\text{ Hz}$** | Standard European/Indian nominal grid frequency |
| **Sampling Frequency** | $f_s$ | **$5,000\text{ Hz}$ ($5\text{ kHz}$)** | Satisfies Nyquist criterion ($f_s \gg 2f_0$) |
| **Sampling Period** | $\Delta t$ | **$0.2\text{ ms}$ ($200\ \mu\text{s}$)** | Discrete sample interval: $\Delta t = 1/f_s$ |
| **Samples per Signal** | $N$ | **$100\text{ samples}$** | Window length: $N = T \cdot f_s = 0.020 \times 5000$ |
| **Signal Duration** | $T$ | **$20.0\text{ ms}$** | Exactly 1 complete electrical cycle ($1 / 50\text{ Hz}$) |
| **Frequency Resolution** | $\Delta f$ | **$50.0\text{ Hz}$** | $\Delta f = f_s / N = 5000 / 100 = 50\text{ Hz}$ |
| **Nyquist Limit** | $f_{\text{Nyq}}$ | **$2,500.0\text{ Hz}$** | Maximum observable frequency ($f_s / 2$) |
| **Nominal Amplitude** | $V_{\text{peak}}$ | **$1.0\text{ p.u.}$** | Pure sine: $V_{\text{RMS}} = 1/\sqrt{2} \approx 0.7071\text{ p.u.}$ |
| **Dynamic Range** | $V_{\text{min}}, V_{\text{max}}$ | **$-2.4601\text{ to }+2.4857\text{ p.u.}$** | Observed during high-energy swells and transients |

### 2.3 17 Disturbance Classes
1. Pure Sinusoidal (Baseline reference)
2. Voltage Sag
3. Voltage Swell
4. Total Interruption
5. Impulsive Transient
6. Oscillatory Transient
7. Harmonics
8. Harmonics with Sag
9. Harmonics with Swell
10. Voltage Flicker
11. Flicker with Sag
12. Flicker with Swell
13. Sag with Oscillatory Transient
14. Swell with Oscillatory Transient
15. Sag with Harmonics
16. Swell with Harmonics
17. Voltage Notch

---

## 3. Mathematical Foundations of the Six Feature Extraction Methods

### 3.1 Method 0A: Fast Fourier Transform (FFT / DFT)
The Discrete Fourier Transform (DFT) maps a discrete time-domain sequence $x[n]$ of length $N$ into complex frequency bins $X[k]$:
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$

The one-sided normalized amplitude spectrum is computed for $k = 0, \dots, N/2$:
$$|X_{\text{norm}}[k]| = \begin{cases} \frac{1}{N} |X[0]|, & k = 0 \\ \frac{2}{N} |X[k]|, & 1 \le k < N/2 \\ \frac{1}{N} |X[N/2]|, & k = N/2 \end{cases}$$

**7 Extracted FFT Features**:
1. `fft_fund_mag`: Fundamental amplitude $|X_{\text{norm}}[1]|$ at $50\text{ Hz}$.
2. `fft_thd`: Total Harmonic Distortion:
   $$\text{THD} = \frac{\sqrt{\sum_{h=2}^{M} |X_{\text{norm}}[h]|^2}}{|X_{\text{norm}}[1]| + \epsilon}$$
3. `fft_spec_energy`: Total spectral power: $\sum_{k} |X_{\text{norm}}[k]|^2$.
4. `fft_spec_centroid`: Spectral center-of-mass frequency: $\frac{\sum_k f_k |X_{\text{norm}}[k]|}{\sum_k |X_{\text{norm}}[k]|}$.
5. `fft_spec_entropy`: Shannon spectral entropy: $-\sum_k p_k \ln(p_k)$, where $p_k = |X[k]|^2 / \sum |X|^2$.
6. `fft_dominant_freq`: Frequency of the peak spectral bin ($f_{\text{dom}} = \arg\max_f |X[f]|$).
7. `fft_dominant_mag`: Magnitude of the dominant frequency peak.

### 3.2 Method 0B: Discrete Wavelet Transform (DWT)
The DWT decomposes $x[n]$ through successive low-pass $g[n]$ and high-pass $h[n]$ filter pairs followed by dyadic downsampling by 2:
$$y_{\text{low}}[k] = \sum_{n} x[n] g[2k - n], \quad y_{\text{high}}[k] = \sum_{n} x[n] h[2k - n]$$

Using the Daubechies 4 (`db4`) mother wavelet with 3 levels of decomposition:
- **Approximation $cA_3$**: Band $0.0\text{ to }312.5\text{ Hz}$ (fundamental $50\text{ Hz}$ & 3rd/5th harmonics)
- **Detail $cD_3$**: Band $312.5\text{ to }625.0\text{ Hz}$ (mid-order harmonics)
- **Detail $cD_2$**: Band $625.0\text{ to }1250.0\text{ Hz}$ (oscillatory transients)
- **Detail $cD_1$**: Band $1250.0\text{ to }2500.0\text{ Hz}$ (high-frequency notches and steep edges)

**20 Extracted DWT Features**:
For each sub-band ($b \in \{cA_3, cD_3, cD_2, cD_1\}$):
- Sub-band Energy: $E_b = \sum_k c_{b}[k]^2$
- Standard Deviation: $\sigma_b = \text{std}(c_b)$
- Relative Energy: $E_b / \sum E_{\text{total}}$
- Mean Absolute Value: $\text{MAV}_b = \frac{1}{K} \sum_k |c_b[k]|$
- Maximum Absolute Value: $\max_k |c_b[k]|$

### 3.3 Method 1: Statistical Time-Domain Analysis
Statistical features capture probability density deviations directly in the time domain without transform loss:
1. `stat_rms`: $\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}$
2. `stat_std`: Standard deviation $\sigma = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (x[n] - \mu)^2}$
3. `stat_var`: Signal variance $\sigma^2$
4. `stat_skewness`: Fisher-Pearson 3rd standardized moment:
   $$\gamma_1 = \frac{\frac{1}{N} \sum_{n=0}^{N-1} (x[n] - \mu)^3}{\sigma^3 + \epsilon}$$
5. `stat_kurtosis`: Fisher excess kurtosis:
   $$\gamma_2 = \frac{\frac{1}{N} \sum_{n=0}^{N-1} (x[n] - \mu)^4}{\sigma^4 + \epsilon} - 3.0$$
6. `stat_crest_factor`: Peak-to-RMS ratio:
   $$\text{CF} = \frac{\max_{n} |x[n]|}{\text{RMS} + \epsilon}$$
7. `stat_peak_to_peak`: Total excursion: $V_{pp} = \max_n(x[n]) - \min_n(x[n])$.

### 3.4 Method 2: Short-Time Fourier Transform (STFT)
STFT captures localized spectral changes across moving time windows:
$$\text{STFT}\{x[n]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$

Parameters chosen:
- Window: Hann window ($w[n] = 0.5 - 0.5 \cos(2\pi n / M)$)
- Window length: $N_{\text{perseg}} = 32\text{ samples}$ ($6.4\text{ ms}$, approx 1/3 of the $50\text{ Hz}$ cycle)
- Overlap: $N_{\text{overlap}} = 16\text{ samples}$ ($50\%$ overlap, hop size $3.2\text{ ms}$)
- FFT points: $N_{\text{fft}} = 64$ points (yielding 33 frequency bins from $0\text{ to }2500\text{ Hz}$)
- Spectrogram power density: $S(f, t) = |Z_{xx}(f, t)|^2$

**8 Extracted STFT Features**:
- `stft_mean_energy`: Mean value of $S(f, t)$
- `stft_max_energy`: Peak value $\max_{f, t} S(f, t)$
- `stft_std_energy`: Standard deviation of spectrogram energy
- `stft_energy_low_band`: Energy in $0 \le f \le 300\text{ Hz}$
- `stft_energy_mid_band`: Energy in $300 < f \le 1200\text{ Hz}$
- `stft_energy_high_band`: Energy in $1200 < f \le 2500\text{ Hz}$
- `stft_rel_energy_low`: Low-band energy divided by total spectrogram energy
- `stft_temporal_variability`: Mean over frequencies of standard deviations across time slices ($\frac{1}{K}\sum_f \text{std}_t(S(f, t))$)

### 3.5 Method 3: Wavelet Packet Transform (WPT)
While standard DWT iteratively splits only the low-frequency approximation sub-band, WPT decomposes **both** approximation and detail sub-bands at every level:
At level 3, WPT builds a complete balanced binary tree with $2^3 = 8$ terminal nodes.
Using natural frequency ordering (Gray-code sorted), the 8 terminal sub-bands partition the $0\text{ to }2500\text{ Hz}$ Nyquist bandwidth into 8 uniform $312.5\text{ Hz}$ frequency slots:
- **Band 0**: $[0.0, 312.5\text{ Hz}]$ (Fundamental & low harmonics)
- **Band 1**: $[312.5, 625.0\text{ Hz}]$ (7th, 9th, 11th harmonics)
- **Band 2**: $[625.0, 937.5\text{ Hz}]$ (Mid-frequency resonance)
- **Band 3**: $[937.5, 1250.0\text{ Hz}]$ (Higher harmonics)
- **Band 4**: $[1250.0, 1562.5\text{ Hz}]$ (Low oscillatory transients)
- **Band 5**: $[1562.5, 1875.0\text{ Hz}]$ (Mid oscillatory transients)
- **Band 6**: $[1875.0, 2187.5\text{ Hz}]$ (High oscillatory transients)
- **Band 7**: $[2187.5, 2500.0\text{ Hz}]$ (Fast notches and steep edge transients)

**16 Extracted WPT Features**:
- 8 Terminal sub-band energies: `wpt_band0_energy` to `wpt_band7_energy` ($E_i = \sum_k w_{i, k}^2$)
- 8 Terminal sub-band relative energies: `wpt_band0_rel_energy` to `wpt_band7_rel_energy` ($E_i / \sum E_{\text{total}}$)

### 3.6 Method 4: Hilbert-Huang Transform (HHT)
HHT is designed specifically for non-linear and non-stationary processes through a two-step framework:
1. **Empirical Mode Decomposition (EMD)**: Sifts the signal $x[n]$ into $M$ Intrinsic Mode Functions (IMFs) $c_m[n]$ and a monotonic residue $r[n]$:
   $$x[n] = \sum_{m=1}^{M} c_m[n] + r[n]$$
   - IMF 1 captures fast oscillatory transients and steep edges.
   - IMF 2 captures dominant fundamental $50\text{ Hz}$ and intermediate variations.
2. **Hilbert Spectral Analysis**:
   Constructs the analytic signal $z_m[n] = c_m[n] + j \mathcal{H}[c_m[n]] = a_m[n] e^{j \theta_m[n]}$, where:
   - Instantaneous amplitude: $a_m[n] = |z_m[n]|$
   - Instantaneous frequency: $f_m[n] = \frac{f_s}{2\pi} (\theta_m[n] - \theta_m[n-1])$ (clamped to $[0, f_s/2]$ to suppress boundary differentiation spikes).

**8 Extracted HHT Features**:
For IMFs 1 & 2:
- `hht_imf{k}_mean_amp`: Mean instantaneous amplitude
- `hht_imf{k}_std_amp`: Standard deviation of instantaneous amplitude
- `hht_imf{k}_mean_freq`: Mean instantaneous frequency (Hz)
- `hht_imf{k}_energy`: Mode discrete energy ($\sum c_k[n]^2$)

---

## 4. Multi-Domain Feature Matrix Architecture & Quality Audit

### 4.1 Feature Matrix Dimensionality
The combined feature matrix stored in `data/processed/extended_features.csv` contains:
- **Total Rows**: $17,000$ signals
- **Total Columns**: $68$
  - **2 Metadata Columns**: `class_label` (string) and `class_index` (integer $0\text{--}16$)
  - **66 Numerical Feature Columns**:
    - FFT: 7
    - DWT: 20
    - Statistical: 7
    - STFT: 8
    - WPT: 16
    - HHT: 8

### 4.2 Automated Quality & Validation Audit Results
- **Missing / NaN Values**: **0** (100% complete)
- **Infinite Values**: **0** (All divisions protected by $\epsilon = 10^{-12}$)
- **Duplicate Columns**: **0** (All 68 feature names are unique)
- **Class Balance**: Exactly $1,000$ signals per class across all 17 classes
- **Degenerate Features**: Zero constant features ($\text{std} > 10^{-9}$ for all columns)
- **Data Leakage Safeguard**: Feature normalization (`StandardScaler`) was strictly fit on training data partitions and applied to test sets.

---

## 5. Machine Learning Experimental Methodology

### 5.1 Classifier Choice & Hyperparameters
Extreme Gradient Boosting (**XGBoost**) was selected as the benchmark classifier due to its superior performance on tabular multi-domain feature vectors, fast histogram-based tree splitting (`tree_method='hist'`), and invariance to monotonic feature scaling.

```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    objective="multi:softprob",
    num_class=17,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
```

### 5.2 Train / Test Splitting Protocol
- **Total Signals**: $17,000$
- **Training Partition (80%)**: $13,600$ signals ($800$ per class, stratified)
- **Test Partition (20%)**: $3,400$ signals ($200$ per class, stratified)
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training partition
- **Random Seed**: Fixed to `42` across all splits and initializations for 100% reproducibility.

---

## 6. Empirical Classification Results (Configurations A through L)

### 6.1 Performance Benchmark Table

| Config | Feature Configuration | # Feats | Train 5-Fold CV Acc | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Fit Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | FFT only | 7 | $69.59\% \pm 0.94\%$ | 68.79% | 67.75% | 68.79% | 68.09% | 3.73 s |
| **B** | DWT only | 20 | $82.17\% \pm 0.69\%$ | 82.03% | 81.99% | 82.03% | 81.81% | 6.05 s |
| **C** | Statistical only | 7 | $70.87\% \pm 0.53\%$ | 71.32% | 71.73% | 71.32% | 70.95% | 2.92 s |
| **D** | STFT only | 8 | $72.38\% \pm 0.60\%$ | 71.44% | 71.02% | 71.44% | 70.56% | 2.88 s |
| **E** | WPT only | 16 | $65.48\% \pm 0.28\%$ | 66.00% | 64.59% | 66.00% | 64.21% | 4.52 s |
| **F** | HHT only | 8 | $67.74\% \pm 0.84\%$ | 69.38% | 68.96% | 69.38% | 68.76% | 3.29 s |
| **G** | FFT + DWT | 27 | $87.06\% \pm 0.78\%$ | 86.71% | 86.80% | 86.71% | 86.60% | 6.25 s |
| **H** | FFT + DWT + Stats | 34 | $88.76\% \pm 0.37\%$ | 89.03% | 89.21% | 89.03% | 88.95% | 8.42 s |
| **I** | FFT + DWT + STFT | 35 | $87.94\% \pm 0.74\%$ | 87.62% | 87.72% | 87.62% | 87.52% | 8.58 s |
| **J** | FFT + DWT + WPT | 43 | $87.07\% \pm 0.44\%$ | 86.85% | 86.98% | 86.85% | 86.76% | 9.64 s |
| **K** | FFT + DWT + HHT | 35 | $88.38\% \pm 0.80\%$ | 89.68% | 89.83% | 89.68% | 89.63% | 8.19 s |
| **L** | **All Combined (Global Best)** | **66** | **$89.85\% \pm 0.55\%$** | **91.09%** | **91.37%** | **91.09%** | **91.10%** | **15.91 s** |

### 6.2 Detailed Per-Class Performance (Configuration L)

| Class Index | Disturbance Class Name | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `Pure_Sinusoidal` | **1.000** | **1.000** | **1.000** | 200 |
| 1 | `Sag` | 0.911 | 0.970 | 0.939 | 200 |
| 2 | `Swell` | 0.941 | 0.960 | 0.950 | 200 |
| 3 | `Interruption` | 0.980 | 0.990 | 0.985 | 200 |
| 4 | `Transient` | 0.900 | 0.985 | 0.940 | 200 |
| 5 | `Oscillatory_Transient` | 0.926 | 0.940 | 0.933 | 200 |
| 6 | `Harmonics` | **1.000** | **1.000** | **1.000** | 200 |
| 7 | `Harmonics_with_Sag` | 0.906 | 0.770 | 0.832 | 200 |
| 8 | `Harmonics_with_Swell` | 0.939 | 0.930 | 0.935 | 200 |
| 9 | `Flicker` | 0.990 | **1.000** | 0.995 | 200 |
| 10 | `Flicker_with_Sag` | 0.868 | 0.790 | 0.827 | 200 |
| 11 | `Flicker_with_Swell` | 0.782 | 0.845 | 0.812 | 200 |
| 12 | `Sag_with_Oscillatory_Transient` | 0.973 | 0.895 | 0.932 | 200 |
| 13 | `Swell_with_Oscillatory_Transient` | 0.962 | 0.885 | 0.922 | 200 |
| 14 | `Sag_with_Harmonics` | 0.718 | 0.855 | 0.781 | 200 |
| 15 | `Swell_with_Harmonics` | 0.741 | 0.715 | 0.728 | 200 |
| 16 | `Notch` | 0.995 | 0.955 | 0.974 | 200 |
| **—** | **Macro Average** | **0.914** | **0.911** | **0.911** | **3400** |

---

## 7. Noise Robustness (AWGN) Stress Testing Analysis

### 7.1 Mathematical Formulation of AWGN
To simulate actual transmission line and transducer thermal noise, zero-mean Additive White Gaussian Noise was injected into the test partition signals according to:
$$P_{\text{signal}} = \frac{1}{N} \sum_{n=0}^{N-1} x[n]^2, \quad P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}} / 10}}, \quad x_{\text{noisy}}[n] = x[n] + \mathcal{N}(0, P_{\text{noise}})$$

Tested levels:
- **Clean ($\text{SNR} = \infty$)**: Baseline lab environment
- **$40\text{ dB}$ SNR**: High-precision substation measurement
- **$30\text{ dB}$ SNR**: Typical distribution network monitoring
- **$20\text{ dB}$ SNR**: Heavy industrial / arc-furnace environment

### 7.2 Noise Degradation Results Table

| Configuration | Clean ($\infty$) | 40 dB SNR | 30 dB SNR | 20 dB SNR | Total Degradation (Clean $\to$ 20 dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. FFT only** | 68.79% | 56.21% | 38.94% | 23.18% | -45.62% |
| **B. DWT only** | 82.03% | 50.47% | 37.76% | 23.79% | -58.24% |
| **C. Statistical only** | 71.32% | 51.56% | 40.74% | 25.47% | -45.85% |
| **D. STFT only** | **71.44%** | **64.74%** | **59.82%** | **38.18%** | **-33.26% (Most Resilient)** |
| **E. WPT only** | 66.00% | 46.74% | 34.06% | 23.03% | -42.97% |
| **F. HHT only** | 69.38% | 15.65% | 9.41% | 6.41% | -62.97% |
| **G. FFT + DWT** | 86.71% | 59.18% | 43.41% | 28.41% | -58.29% |
| **H. FFT + DWT + Stats** | 89.03% | 59.44% | 42.74% | 23.50% | -65.53% |
| **I. FFT + DWT + STFT** | 87.62% | 62.03% | 49.26% | 31.00% | -56.62% |
| **J. FFT + DWT + WPT** | 86.85% | 58.24% | 47.21% | 29.12% | -57.74% |
| **K. FFT + DWT + HHT** | 89.68% | 42.32% | 31.59% | 22.21% | -67.47% |
| **L. All Combined** | **91.09%** | 45.79% | 34.62% | 21.62% | -69.47% |

### 7.3 Critical Physical Findings on Noise Resilience
1. **Superiority of STFT under Noise**: STFT features demonstrated the highest resilience across all tested noise tiers ($64.74\%$ at $40\text{ dB}$, $59.82\%$ at $30\text{ dB}$, and $38.18\%$ at $20\text{ dB}$). The windowed integration ($N_{\text{perseg}} = 32$) acts as a natural low-pass averaging filter that suppresses uncorrelated white noise.
2. **Fragility of Empirical Mode Decomposition (HHT)**: HHT experienced the most severe degradation under noise ($69.38\% \to 6.41\%$). In EMD, white noise introduces high-frequency local extrema, causing severe **mode mixing** where true disturbance oscillatory modes are fractured into false high-frequency IMFs.
3. **Engineering Deployment Recommendation**: For clean, shielded substations, **Configuration L (All Combined)** is optimal for maximum diagnostic precision ($91.09\%$). For unshielded, high-noise industrial distribution lines, **Configuration I (FFT + DWT + STFT)** or **Configuration D (STFT)** provides superior operational reliability.

---

## 8. Summary of Project Deliverables

| Deliverable Type | Artifact Path | Description |
| :--- | :--- | :--- |
| **Master Runner** | [`run_extended_pipeline.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/run_extended_pipeline.py) | Full end-to-end extraction, validation, and training runner |
| **Noise Benchmark** | [`run_noise_robustness.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/run_noise_robustness.py) | IEEE Std 1159 AWGN noise stress testing runner |
| **Feature Matrix** | [`data/processed/extended_features.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/data/processed/extended_features.csv) | Full 17,000 x 68 feature matrix (20.96 MB) |
| **Feature Dictionary** | [`docs/feature_dictionary.md`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/docs/feature_dictionary.md) | 66-feature mathematical definition and units |
| **Benchmark CSV** | [`outputs/comparative_model_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/comparative_model_results.csv) | Clean comparative metrics table across configs A-L |
| **Noise CSV** | [`outputs/noise_robustness_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/noise_robustness_results.csv) | Multi-SNR degradation metrics table |
| **Trained Model** | [`outputs/models/xgboost_best_model_L.json`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/models/xgboost_best_model_L.json) | Production serialized XGBoost model artifact |
| **Visualizations** | [`outputs/figures/`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/figures/) | 14 publication-grade figures (spectrograms, curves, confusion heatmaps) |

---

## 9. Conclusion & Future Directions

This major project has successfully established a multi-domain signal processing and machine-learning classification architecture for 17 power quality disturbance classes. By systematically auditing, formulating, and empirically evaluating 6 distinct signal processing families, the project demonstrates that multi-domain fusion achieves superior accuracy (**91.09%**) under clean conditions, while time-frequency windowed methods (STFT) provide unmatched stability under noisy field conditions.

### Future Work
1. **Hardware Implementation**: Porting the feature extraction pipeline to an embedded DSP/FPGA (e.g., Texas Instruments TMS320C6748 or Xilinx Zynq) for sub-cycle real-time smart meter inference.
2. **Ensemble Wavelet / EEMD**: Implementing Ensemble Empirical Mode Decomposition (EEMD) to mitigate mode-mixing in high-noise environments.
3. **Deep Learning Comparison**: Benchmarking the multi-domain feature vectors against 1D Convolutional Neural Networks (1D-CNN) and Temporal Convolutional Networks (TCN) processing raw voltage samples directly.
