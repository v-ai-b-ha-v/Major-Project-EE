# Comprehensive Audit Report: Noise Robustness and Additive White Gaussian Noise (AWGN) Evaluation

**Project Title:** Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification  
**Institution:** Department of Electrical Engineering, National Institute of Technology (NIT) Jalandhar  
**Document ID:** `docs/Noise_Robustness_Audit.md`  
**Audit Date:** October 2026  
**Auditor:** Technical Verification Agent (Antigravity IDE)  
**Status:** Complete & Verified  

---

## Executive Summary

This document presents an independent, ground-truth audit of the noise robustness evaluation implemented in [`run_noise_robustness.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/run_noise_robustness.py) and [`src/noise_robustness.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/src/noise_robustness.py), along with its output artifacts:
- [`outputs/noise_robustness_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/noise_robustness_results.csv)
- [`outputs/comparative_model_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/comparative_model_results.csv)
- [`outputs/figures/snr_accuracy_curves.png`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/figures/snr_accuracy_curves.png)
- [`outputs/figures/noise_waveform_comparison.png`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/figures/noise_waveform_comparison.png)

All claims, numerical values, equations, experimental designs, and theoretical assertions were verified directly against raw code and saved data. No code or results were modified or overwritten during this audit.

---

## 1. Verification of Reported Results

### 1.1 Ground-Truth Numerical Comparison
The reported values in the project documentation and README were cross-referenced against the raw saved CSV file [`outputs/noise_robustness_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/noise_robustness_results.csv).

| Config ID | Feature Configuration | Clean ($\infty$) Acc | 40 dB SNR Acc | 30 dB SNR Acc | 20 dB SNR Acc | Absolute Drop ($\text{pp}$) | Relative Drop ($\%$) | Verification Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | A. FFT only | 68.79% | 56.21% | 38.94% | 23.18% | 45.62 pp | 66.31% | **VERIFIED EXACT** |
| **B** | B. DWT only | 82.03% | 50.47% | 37.76% | 23.79% | 58.24 pp | 70.99% | **VERIFIED EXACT** |
| **C** | C. Statistical only | 71.32% | 51.56% | 40.74% | 25.47% | 45.85 pp | 64.29% | **VERIFIED EXACT** |
| **D** | D. STFT only | **71.44%** | **64.74%** | **59.82%** | **38.18%** | **33.26 pp** | **46.56%** | **VERIFIED EXACT** |
| **E** | E. WPT only | 66.00% | 46.74% | 34.06% | 23.03% | 42.97 pp | 65.11% | **VERIFIED EXACT** |
| **F** | F. HHT only | 69.38% | 15.65% | 9.41% | 6.41% | 62.97 pp | 90.76% | **VERIFIED EXACT** |
| **G** | G. FFT + DWT | 86.71% | 59.18% | 43.41% | 28.41% | 58.29 pp | 67.23% | **VERIFIED EXACT** |
| **H** | H. FFT + DWT + Stats | 89.03% | 59.44% | 42.74% | 23.50% | 65.53 pp | 73.60% | **VERIFIED EXACT** |
| **I** | I. FFT + DWT + STFT | 87.62% | 62.03% | 49.26% | 31.00% | 56.62 pp | 64.62% | **VERIFIED EXACT** |
| **J** | FFT + DWT + WPT | 86.85% | 58.24% | 47.21% | 29.12% | 57.74 pp | 66.47% | **VERIFIED EXACT** |
| **K** | FFT + DWT + HHT | 89.68% | 42.32% | 31.59% | 22.21% | 67.47 pp | 75.24% | **VERIFIED EXACT** |
| **L** | All Features Combined | **91.09%** | 45.79% | 34.62% | 21.62% | 69.47 pp | 76.27% | **VERIFIED EXACT** |

### 1.2 Verification of Clean Baseline Reproducibility
The Clean baseline row in [`outputs/noise_robustness_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/noise_robustness_results.csv) was compared to the primary benchmark table [`outputs/comparative_model_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/comparative_model_results.csv).
- Maximum absolute difference in Test Accuracy: **0.00000000**
- Maximum absolute difference in Macro-F1: **0.00000000**
- **Conclusion**: The clean baseline reproduces the earlier benchmark with 100% precision.

### 1.3 Audit of Degradation Metrics
- The reported degradation values in the summary table (e.g., `-33.26%` for STFT and `-69.47%` for Config L) are **absolute percentage point differences** ($\text{Acc}_{\text{clean}} - \text{Acc}_{20\text{dB}}$).
- When measured as **relative percentage degradation** ($\frac{\text{Acc}_{\text{clean}} - \text{Acc}_{20\text{dB}}}{\text{Acc}_{\text{clean}}} \times 100\%$), STFT degraded by **$46.56\%$**, whereas Config L degraded by **$76.27\%$**, and HHT degraded by **$90.76\%$**.
- **Correction Required**: The documentation should explicitly state that the reported degradation values are *percentage points* ($\text{pp}$), or provide both definitions to avoid ambiguity.

---

## 2. Audit of the AWGN Mathematical Implementation

### 2.1 Governing Equations
In [`src/noise_robustness.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/src/noise_robustness.py) (lines 21–41), the noise calculation is implemented as:

```python
def add_awgn_to_signal(signal, snr_db, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p_signal = np.mean(signal ** 2)
    if p_signal < 1e-12:
        return signal.copy()

    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=signal.shape)
    return signal + noise
```

The implemented mathematical definition is:
1. Average Discrete Signal Power:
   $$P_{\text{signal}} = \frac{1}{N} \sum_{n=0}^{N-1} x[n]^2$$
2. Target Noise Power for specified $\text{SNR}_{\text{dB}}$:
   $$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\frac{\text{SNR}_{\text{dB}}}{10}}}$$
3. Gaussian Noise Generation:
   $$w[n] \sim \mathcal{N}\left(0, \sigma_{\text{noise}}^2\right), \quad \sigma_{\text{noise}} = \sqrt{P_{\text{noise}}}$$
4. Corrupted Signal:
   $$x_{\text{noisy}}[n] = x[n] + w[n]$$

### 2.2 Waveform-Independent Scaling Verification
- **Audit Question:** Is noise power computed globally across the entire dataset or independently for each waveform?
- **Code Inspection:** In `add_awgn_to_matrix()`, the loop iterates over each row $i$:
  `X_noisy[i] = add_awgn_to_signal(X[i], snr_db, rng=rng)`
  Inside `add_awgn_to_signal()`, `p_signal = np.mean(signal ** 2)` is evaluated on the individual 100-sample array `X[i]`.
- **Finding:** **Confirmed.** Noise is scaled independently for every individual signal, correctly accounting for variations in waveform amplitudes (e.g. Swells with $P \approx 2.5\text{ p.u.}^2$ receive proportionally larger noise variance than Sags with $P \approx 0.2\text{ p.u.}^2$, maintaining exact requested SNR for every waveform).

### 2.3 Numerical Verification of Realized SNR
Empirical SNR was calculated across all 3,400 test waveforms using:
$$\text{SNR}_{\text{empirical}} = 10 \log_{10}\left(\frac{\frac{1}{N}\sum x[n]^2}{\frac{1}{N}\sum (x_{\text{noisy}}[n] - x[n])^2}\right)$$

Empirical results across the 3,400 test signals:
- **Target $40\text{ dB}$ SNR**: Mean empirical $\text{SNR} = \mathbf{40.0402\text{ dB}}$ ($\text{std} = 0.6126\text{ dB}$, $\text{min} = 38.07\text{ dB}$, $\text{max} = 42.31\text{ dB}$)
- **Target $30\text{ dB}$ SNR**: Mean empirical $\text{SNR} = \mathbf{30.0402\text{ dB}}$ ($\text{std} = 0.6126\text{ dB}$, $\text{min} = 28.07\text{ dB}$, $\text{max} = 32.31\text{ dB}$)
- **Target $20\text{ dB}$ SNR**: Mean empirical $\text{SNR} = \mathbf{20.0402\text{ dB}}$ ($\text{std} = 0.6126\text{ dB}$, $\text{min} = 18.07\text{ dB}$, $\text{max} = 22.31\text{ dB}$)
- **Finding:** The realized mean SNR matches the target SNR within **$\pm 0.04\text{ dB}$** ($< 0.1\%$ error). The small sample variance ($\text{std} \approx 0.61\text{ dB}$) is mathematically expected due to finite sequence length ($N = 100\text{ samples}$).

### 2.4 Random Seeds and Number of Realizations
- **Random Number Generator**: `np.random.default_rng(random_state=42)` (NumPy PCG64 bit-generator).
- **Number of Realizations**: **1 realization per test waveform** per SNR level.
- **Finding**: While standard in initial benchmarking, running multiple noise realizations (e.g., 5 or 10 Monte Carlo runs per SNR) would provide error margins ($\pm \sigma_{\text{acc}}$). This should be noted as a future enhancement.

---

## 3. Verification of Experimental Fairness

### 3.1 Test Index & Partition Integrity
- In [`run_noise_robustness.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/run_noise_robustness.py) (lines 65–70), `train_test_split(indices, test_size=0.20, stratify=y, random_state=42)` is called with seed `42`.
- This exact partition ($13,600$ train, $3,400$ test) is identical to the one used in `main.py`.
- **Finding:** All 12 configurations evaluated identical test waveforms at each noise level.

### 3.2 Feature Extraction Integrity
- **Audit Question:** Are features re-extracted from raw noisy signals, or was noise artificially added to the feature matrix?
- **Code Inspection:** Line 232 of `src/noise_robustness.py`:
  ```python
  X_test_noisy = add_awgn_to_matrix(X_test_clean, snr_db=snr, random_state=42)
  df_noisy_test = extract_all_extended_dataset_features(X_test_noisy, ...)
  ```
- **Finding:** **Confirmed.** AWGN is injected directly into the 100 discrete time-domain samples. Full mathematical transforms (FFT, DWT, STFT, WPT, HHT) are freshly recomputed on the corrupted time-series data.

### 3.3 Data Leakage & Model Training Audit
- In `run_noise_robustness.py`, `StandardScaler()` and `XGBClassifier()` are fitted strictly on `train_idx` ($13,600$ clean samples).
- Test features are scaled using `scaler.transform(X_test_noisy_features)`.
- No noisy test samples were used for model fitting, parameter tuning, or scaler fitting.
- **Finding:** Zero data leakage confirmed.

### 3.4 Consistency of Evaluation Design
- All 12 configurations were evaluated using the identical paradigm: **Clean Training / Noisy Testing** (no noise augmentation during training).
- **Finding:** Fully consistent across all configurations.

---

## 4. Root-Cause Analysis of Unexpected Results

### 4.1 Why Does Configuration L (All Features) Degrade More than STFT Alone (Config D)?
Under clean conditions, Config L is superior (**91.09%** vs. **71.44%**). However, at $20\text{ dB}$ SNR, Config L drops to **$21.62\%$**, while STFT alone retains **$38.18\%$**.

**Root-Cause Investigation:**
1. **High-Frequency Wavelet Distortion**:
   - In clean signals, high-frequency wavelet detail coefficients ($cD_1, cD_2$ in DWT and bands 4–7 in WPT) have near-zero energy unless an impulsive transient, oscillatory transient, or notch occurs.
   - For a clean pure sine wave: `dwt_cD1_energy` $= 5.17 \times 10^{-5}$.
   - Under $20\text{ dB}$ AWGN: `dwt_cD1_energy` jumps to $0.1615$—an increase of **$3,120 \times$**.
   - During clean training, XGBoost placed split points such as `dwt_cD1_energy > 0.01` to isolate `Oscillatory_Transient` or `Notch`. Under noise, **pure sine waves and sags exhibit $cD_1 > 0.16$**, triggering false transient classifications.
2. **Empirical Proof of Tree Poisoning**:
   - Evaluating Config L on $200$ pure sine waves corrupted with $20\text{ dB}$ noise: **0 out of 200** are classified as Pure Sinusoid. **$174$ out of $200$ ($87\%$) are misclassified as `Oscillatory_Transient`**, and $23$ as `Notch`.
3. **Curse of Dimensionality under Distribution Shift**:
   - Config L contains 66 features. Having more features increases clean diagnostic power, but under noise distribution shift, each additional noisy channel provides another false trigger for tree splits. Config D uses only 8 features, significantly reducing the surface area for false positive activations.

### 4.2 Why Does HHT Collapse under Noise ($69.38\% \to 6.41\%$)?
At $20\text{ dB}$ SNR, HHT accuracy ($6.41\%$) is barely above random guessing ($1/17 \approx 5.88\%$).

**Root-Cause Investigation:**
1. **Sifting Order Inversion (Mode Mixing)**:
   - Empirical Mode Decomposition sifts signals from highest local frequency to lowest.
   - On a clean sine wave ($50\text{ Hz}$): EMD produces **1 IMF** (the $50\text{ Hz}$ fundamental at $50\text{ Hz}$, energy $\approx 50$).
   - On a $20\text{ dB}$ noisy sine wave: High-frequency Gaussian noise creates numerous local extrema. EMD sifts the noise first!
     - **IMF 1**: Extracted at $1,150.3\text{ Hz}$ (pure noise), energy $= 0.23$
     - **IMF 2**: Extracted at $698.3\text{ Hz}$ (residual noise), energy $= 0.12$
     - **IMF 3**: Extracted at $47.8\text{ Hz}$ (the actual $50\text{ Hz}$ sine wave!), energy $= 52.06$
2. **Schema Truncation Mismatch**:
   - The feature extraction schema was designed for clean signals where disturbances have at most 2 primary modes (`max_imf=2`), extracting only IMF 1 and IMF 2.
   - Under noise, the true power frequency waveform is pushed to **IMF 3**, which is **completely discarded by the schema**.
   - Consequently, `hht_imf1_*` and `hht_imf2_*` represent pure Gaussian noise parameters, yielding near-zero classification capability.

### 4.3 Does STFT "Inherently Filter" AWGN? Critical Assessment
- **Claim Under Review:** "STFT naturally filters out white Gaussian noise through moving window integration."
- **Critical Technical Audit:**
  - This claim is **partially inaccurate**. STFT is a linear, energy-preserving time-frequency transform; it does **not** filter or remove noise. Total noise energy is preserved:
    $$\sum_{f} \sum_{t} |Z_{xx}(f, t)|^2 \propto \sum_{n} |x_{\text{noisy}}[n]|^2$$
  - **Why STFT Outperformed Other Methods:**
    1. STFT features partition the spectrum into broad frequency bands: Low ($0\text{--}300\text{ Hz}$), Mid ($300\text{--}1200\text{ Hz}$), and High ($1200\text{--}2500\text{ Hz}$).
    2. Because white Gaussian noise has a flat power spectral density, its power is distributed uniformly across the entire $2,500\text{ Hz}$ Nyquist bandwidth.
    3. The low band ($0\text{--}300\text{ Hz}$) spans only $12\%$ of the total bandwidth ($300/2500 = 0.12$), capturing only $12\%$ of the total noise power, while retaining $100\%$ of the fundamental $50\text{ Hz}$ signal energy.
    4. Thus, `stft_rel_energy_low` shifted by only $0.0039$ (from $0.9964$ clean to $0.9925$ at $20\text{ dB}$).
  - **Conclusion:** STFT does not filter noise; its **coarse spectral integration** preserves local SNR in the fundamental band, unlike point-wise EMD extrema or high-frequency wavelet detail coefficients.

---

## 5. Validation of the IEEE Std 1159 Claim

### 5.1 Stated Claim
The documentation claimed that noise testing was conducted *"pursuant to IEEE Std 1159-2019"*.

### 5.2 Standard Specification Audit
- **IEEE Std 1159-2019 Title:** *IEEE Recommended Practice for Monitoring Electric Power Quality*.
- **Standard Content:**
  - Defines power quality disturbance categories (sags, swells, transients, harmonics, flickers).
  - Establishes nominal voltage thresholds, duration boundaries, and spectral definitions.
  - Specifies measurement principles and transducer guidelines.
- **What IEEE Std 1159 Does NOT Contain:**
  - IEEE Std 1159 does **not** specify, mandate, or certify machine learning benchmark protocols.
  - IEEE Std 1159 does **not** specify SNR tiers of $40\text{ dB}$, $30\text{ dB}$, and $20\text{ dB}$ for algorithm evaluation.
- **Finding:**
  - Claiming "compliance with IEEE Std 1159" for the noise robustness experiment is **technically unsubstantiated**.
  - **Accurate Scope:** The *taxonomy of disturbances and waveform characteristics* are defined according to IEEE Std 1159, while the *AWGN SNR stress-testing methodology* ($40\text{ dB}, 30\text{ dB}, 20\text{ dB}$) is adopted from established peer-reviewed academic literature in power engineering (e.g., IEEE Transactions on Power Delivery).

---

## 6. Comprehensive Verification Summary Table

| # | Reported Claim / Assertion | Inspected Evidence & Raw Values | Verification Status | Required Correction / Nuance |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Clean Test Accuracy is 91.09% for Config L | `outputs/comparative_model_results.csv`: `0.9108823529411765` | **VERIFIED EXACT** | None. Exactly matches ground-truth. |
| **2** | 40 dB SNR Test Accuracy is 45.79% for Config L | `outputs/noise_robustness_results.csv`: `0.45794117647058824` | **VERIFIED EXACT** | None. Exactly matches ground-truth. |
| **3** | 30 dB SNR Test Accuracy is 34.62% for Config L | `outputs/noise_robustness_results.csv`: `0.3461764705882353` | **VERIFIED EXACT** | None. Exactly matches ground-truth. |
| **4** | 20 dB SNR Test Accuracy is 21.62% for Config L | `outputs/noise_robustness_results.csv`: `0.2161764705882353` | **VERIFIED EXACT** | None. Exactly matches ground-truth. |
| **5** | STFT retains 38.18% accuracy at 20 dB SNR | `outputs/noise_robustness_results.csv`: `0.38176470588235295` | **VERIFIED EXACT** | None. Highest surviving single method. |
| **6** | Clean baseline reproduces earlier benchmark | Cross-file diff between `comparative_model_results.csv` and `noise_robustness_results.csv` | **VERIFIED EXACT** | $\Delta = 0.000000$ across all 12 configs. |
| **7** | Degradation is -33.26% (STFT) and -69.47% (Config L) | `Clean - 20dB`: $71.44 - 38.18 = 33.26$; $91.09 - 21.62 = 69.47$ | **NUANCED** | Clarify that degradation is reported in **percentage points (pp)**, not relative percentage ($46.56\%$ and $76.27\%$). |
| **8** | Noise is scaled independently per waveform | `src/noise_robustness.py`: `p_signal = np.mean(signal**2)` inside per-signal loop | **VERIFIED CORRECT** | Correctly accommodates amplitude differences between Sags and Swells. |
| **9** | Empirical SNR matches target SNR | Mean empirical SNR: $40.0402\text{ dB}$, $30.0402\text{ dB}$, $20.0402\text{ dB}$ | **VERIFIED EXACT** | Deviation $< 0.05\text{ dB}$ across 3,400 test waveforms. |
| **10** | All 12 configs evaluated on identical noisy signals | Shared `test_idx` ($N=3,400$) and identical `X_test_noisy` array | **VERIFIED FAIR** | Strictly fair comparative testing. |
| **11** | Features re-extracted from raw noisy waveforms | Line 236: `extract_all_extended_dataset_features(X_test_noisy)` | **VERIFIED CORRECT** | Real physical feature extraction from noisy voltage time-series. |
| **12** | Test partition kept separate from training | `StandardScaler` and `XGBClassifier` fit strictly on `train_idx` | **VERIFIED ZERO LEAKAGE** | No data leakage detected. |
| **13** | STFT "inherently filters" AWGN | Spectral power integration in $[0, 300\text{ Hz}]$ retains fundamental | **CORRECTION REQUIRED** | STFT does not filter noise; coarse frequency binning concentrates fundamental signal power relative to uniform broadband noise. |
| **14** | HHT degrades due to algorithm failure | EMD extracts 3 IMFs; fundamental shifts to IMF 3, schema only reads IMFs 1 & 2 | **IDENTIFIED MECHANISM** | Degradation is caused by mode mixing and schema truncation, not runtime crash. |
| **15** | Noise stress-testing is "compliant with IEEE Std 1159" | IEEE Std 1159 defines disturbance definitions, not ML noise benchmark standards | **CORRECTION REQUIRED** | Modify claim to: "Disturbance taxonomy defined by IEEE Std 1159; AWGN SNR tiers adopted from peer-reviewed literature." |

---

## 7. Recommendations for Final Documentation & Thesis

1. **Terminology Clarification**:
   In [`README.md`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/README.md) and [`docs/Final_Technical_Report.md`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/docs/Final_Technical_Report.md), label the degradation column as **"Degradation (percentage points / pp)"** to prevent confusion with relative percentage loss.
2. **Scientific Precision on Standards**:
   Replace any assertion of "IEEE Std 1159 noise testing compliance" with:
   *"Disturbance taxonomy and signal parameters are governed by IEEE Std 1159-2019; noise robustness evaluation under Additive White Gaussian Noise (40, 30, and 20 dB SNR) follows standard peer-reviewed methodology in power quality machine learning literature."*
3. **Physical Explanation of HHT Failure**:
   Include the mode-mixing explanation in the thesis discussion: high-frequency Gaussian noise introduces local extrema that EMD sifts into the first two IMFs, shifting the true fundamental power signal into IMF 3 and higher, where a 2-mode feature schema cannot detect it.
4. **Physical Explanation of STFT Resilience**:
   Clarify that STFT outperforms wavelets under noise not because it eliminates noise, but because broad sub-band integration ($0\text{--}300\text{ Hz}$) captures only $12\%$ of the flat white noise power while preserving $100\%$ of the fundamental power signal.
