# B.Tech Major Project Defense Presentation Slides

**Project Title:** Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification  
**Institution:** Department of Electrical Engineering, National Institute of Technology (NIT) Jalandhar  
**Academic Year:** 2026  
**Format:** Slide-by-Slide Deck with Key Bullet Points, Equations, and Speaking Notes  

---

## Slide 1: Title Slide
- **Title:** Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification
- **Subtitle:** Multi-Domain Feature Extraction (FFT, DWT, STFT, WPT, HHT) & XGBoost Classification under Real-World Noise Conditions
- **Department:** Department of Electrical Engineering
- **Institution:** National Institute of Technology (NIT) Jalandhar
- **Speakers:** Major Project Team
- *Speaking Notes: Welcome committee members and faculty advisors. Today we present our major project investigating multi-domain signal processing for automated power quality disturbance monitoring.*

---

## Slide 2: Engineering Problem & Motivation
- **Context:** Integration of renewable distributed generation (solar PV, wind) & power-electronic loads (inverters, EV chargers, VFDs).
- **The Issue:** Non-linear switching causes voltage sags, swells, transients, harmonics, and complex compound disturbances.
- **Standards:** Governed by **IEEE Std 1159-2019**.
- **Impact:** Transformer overheating, insulation degradation, false breaker tripping, and industrial downtime.
- **Core Research Challenge:** Non-stationary disturbances cannot be adequately captured by conventional single-domain Fourier transforms alone.
- *Speaking Notes: Traditional FFT assumes periodicity and loses time localization. We need a multi-domain framework that captures time, frequency, time-scale, and non-stationary phase information.*

---

## Slide 3: Benchmark Dataset (SEED-PQD v1)
- **Source:** Zenodo Benchmark Record 11843312.
- **Total Signals:** $17,000$ discrete waveforms ($1,000$ signals per class, 100% balanced).
- **Disturbance Categories:** 17 classes (Pure Sine, Sag, Swell, Interruption, Transients, Harmonics, Flicker, Notch, and Compound Distortions).
- **Sampling Frequency ($f_s$):** $5,000\text{ Hz}$ ($5\text{ kHz}$).
- **Power Fundamental ($f_0$):** $50\text{ Hz}$ (Indian / European grid nominal).
- **Sequence Length:** Exactly 1 cycle ($20\text{ ms} = 100\text{ samples}$).
- **Data Quality:** Zero missing values, zero NaNs, double-precision numeric consistency.
- *Speaking Notes: SEED-PQD is an ideal benchmark because it provides 1,000 realistic synthetic waveforms for every single one of the 17 standard disturbance classes.*

---

## Slide 4: Multi-Domain Signal Processing Architecture
- **6 Signal Processing Methodologies (66 Features Total)**:
  1. **FFT (7 Features)**: Steady-state harmonic distribution, THD, spectral centroid, spectral entropy.
  2. **DWT (20 Features)**: 3-level dyadic decomposition (`db4`), approximation ($cA_3$) and detail ($cD_3, cD_2, cD_1$) energies, standard deviations, MAV, and peak values.
  3. **Statistical Time-Domain (7 Features)**: RMS, variance, skewness, excess kurtosis, crest factor, peak-to-peak.
  4. **STFT (8 Features)**: Moving Hann window ($N_{\text{perseg}}=32, 50\%$ overlap), time-frequency spectrogram energy in low, mid, and high bands + temporal variability.
  5. **WPT (16 Features)**: Full balanced binary tree (`db4`, Level 3), 8 uniform terminal sub-bands ($312.5\text{ Hz}$ each) covering $0\text{ to }2500\text{ Hz}$.
  6. **HHT (8 Features)**: Empirical Mode Decomposition (EMD) extracting IMFs 1 & 2 + Hilbert transform instantaneous amplitude and frequency.
- *Speaking Notes: Each technique addresses a distinct physical domain. FFT captures steady harmonics; DWT isolates sudden transients; STFT tracks time-frequency evolution; WPT provides uniform sub-band resolution; HHT resolves non-linear instantaneous phase.*

---

## Slide 5: Quality Audit & Feature Matrix Dimensionality
- **Matrix Shape:** $17,000\text{ rows} \times 68\text{ columns}$ ($20.96\text{ MB}$).
- **Metadata:** 2 columns (`class_label` string + `class_index` integer).
- **Features:** 66 numerical features.
- **Audit Results**:
  - Duplicate columns: **0**
  - NaN / Infinite values: **0**
  - Degenerate / Constant features: **0**
  - Alignment: 100% verified against documented taxonomy.
- **Leakage Prevention:** `StandardScaler` fitted strictly on training data partition only.
- *Speaking Notes: We conducted a formal automated audit verifying zero numerical artifacts, ensuring our machine learning results are 100% methodologically sound.*

---

## Slide 6: Machine Learning Protocol (XGBoost)
- **Classifier:** Extreme Gradient Boosting (**XGBoost**) multi-class classifier (`multi:softprob`).
- **Data Splitting:**
  - 80% Stratified Training Partition ($13,600$ signals, $800$ per class).
  - 20% Held-Out Test Partition ($3,400$ signals, $200$ per class).
- **Validation:** 5-Fold Stratified Cross-Validation on the training partition.
- **Evaluation Metrics:** Accuracy, Macro-Precision, Macro-Recall, and Macro-F1 Score.
- **Hyperparameters:** $100$ estimators, tree depth $6$, learning rate $0.1$, histogram-based tree splitting (`tree_method='hist'`).
- *Speaking Notes: The primary objective was comparing signal representations, not tuning dozens of algorithms. XGBoost provides a fair, invariant baseline.*

---

## Slide 7: Comparative Benchmark Results (Configurations A through L)
- **Single Method Benchmark**:
  - DWT only: **82.03%**
  - STFT only: **71.44%**
  - Statistical only: **71.32%**
  - HHT only: **69.38%**
  - FFT only: **68.79%**
  - WPT only: **66.00%**
- **Domain Fusion Synergy**:
  - FFT + DWT: **86.71%** (+4.68% over DWT alone)
  - FFT + DWT + HHT: **89.68%** (+2.97% boost)
  - FFT + DWT + Stats: **89.03%** (+2.32% boost)
  - **All Combined (Config L, 66 features): 91.09% Test Accuracy | 91.10% Macro-F1**
- *Speaking Notes: As shown in our results, combining complementary domains yields a monotonic improvement. Fusing all 6 methods breaks the 91% mark.*

---

## Slide 8: Per-Class Classification Highlights
- **Perfect 100% F1-Score**:
  - `Pure_Sinusoidal`: 100.0%
  - `Harmonics`: 100.0%
- **Near-Perfect (>97% F1-Score)**:
  - `Flicker`: 99.5%
  - `Interruption`: 98.5%
  - `Notch`: 97.4%
- **Transients & Sags (>93% F1-Score)**:
  - `Swell`: 95.0%
  - `Transient`: 94.0%
  - `Sag`: 93.9%
  - `Oscillatory_Transient`: 93.3%
- **Compound Disturbance Challenges**:
  - Overlapping phenomena (e.g. `Swell_with_Harmonics` ~72.8%, `Sag_with_Harmonics` ~78.1%) represent the primary remaining classification boundary.
- *Speaking Notes: The model separates distinct physical disturbance types with near-zero error. Only compound events with overlapping spectral profiles present residual ambiguity.*

---

## Slide 9: Noise Robustness & AWGN Stress Testing (Step 9)
- **Motivation:** Substation CT/PT sensors and industrial meters operate in electrically noisy environments.
- **Standards & Protocol:** Disturbance taxonomy defined per **IEEE Std 1159-2019**; AWGN stress testing applied at **$40\text{ dB}$, $30\text{ dB}$, and $20\text{ dB}$ SNR**.
- **Experimental Findings**:
  - **STFT is the Most Noise-Resilient Technique**:
    - Retains **64.74%** accuracy at $40\text{ dB}$ SNR, **59.82%** at $30\text{ dB}$, and **38.18%** at $20\text{ dB}$.
    - Lowest degradation across the spectrum: drop of only **33.26 percentage points** (**46.56% relative loss** from Clean to $20\text{ dB}$ vs. $76.27\%$ loss for Config L).
    - *Physical Mechanism:* STFT energy features concentrate in the low-frequency band ($0\text{--}300\text{ Hz}$). Because white noise is uniformly spread across the $0\text{--}2,500\text{ Hz}$ Nyquist bandwidth, $\approx 88\%$ of noise energy falls outside this diagnostic band, leaving fundamental power signals intact.
  - **HHT Suffers Severe Noise Degradation**:
    - Drops to $15.65\%$ at $40\text{ dB}$ and $6.41\%$ at $20\text{ dB}$ (**90.76% relative loss**).
    - *Physical Mechanism:* High-frequency white noise injects artificial local extrema, causing EMD to sift noise into IMFs 1 & 2 and shifting the true 50 Hz disturbance into IMF 3 (omitted by 2-mode scheme).
- *Speaking Notes: This is a major engineering contribution: while HHT provides clean-grid phase resolution, STFT is essential for unshielded, high-noise industrial feeders.*

---

## Slide 10: Engineering Recommendations for Smart Grid Hardware
- **Deployment Strategy Based on Field Conditions**:
  - **Clean / High-Precision Substations**: Deploy **Configuration L (All 66 Features)** for maximum diagnostic resolution (**91.09% accuracy**).
  - **Industrial / Noisy Distribution Feeders**: Deploy **Configuration I (FFT + DWT + STFT)** or **Configuration D (STFT)** for maximum noise resilience.
- **Computational Complexity**:
  - Full feature extraction across 17,000 signals takes only **8.92 seconds** (**0.52 ms per signal**).
  - Can easily run sub-cycle (<20 ms) on standard embedded DSPs (e.g. TI C6000 or ARM Cortex-M7).
- *Speaking Notes: The feature extraction pipeline executes in 0.52 ms per signal, which is 38 times faster than the 20 ms cycle duration, proving real-time feasibility.*

---

## Slide 11: Summary & Conclusion
- **Achievements**:
  - Comprehensive implementation and mathematical formulation of 6 signal processing methods.
  - Extraction and validation of a 66-dimensional feature matrix across 17,000 signals ($100\%$ complete, $0$ errors).
  - Rigorous 12-configuration XGBoost benchmark achieving **91.09% Accuracy** and **91.10% Macro-F1**.
  - Comprehensive IEEE Std 1159 AWGN noise stress testing revealing STFT's noise resilience vs. HHT mode mixing.
  - Complete reproducible codebase and publication-ready documentation.
- **Future Directions**: Embedded FPGA implementation & Ensemble Empirical Mode Decomposition (EEMD).
- **Questions & Answers**: Thank you! We are happy to take questions from the committee.
