# Major Project: Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification

**Department of Electrical Engineering | National Institute of Technology (NIT) Jalandhar**  
*Academic Year: 2026*  

---

## 1. Project Directory Structure

The project has been organized into a modular, production-grade architecture supporting dataset loading, multi-domain signal processing, automated validation, and comparative machine-learning classification:

```text
ML Code/
│
├── data/
│   ├── raw/                         # Raw dataset files
│   │   ├── SEED_PQD_v1.zip          # Zenodo archive (Record 11843312)
│   │   ├── Details.txt              # Specification sheet
│   │   └── XPQRS/                   # 17 class CSVs + 5Kfs_1Cycle_50f_1000Sam_1A.mat
│   └── processed/                   # Prepared feature matrices & datasets
│       ├── extracted_features.csv   # Milestone 1 feature matrix (17,000 x 29; 27 features)
│       ├── extended_features.csv    # Extended feature matrix (17,000 x 68; 66 features)
│       └── feature_dictionary.csv   # Complete schema documentation in CSV format
│
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── data_loader.py               # Robust dataset loading & sanity checking
│   ├── signal_processing.py         # Multi-domain feature extractors (FFT, DWT, Stats, STFT, WPT, HHT)
│   ├── visualization.py             # Publication-ready Matplotlib visualizers & confusion heatmaps
│   ├── classification.py            # XGBoost comparative training & evaluation pipeline
│   └── feature_dictionary.py        # Automated feature dictionary generator
│
├── outputs/
│   ├── figures/                     # All generated plots and visual spectra
│   │   ├── class_distribution.png   # Class balance across 17 classes
│   │   ├── time_domain_waveforms.png# 9-class time-domain waveform grid
│   │   ├── fft_magnitude_spectra.png# One-sided frequency spectra (0 to 2500 Hz)
│   │   ├── dwt_decomposition.png    # db4 multi-level decomposition (cA3, cD3, cD2, cD1)
│   │   ├── dwt_features.png         # Wavelet sub-band energy comparison
│   │   ├── stft_spectrograms.png    # Short-Time Fourier Transform time-frequency power maps
│   │   ├── wpt_subband_energies.png # Wavelet Packet Transform 8-band uniform energy profiles
│   │   ├── hht_decomposition.png    # Empirical Mode Decomposition & IMF spectra
│   │   ├── confusion_matrix_best_config_L.png # Best model (Config L) confusion matrix
│   │   ├── confusion_matrix_all_features_config_L.png # 17-class normalized heatmap
│   │   └── method_comparison.png    # Bar chart comparing Accuracy, F1, & Dimensionality (A-L)
│   ├── models/                      # Saved trained models
│   │   └── xgboost_best_model_L.json# Serialized best XGBoost model artifact
│   ├── comparative_model_results.csv# Benchmark comparison table across configs A through L
│   └── per_class_report_best_config_L.csv # Detailed per-class precision, recall, and F1
│
├── docs/                            # Comprehensive documentation & reports
│   ├── Milestone_1_Technical_Report.md # Formal technical milestone 1 report
│   ├── dataset_inspection_report.md    # Dataset verification findings
│   ├── feature_dictionary.md        # Comprehensive 66-feature mathematical dictionary
│   └── Details.txt                  # Original dataset parameters
│
├── run_milestone_1.py               # Milestone 1 runner (FFT + DWT)
├── run_extended_pipeline.py         # Master runner for extended 4 methods & XGBoost benchmark
├── requirements.txt                 # Python dependencies
└── README.md                        # Master project guide (this file)
```

---

## 2. Dataset Quick Facts (SEED-PQD)

- **Source:** [https://zenodo.org/records/11843312](https://zenodo.org/records/11843312)
- **Power System Fundamental ($f_0$):** $50\text{ Hz}$
- **Sampling Frequency ($f_s$):** $5,000\text{ Hz}$ ($5\text{ kHz}$)
- **Signal Duration ($T$):** $20\text{ ms}$ (1 complete AC cycle = $100\text{ samples}$)
- **Frequency Resolution ($\Delta f$):** $50\text{ Hz}$ (Nyquist limit = $2,500\text{ Hz}$)
- **Total Disturbances:** 17 classes ($1,000\text{ signals/class} = 17,000\text{ total signals}$)
- **Quality Audit:** Zero missing/NaN/Inf values; 100% balanced across all classes.

---

## 3. Signal Processing Feature Extraction Families (66 Features)

| Family | Method | Features | Primary Domain | Physical Information Captured |
| :---: | :--- | :---: | :---: | :--- |
| **0A** | **Fast Fourier Transform (FFT)** | 7 | Frequency | Fundamental magnitude (50 Hz), THD, spectral centroid, spectral energy, entropy |
| **0B** | **Discrete Wavelet Transform (DWT)** | 20 | Time-Scale | Multi-level `db4` dyadic sub-bands ($cA_3, cD_3, cD_2, cD_1$ energy, std, rel-energy, MAV, max) |
| **1** | **Statistical Time-Domain** | 7 | Time | RMS, standard deviation, variance, skewness, excess kurtosis, crest factor, peak-to-peak |
| **2** | **Short-Time Fourier Transform (STFT)** | 8 | Time-Frequency | Spectrogram mean/max/std energy, low/mid/high band energies, temporal variability |
| **3** | **Wavelet Packet Transform (WPT)** | 16 | Wavelet Packet | 8 uniform 312.5 Hz terminal sub-band absolute and relative energies (`db4`, Level 3) |
| **4** | **Hilbert-Huang Transform (HHT)** | 8 | Non-stationary | EMD-derived IMFs 1 & 2: mean/std instantaneous amplitude, instantaneous frequency, energy |
| **Total** | **All 6 Combined** | **66** | **Multi-Domain** | **Comprehensive time, frequency, time-scale, and instantaneous phase representation** |

*(Matrix also includes 2 metadata columns: `class_label` and `class_index`, giving 68 columns total).*

---

## 4. How to Set Up & Run

### A. Environment Installation
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### B. Running Milestone 1 (Baseline FFT & DWT)
```bash
python run_milestone_1.py
```

### C. Running Extended Pipeline (All 6 Methods + XGBoost Comparative Benchmark)
To extract all 66 features, validate the matrix, generate all figures, and evaluate configurations A through L:
```bash
python run_extended_pipeline.py
```
*(Execution finishes in ~95 seconds using multi-core parallelization).*

---

## 5. Comparative XGBoost Benchmark Results (17 Classes, Stratified 80/20 Split)

All experiments were executed with a strict 80/20 stratified train/test split (13,600 train / 3,400 test) with 5-fold cross-validation on the training set:

| Config | Feature Configuration | # Feats | Train CV Acc | Test Acc | Macro Precision | Macro Recall | Macro F1 | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | FFT only | 7 | 69.59% | 68.79% | 67.75% | 68.79% | 68.09% | 3.73 s |
| **B** | DWT only | 20 | 82.17% | 82.03% | 81.99% | 82.03% | 81.81% | 6.05 s |
| **C** | Statistical only | 7 | 70.87% | 71.32% | 71.73% | 71.32% | 70.95% | 2.92 s |
| **D** | STFT only | 8 | 72.38% | 71.44% | 71.02% | 71.44% | 70.56% | 2.88 s |
| **E** | WPT only | 16 | 65.48% | 66.00% | 64.59% | 66.00% | 64.21% | 4.52 s |
| **F** | HHT only | 8 | 67.74% | 69.38% | 68.96% | 69.38% | 68.76% | 3.29 s |
| **G** | FFT + DWT | 27 | 87.06% | 86.71% | 86.80% | 86.71% | 86.60% | 6.25 s |
| **H** | FFT + DWT + Stats | 34 | 88.76% | 89.03% | 89.21% | 89.03% | 88.95% | 8.42 s |
| **I** | FFT + DWT + STFT | 35 | 87.94% | 87.62% | 87.72% | 87.62% | 87.52% | 8.58 s |
| **J** | FFT + DWT + WPT | 43 | 87.07% | 86.85% | 86.98% | 86.85% | 86.76% | 9.64 s |
| **K** | FFT + DWT + HHT | 35 | 88.38% | 89.68% | 89.83% | 89.68% | 89.63% | 8.19 s |
| **L** | **All Features Combined** | **66** | **89.85%** | **91.09%** | **91.37%** | **91.09%** | **91.10%** | **15.91 s** |

### Key Experimental Insights:
1. **Single Method Performance**: DWT achieved the highest single-method accuracy (82.03%), followed by STFT (71.44%) and Statistical Features (71.32%).
2. **Domain Fusion Gains**:
   - Combining FFT + DWT produced an immediate jump to **86.71%**.
   - Incorporating HHT into FFT + DWT lifted accuracy to **89.68%** (+2.97%), demonstrating the value of instantaneous frequency tracking for non-stationary PQDs.
   - Incorporating Time-Domain Statistical Moments into FFT + DWT lifted accuracy to **89.03%** (+2.32%).
   - Fusing all 6 methods (Config L, 66 features) produced the global maximum: **91.09% Test Accuracy** and **91.10% Macro F1**.
3. **Perfect Classification**: `Pure_Sinusoidal` and `Harmonics` achieved **100.0% F1-score**, while `Flicker` achieved **99.5%**, and `Interruption` achieved **98.5%**.

---

## 6. Noise Robustness (AWGN) Stress-Testing Results (IEEE Std 1159)

To simulate practical field environments, zero-mean Additive White Gaussian Noise was injected into the test partition signals:

| Configuration | Clean ($\infty$) | 40 dB SNR | 30 dB SNR | 20 dB SNR | Degradation (Clean $\to$ 20 dB) |
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

To run the noise robustness stress testing pipeline:
```bash
python run_noise_robustness.py
```

---

## 7. Master 10-Step Roadmap Progress (100% Complete)

| Step | Phase Description | Status | Deliverables / Output |
| :---: | :--- | :---: | :--- |
| **1** | Dataset acquisition and inspection | **Completed** | SEED-PQD v1 verified (17,000 signals, 17 classes) |
| **2** | Data loading and preprocessing | **Completed** | [`src/data_loader.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/src/data_loader.py) |
| **3** | Initial signal processing (FFT & DWT) | **Completed** | [`src/signal_processing.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/src/signal_processing.py) (27 features) |
| **4** | Additional signal processing (STFT, Stats, WPT, HHT) | **Completed** | Sections D, E, F, G in `src/signal_processing.py` (39 features) |
| **5** | Feature dataset preparation & normalization | **Completed** | [`data/processed/extended_features.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/data/processed/extended_features.csv) (17,000 x 68) |
| **6** | Machine-learning implementation (XGBoost) | **Completed** | [`src/classification.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/src/classification.py) (12 configurations A through L) |
| **7** | Model evaluation (Metrics & Confusion Matrices) | **Completed** | [`outputs/comparative_model_results.csv`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/comparative_model_results.csv), confusion matrices |
| **8** | Comparative analysis across signal processing techniques | **Completed** | [`outputs/figures/method_comparison.png`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/outputs/figures/method_comparison.png) |
| **9** | Validation, robustness (Noise/AWGN), and research contribution | **Completed** | [`run_noise_robustness.py`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/run_noise_robustness.py), `outputs/noise_robustness_results.csv` |
| **10** | Final documentation, thesis, and defense presentation | **Completed** | [`docs/Major_Project_Comprehensive_Technical_Report.md`](file:///c:/Drive%20E_F/NIT%20Jalandhar/4th%20Year/7th%20Semester/Major%20Project/ML%20Code/docs/Major_Project_Comprehensive_Technical_Report.md), `docs/Defense_Presentation_Slides.md` |
