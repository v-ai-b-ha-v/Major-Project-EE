# Major Project: Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification

**Department of Electrical Engineering | National Institute of Technology (NIT) Jalandhar**  
*Academic Year: 2026*  

---

## 1. Project Directory Structure

The project has been organized into a modular, industry-standard directory layout designed to support all milestones through final model training, evaluation, and documentation:

```text
ML Code/
│
├── data/
│   ├── raw/                         # Raw dataset files
│   │   ├── SEED_PQD_v1.zip          # Zenodo archive (Record 11843312)
│   │   ├── Details.txt              # Specification sheet
│   │   └── XPQRS/                   # 17 class CSVs + 5Kfs_1Cycle_50f_1000Sam_1A.mat
│   └── processed/                   # Prepared feature matrices & datasets
│       └── extracted_features.csv   # Milestone 1 extracted features (17,000 x 29)
│
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── data_loader.py               # Robust dataset loading & sanity checking
│   ├── signal_processing.py         # FFT & DWT feature extractors
│   └── visualization.py             # Publication-ready Matplotlib visualizers
│
├── outputs/
│   ├── figures/                     # All generated plots and visual spectra
│   │   ├── class_distribution.png   # Class balance across 17 classes
│   │   ├── time_domain_waveforms.png# 9-class time-domain waveform grid
│   │   ├── fft_magnitude_spectra.png# One-sided frequency spectra (0 to 2500 Hz)
│   │   ├── dwt_decomposition.png    # db4 multi-level decomposition (cA3, cD3, cD2, cD1)
│   │   └── dwt_features.png         # Wavelet sub-band energy comparison
│   └── models/                      # Saved trained models (for Step 6)
│
├── docs/                            # Comprehensive documentation & reports
│   ├── Milestone_1_Technical_Report.md # Formal technical milestone report
│   ├── dataset_inspection_report.md    # Dataset verification findings
│   └── Details.txt                  # Original dataset parameters
│
├── run_milestone_1.py               # Main root entry-point runner
├── main_milestone.py                # Backward-compatibility alias
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

## 3. How to Set Up & Run

### A. Environment Installation
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### B. Running Milestone 1
To load data, run validation checks, generate all figures, and extract features:
```bash
python run_milestone_1.py
```
*(Execution finishes in ~5.5 seconds, saving plots to `outputs/figures/` and features to `data/processed/extracted_features.csv`).*

---

## 4. Master 10-Step Roadmap Progress

| Step | Phase Description | Status |
| :---: | :--- | :---: |
| **1** | Dataset acquisition and inspection | **Completed** |
| **2** | Data loading and preprocessing | **Completed** |
| **3** | Initial signal processing (FFT & DWT) | **Completed** |
| **4** | Additional signal processing (STFT & Time-Domain) | **Next Step** |
| **5** | Feature dataset preparation & normalization | **Partially Complete** |
| **6** | Machine-learning implementation (Random Forest & SVM) | Pending |
| **7** | Model evaluation (Metrics & Confusion Matrices) | Pending |
| **8** | Comparative analysis across signal processing techniques | Pending |
| **9** | Validation, robustness (Noise/AWGN), and research contribution | Pending |
| **10** | Final documentation, thesis, and defense presentation | Pending |
