# Power Quality Disturbance Feature Dictionary

**National Institute of Technology (NIT) Jalandhar | Department of Electrical Engineering**  
**Project:** Comparative Analysis of Signal Processing Techniques for Machine Learning-Based Power Quality Disturbance Classification  

This dictionary documents all **66 extracted features** across **6 signal processing families** plus **2 metadata columns** (Total: 68 columns in `extended_features.csv`).

## Summary by Feature Family

| Family | Physical Domain | Feature Count | Target Phenomenon |
| :--- | :--- | :---: | :--- |
| **Metadata** | Labels & Index | 2 | Class label string and integer class index |
| **FFT** | Frequency (Fourier) | 7 | Steady-state harmonics, THD, spectral centroid |
| **DWT** | Time-Scale (Wavelet) | 20 | Multi-resolution sub-band energies & statistics |
| **Statistical** | Time Domain | 7 | Amplitude spread, RMS, crest factor, skewness, kurtosis |
| **STFT** | Time-Frequency | 8 | Temporal energy progression across frequency bands |
| **WPT** | Wavelet Packet | 16 | Uniform 312.5 Hz sub-band energies & relative energies |
| **HHT** | Hilbert-Huang (EMD) | 8 | Instantaneous amplitude and frequency of IMFs |
| **TOTAL** | **Multi-Domain** | **68 (66 features + 2 metadata)** | **Comprehensive PQ Disturbance Representation** |

---

## Complete Feature Specification

| # | Column Name | Family | Domain | Units | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `class_label` | Metadata | Labels | String | Human-readable disturbance class name (e.g., Pure_Sinusoidal, Sag, Swell) |
| 2 | `class_index` | Metadata | Labels | Integer [0-16] | Zero-indexed integer label for ML classification |
| 3 | `fft_fund_mag` | FFT | Frequency | p.u. | Normalized magnitude of fundamental power frequency bin at 50 Hz |
| 4 | `fft_thd` | FFT | Frequency | Ratio (dimensionless) | Total Harmonic Distortion: sqrt(sum(harmonics^2)) / fundamental_magnitude |
| 5 | `fft_spec_energy` | FFT | Frequency | p.u.^2 | Total discrete spectral energy: sum of squared magnitude bins |
| 6 | `fft_spec_centroid` | FFT | Frequency | Hz | Spectral center-of-mass frequency: sum(f * mag) / sum(mag) |
| 7 | `fft_spec_entropy` | FFT | Frequency | nats | Spectral entropy measuring peakedness vs. dispersion of power distribution |
| 8 | `fft_dominant_freq` | FFT | Frequency | Hz | Frequency corresponding to the maximum spectral magnitude peak |
| 9 | `fft_dominant_mag` | FFT | Frequency | p.u. | Magnitude of the dominant spectral peak |
| 10 | `dwt_cA3_energy` | DWT | Time-Scale | p.u.^2 | Sub-band energy (sum of squared coefficients) for Approximation band [0.0 - 312.5 Hz] containing 50 Hz fundamental and 3rd/5th harmonics |
| 11 | `dwt_cA3_std` | DWT | Time-Scale | p.u. | Standard deviation of wavelet coefficients for Approximation band [0.0 - 312.5 Hz] containing 50 Hz fundamental and 3rd/5th harmonics |
| 12 | `dwt_cA3_rel_energy` | DWT | Time-Scale | Ratio [0-1] | Relative energy ratio: cA3 energy divided by total wavelet energy |
| 13 | `dwt_cA3_mav` | DWT | Time-Scale | p.u. | Mean Absolute Value of wavelet coefficients for Approximation band [0.0 - 312.5 Hz] containing 50 Hz fundamental and 3rd/5th harmonics |
| 14 | `dwt_cA3_max` | DWT | Time-Scale | p.u. | Maximum absolute coefficient amplitude in cA3 |
| 15 | `dwt_cD3_energy` | DWT | Time-Scale | p.u.^2 | Sub-band energy (sum of squared coefficients) for Detail band [312.5 - 625.0 Hz] containing mid-order harmonics |
| 16 | `dwt_cD3_std` | DWT | Time-Scale | p.u. | Standard deviation of wavelet coefficients for Detail band [312.5 - 625.0 Hz] containing mid-order harmonics |
| 17 | `dwt_cD3_rel_energy` | DWT | Time-Scale | Ratio [0-1] | Relative energy ratio: cD3 energy divided by total wavelet energy |
| 18 | `dwt_cD3_mav` | DWT | Time-Scale | p.u. | Mean Absolute Value of wavelet coefficients for Detail band [312.5 - 625.0 Hz] containing mid-order harmonics |
| 19 | `dwt_cD3_max` | DWT | Time-Scale | p.u. | Maximum absolute coefficient amplitude in cD3 |
| 20 | `dwt_cD2_energy` | DWT | Time-Scale | p.u.^2 | Sub-band energy (sum of squared coefficients) for Detail band [625.0 - 1250.0 Hz] containing oscillatory transients |
| 21 | `dwt_cD2_std` | DWT | Time-Scale | p.u. | Standard deviation of wavelet coefficients for Detail band [625.0 - 1250.0 Hz] containing oscillatory transients |
| 22 | `dwt_cD2_rel_energy` | DWT | Time-Scale | Ratio [0-1] | Relative energy ratio: cD2 energy divided by total wavelet energy |
| 23 | `dwt_cD2_mav` | DWT | Time-Scale | p.u. | Mean Absolute Value of wavelet coefficients for Detail band [625.0 - 1250.0 Hz] containing oscillatory transients |
| 24 | `dwt_cD2_max` | DWT | Time-Scale | p.u. | Maximum absolute coefficient amplitude in cD2 |
| 25 | `dwt_cD1_energy` | DWT | Time-Scale | p.u.^2 | Sub-band energy (sum of squared coefficients) for Detail band [1250.0 - 2500.0 Hz] containing high-frequency notches and steep edges |
| 26 | `dwt_cD1_std` | DWT | Time-Scale | p.u. | Standard deviation of wavelet coefficients for Detail band [1250.0 - 2500.0 Hz] containing high-frequency notches and steep edges |
| 27 | `dwt_cD1_rel_energy` | DWT | Time-Scale | Ratio [0-1] | Relative energy ratio: cD1 energy divided by total wavelet energy |
| 28 | `dwt_cD1_mav` | DWT | Time-Scale | p.u. | Mean Absolute Value of wavelet coefficients for Detail band [1250.0 - 2500.0 Hz] containing high-frequency notches and steep edges |
| 29 | `dwt_cD1_max` | DWT | Time-Scale | p.u. | Maximum absolute coefficient amplitude in cD1 |
| 30 | `stat_rms` | Statistical | Time | p.u. | Root Mean Square amplitude: sqrt(1/N * sum(x[n]^2)) |
| 31 | `stat_std` | Statistical | Time | p.u. | Sample standard deviation (signal spread around mean) |
| 32 | `stat_var` | Statistical | Time | p.u.^2 | Signal variance (second central moment) |
| 33 | `stat_skewness` | Statistical | Time | Dimensionless | Fisher-Pearson standardized 3rd moment measuring waveform asymmetry |
| 34 | `stat_kurtosis` | Statistical | Time | Dimensionless | Fisher excess kurtosis measuring heavy-tailedness (normal distribution = 0.0) |
| 35 | `stat_crest_factor` | Statistical | Time | Ratio | Ratio of absolute peak amplitude to RMS: max|x| / RMS |
| 36 | `stat_peak_to_peak` | Statistical | Time | p.u. | Dynamic excursion range: max(x) - min(x) |
| 37 | `stft_mean_energy` | STFT | Time-Frequency | p.u.^2 / Hz | Mean spectrogram power density across time-frequency bins |
| 38 | `stft_max_energy` | STFT | Time-Frequency | p.u.^2 / Hz | Maximum spectrogram power density (peak energy concentration) |
| 39 | `stft_std_energy` | STFT | Time-Frequency | p.u.^2 / Hz | Standard deviation of spectrogram power density |
| 40 | `stft_energy_low_band` | STFT | Time-Frequency | p.u.^2 | Integrated energy in fundamental low band [0.0 - 300.0 Hz] |
| 41 | `stft_energy_mid_band` | STFT | Time-Frequency | p.u.^2 | Integrated energy in intermediate harmonic/transient band (300.0 - 1200.0 Hz] |
| 42 | `stft_energy_high_band` | STFT | Time-Frequency | p.u.^2 | Integrated energy in high-frequency band (1200.0 - 2500.0 Hz] |
| 43 | `stft_rel_energy_low` | STFT | Time-Frequency | Ratio [0-1] | Ratio of low-band power to total spectrogram power |
| 44 | `stft_temporal_variability` | STFT | Time-Frequency | p.u.^2 | Average across frequency bins of standard deviation across time slices |
| 45 | `wpt_band0_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 0.0 - 312.5 Hz (Fundamental 50 Hz and low-order 3rd/5th harmonics) |
| 46 | `wpt_band0_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band0 energy divided by total WPT tree energy |
| 47 | `wpt_band1_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 312.5 - 625.0 Hz (7th, 9th, 11th harmonic components) |
| 48 | `wpt_band1_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band1 energy divided by total WPT tree energy |
| 49 | `wpt_band2_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 625.0 - 937.5 Hz (Mid-frequency resonance oscillations) |
| 50 | `wpt_band2_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band2 energy divided by total WPT tree energy |
| 51 | `wpt_band3_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 937.5 - 1250.0 Hz (Higher-order harmonic distortions) |
| 52 | `wpt_band3_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band3 energy divided by total WPT tree energy |
| 53 | `wpt_band4_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 1250.0 - 1562.5 Hz (Lower oscillatory transient band) |
| 54 | `wpt_band4_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band4 energy divided by total WPT tree energy |
| 55 | `wpt_band5_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 1562.5 - 1875.0 Hz (Mid oscillatory transient band) |
| 56 | `wpt_band5_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band5 energy divided by total WPT tree energy |
| 57 | `wpt_band6_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 1875.0 - 2187.5 Hz (High oscillatory transient band) |
| 58 | `wpt_band6_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band6 energy divided by total WPT tree energy |
| 59 | `wpt_band7_energy` | WPT | Wavelet Packet | p.u.^2 | Sub-band energy (sum of squared coefficients) in 2187.5 - 2500.0 Hz (Fast notches and steep edge transients) |
| 60 | `wpt_band7_rel_energy` | WPT | Wavelet Packet | Ratio [0-1] | Relative energy ratio: band7 energy divided by total WPT tree energy |
| 61 | `hht_imf1_mean_amp` | HHT | Hilbert-Huang | p.u. | Mean instantaneous amplitude of Intrinsic Mode Function 1 (high-frequency, fast transient mode) |
| 62 | `hht_imf1_std_amp` | HHT | Hilbert-Huang | p.u. | Standard deviation of instantaneous amplitude of Intrinsic Mode Function 1 (high-frequency, fast transient mode) |
| 63 | `hht_imf1_mean_freq` | HHT | Hilbert-Huang | Hz | Mean instantaneous frequency of Intrinsic Mode Function 1 (high-frequency, fast transient mode) (bounded to [0, fs/2]) |
| 64 | `hht_imf1_energy` | HHT | Hilbert-Huang | p.u.^2 | Discrete energy (sum of squared values) of Intrinsic Mode Function 1 (high-frequency, fast transient mode) |
| 65 | `hht_imf2_mean_amp` | HHT | Hilbert-Huang | p.u. | Mean instantaneous amplitude of Intrinsic Mode Function 2 (dominant fundamental 50 Hz intermediate mode) |
| 66 | `hht_imf2_std_amp` | HHT | Hilbert-Huang | p.u. | Standard deviation of instantaneous amplitude of Intrinsic Mode Function 2 (dominant fundamental 50 Hz intermediate mode) |
| 67 | `hht_imf2_mean_freq` | HHT | Hilbert-Huang | Hz | Mean instantaneous frequency of Intrinsic Mode Function 2 (dominant fundamental 50 Hz intermediate mode) (bounded to [0, fs/2]) |
| 68 | `hht_imf2_energy` | HHT | Hilbert-Huang | p.u.^2 | Discrete energy (sum of squared values) of Intrinsic Mode Function 2 (dominant fundamental 50 Hz intermediate mode) |