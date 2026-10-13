"""
Dataset Loader and Preprocessing Module for SEED-PQD Dataset
============================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Module: src/data_loader.py

This module provides reusable functions to locate, validate,
and load the SEED-PQD dataset from CSV files or MATLAB (.mat) files.
"""

import os
import glob
import zipfile
import numpy as np
import pandas as pd
import scipy.io as sio

# Verified Class Names in documented order from Details.txt
CLASS_NAMES = [
    "Pure_Sinusoidal",
    "Sag",
    "Swell",
    "Interruption",
    "Transient",
    "Oscillatory_Transient",
    "Harmonics",
    "Harmonics_with_Sag",
    "Harmonics_with_Swell",
    "Flicker",
    "Flicker_with_Sag",
    "Flicker_with_Swell",
    "Sag_with_Oscillatory_Transient",
    "Swell_with_Oscillatory_Transient",
    "Sag_with_Harmonics",
    "Swell_with_Harmonics",
    "Notch",
]

# Physical Signal Characteristics (Documented in Details.txt)
SAMPLING_RATE_HZ = 5000       # 5 kHz
FUNDAMENTAL_FREQ_HZ = 50      # 50 Hz power frequency
SAMPLES_PER_SIGNAL = 100      # 100 samples per signal
SIGNAL_DURATION_SEC = 0.020   # 20 ms (exactly 1 power cycle at 50 Hz)
SIGNALS_PER_CLASS = 1000      # 1000 signals per class
NUM_CLASSES = 17              # 17 disturbance classes


def get_signal_metadata():
    """
    Returns verified physical metadata for the SEED-PQD dataset.
    """
    dt = 1.0 / SAMPLING_RATE_HZ
    time_axis_ms = np.arange(SAMPLES_PER_SIGNAL) * dt * 1000.0  # in milliseconds
    return {
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "fundamental_freq_hz": FUNDAMENTAL_FREQ_HZ,
        "samples_per_signal": SAMPLES_PER_SIGNAL,
        "signal_duration_sec": SIGNAL_DURATION_SEC,
        "signal_duration_ms": SIGNAL_DURATION_SEC * 1000.0,
        "signals_per_class": SIGNALS_PER_CLASS,
        "num_classes": NUM_CLASSES,
        "time_axis_ms": time_axis_ms,
        "class_names": list(CLASS_NAMES),
    }


def locate_dataset_dir(base_dir=None):
    """
    Searches known relative and absolute paths for the extracted SEED-PQD files
    or the ZIP archive. If not found, returns None.
    """
    if base_dir is None:
        # Detect workspace root relative to this file
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.abspath(os.path.join(current_file_dir, ".."))
        candidate_bases = [os.getcwd(), workspace_root, current_file_dir]
    else:
        candidate_bases = [base_dir]

    candidate_dirs = []
    for b in candidate_bases:
        candidate_dirs.extend([
            os.path.join(b, "data", "raw", "XPQRS"),
            os.path.join(b, "data", "raw"),
            os.path.join(b, "dataset", "extracted", "XPQRS"),
            os.path.join(b, "dataset", "XPQRS"),
            os.path.join(b, "SEED_PQD_v1", "XPQRS"),
            os.path.join(b, "..", "data", "raw", "XPQRS"),
        ])

    for cdir in candidate_dirs:
        if os.path.isdir(cdir):
            # Check if at least one class CSV or MAT exists
            if os.path.exists(os.path.join(cdir, "Pure_Sinusoidal.csv")) or os.path.exists(
                os.path.join(cdir, "5Kfs_1Cycle_50f_1000Sam_1A.mat")
            ):
                return os.path.abspath(cdir)

    # Check for zip archive to extract if folder doesn't exist
    candidate_zips = []
    for b in candidate_bases:
        candidate_zips.extend([
            os.path.join(b, "data", "raw", "SEED_PQD_v1.zip"),
            os.path.join(b, "dataset", "SEED_PQD_v1.zip"),
            os.path.join(b, "SEED_PQD_v1.zip"),
            os.path.join(b, "..", "SEED_PQD_v1.zip"),
        ])

    for czip in candidate_zips:
        if os.path.isfile(czip):
            target_extract = os.path.join(os.path.dirname(czip), "XPQRS")
            print(f"Extracting '{czip}' to '{target_extract}'...")
            os.makedirs(target_extract, exist_ok=True)
            with zipfile.ZipFile(czip, "r") as zf:
                zf.extractall(os.path.dirname(czip))
            if os.path.isdir(target_extract):
                return os.path.abspath(target_extract)

    return None


def print_download_instructions():
    """
    Prints clear instructions on how to obtain the SEED-PQD dataset.
    """
    instructions = """
================================================================================
SEED-PQD DATASET NOT FOUND!
================================================================================
Please acquire the SEED-PQD dataset from Zenodo:
    URL: https://zenodo.org/records/11843312

To set up the dataset:
1. Download 'SEED_PQD_v1.zip' and 'Details.txt' from the Zenodo record.
2. Place them into 'data/raw/' within this project:
       data/raw/SEED_PQD_v1.zip
       data/raw/Details.txt
3. Re-run this script. The script will automatically detect and extract the files.
================================================================================
"""
    print(instructions)


def load_dataset(dataset_dir=None, source_format="csv"):
    """
    Loads all signals and labels from the SEED-PQD dataset.

    Parameters:
    -----------
    dataset_dir : str or None
        Path to directory containing dataset files (XPQRS directory).
        If None, automatically searches candidate locations.
    source_format : str, default 'csv'
        'csv' to load from individual class CSV files,
        'mat' to load from 5Kfs_1Cycle_50f_1000Sam_1A.mat.

    Returns:
    --------
    X : np.ndarray of shape (17000, 100), dtype=np.float64
        Matrix of raw signal samples.
    y_str : np.ndarray of shape (17000,), dtype=object
        Class label strings (e.g., 'Pure_Sinusoidal', 'Sag').
    y_idx : np.ndarray of shape (17000,), dtype=np.int64
        Integer class indices (0 to 16).
    """
    if dataset_dir is None:
        dataset_dir = locate_dataset_dir()

    if dataset_dir is None or not os.path.isdir(dataset_dir):
        print_download_instructions()
        raise FileNotFoundError(
            "SEED-PQD dataset directory not found. Please review the download instructions above."
        )

    print(f"Loading SEED-PQD dataset from: {dataset_dir}")

    if source_format.lower() == "mat":
        mat_path = os.path.join(dataset_dir, "5Kfs_1Cycle_50f_1000Sam_1A.mat")
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"MATLAB file not found at: {mat_path}")
        
        mat_data = sio.loadmat(mat_path)
        if "Out" not in mat_data:
            raise KeyError(f"Expected variable 'Out' in MAT file. Available keys: {list(mat_data.keys())}")
        
        raw_mat = mat_data["Out"]  # Shape: (1000, 100, 17)
        if raw_mat.shape != (SIGNALS_PER_CLASS, SAMPLES_PER_SIGNAL, NUM_CLASSES):
            raise ValueError(
                f"Unexpected MAT matrix shape: {raw_mat.shape}. "
                f"Expected: ({SIGNALS_PER_CLASS}, {SAMPLES_PER_SIGNAL}, {NUM_CLASSES})"
            )
        
        # Reshape into (17000, 100)
        X_list = []
        y_str_list = []
        y_idx_list = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_signals = raw_mat[:, :, class_idx].astype(np.float64)
            X_list.append(class_signals)
            y_str_list.extend([class_name] * SIGNALS_PER_CLASS)
            y_idx_list.extend([class_idx] * SIGNALS_PER_CLASS)
        
        X = np.vstack(X_list)
        y_str = np.array(y_str_list, dtype=object)
        y_idx = np.array(y_idx_list, dtype=np.int64)

    else:
        # Load from CSV files
        X_list = []
        y_str_list = []
        y_idx_list = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            csv_path = os.path.join(dataset_dir, f"{class_name}.csv")
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Required CSV file missing: {csv_path}")
            
            # Read CSV without header, cast explicitly to float64
            df = pd.read_csv(csv_path, header=None, dtype=np.float64)
            if df.shape != (SIGNALS_PER_CLASS, SAMPLES_PER_SIGNAL):
                raise ValueError(
                    f"Unexpected shape for '{csv_path}': {df.shape}. "
                    f"Expected ({SIGNALS_PER_CLASS}, {SAMPLES_PER_SIGNAL})"
                )
            
            signals = df.to_numpy(dtype=np.float64)
            X_list.append(signals)
            y_str_list.extend([class_name] * SIGNALS_PER_CLASS)
            y_idx_list.extend([class_idx] * SIGNALS_PER_CLASS)

        X = np.vstack(X_list)
        y_str = np.array(y_str_list, dtype=object)
        y_idx = np.array(y_idx_list, dtype=np.int64)

    return X, y_str, y_idx


def validate_dataset(X, y_str, y_idx):
    """
    Performs data integrity checks:
    - Verifies array shapes and dimensions
    - Verifies numeric data type (float64)
    - Checks for NaN and infinite values
    - Validates class distribution balance

    Returns:
    --------
    validation_summary : dict
        Dictionary of check results and statistics.
    """
    expected_rows = NUM_CLASSES * SIGNALS_PER_CLASS
    expected_cols = SAMPLES_PER_SIGNAL

    shape_valid = (X.shape == (expected_rows, expected_cols))
    dtype_valid = (X.dtype == np.float64)
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    # Check class distribution
    unique_classes, counts = np.unique(y_str, return_counts=True)
    class_balance = dict(zip(unique_classes, counts))
    is_balanced = all(c == SIGNALS_PER_CLASS for c in counts)

    min_val = float(np.min(X))
    max_val = float(np.max(X))
    mean_val = float(np.mean(X))
    std_val = float(np.std(X))

    validation_summary = {
        "shape": X.shape,
        "shape_valid": shape_valid,
        "dtype": str(X.dtype),
        "dtype_valid": dtype_valid,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "is_finite": (nan_count == 0 and inf_count == 0),
        "total_signals": len(X),
        "samples_per_signal": X.shape[1],
        "num_classes": len(unique_classes),
        "is_balanced": is_balanced,
        "min_amplitude": min_val,
        "max_amplitude": max_val,
        "mean_amplitude": mean_val,
        "std_amplitude": std_val,
        "class_counts": class_balance,
    }

    return validation_summary


def print_validation_report(summary):
    """
    Prints a clear, formatted summary of the dataset validation.
    """
    print("\n" + "=" * 65)
    print("           SEED-PQD DATASET VALIDATION REPORT           ")
    print("=" * 65)
    print(f"  Total Signals Loaded     : {summary['total_signals']}")
    print(f"  Samples per Signal       : {summary['samples_per_signal']}")
    print(f"  Signal Dimensions [X]    : {summary['shape']} (Expected: (17000, 100))")
    print(f"  Data Type                : {summary['dtype']}")
    print(f"  Shape Valid              : {'PASSED' if summary['shape_valid'] else 'FAILED'}")
    print(f"  Data Type Valid          : {'PASSED' if summary['dtype_valid'] else 'FAILED'}")
    print(f"  Missing (NaN) Values     : {summary['nan_count']}")
    print(f"  Infinite (Inf) Values    : {summary['inf_count']}")
    print(f"  Numerical Integrity      : {'PASSED (Zero NaN/Inf)' if summary['is_finite'] else 'FAILED'}")
    print(f"  Disturbance Classes      : {summary['num_classes']} classes")
    print(f"  Class Balance            : {'PASSED (Exactly 1000 signals/class)' if summary['is_balanced'] else 'UNBALANCED'}")
    print(f"  Global Signal Range      : [{summary['min_amplitude']:.4f}, {summary['max_amplitude']:.4f}]")
    print(f"  Global Mean +/- Std      : {summary['mean_amplitude']:.4f} +/- {summary['std_amplitude']:.4f}")
    print("=" * 65 + "\n")
