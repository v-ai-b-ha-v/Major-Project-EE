"""
XGBoost Classification & Comparative Analysis Module
===================================================
Project: Comparative Analysis of Signal Processing Techniques for
         Machine Learning-Based Power Quality Disturbance Classification
Module: src/classification.py

Evaluates 12 distinct feature configurations (A through L):
A. FFT only (7 features)
B. DWT only (20 features)
C. Statistical features only (7 features)
D. STFT only (8 features)
E. WPT only (16 features)
F. HHT only (8 features)
G. FFT + DWT (27 features)
H. FFT + DWT + Statistical features (34 features)
I. FFT + DWT + STFT (35 features)
J. FFT + DWT + WPT (43 features)
K. FFT + DWT + HHT (35 features)
L. All feature families combined (66 features)
"""

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def get_feature_groups(df):
    """
    Groups columns into their respective signal processing families.
    """
    return {
        "fft": [c for c in df.columns if c.startswith("fft_")],
        "dwt": [c for c in df.columns if c.startswith("dwt_")],
        "stat": [c for c in df.columns if c.startswith("stat_")],
        "stft": [c for c in df.columns if c.startswith("stft_")],
        "wpt": [c for c in df.columns if c.startswith("wpt_")],
        "hht": [c for c in df.columns if c.startswith("hht_")],
    }


def define_configurations(feature_groups):
    """
    Defines the 12 evaluation configurations (A through L) with descriptive labels.
    """
    g = feature_groups
    configs = {
        "A": {"label": "A. FFT only", "cols": g["fft"]},
        "B": {"label": "B. DWT only", "cols": g["dwt"]},
        "C": {"label": "C. Statistical only", "cols": g["stat"]},
        "D": {"label": "D. STFT only", "cols": g["stft"]},
        "E": {"label": "E. WPT only", "cols": g["wpt"]},
        "F": {"label": "F. HHT only", "cols": g["hht"]},
        "G": {"label": "G. FFT + DWT", "cols": g["fft"] + g["dwt"]},
        "H": {"label": "H. FFT + DWT + Stats", "cols": g["fft"] + g["dwt"] + g["stat"]},
        "I": {"label": "I. FFT + DWT + STFT", "cols": g["fft"] + g["dwt"] + g["stft"]},
        "J": {"label": "J. FFT + DWT + WPT", "cols": g["fft"] + g["dwt"] + g["wpt"]},
        "K": {"label": "K. FFT + DWT + HHT", "cols": g["fft"] + g["dwt"] + g["hht"]},
        "L": {
            "label": "L. All Features Combined",
            "cols": g["fft"] + g["dwt"] + g["stat"] + g["stft"] + g["wpt"] + g["hht"],
        },
    }
    return configs


def build_xgb_classifier(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Constructs an XGBoost multi-class classifier with reproducible hyperparameters.
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        objective="multi:softprob",
        num_class=17,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def run_comparative_experiments(
    df_features,
    test_size=0.20,
    random_state=42,
    cv_folds=5,
    save_dir="outputs",
):
    """
    Executes the full comparative benchmark across configurations A through L.

    Methodological safeguards:
    - Fixed train/test split (80% train, 20% test) stratified by class.
    - Zero data leakage: StandardScaler is fitted strictly on the training partition.
    - 5-fold stratified cross-validation on the training set for model validation.
    - Evaluation on held-out test set reporting: Accuracy, Macro-Precision, Macro-Recall, Macro-F1.
    - Confusion matrices and per-class reports saved to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    y = df_features["class_index"].values
    class_labels = df_features["class_label"].values
    
    # Unique ordered class names
    unique_indices = np.unique(y)
    class_names = [
        df_features[df_features["class_index"] == idx]["class_label"].iloc[0]
        for idx in unique_indices
    ]

    feature_groups = get_feature_groups(df_features)
    configs = define_configurations(feature_groups)

    results_records = []
    trained_models = {}
    test_predictions = {}
    confusion_matrices = {}
    scalers = {}

    print("=" * 80)
    print("  STARTING XGBOOST COMPARATIVE EVALUATION ACROSS 12 FEATURE CONFIGURATIONS  ")
    print("=" * 80)
    print(f"Total signals: {len(df_features)} across {len(class_names)} classes.")
    print(f"Train/Test Split: {1 - test_size:.0%} train ({int(len(df_features)*(1-test_size))}) / {test_size:.0%} test ({int(len(df_features)*test_size)}) [Stratified]")
    print(f"Cross-Validation: {cv_folds}-Fold Stratified CV on Training Partition")
    print("-" * 80)

    # Global train-test index split to maintain absolute consistency across all configs
    indices = np.arange(len(df_features))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    y_train, y_test = y[train_idx], y[test_idx]

    for cfg_key, cfg in configs.items():
        cols = cfg["cols"]
        label = cfg["label"]
        num_feats = len(cols)
        print(f"\n>>> Running Configuration [{cfg_key}]: {label} ({num_feats} features)...")
        t0 = time.time()

        X_raw = df_features[cols].values
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]

        # Fit Scaler ONLY on train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        scalers[cfg_key] = scaler

        # 5-Fold Stratified Cross-Validation on Training partition
        clf_cv = build_xgb_classifier(random_state=random_state)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(clf_cv, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Train final model on entire training partition
        clf = build_xgb_classifier(random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - t0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        print(f"    Train CV Acc: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
        print(f"    Test Acc    : {acc*100:.2f}% | Macro-F1: {f1*100:.2f}% | Elapsed: {elapsed:.2f}s")

        results_records.append({
            "config_id": cfg_key,
            "config_label": label,
            "num_features": num_feats,
            "train_cv_acc": cv_mean,
            "train_cv_std": cv_std,
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "train_time_sec": elapsed,
        })

        trained_models[cfg_key] = clf
        test_predictions[cfg_key] = y_pred
        confusion_matrices[cfg_key] = cm

    comparison_df = pd.DataFrame(results_records)
    csv_path = os.path.join(save_dir, "comparative_model_results.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparative benchmark results table to: {csv_path}")

    # Per-Class Report for Configuration L (All Features) and Best Configuration
    best_cfg_key = comparison_df.sort_values(by="macro_f1", ascending=False).iloc[0]["config_id"]
    print(f"\nBest Performing Configuration: [{best_cfg_key}] {configs[best_cfg_key]['label']}")

    best_pred = test_predictions[best_cfg_key]
    report_dict = classification_report(
        y_test, best_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(save_dir, f"per_class_report_best_config_{best_cfg_key}.csv")
    report_df.to_csv(report_csv)
    print(f"Saved per-class performance report for Best Configuration [{best_cfg_key}] to: {report_csv}")

    # Save model for best config
    best_model_path = os.path.join(save_dir, "models", f"xgboost_best_model_{best_cfg_key}.json")
    trained_models[best_cfg_key].save_model(best_model_path)
    print(f"Saved best model artifact to: {best_model_path}")

    # Also save Configuration L model if different
    if best_cfg_key != "L":
        model_l_path = os.path.join(save_dir, "models", "xgboost_model_L_all_features.json")
        trained_models["L"].save_model(model_l_path)
        print(f"Saved Config L model to: {model_l_path}")

    return {
        "comparison_df": comparison_df,
        "best_config_id": best_cfg_key,
        "class_names": class_names,
        "confusion_matrices": confusion_matrices,
        "test_predictions": test_predictions,
        "trained_models": trained_models,
        "scalers": scalers,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_test": y_test,
    }
