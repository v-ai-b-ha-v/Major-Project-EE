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

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_feature_groups(df):
    """Group columns into their respective signal-processing families."""
    return {
        "fft": [c for c in df.columns if c.startswith("fft_")],
        "dwt": [c for c in df.columns if c.startswith("dwt_")],
        "stat": [c for c in df.columns if c.startswith("stat_")],
        "stft": [c for c in df.columns if c.startswith("stft_")],
        "wpt": [c for c in df.columns if c.startswith("wpt_")],
        "hht": [c for c in df.columns if c.startswith("hht_")],
    }


def define_configurations(feature_groups):
    """Define the 12 evaluation configurations (A through L)."""
    g = feature_groups
    return {
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


def build_xgb_classifier(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1):
    """Construct the XGBoost multi-class classifier with fixed hyperparameters."""
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
    Evaluate configurations A-L with a stratified hold-out test set.

    StandardScaler is placed inside a Pipeline for cross-validation, ensuring
    that each fold fits preprocessing only on that fold's training subset.
    Configurations are selected using training-partition CV macro-F1. The
    held-out test set is reserved for final performance estimation.

    Results from earlier evaluation code are not automatically comparable to
    results generated with this corrected evaluation protocol. Rerun the
    experiments before updating any reported metrics.
    """
    required = {"class_index", "class_label"}
    missing = required.difference(df_features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    os.makedirs(save_dir, exist_ok=True)
    models_dir = os.path.join(save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    y = df_features["class_index"].to_numpy()
    if pd.isna(y).any():
        raise ValueError("class_index contains missing values.")

    unique_indices = np.unique(y)
    class_names = [
        str(df_features.loc[df_features["class_index"] == idx, "class_label"].iloc[0])
        for idx in unique_indices
    ]
    if len(class_names) != 17:
        raise ValueError(f"Expected 17 classes, found {len(class_names)}.")

    feature_groups = get_feature_groups(df_features)
    configs = define_configurations(feature_groups)
    empty_configs = [key for key, cfg in configs.items() if not cfg["cols"]]
    if empty_configs:
        raise ValueError(f"No feature columns found for configurations: {empty_configs}")

    all_feature_cols = sorted({col for cfg in configs.values() for col in cfg["cols"]})
    all_features = df_features[all_feature_cols].to_numpy(dtype=float)
    if not np.isfinite(all_features).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    print("=" * 80)
    print("  STARTING XGBOOST COMPARATIVE EVALUATION ACROSS 12 CONFIGURATIONS")
    print("=" * 80)
    print(f"Total signals: {len(df_features)} across {len(class_names)} classes.")
    print(f"Train/Test Split: {1-test_size:.0%} train / {test_size:.0%} test [Stratified]")
    print(f"Cross-Validation: {cv_folds}-Fold Stratified CV on Training Partition")
    print("Configuration selection: training CV macro-F1; test set is not used.")
    print("-" * 80)

    indices = np.arange(len(df_features))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    y_train, y_test = y[train_idx], y[test_idx]

    results_records = []
    trained_models = {}
    test_predictions = {}
    confusion_matrices = {}
    scalers = {}
    cv_results = {}

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    }

    for cfg_key, cfg in configs.items():
        cols = cfg["cols"]
        print(
            f"\n>>> Running Configuration [{cfg_key}]: "
            f"{cfg['label']} ({len(cols)} features)..."
        )
        t0 = time.time()

        X_raw = df_features[cols].to_numpy(dtype=float)
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]

        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        cv_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", build_xgb_classifier(random_state=random_state)),
            ]
        )
        fold_scores = cross_validate(
            cv_pipeline,
            X_train_raw,
            y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )

        cv_acc_mean = float(np.mean(fold_scores["test_accuracy"]))
        cv_acc_std = float(np.std(fold_scores["test_accuracy"], ddof=1))
        cv_f1_mean = float(np.mean(fold_scores["test_macro_f1"]))
        cv_f1_std = float(np.std(fold_scores["test_macro_f1"], ddof=1))

        # Fit the final deployable model on the complete outer training split.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        clf = build_xgb_classifier(random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - t0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=unique_indices)

        print(
            f"    CV Accuracy: {cv_acc_mean*100:.2f}% "
            f"(SD {cv_acc_std*100:.2f}%)"
        )
        print(
            f"    CV Macro-F1: {cv_f1_mean*100:.2f}% "
            f"(SD {cv_f1_std*100:.2f}%)"
        )
        print(
            f"    Test Accuracy: {acc*100:.2f}% | Test Macro-F1: {f1*100:.2f}% "
            f"| Elapsed: {elapsed:.2f}s"
        )

        results_records.append(
            {
                "config_id": cfg_key,
                "config_label": cfg["label"],
                "num_features": len(cols),
                "train_cv_accuracy_mean": cv_acc_mean,
                "train_cv_accuracy_std": cv_acc_std,
                "train_cv_macro_f1_mean": cv_f1_mean,
                "train_cv_macro_f1_std": cv_f1_std,
                "accuracy": acc,
                "macro_precision": prec,
                "macro_recall": rec,
                "macro_f1": f1,
                "train_time_sec": elapsed,
            }
        )

        trained_models[cfg_key] = clf
        test_predictions[cfg_key] = y_pred
        confusion_matrices[cfg_key] = cm
        scalers[cfg_key] = scaler
        cv_results[cfg_key] = fold_scores

    comparison_df = pd.DataFrame(results_records)
    csv_path = os.path.join(save_dir, "comparative_model_results.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparative benchmark results table to: {csv_path}")

    # Select using training CV macro-F1 only. Deterministic tie-breaks prefer
    # fewer features, then the original A-L configuration order.
    ranking = comparison_df.copy()
    ranking["config_order"] = ranking["config_id"].map(
        {key: order for order, key in enumerate(configs)}
    )
    ranking = ranking.sort_values(
        by=["train_cv_macro_f1_mean", "num_features", "config_order"],
        ascending=[False, True, True],
        kind="stable",
    )
    best_cfg_key = str(ranking.iloc[0]["config_id"])
    print(
        f"\nBest configuration by training CV macro-F1: [{best_cfg_key}] "
        f"{configs[best_cfg_key]['label']}"
    )

    best_pred = test_predictions[best_cfg_key]
    report_dict = classification_report(
        y_test,
        best_pred,
        labels=unique_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(
        save_dir, f"per_class_report_best_config_{best_cfg_key}.csv"
    )
    report_df.to_csv(report_csv)
    print(f"Saved held-out test report to: {report_csv}")

    best_model_path = os.path.join(
        models_dir, f"xgboost_best_model_{best_cfg_key}.json"
    )
    trained_models[best_cfg_key].save_model(best_model_path)
    best_scaler_path = os.path.join(
        models_dir, f"scaler_best_config_{best_cfg_key}.joblib"
    )
    joblib.dump(scalers[best_cfg_key], best_scaler_path)
    print(f"Saved selected model to: {best_model_path}")
    print(f"Saved selected scaler to: {best_scaler_path}")

    # Preserve Config-L artifacts for existing inference/dashboard consumers.
    model_l_path = os.path.join(models_dir, "xgboost_model_L_all_features.json")
    scaler_l_path = os.path.join(models_dir, "scaler_config_L_all_features.joblib")
    trained_models["L"].save_model(model_l_path)
    joblib.dump(scalers["L"], scaler_l_path)
    print(f"Saved Config-L model to: {model_l_path}")
    print(f"Saved Config-L scaler to: {scaler_l_path}")

    return {
        "comparison_df": comparison_df,
        "best_config_id": best_cfg_key,
        "class_names": class_names,
        "confusion_matrices": confusion_matrices,
        "test_predictions": test_predictions,
        "trained_models": trained_models,
        "scalers": scalers,
        "cv_results": cv_results,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_test": y_test,
    }
