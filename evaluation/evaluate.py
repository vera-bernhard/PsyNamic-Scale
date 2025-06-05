#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: evaluate.py
Description: ...
Author: Vera Bernhard
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from confidenceinterval.bootstrap import bootstrap_ci

# STRIDE-Lab
def bootstrap(metric: callable, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Computes bootstrap confidence intervals."""
    print(f"Running bootstrap for {metric.__name__}...")
    score, ci = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=metric,
        confidence_level=0.95,
        n_resamples=9999,
        method="bootstrap_bca",
        random_state=42,
    )
    return score, ci


# STRIDE-Lab
def custom_f1(true_labels: np.ndarray, pred_labels: np.ndarray):
    return f1_score(true_labels, pred_labels, average="macro", zero_division=0)


# STRIDE-Lab
def custom_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray):
    return accuracy_score(true_labels, pred_labels)


# STRIDE-Lab
def custom_precision(true_labels: np.ndarray, pred_labels: np.ndarray):
    return precision_score(true_labels, pred_labels, average="macro", zero_division=0)


# STRIDE-Lab
def custom_recall(true_labels: np.ndarray, pred_labels: np.ndarray):
    return recall_score(true_labels, pred_labels, average="macro", zero_division=0)


# STRIDE-Lab
def get_metrics_ci(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes evaluation metrics with confidence intervals."""
    print("Computing F1 score and confidence interval...")
    f1_score, f1_ci = bootstrap(custom_f1, y_true, y_pred)
    print("Computing accuracy and confidence interval...")
    accuracy, acc_ci = bootstrap(custom_accuracy, y_true, y_pred)
    print("Computing precision and confidence interval...")
    precision, prec_ci = bootstrap(custom_precision, y_true, y_pred)
    print("Computing recall and confidence interval...")
    recall, recall_ci = bootstrap(custom_recall, y_true, y_pred)

    metric_dict = {
        "f1-macro": (f1_score, f1_ci),
        "accuracy": (accuracy, acc_ci),
        "precision": (precision, prec_ci),
        "recall": (recall, recall_ci),
    }
    return metric_dict


def get_performance_report(col_tru: str, col_pred: str, df: pd.DataFrame) -> dict:
    """Generates a performance report with metrics and confusion matrix."""

    # Check if columns exist
    if col_tru not in df.columns or col_pred not in df.columns:
        raise ValueError(
            f"Columns '{col_tru}' and '{col_pred}' must be present in the DataFrame. Available columns: {df.columns.tolist()}")

    # Count and remove empty predictions
    nr_empty_tru = df[col_pred].apply(lambda x: pd.isna(x) or x == "").sum()
    df = df[~df[col_pred].isna() & (df[col_pred] != "")]

    # Read labels from string to numpy arrays
    df.loc[:, col_tru] = df[col_tru].apply(
        lambda x: eval(x) if isinstance(x, str) else x)
    df.loc[:, col_pred] = df[col_pred].apply(
        lambda x: eval(x) if isinstance(x, str) else x)
    y_true = np.array(df[col_tru].tolist())
    y_pred = np.array(df[col_pred].tolist())

    metrics = get_metrics_ci(y_true, y_pred)

    # # Confusion matrix
    # conf_matrix = confusion_matrix(y_true, y_pred)

    report = {
        "metrics": metrics,
        #"confusion_matrix": conf_matrix.tolist(),
        "nr_empty_tru": nr_empty_tru,
    }

    return report
