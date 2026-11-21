"""Practical 5: comparing base classifiers with Voting, Stacking and AdaBoost.

Run from the project folder using: python main.py
The dataset ships with scikit-learn; no download or API key is needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"


def load_data():
    """Return X and y. In this exercise, 1 means malignant (positive class)."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # sklearn uses malignant=0, benign=1 by default; flip for intuitive recall.
    y = pd.Series((dataset.target == 0).astype(int), name="malignant")
    return X, y


def build_models():
    """Give every model independent, fresh estimators."""
    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    )
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))

    bases = [
        ("logistic", logistic),
        ("tree", tree),
        ("knn", knn),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Logistic Regression": clone(logistic),
        "Decision Tree": clone(tree),
        "KNN": clone(knn),
        # Hard voting = majority vote of three predicted class labels.
        "Voting (Hard)": VotingClassifier(
            estimators=[(name, clone(model)) for name, model in bases],
            voting="hard",
        ),
        # The meta-model trains on out-of-fold predictions of the base models.
        "Stacking": StackingClassifier(
            estimators=[(name, clone(model)) for name, model in bases],
            final_estimator=make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
            ),
            cv=cv,
            stack_method="predict_proba",
        ),
        # AdaBoost repeatedly reweights examples that earlier weak learners missed.
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            n_estimators=100,
            learning_rate=0.5,
            random_state=SEED,
        ),
    }
    return models


def evaluate(name, model, X_train, X_test, y_train, y_test, cv):
    """Evaluate on training CV and a held-out test set (never used for fitting)."""
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    predictions = model.predict(X_test)

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "CV F1 mean": cv_f1.mean(),
        "CV F1 std": cv_f1.std(),
        "Fit seconds": fit_seconds,
    }

    # Hard voting exposes only predicted labels, not continuous probability scores.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        scores["ROC-AUC"] = roc_auc_score(y_test, probabilities)
    else:
        probabilities = None
        scores["ROC-AUC"] = np.nan

    report = classification_report(
        y_test, predictions,
        target_names=["Benign (0)", "Malignant (1)"],
        digits=3, zero_division=0,
    )
    return scores, predictions, probabilities, report


def save_class_balance(y_train, y_test):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(2)
    w = 0.32
    train = [(y_train == k).sum() for k in (0, 1)]
    test = [(y_test == k).sum() for k in (0, 1)]
    bars_a = ax.bar(x - w / 2, train, w, label="Training set", color="#3769a4")
    bars_b = ax.bar(x + w / 2, test, w, label="Test set", color="#26a69a")
    ax.bar_label(bars_a, padding=3)
    ax.bar_label(bars_b, padding=3)
    ax.set_xticks(x, ["Benign", "Malignant"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Class balance after stratified 80/20 split", pad=14)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "class_balance.png", dpi=180)
    plt.close(fig)


def save_comparison(df):
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    data = df.set_index("Model")
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(data))
    width = 0.19
    colors = ["#275d9f", "#42a6a0", "#f0ae45", "#a77ac8"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + (i - 1.5) * width, data[metric], width,
               label=metric, color=color)
    ax.set_xticks(x, data.index, rotation=18, ha="right")
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score on held-out test set")
    ax.set_title("Base classifiers vs ensemble classifiers", pad=14)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.21), frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(OUTPUTS / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices(y_test, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.9))
    for ax, (name, predicted) in zip(axes.flat, predictions.items()):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, predicted, labels=[0, 1]),
            display_labels=["Benign", "Malignant"],
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    fig.suptitle("Confusion matrices — same 114 test samples", fontweight="bold", y=0.998)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(y_test, probabilities, df):
    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    for name, probs in probabilities.items():
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        area = df.loc[df.Model == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, linewidth=2.1, label=f"{name} (AUC {area:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set", pad=13)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "roc_curves.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = build_models()

    print("PRACTICAL 5 — ENSEMBLE LEARNING AND PERFORMANCE EVALUATION")
    print("=" * 66)
    print("Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn")
    print(f"Rows: {len(X)} | Features: {X.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
    print("Positive class: malignant (1) | Random seed: 42 | Stratified split: 80/20")
    print("Training-only cross-validation: 5 stratified folds\n")

    rows, predictions, probabilities, reports = [], {}, {}, {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        scores, preds, probs, report = evaluate(
            name, model, X_train, X_test, y_train, y_test, cv
        )
        rows.append(scores)
        predictions[name] = preds
        probabilities[name] = probs
        reports[name] = report

    df = pd.DataFrame(rows)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1 mean", "CV F1 std"]
    print("\nTEST SET RESULTS (higher is better)")
    print("-" * 90)
    print(df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="N/A"))
    print("\nN/A for hard voting AUC: the majority-vote output has no continuous score.")

    winners = df.loc[df.F1 == df.F1.max(), "Model"].tolist()
    print("\nHighest test F1:", ", ".join(winners), f"({df.F1.max():.3f})")
    print("This is a small held-out set: a few cases can change the ranking.")

    df.to_csv(RESULTS / "model_scores.csv", index=False, float_format="%.6f")
    with (RESULTS / "classification_reports.txt").open("w", encoding="utf-8") as f:
        for name, report in reports.items():
            f.write(f"{name}\n{'=' * len(name)}\n{report}\n")
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic), sklearn built-in",
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_malignant": int(y_test.sum()),
        "test_benign": int((y_test == 0).sum()),
        "positive_class": "malignant (1)",
        "seed": SEED,
        "cross_validation": "StratifiedKFold, 5 splits, shuffled, training set only",
        "best_test_f1_models": winners,
    }
    (RESULTS / "experiment_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_class_balance(y_train, y_test)
    save_comparison(df)
    save_confusion_matrices(y_test, predictions)
    save_roc_curves(y_test, probabilities, df)
    print("\nSaved result tables in results/ and four figures in outputs/.")


if __name__ == "__main__":
    main()