import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_eval_artifacts(prefix: str, y_true, y_pred, clf, X_eval, metrics_dir: Path):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).to_csv(metrics_dir / f"{prefix}_report.csv")

    labels_sorted = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(metrics_dir / f"{prefix}_cm.csv")

    np.save(metrics_dir / f"{prefix}_y_test.npy", np.array(y_true))
    np.save(metrics_dir / f"{prefix}_y_pred.npy", np.array(y_pred))

    # Optional: save scores if available (binary/proba)
    try:
        proba = clf.predict_proba(X_eval)
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba = proba[:, 1]
        np.save(metrics_dir / f"{prefix}_y_scores.npy", proba)
    except Exception:
        pass


def append_final_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        base = pd.read_csv(csv_path)
        df = pd.concat([base, df], ignore_index=True)
    df.to_csv(csv_path, index=False)