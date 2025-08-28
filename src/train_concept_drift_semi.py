# src/train_concept_drift_semi.py
# Concept drift timeline with semi-supervised adaptation (pseudo-labeling).
# This file is self-contained

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

try:
    from river.drift import ADWIN
except Exception as e:
    raise SystemExit("River is required. Install with: pip install river") from e

# Paths & constants 
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"
MODELS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

# Columns that should never be used as features
NON_FEATURES = {
    "ts","ts_readable","label","detailed-label","scenario","LabelClean","LabelCoarse",
    "Unnamed: 0","target_binary","target_multiclass"
}

# Helper functions 

def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_readable" in df.columns:
        return df.sort_values("ts_readable", kind="stable").reset_index(drop=True)
    if "ts" in df.columns:
        return df.sort_values("ts", kind="stable").reset_index(drop=True)
    if "scenario" in df.columns:
        return df.sort_values("scenario", kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)

def _prep_features(df: pd.DataFrame):
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns], errors="ignore")
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if num:
        X[num] = X[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in cat:
        X[c] = X[c].fillna("unknown").astype(str)
    return X, num, cat

def _build_preprocessor(num, cat):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cat),
        ("num", MinMaxScaler(), num)
    ])

def _shortlist(pre_X, y, k_max=80):
    k = min(k_max, pre_X.shape[1])
    sel = SelectKBest(chi2, k=k).fit(pre_X, y)
    return sel, sel.transform(pre_X)

def _fit_rf(X, y):
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1,
        class_weight="balanced_subsample" if len(pd.Series(y).unique()) > 2 else None,
        random_state=42
    )
    clf.fit(X, y)
    return clf

def _evaluate(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return macro, p, r


# Main: Semi-supervised drift

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROC / "dataset_cleaned.parquet"))
    ap.add_argument("--target", choices=["target_binary","target_multiclass"], default="target_multiclass")
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--adwin_delta", type=float, default=0.002)
    ap.add_argument("--mode", choices=["retrain","sliding"], default="retrain",
                    help="retrain: use all past labeled + pseudo-labeled; sliding: last K chunks")
    ap.add_argument("--window_chunks", type=int, default=2)
    ap.add_argument("--conf_thresh", type=float, default=0.95,
                    help="Confidence threshold for accepting pseudo-labels")
    ap.add_argument("--max_pseudo_per_chunk", type=int, default=200_000,
                    help="Cap pseudo-labels per chunk (prevents drift runaway)")
    args = ap.parse_args()

    # Load & sort 
    df = pd.read_parquet(args.parquet)
    if args.target not in df.columns:
        raise SystemExit(f"{args.target} missing. Run prepare first.")
    df = _sort_df(df)

    # Build chunks
    n = len(df)
    chunks = [(i, min(i+args.chunk_rows, n)) for i in range(0, n, args.chunk_rows)]
    if len(chunks) < 2:
        raise SystemExit("Need at least 2 chunks (increase data or reduce --chunk_rows).")

    # Prepare features/labels 
    X_all, num, cat = _prep_features(df)
    y_all = df[args.target].astype("category")
    classes_ = y_all.cat.categories

    pre = _build_preprocessor(num, cat)

    # Fit encoders on first chunk only (avoid leakage)
    tr0_s, tr0_e = chunks[0]
    X0 = pre.fit_transform(X_all.iloc[tr0_s:tr0_e])
    y0 = y_all.iloc[tr0_s:tr0_e]

    sel0, X0s = _shortlist(X0, y0)
    clf = _fit_rf(X0s, y0)  # baseline supervised learner

    # Pools for labeled + pseudo-labeled indices
    labeled_idxs = np.arange(tr0_s, tr0_e)
    pseudo_pool = []  # list of np.ndarray indices

    adwin = ADWIN(delta=args.adwin_delta)
    rows = []

    for step, (cs, ce) in enumerate(chunks[1:], start=1):
        # Transform current chunk for inference
        Xt = pre.transform(X_all.iloc[cs:ce])
        Xt_s = sel0.transform(Xt)
        yt = y_all.iloc[cs:ce]

        # Predict & monitor drift on real labels
        y_pred = clf.predict(Xt_s)
        errors = (y_pred != yt).astype(int).to_numpy()
        drift = False
        for e in errors:
            adwin.update(float(e))
            if adwin.drift_detected:
                drift = True
                adwin = ADWIN(delta=args.adwin_delta)
                break

        # Metrics (on true labels)
        macro, prec, rec = _evaluate(yt, y_pred)
        rows.append({
            "step": step, "chunk_start": int(cs), "chunk_end": int(ce),
            "n_rows": int(ce - cs), "drift_flag": int(drift),
            "macro_f1": float(macro), "precision": float(prec), "recall": float(rec),
            "notes": f"semi:{args.mode}"
        })

        # Semi-supervised increment: add confident pseudo-labels from THIS chunk
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(Xt_s)
            conf = probs.max(axis=1)
            pseudo_mask = conf >= args.conf_thresh
            if pseudo_mask.any():
                sel = np.where(pseudo_mask)[0]
                if len(sel) > args.max_pseudo_per_chunk:
                    sel = sel[:args.max_pseudo_per_chunk]
                pseudo_idxs = np.arange(cs, ce)[sel]
                pseudo_pool.append(pseudo_idxs)

        #  On drift: retrain using labeled + pseudo-labeled indices
        if drift:
            if args.mode == "retrain":
                # All seen labeled chunks (0..current)
                base_idxs = np.arange(0, ce)
            else:
                # sliding: last K chunks BEFORE current chunk
                start = max(0, step - args.window_chunks)
                win = chunks[start:step]
                if not win:
                    win = [chunks[0]]
                base_idxs = np.concatenate([np.arange(s, e) for s, e in win])

            # Merge with pseudo-labeled pool (keep those inside base window)
            if pseudo_pool:
                pseudo_concat = np.concatenate(pseudo_pool)
                pseudo_concat = pseudo_concat[(pseudo_concat >= base_idxs.min()) & (pseudo_concat < base_idxs.max())]
                train_idxs = np.unique(np.concatenate([base_idxs, pseudo_concat]))
            else:
                train_idxs = base_idxs

            # Refit shortlist + classifier on new window
            Xw = pre.transform(X_all.iloc[train_idxs])
            yw = y_all.iloc[train_idxs]
            sel0, Xws = _shortlist(Xw, yw)
            clf = _fit_rf(Xws, yw)

    # Save timeline & final model
    out = pd.DataFrame(rows)
    out_csv = METRICS / "drift_timeline_semi.csv"
    out.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {out_csv}")
    joblib.dump(
        {"preprocessor": pre, "selector": sel0, "classifier": clf, "classes_": classes_.tolist()},
        MODELS / f"drift_{args.target}_semi_{args.mode}.joblib"
    )

if __name__ == "__main__":
    main()
