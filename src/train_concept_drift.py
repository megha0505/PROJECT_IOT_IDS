# src/train_concept_drift.py
# Detects and handles concept drift in IoT intrusion detection
# using ADWIN (Adaptive Windowing).

#   1) Load and sort dataset chronologically
#   2) Split into sequential chunks (stream simulation)
#   3) Train initial model on first chunk
#   4) Monitor performance on new chunks
#   5) If drift is detected, retrain model (either on all past data or sliding window)
#   6) Save drift timeline and final model

import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

try:
    from river.drift import ADWIN
except Exception as e:
    raise SystemExit("River is required. Install with: pip install river") from e

# Project paths
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

def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer timestamp; else scenario; else leave order
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
    if num: X[num] = X[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in cat: X[c] = X[c].fillna("unknown").astype(str)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROC / "dataset_cleaned.parquet"))
    ap.add_argument("--target", choices=["target_binary","target_multiclass"], default="target_multiclass")
    ap.add_argument("--chunk_rows", type=int, default=200_000, help="Sequential chunk size (rows)")
    ap.add_argument("--adwin_delta", type=float, default=0.002, help="ADWIN sensitivity (lower = more sensitive)")
    ap.add_argument("--mode", choices=["retrain","sliding"], default="retrain",
                    help="retrain: train on all seen data; sliding: last K chunks")
    ap.add_argument("--window_chunks", type=int, default=2, help="K for sliding window retrain")
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

    # Split: first chunk = initial train, then stream next chunks as evaluation (A->B->C...)
    # Prepare features
    X_all, num, cat = _prep_features(df)
    y_all = df[args.target].astype("category")

    pre = _build_preprocessor(num, cat)
    # Fit on first train chunk only (avoid leakage)
    tr0_s, tr0_e = chunks[0]
    X0 = pre.fit_transform(X_all.iloc[tr0_s:tr0_e])
    y0 = y_all.iloc[tr0_s:tr0_e]

    sel0, X0s = _shortlist(X0, y0)
    clf = _fit_rf(X0s, y0)

    # For 'sliding' mode we keep buffer of last K chunk indices
    seen_chunks = [0]
    adwin = ADWIN(delta=args.adwin_delta)

    rows = []
    for step, (cs, ce) in enumerate(chunks[1:], start=1):
        # Transform current chunk
        Xt = pre.transform(X_all.iloc[cs:ce])
        Xt = sel0.transform(Xt)  # use same shortlist features
        yt = y_all.iloc[cs:ce]

        # Predict and update ADWIN on error stream (1 = error, 0 = correct)
        y_pred = clf.predict(Xt)
        errors = (y_pred != yt).astype(int).to_numpy()
        drift = False
        for e in errors:
            adwin.update(float(e))
            if adwin.drift_detected:
                drift = True
                # Reset flag to avoid cascading triggers inside same chunk
                adwin = ADWIN(delta=args.adwin_delta)
                break    # avoid multiple triggers within the same chunk

        macro, prec, rec = _evaluate(yt, y_pred)
        rows.append({
            "step": step,
            "chunk_start": int(cs),
            "chunk_end": int(ce),
            "n_rows": int(ce - cs),
            "drift_flag": int(drift),
            "macro_f1": float(macro),
            "precision": float(prec),
            "recall": float(rec),
            "notes": args.mode
        })

        # On drift: update model
        if drift:
            if args.mode == "retrain":
                # Train on ALL data up to current chunk (0..step)
                up_to = chunks[0:step+1]
                idxs = np.concatenate([np.arange(s, e) for s, e in up_to])
            else:
                # sliding: last K chunks (including current-1)
                start = max(0, step - args.window_chunks)
                win = chunks[start:step]  # train on recent chunks before this one
                if not win: win = [chunks[0]]
                idxs = np.concatenate([np.arange(s, e) for s, e in win])

            # Refit pre (same categories scale!)? Keep pre but refit selector + clf on new window
            Xw = pre.transform(X_all.iloc[idxs])
            yw = y_all.iloc[idxs]
            sel0, Xws = _shortlist(Xw, yw)  # update shortlist to new distribution
            clf = _fit_rf(Xws, yw)

        # Maintain seen chunks list (for optional debugging)
        seen_chunks.append(step)

    # Save timeline CSV
    out = pd.DataFrame(rows)
    out_csv = METRICS / "drift_timeline.csv"
    out.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {out_csv}")
    # Save latest model bundle (so you can reuse)
    joblib.dump({"preprocessor": pre, "selector": sel0, "classifier": clf, "classes_": y_all.cat.categories.tolist()},
                MODELS / f"drift_{args.target}_{args.mode}.joblib")

if __name__ == "__main__":
    main()
