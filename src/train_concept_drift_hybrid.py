# src/train_concept_drift_hybrid.py
# Concept drift timeline with a Hybrid model:
# Supervised RF + IsolationForest trained on benign behavior.


import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, IsolationForest

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
    # Chi^2 shortlist; keep up to k_max or total features if fewer.
    k = min(k_max, pre_X.shape[1])
    sel = SelectKBest(chi2, k=k).fit(pre_X, y)
    return sel, sel.transform(pre_X)

def _fit_rf(X, y):
    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced_subsample" if len(pd.Series(y).unique()) > 2 else None,
        random_state=42
    )
    clf.fit(X, y)
    return clf

def _evaluate(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return macro, p, r

def _is_attack_label(y_val: str) -> bool:
    # Treat anything not clearly benign as attack (works for binary & multiclass IoT-23 variants)
    return str(y_val).strip().lower() not in {"benign", "-   benign   -", "normal"}

def _find_attack_superclass(classes) -> object:
    # Try to find a coarse "Attack" / "Malicious" class in the categorical target
    for c in classes:
        n = str(c).strip().lower()
        if n in {"attack", "malicious"}:
            return c
    return None


# Main: Hybrid drift

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROC / "dataset_cleaned.parquet"))
    ap.add_argument("--target", choices=["target_binary","target_multiclass"], default="target_multiclass")
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--adwin_delta", type=float, default=0.002)
    ap.add_argument("--mode", choices=["retrain","sliding"], default="retrain",
                    help="retrain: expanding window; sliding: last K chunks")
    ap.add_argument("--window_chunks", type=int, default=2)
    ap.add_argument("--iforest_contamination", type=float, default=0.02,
                    help="Expected anomaly proportion for IsolationForest")
    ap.add_argument("--iforest_override", choices=["only_if_sup_benign","always_or"], default="only_if_sup_benign",
                    help="Fuse rule: only override when RF predicts benign (recommended) OR logical OR with iForest anomalies")
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
    attack_super = _find_attack_superclass(classes_)

    pre = _build_preprocessor(num, cat)

    # Fit encoders on first chunk (avoid leakage)
    tr0_s, tr0_e = chunks[0]
    X0_pre = pre.fit_transform(X_all.iloc[tr0_s:tr0_e])
    y0 = y_all.iloc[tr0_s:tr0_e]

    # Train supervised RF on shortlisted features
    sel0, X0_sel = _shortlist(X0_pre, y0)
    clf = _fit_rf(X0_sel, y0)

    # Train IsolationForest on BENIGN samples from first chunk (unsupervised benign model)
    benign_mask0 = ~y0.astype(str).map(_is_attack_label)
    if benign_mask0.sum() < 10:
        raise SystemExit("Not enough benign samples in the first chunk for IsolationForest.")
    iforest = IsolationForest(
        n_estimators=200,
        contamination=args.iforest_contamination,
        random_state=42,
        n_jobs=-1
    ).fit(X0_pre[benign_mask0])

    # Drift monitor
    adwin = ADWIN(delta=args.adwin_delta)
    rows = []

    for step, (cs, ce) in enumerate(chunks[1:], start=1):
        # Transform current chunk
        Xt_pre = pre.transform(X_all.iloc[cs:ce])   # full preprocessed features
        Xt_sel = sel0.transform(Xt_pre)             # shortlisted for RF
        yt = y_all.iloc[cs:ce]

        # Supervised RF prediction
        y_sup = clf.predict(Xt_sel)

        # iForest anomaly decision on preprocessed (non-shortlisted) features
        if_pred = iforest.predict(Xt_pre)  # -1 = anomaly (attack-like), 1 = normal (benign-like)

        # Fuse RF + iForest
        y_hyb = y_sup.copy()
        sup_is_benign = ~pd.Series(y_sup).astype(str).map(_is_attack_label).to_numpy()

        if args.iforest_override == "only_if_sup_benign":
            override_mask = (if_pred == -1) & sup_is_benign
        else:  # "always_or"
            override_mask = (if_pred == -1)

        # If we have a coarse "Attack" superclass, move overridden samples into it
        if attack_super is not None:
            # For binary this is usually the same positive class (1); for multiclass it's "Attack"
            if override_mask.any():
                y_hyb[override_mask] = attack_super
        else:
            # No superclass available:
            # If RF predicted benign and iForest says anomaly, we keep RF label (benign) but it will be counted as error.
            # (Alternative strategies could map to the most likely attack class if probabilities are available.)
            pass

        # Drift detection based on hybrid errors
        errors = (y_hyb != yt).astype(int).to_numpy()
        drift = False
        for e in errors:
            adwin.update(float(e))
            if adwin.drift_detected:
                drift = True
                adwin = ADWIN(delta=args.adwin_delta)
                break

        # Metrics on true labels
        macro, prec, rec = _evaluate(yt, y_hyb)
        rows.append({
            "step": step,
            "chunk_start": int(cs),
            "chunk_end": int(ce),
            "n_rows": int(ce - cs),
            "drift_flag": int(drift),
            "macro_f1": float(macro),
            "precision": float(prec),
            "recall": float(rec),
            "notes": f"hybrid:{args.mode}"
        })

        # On drift: update RF (and iForest on benign) using expanding or sliding window
        if drift:
            if args.mode == "retrain":
                idxs = np.arange(0, ce)  # all data up to current chunk end
            else:
                start = max(0, step - args.window_chunks)
                win = chunks[start:step]  # last K chunks BEFORE current
                if not win:
                    win = [chunks[0]]
                idxs = np.concatenate([np.arange(s, e) for s, e in win])

            # Refit RF shortlist + classifier on window
            Xw_pre = pre.transform(X_all.iloc[idxs])
            yw = y_all.iloc[idxs]
            sel0, Xw_sel = _shortlist(Xw_pre, yw)
            clf = _fit_rf(Xw_sel, yw)

            # Refit iForest on BENIGN-only window
            benign_mask_w = ~yw.astype(str).map(_is_attack_label)
            if benign_mask_w.sum() >= 10:
                iforest = IsolationForest(
                    n_estimators=200,
                    contamination=args.iforest_contamination,
                    random_state=42,
                    n_jobs=-1
                ).fit(Xw_pre[benign_mask_w])

    # Save timeline & final bundle
    out = pd.DataFrame(rows)
    out_csv = METRICS / "drift_timeline_hybrid.csv"
    out.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {out_csv}")

    joblib.dump(
        {"preprocessor": pre, "selector": sel0, "classifier": clf, "iforest": iforest, "classes_": list(classes_)},
        MODELS / f"drift_{args.target}_hybrid_{args.mode}.joblib"
    )
    print(f"[DONE] Saved model bundle: {MODELS / f'drift_{args.target}_hybrid_{args.mode}.joblib'}")

if __name__ == "__main__":
    main()

# end of file