# Semi-supervised training with pseudo-labeling:
#  1) Load processed data
#  2) Preprocess (OHE + MinMax) and Chi² shortlist
#  3) Keep only a fraction of labels; train RF
#  4) Pseudo-label high-confidence unlabeled samples; re-train
#  5) Evaluate and log metrics & learning curve row

import time, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Project paths
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"
MODELS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

# Columns never used as features
NON_FEATURES = {
    "ts","ts_readable","label","detailed-label","scenario","LabelClean","LabelCoarse",
    "Unnamed: 0","target_binary","target_multiclass"
}

def _save_eval(prefix, y_test, y_pred, clf, Xte):
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).to_csv(METRICS / f"{prefix}_report.csv", index=True)

    labels_sorted = sorted(np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)])))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(METRICS / f"{prefix}_cm.csv")

    np.save(METRICS / f"{prefix}_y_test.npy", np.array(y_test))
    np.save(METRICS / f"{prefix}_y_pred.npy", np.array(y_pred))
    try:
        proba = clf.predict_proba(Xte); np.save(METRICS / f"{prefix}_y_scores.npy", proba)
    except Exception:
        pass
    return rep

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default=str(PROC / 'dataset_cleaned.parquet'))
    ap.add_argument('--target', choices=['target_binary','target_multiclass'], default='target_multiclass')
    ap.add_argument('--label_frac', type=float, default=0.3)    # keep this fraction of labels
    ap.add_argument('--confidence', type=float, default=0.9)    # pseudo-label threshold
    ap.add_argument('--test_size', type=float, default=0.3)
    ap.add_argument('--min_per_class', type=int, default=2, help='Drop classes with < this many samples')
    args = ap.parse_args()

    print("[INFO] Loading dataset...", flush=True)
    df = pd.read_parquet(args.parquet)
    if args.target not in df.columns:
        raise SystemExit("target not found. Run prepare script first.")

    # Build features/target
    y = df[args.target].astype('category')
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns], errors='ignore')

    # Basic cleaning
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if num: X[num] = X[num].replace([np.inf,-np.inf], np.nan).fillna(0)
    for c in cat: X[c] = X[c].fillna('unknown').astype(str)

    # Rare-class pruning (prevents stratified split errors)
    counts = y.value_counts()
    rare = counts[counts < args.min_per_class].index.tolist()
    if rare:
        print(f"[WARN] Dropping {len(rare)} rare classes with < {args.min_per_class} samples:")
        print("       " + ", ".join(map(str, rare)))
        keep_mask = ~y.isin(rare)
        X = X.loc[keep_mask].reset_index(drop=True)
        y = y.loc[keep_mask].reset_index(drop=True)

    if y.nunique() < 2:
        raise SystemExit("[ERROR] Only one class remains after pruning. Not suitable for training.")

    print(f"[INFO] Rows={len(X):,}  Features(before)={X.shape[1]}  Target={args.target}", flush=True)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    tr_idx, te_idx = next(sss.split(X, y))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    if yte.nunique() < 2:
        raise SystemExit("[ERROR] Test set has one class even after pruning. Increase --test_size.")

    # Preprocess & shortlist features
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=50), cat),
        ('num', MinMaxScaler(), num),
    ])
    Xtr_p = pre.fit_transform(Xtr); Xte_p = pre.transform(Xte)
    k = min(80, Xtr_p.shape[1])  # fast shortlist
    sel = SelectKBest(chi2, k=k).fit(Xtr_p, ytr)
    Xtr_s = sel.transform(Xtr_p); Xte_s = sel.transform(Xte_p)

    # Keep only a fraction of labels (semi-supervised)
    rng = np.random.RandomState(42)
    keep_mask_lab = rng.rand(len(ytr)) < args.label_frac
    if keep_mask_lab.sum() < max(10, ytr.nunique()*2):
        raise SystemExit("[ERROR] Too few labeled samples after masking; increase --label_frac.")

    # Train on labeled subset
    print(f"[INFO] Labeled in train: {keep_mask_lab.sum():,} / {len(ytr):,}", flush=True)
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1,
        class_weight='balanced_subsample' if args.target=='target_multiclass' else None,
        random_state=42
    )
    t0 = time.time()
    clf.fit(Xtr_s[keep_mask_lab], ytr[keep_mask_lab])
    t1 = time.time()

    # Pseudo-label unlabeled subset (high confidence)
    Umask = ~keep_mask_lab
    confident = np.array([], dtype=bool)
    if Umask.sum() > 0 and hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(Xtr_s[Umask])
        mx = proba.max(axis=1); yhat = clf.classes_[proba.argmax(axis=1)]
        confident = mx >= args.confidence
        n_conf = int(confident.sum())
        print(f"[INFO] Unlabeled in train: {Umask.sum():,}; confident pseudo-labels: {n_conf:,}", flush=True)
        if n_conf > 0:
            X_aug = np.vstack([Xtr_s[keep_mask_lab], Xtr_s[Umask][confident]])
            y_aug = pd.concat([ytr[keep_mask_lab],
                               pd.Series(yhat[confident], index=ytr[Umask][confident].index)],
                              axis=0)
            clf2 = RandomForestClassifier(
                n_estimators=200, n_jobs=-1,
                class_weight='balanced_subsample' if args.target=='target_multiclass' else None,
                random_state=123
            )
            clf2.fit(X_aug, y_aug)
            clf = clf2

    # Evaluate
    y_pred = clf.predict(Xte_s)
    macro = f1_score(yte, y_pred, average='macro', zero_division=0)

    # Save artifacts
    prefix = f"semi_{'binary' if args.target=='target_binary' else 'multiclass'}_p{int(args.label_frac*100)}"
    joblib.dump(
        {'preprocessor': pre, 'selector': sel, 'classifier': clf, 'classes_': y.cat.categories.tolist()},
        MODELS / f"{prefix}.joblib"
    )
    rep = _save_eval(prefix, yte, y_pred, clf, Xte_s)

    # Append one row to the semi-supervised curve
    row = pd.DataFrame([{
        'target': 'binary' if args.target=='target_binary' else 'multiclass',
        'label_frac': args.label_frac,
        'macro_f1': float(macro),
        'precision': rep.get('macro avg', {}).get('precision', np.nan) if isinstance(rep, dict) else np.nan,
        'recall':    rep.get('macro avg', {}).get('recall', np.nan)    if isinstance(rep, dict) else np.nan,
        'n_pseudo':  int(confident.sum()) if confident.size else 0
    }])
    curve = METRICS / 'semi_supervised_curve.csv'
    if curve.exists():
        base = pd.read_csv(curve); out = pd.concat([base, row], ignore_index=True)
    else:
        out = row
    out.to_csv(curve, index=False)

    print(f"[DONE] {prefix} macro-F1={macro:.4f}  fit_time_s={t1-t0:.2f}")

if __name__ == '__main__':
    main()

# end of file