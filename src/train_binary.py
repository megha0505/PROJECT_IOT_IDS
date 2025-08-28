# Trains a supervised binary classifier (Decision Tree) on the processed IoT dataset.
# Pipeline:
#   1. Load processed dataset
#   2. Preprocess (OHE for categorical, MinMax for numeric)
#   3. Feature selection (Chi² → RFE)
#   4. Train Decision Tree classifier
#   5. Evaluate and save model, metrics, and selected features

import time, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Define important paths
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"
for p in (MODELS, METRICS): p.mkdir(parents=True, exist_ok=True)

# Columns that should never be used as features
DROP_NON_FEATURES = {
    "ts", "ts_readable", "label", "detailed-label", "scenario",
    "LabelClean", "LabelCoarse", "Unnamed: 0",
    "target_binary", "target_multiclass",
}

def _save_eval(prefix: str, y_test, y_pred, clf, X_eval):
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).to_csv(METRICS / f"{prefix}_report.csv", index=True)
    labels_sorted = sorted(np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)])))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(METRICS / f"{prefix}_cm.csv")
    np.save(METRICS / f"{prefix}_y_test.npy", np.array(y_test))
    np.save(METRICS / f"{prefix}_y_pred.npy", np.array(y_pred))
    # Save predicted probabilities if available
    try:
        proba = clf.predict_proba(X_eval)
        if proba.ndim == 2 and proba.shape[1] == 2: proba = proba[:, 1]
        np.save(METRICS / f"{prefix}_y_scores.npy", proba)
    except Exception:
        pass

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROC / "dataset_cleaned.parquet"))
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--max_rows", type=int, default=0, help="Use at most N rows (stratified). 0 = all rows")
    args = ap.parse_args()

    print("[INFO] Loading dataset...", flush=True)
    df = pd.read_parquet(args.parquet)
    if "target_binary" not in df.columns:
        raise SystemExit("target_binary not found. Run prepare script first.")

    y = df["target_binary"].astype(int)

    # Optional: faster dev run
    if args.max_rows and len(df) > args.max_rows:
        frac = args.max_rows / len(df)
        df = (df.groupby(y, group_keys=False)
                .apply(lambda g: g.sample(max(1, int(len(g)*frac)), random_state=42))
                .reset_index(drop=True))
        y = df["target_binary"].astype(int)
        print(f"[INFO] Downsampled to {len(df):,} rows (stratified).", flush=True)

    # Remove non-feature columns
    to_drop = [c for c in DROP_NON_FEATURES if c in df.columns]
    X = df.drop(columns=to_drop, errors="ignore")

    # Separate numeric and categorical features
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
   
    # Handle missing values
    if num_cols: X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in cat_cols: X[c] = X[c].fillna("unknown").astype(str)

    print(f"[INFO] Rows={len(df):,}  Features(before)={X.shape[1]}  Target=target_binary", flush=True)

    # Stratified split into train/test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    (train_idx, test_idx,) = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if y_test.nunique() < 2:
        raise SystemExit("[ERROR] Test set has one class. Increase --test_size or use --max_rows to include more data.")

    # Preprocessing: OHE for categorical, MinMax for numeric
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", MinMaxScaler(), num_cols),
    ])

    print("[INFO] Fitting preprocessor...", flush=True)
    Xp_train = preproc.fit_transform(X_train)
    feat_names = preproc.get_feature_names_out()

    # Feature selection: Chi² filter, then RFE wrapper
    print("[INFO] Selecting features (Chi² -> RFE)...", flush=True)
    k = min(30, Xp_train.shape[1])
    chi = SelectKBest(chi2, k=k).fit(Xp_train, y_train); mask1 = chi.get_support()
    final = min(15, int(mask1.sum()) if mask1.sum() > 0 else k)
    base = Xp_train[:, mask1] if mask1.sum() > 0 else Xp_train
    rfe = RFE(DecisionTreeClassifier(max_depth=6, random_state=42), n_features_to_select=final).fit(base, y_train)

    mask_final = np.zeros(Xp_train.shape[1], dtype=bool)
    idx_base = np.where(mask1)[0] if mask1.sum() > 0 else np.arange(Xp_train.shape[1])
    mask_final[idx_base[rfe.get_support()]] = True
    selected = feat_names[mask_final].tolist()

    # Helper to transform and select only chosen features
    def _select(pre, Xdf, names):
        Xt = pre.transform(Xdf); cols = pre.get_feature_names_out()
        m = np.isin(cols, names); return Xt[:, m]

    Xtr_sel = _select(preproc, X_train, selected)
    Xte_sel = _select(preproc, X_test, selected)

    # Train Decision Tree
    print("[INFO] Training classifier...", flush=True)
    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    t0 = time.time(); clf.fit(Xtr_sel, y_train); train_time = time.time() - t0

    print("[INFO] Evaluating & saving artifacts...", flush=True)
    y_pred = clf.predict(Xte_sel)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    prefix = "supervised_binary_baseline"
    joblib.dump({"preprocessor": preproc, "feature_names": selected, "classifier": clf, "classes_": [0,1]},
                MODELS / f"{prefix}.joblib")
    (MODELS / "selected_features_target_binary.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    _save_eval(prefix, y_test, y_pred, clf, Xte_sel)

    # Update comparison CSV
    fc = METRICS / "final_comparison.csv"
    row = pd.DataFrame([{
        "model": prefix, "labeled_pct": 100, "zero_day_supported": 0, "drift_adaptation": 0,
        "accuracy": np.nan, "macro_f1": float(macro_f1), "fit_time_s": round(float(train_time), 3),
        "notes": "DecisionTree + Chi²→RFE; OHE+MinMax",
    }])
    if fc.exists(): 
        base = pd.read_csv(fc); base = base[base["model"] != prefix]; out = pd.concat([base, row], ignore_index=True)
    else:
        out = row
    out.to_csv(fc, index=False)

    # Print final summary
    print(f"target=target_binary macro-F1={macro_f1:.4f} train_time={train_time:.2f}s")
    print("model:", MODELS / f"{prefix}.joblib")
    print("report:", METRICS / f"{prefix}_report.csv")
    print("cm:", METRICS / f"{prefix}_cm.csv")
    print("y_test/y_pred:", METRICS / f"{prefix}_y_test.npy", METRICS / f"{prefix}_y_pred.npy")

if __name__ == "__main__":
    from sklearn.feature_selection import RFE
    from sklearn.metrics import f1_score
    main()

