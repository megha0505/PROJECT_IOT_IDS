# # Trains an Isolation Forest model using only benign samples
# to model "normal" IoT network behavior for anomaly detection.
# Steps:
#   1) Load cleaned dataset
#   2) Filter benign traffic
#   3) Preprocess (OHE for categorical, MinMax scaling for numeric)
#   4) Train Isolation Forest
#   5) Save model & anomaly threshold

import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

# Project paths
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"; MODELS = RESULTS / "models"; METRICS = RESULTS / "metrics"
MODELS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)

NON_FEATURES = {"ts","ts_readable","label","detailed-label","scenario","LabelClean","LabelCoarse",
                "Unnamed: 0","target_binary","target_multiclass"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROC/"dataset_cleaned.parquet"))
    ap.add_argument("--contamination", type=float, default=0.01, help="IF expected attack rate")
    ap.add_argument("--model_name", default="iforest_benign")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if "target_binary" not in df.columns: raise SystemExit("Run prepare first.")

    # Benign only
    dfb = df[df["target_binary"]==0].copy()
    X = dfb.drop(columns=[c for c in NON_FEATURES if c in dfb.columns], errors="ignore")
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if num: X[num] = X[num].replace([np.inf,-np.inf], np.nan).fillna(0)
    for c in cat: X[c] = X[c].fillna("unknown").astype(str)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cat),
        ("num", MinMaxScaler(), num)
    ])
    Xt = pre.fit_transform(X)

    # Isolation Forest
    iforest = IsolationForest(
        n_estimators=200, contamination=args.contamination, random_state=42, n_jobs=-1
    )
    t0 = time.time(); iforest.fit(Xt); t1 = time.time()

    # Score distribution on benign for threshold (higher score = more normal)
    # Convert to anomaly score = -score_samples (higher = more anomalous)
    ben_scores = -iforest.score_samples(Xt)
    # threshold at 95th percentile of benign anomaly scores (tuneable)
    tau = float(np.percentile(ben_scores, 95))

    joblib.dump({"preprocessor": pre, "iforest": iforest, "tau": tau},
                MODELS / f"{args.model_name}.joblib")
    print(f"[DONE] Trained IsolationForest on Benign ({len(dfb):,} rows) in {t1-t0:.2f}s")
    print(f"[INFO] Saved: {MODELS / (args.model_name + '.joblib')}  (tau={tau:.4f})")

if __name__ == "__main__":
    main()
