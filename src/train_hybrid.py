# Hybrid Intrusion Detection: Combines supervised Random Forest (RF) with
# an unsupervised Isolation Forest (IF) to detect both known and unseen IoT attacks.
# Steps:
#   1) Load cleaned dataset
#   2) Split into train/test (stratified)
#   3) Train supervised RF excluding one attack family (LOFO = Leave One Family Out)
#   4) Apply pre-trained IF model on benign data
#   5) Combine predictions (hybrid rule: attack if RF OR IF flags it)
#   6) Save detection rates and false positive rates
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Project paths
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"; MODELS = RESULTS / "models"; METRICS = RESULTS / "metrics"
MODELS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)

NON_FEATURES = {"ts","ts_readable","label","detailed-label","scenario","LabelClean","LabelCoarse",
                "Unnamed: 0","target_binary","target_multiclass"}

def _prep(df):
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns], errors="ignore")
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if num: X[num] = X[num].replace([np.inf,-np.inf], np.nan).fillna(0)
    for c in cat: X[c] = X[c].fillna("unknown").astype(str)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cat),
        ("num", MinMaxScaler(), num)
    ])
    return X, pre

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROC/"dataset_cleaned.parquet"))
    ap.add_argument("--tau", type=float, default=None, help="Override anomaly threshold; default loads from model")
    ap.add_argument("--iforest_joblib", default=str(MODELS/"iforest_benign.joblib"))
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--min_per_class", type=int, default=2)
    ap.add_argument("--families", nargs="*", default=None,
                    help='Families to LOFO (defaults to all non-Benign classes from target_multiclass)')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if "target_multiclass" not in df.columns or "target_binary" not in df.columns:
        raise SystemExit("Run prepare script first.")

    # Rare-class pruning
    y = df["target_multiclass"].astype("category")
    counts = y.value_counts(); rare = counts[counts < args.min_per_class].index.tolist()
    if rare:
        df = df[~df["target_multiclass"].isin(rare)].reset_index(drop=True)
        y = df["target_multiclass"].astype("category")

    # Families = all attack classes except Benign
    families = args.families or [c for c in y.unique().tolist() if str(c).lower() not in {"benign"}]

    # Set up supervised split once (consistent train/test across families)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    tr_idx, te_idx = next(sss.split(df, y))
    df_tr, df_te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

    # Load unsupervised model (IsolationForest on Benign)
    bundle = joblib.load(args.iforest_joblib)
    pre_if = bundle["preprocessor"]; iforest = bundle["iforest"]; tau = float(bundle["tau"] if args.tau is None else args.tau)

    rows = []
    for F in families:
        # Supervised training data: All - F (attacks excluding F) + Benign
        tr_keep = (df_tr["target_multiclass"] != F) | (df_tr["target_binary"] == 0)
        sup_tr = df_tr[tr_keep]
        sup_te = df_te[(df_te["target_multiclass"] == F) | (df_te["target_binary"] == 0)]

        # Build supervised features
        X_tr_raw, pre_sup = _prep(sup_tr)
        X_te_raw, _      = _prep(sup_te)
        Xtr = pre_sup.fit_transform(X_tr_raw); Xte = pre_sup.transform(X_te_raw)

        # Fast shortlist
        ytr = sup_tr["target_binary"].astype(int)  # binary: attack vs benign
        sel = SelectKBest(chi2, k=min(80, Xtr.shape[1])).fit(Xtr, ytr)
        Xtr_s = sel.transform(Xtr); Xte_s = sel.transform(Xte)

        # Supervised RF (binary)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
        t0 = time.time(); clf.fit(Xtr_s, ytr); _ = time.time()-t0

        # Supervised predictions on test (binary)
        yte_bin = sup_te["target_binary"].astype(int).to_numpy()
        sup_pred_attack = clf.predict(Xte_s).astype(int)

        # Unsupervised anomaly scores on test (use IF preprocessor)
        X_te_if_raw = sup_te.drop(columns=[c for c in NON_FEATURES if c in sup_te.columns], errors="ignore")
        num = X_te_if_raw.select_dtypes(include=[np.number]).columns.tolist()
        cat = [c for c in X_te_if_raw.columns if c not in num]
        if num: X_te_if_raw[num] = X_te_if_raw[num].replace([np.inf,-np.inf], np.nan).fillna(0)
        for c in cat: X_te_if_raw[c] = X_te_if_raw[c].fillna("unknown").astype(str)
        Xte_if = pre_if.transform(X_te_if_raw)
        anomaly = -iforest.score_samples(Xte_if)  # higher = more anomalous
        if_pred_attack = (anomaly > tau).astype(int)

        # Hybrid rule: attack if supervised_attack OR anomaly > tau
        hyb_pred_attack = np.logical_or(sup_pred_attack==1, if_pred_attack==1).astype(int)

        # Metrics: detection rate on family F (attack TPR on F), FPR on benign
        mask_F = (sup_te["target_multiclass"] == F)
        mask_B = (sup_te["target_binary"] == 0)
        def det_rate(mask, pred):
            y_true = yte_bin[mask]
            if y_true.size == 0: return np.nan
            return (pred[mask] & (y_true==1)).sum() / max((y_true==1).sum(), 1)

        sup_det = det_rate(mask_F, sup_pred_attack)
        ifo_det = det_rate(mask_F, if_pred_attack)
        hyb_det = det_rate(mask_F, hyb_pred_attack)
        benign_fpr_sup = (sup_pred_attack[mask_B]==1).mean() if mask_B.any() else np.nan
        benign_fpr_if  = (if_pred_attack[mask_B]==1).mean() if mask_B.any() else np.nan
        benign_fpr_hyb = (hyb_pred_attack[mask_B]==1).mean() if mask_B.any() else np.nan

        rows.append({
            "family": F,
            "sup_det_rate_on_F": sup_det,
            "iforest_det_rate_on_F": ifo_det,
            "hybrid_det_rate_on_F": hyb_det,
            "benign_fpr_sup": benign_fpr_sup,
            "benign_fpr_iforest": benign_fpr_if,
            "benign_fpr_hybrid": benign_fpr_hyb,
            "tau": tau
        })
        print(f"[{F}] sup={sup_det:.3f}  iforest={ifo_det:.3f}  hybrid={hyb_det:.3f}  | benign FPR hyb={benign_fpr_hyb:.3f}")

    out = pd.DataFrame(rows)
    out_csv = METRICS / "hybrid_lofo_summary.csv"
    out.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {out_csv}")

if __name__ == "__main__":
    main()
