
# Simulated realtime streaming for IoT IDS (enhanced logging)
# - Streams the processed parquet in chunks
# - Uses your trained model bundle (joblib) with preprocessor



import argparse, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

# Non-feature columns we always drop before preprocessing
NON_FEATURES = {
    "ts","ts_readable","label","detailed-label","scenario","LabelClean","LabelCoarse",
    "Unnamed: 0","target_binary","target_multiclass"
}

def _prep_like_training(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror training-time cleaning for features."""
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns], errors="ignore").copy()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if num:
        X[num] = X[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in cat:
        X[c] = X[c].fillna("unknown").astype(str)
    return X

def _top_labels(batch: pd.DataFrame, y_true_bin: np.ndarray, k: int = 3):
    """
    Return dominant attack labels/scenarios within this chunk.
    Only computed over rows where y_true_bin == 1 (attacks).
    """
    out = {
        "top_attack_label": None,
        "top_attack_count": 0,
        "top3_attack_labels": None,   # JSON string of dict {label: count}
        "top_scenario": None,
        "top_scenario_count": 0
    }
    if len(batch) == 0 or y_true_bin.sum() == 0:
        return out

    atk_mask = (y_true_bin == 1)
    # Prefer multiclass labels if available; else raw 'label'; else None
    label_col = "target_multiclass" if "target_multiclass" in batch.columns \
                else ("label" if "label" in batch.columns else None)
    if label_col:
        counts = batch.loc[atk_mask, label_col].value_counts()
        if not counts.empty:
            out["top_attack_label"] = str(counts.index[0])
            out["top_attack_count"] = int(counts.iloc[0])
            out["top3_attack_labels"] = json.dumps({str(k): int(v) for k, v in counts.head(k).to_dict().items()})

    if "scenario" in batch.columns:
        sc = batch.loc[atk_mask, "scenario"].value_counts()
        if not sc.empty:
            out["top_scenario"] = str(sc.index[0])
            out["top_scenario_count"] = int(sc.iloc[0])

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to processed dataset parquet")
    ap.add_argument("--model",   required=True, help="Path to trained joblib bundle")
    ap.add_argument("--chunk_rows", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between chunks")
    ap.add_argument("--score_threshold", type=float, default=0.5, help="Prob threshold for 'attack'")
    ap.add_argument("--out_csv", default=str(METRICS / "realtime_stream_results.csv"))
    ap.add_argument("--alerts_csv", default=str(METRICS / "realtime_stream_alerts.csv"),
                    help="Optional per-row alert snapshots (only predicted attacks)")
    ap.add_argument("--top_alerts_per_chunk", type=int, default=10,
                    help="Save at most N highest-score alerts per chunk (0 = disable)")
    args = ap.parse_args()

    # Load data
    df = pd.read_parquet(args.parquet)
    if "target_binary" not in df.columns:
        raise SystemExit("Expected 'target_binary' in dataset. Run prepare script first.")

    # Sort by time if available
    if "ts_readable" in df.columns:
        df = df.sort_values("ts_readable", kind="stable").reset_index(drop=True)
    elif "ts" in df.columns:
        df = df.sort_values("ts", kind="stable").reset_index(drop=True)

    # Load model bundle
    bundle = joblib.load(args.model)
    pre = bundle["preprocessor"]
    clf = bundle["classifier"]
    feature_names = bundle.get("feature_names")  # may be None

    # Prepare whole feature frame (to avoid re-drop every time)
    X_all = _prep_like_training(df)
    y_all = df["target_binary"].astype(int).to_numpy()

    # Helper to align to selected features if present
    def _transform_select(Xdf):
        Xt = pre.transform(Xdf)
        if feature_names is None:
            return Xt
        cols = pre.get_feature_names_out()
        mask = np.isin(cols, feature_names)
        return Xt[:, mask]

    # Chunking
    n = len(df)
    chunks = [(i, min(i + args.chunk_rows, n)) for i in range(0, n, args.chunk_rows)]
    print(f"[INFO] Streaming {n:,} rows in {len(chunks)} chunks (chunk_rows={args.chunk_rows}, sleep={args.sleep}s)")

    rows_out = []
    alerts_rows = [] if args.top_alerts_per_chunk > 0 else None

    for idx, (s, e) in enumerate(chunks, start=1):
        batch_df = df.iloc[s:e].copy()
        Xb = X_all.iloc[s:e]
        yb = y_all[s:e]

        Xt = _transform_select(Xb)

        # Scores & predictions
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xt)
            scores = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba.ravel()
        else:
            # Fallback to decision_function or direct predict
            if hasattr(clf, "decision_function"):
                raw = clf.decision_function(Xt)
                # min-max to [0,1]
                m, M = float(np.min(raw)), float(np.max(raw))
                scores = (raw - m) / (M - m + 1e-9)
            else:
                preds = clf.predict(Xt)
                scores = preds.astype(float)

        preds = (scores >= args.score_threshold).astype(int)

        # Metrics
        acc = float((preds == yb).mean())
        tp = int(((preds == 1) & (yb == 1)).sum())
        fp = int(((preds == 1) & (yb == 0)).sum())
        fn = int(((preds == 0) & (yb == 1)).sum())
        tn = int(((preds == 0) & (yb == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)

        # Dominant labels / scenarios in this chunk
        tops = _top_labels(batch_df, yb, k=3)

        # Aggregate row
        rows_out.append({
            "chunk": idx,
            "rows": int(e - s),
            "accuracy": acc,
            "n_attacks": int(yb.sum()),
            "n_predicted_attacks": int(preds.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "alert_rate": float(preds.mean()),
            "attack_rate": float(yb.mean()),
            **tops
        })

        # Optional: save top-N alert rows by score (pred=1)
        if alerts_rows is not None and preds.sum() > 0:
            topN = args.top_alerts_per_chunk
            sel = np.where(preds == 1)[0]
            if sel.size > 0:
                scores_sel = scores[sel]
                order = np.argsort(scores_sel)[::-1][:topN]
                picks = sel[order]
                snap = batch_df.iloc[picks].copy()
                snap.insert(0, "chunk", idx)
                snap.insert(1, "score", scores[picks])
                snap.insert(2, "pred", 1)
                snap.insert(3, "true", yb[picks])
                alerts_rows.append(snap)

        # Console summary per chunk
        print(f"[{idx:>4}/{len(chunks)}] rows={e-s:>5} acc={acc:.3f} "
              f"prec={precision:.3f} rec={recall:.3f} f1={f1:.3f} "
              f"| alerts={int(preds.sum())} attacks={int(yb.sum())} "
              f"| top_label={tops.get('top_attack_label')}")

        time.sleep(args.sleep)

    # Write per-chunk summary
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Wrote per-chunk results → {args.out_csv}")

    # Write per-row alerts snapshot
    if alerts_rows is not None and len(alerts_rows) > 0:
        alerts_df = pd.concat(alerts_rows, ignore_index=True)
        alerts_df.to_csv(args.alerts_csv, index=False)
        print(f"[DONE] Wrote alert snapshots → {args.alerts_csv}")
    elif alerts_rows is not None:
        print("[INFO] No predicted alerts to save at current threshold.")

if __name__ == "__main__":
    main()


"""cd C:\PROJECT_IOT_IDS
.\.venv\Scripts\activate

python .\src\realtime_stream_sim.py `
  --parquet .\data\processed\dataset_cleaned.parquet `
  --model   .\results\models\supervised_binary_baseline.joblib `
  --chunk_rows 400 `
  --sleep 0.25 `
  --score_threshold 0.5 `
  --out_csv .\results\metrics\realtime_stream_results.csv `
  --alerts_csv .\results\metrics\realtime_stream_alerts.csv `
  --top_alerts_per_chunk 10
"""