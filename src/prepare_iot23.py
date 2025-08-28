# prepare_iot23.py

# This script preprocesses the IoT-23 dataset by:
#  - Mapping detailed attack labels to coarse categories
#  - Creating binary and multiclass target labels
#  - Dropping non-feature columns
#  - Saving the cleaned dataset in Parquet format
#  - Generating a summary and label mapping metadata

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Define important project paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
META = RESULTS / "metadata"
for p in (PROC, FIGS, META): p.mkdir(parents=True, exist_ok=True)

# Mapping detailed IoT-23 labels to broader categories
COARSE_MAP = {
    "Benign": "Benign",
    "PartOfAHorizontalPortScan": "Scanning",
    "PortScan": "Scanning",
    "DDoS": "DDoS",
    "Okiru": "Botnet",
    "Mirai": "Botnet",
    "Torii": "Botnet",
    "C&C": "C&C",
    "C&C-Heartbeat": "C&C",
    "C&C-FileDownload": "C&C",
    "FileDownload": "FileTransfer",
    "Attack": "Attack",  # generic catch-all if present
}

# Columns to drop that are not useful features
DROP_ALWAYS = {
    # non-features/IDs/timestamps often seen in iot23_combined
    "Unnamed: 0", "ts", "ts_readable", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p",
    "uid", "tunnel_parents", "LabelClean", "LabelCoarse", "detailed-label", "scenario",
}

def load_any_csv(path_hint: str | None):
    """
    Load dataset CSV from given path or automatically detect in data/raw/.
    Looks for 'cleaned_data.csv' or 'iot23_combined.csv'.
    """
    if path_hint:
        p = Path(path_hint)
        if not p.exists():
            raise SystemExit(f"Source not found: {p}")
        return pd.read_csv(p, low_memory=False), p

    # auto-detect in data/raw/
    cands = []
    for name in ("cleaned_data.csv", "iot23_combined.csv"):
        p = RAW / name
        if p.exists(): cands.append(p)
    if not cands:
        raise SystemExit(f"Place cleaned_data.csv or iot23_combined.csv in {RAW}")
    return pd.read_csv(cands[0], low_memory=False), cands[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None, help="Path to cleaned_data.csv or iot23_combined.csv")
    args = ap.parse_args()

    df, src_path = load_any_csv(args.source)

    # Ensure 'label' column exists and is clean
    if "label" not in df.columns:
        raise SystemExit("Expected a 'label' column in the CSV.")
    df["label"] = df["label"].astype(str).str.strip()

    # Map detailed labels to coarse categories
    def to_coarse(s: str) -> str:
        return COARSE_MAP.get(s, s)

    df["target_multiclass"] = df["label"].map(to_coarse).fillna(df["label"])
    df["target_binary"] = (df["target_multiclass"].astype(str) != "Benign").astype(int)

     # Remove unwanted columns
    drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Save parquet
    out_parquet = PROC / "dataset_cleaned.parquet"
    df.to_parquet(out_parquet, index=False)

    # Save summary &  mapping
    summary = {
        "source": str(src_path),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "n_multiclass": int(df["target_multiclass"].nunique()),
        "counts_sample": df["target_multiclass"].value_counts().head(20).to_dict(),
    }
    (META / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (META / "coarse_map.json").write_text(json.dumps(COARSE_MAP, indent=2), encoding="utf-8")

    # Print summary info to console
    print(f"[DONE] wrote {out_parquet}")
    print(f"[INFO] Multiclass classes: {summary['n_multiclass']}")
    print(f"[INFO] Top counts: {summary['counts_sample']}")

if __name__ == "__main__":
    main()

# end of file
