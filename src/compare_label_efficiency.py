# This script compares semi-supervised learning (various % labeled) 
# against a fully supervised multiclass baseline:
#  1) Ensures supervised baseline model exists
#  2) Runs semi-supervised training at given label percentages
#  3) Extracts Macro-F1 scores from saved reports
#  4) Saves comparison CSV + generates line plot

import subprocess, sys, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGS = RESULTS / "figures" / "viz"
METRICS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def extract_macro_f1_from_report(report_csv: Path):
    """Read Macro-F1 from classification_report CSV (handles both row/col formats)."""
    if not report_csv.exists():
        return None
    rep = pd.read_csv(report_csv, index_col=0)
    # A) rows = labels, columns = metrics
    if "f1-score" in rep.columns and "macro avg" in rep.index:
        return float(rep.loc["macro avg", "f1-score"])
    # B) rows = metrics, columns = labels
    if "f1-score" in rep.index and "macro avg" in rep.columns:
        return float(rep.loc["f1-score", "macro avg"])
    return None

def ensure_supervised_baseline():
    """Run supervised multiclass baseline if artifacts are missing."""
    rep = METRICS / "supervised_multiclass_baseline_report.csv"
    if rep.exists():
        return
    print("[INFO] Running supervised multiclass baseline...")
    cmd = [sys.executable, "-m", "src.train_multiclass"]
    subprocess.run(cmd, check=True)

def run_semi(label_pct: int, confidence: float = 0.90, test_size: float = 0.30):
    """Run semi-supervised trainer for multiclass at given % labeled."""
    prefix = f"semi_multiclass_p{label_pct}"
    report_csv = METRICS / f"{prefix}_report.csv"
    if report_csv.exists():
        print(f"[INFO] Skipping run: {prefix} (already exists)")
        return
    print(f"[INFO] Running semi-supervised multiclass at {label_pct}% labeled...")
    cmd = [
        sys.executable, "-m", "src.train_semi_supervised",
        "--target", "target_multiclass",
        "--label_frac", str(label_pct / 100.0),
        "--confidence", str(confidence),
        "--test_size", str(test_size),
    ]
    subprocess.run(cmd, check=True)

def build_csv_and_plot(pcts):
    """Collect Macro-F1 and produce CSV + comparison plot."""
    rows = []

    # Semi-supervised points
    for p in pcts:
        rep = METRICS / f"semi_multiclass_p{p}_report.csv"
        f1 = extract_macro_f1_from_report(rep)
        rows.append({"series": "semi", "pct": float(p), "macro_f1": f1})

    # Supervised baseline (single point at 100%)
    sup_rep = METRICS / "supervised_multiclass_baseline_report.csv"
    sup_f1 = extract_macro_f1_from_report(sup_rep)
    rows.append({"series": "supervised", "pct": 100.0, "macro_f1": sup_f1})

    df = pd.DataFrame(rows)
    out_csv = METRICS / "label_efficiency_multiclass.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {out_csv}")

    # Plot
    plt.figure(figsize=(8,5))
    # Semi-supervised line
    semi_df = df[df["series"]=="semi"].sort_values("pct")
    plt.plot(semi_df["pct"], semi_df["macro_f1"], marker="o", label="Semi-supervised")

    # Supervised marker at 100%
    sup_df = df[df["series"]=="supervised"]
    if not sup_df["macro_f1"].isna().all():
        plt.plot(sup_df["pct"], sup_df["macro_f1"], marker="s", markersize=8,
                 linestyle="None", label="Fully supervised (100%)")

    plt.xlabel("% labeled")
    plt.ylabel("Macro-F1")
    plt.title("Label Efficiency — Multiclass (Semi-supervised vs Fully Supervised)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = FIGS / "label_efficiency_multiclass.png"
    plt.savefig(out_png, dpi=160)
    print(f"[INFO] Wrote {out_png}")
    plt.close()

def main():
    # You can change these if needed
    pcts = [10, 30, 50, 100]
    confidence = 0.90
    test_size = 0.30

    ensure_supervised_baseline()
    for p in pcts:
        run_semi(p, confidence=confidence, test_size=test_size)

    build_csv_and_plot(pcts)

if __name__ == "__main__":
    main()
