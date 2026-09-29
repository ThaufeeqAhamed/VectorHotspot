"""
Phase 11: Rigorous Evaluation, Ablation Studies & Ground-Truth Validation.
Master runner — executes all Phase 11 sub-tasks in order.

Tasks:
  1. Future Hotspot Verification (Phase 10)
  2. National Spatial Holdout Validation
  3. Ablation 1: Spatial Disaggregation Methodology
  4. Ablation 2: Feature Engineering (Spatial Neighbors)
  5. Ablation 3: Environmental Signals (Weather)

Produces: outputs/evaluation/phase11_report.txt
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

src = ROOT / "src" / "evaluation"
sys.path.insert(0, str(src))

RESULTS = {}

def run_task(name, module_name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        RESULTS[name] = {"status": "SUCCESS", "elapsed_s": round(elapsed, 1)}
        print(f"\n  [OK] {name} completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        RESULTS[name] = {"status": "FAILED", "error": str(e), "elapsed_s": round(elapsed, 1)}
        print(f"\n  [FAIL] {name} FAILED: {e}")

def write_report():
    import pandas as pd

    lines = [
        "=" * 60,
        "PHASE 11 EVALUATION REPORT",
        "VectorHotspot — Rigorous Evaluation & Ablation Studies",
        "=" * 60,
        "",
    ]

    # Task 1 already done (Phase 10 metrics)
    lines += [
        "TASK 1: Future Hotspot Verification",
        "(Completed in Phase 10 — metrics from hotspot_fusion_evaluation_metrics.csv)",
    ]
    try:
        df = pd.read_csv(ROOT / "outputs" / "tables" / "hotspot_fusion_evaluation_metrics.csv")
        lines.append(df.to_string(index=False))
    except Exception as e:
        lines.append(f"  [could not load: {e}]")
    lines.append("")

    # Task 2: National Holdout
    lines.append("TASK 2: National Spatial Holdout Validation (108 districts, 28 states)")
    try:
        with open(OUT_DIR / "spatial_holdout_summary.json") as f:
            ho = json.load(f)
        for disease, stats in ho.items():
            lines += [
                f"  [{disease.title()}]",
                f"    Mean Spearman rho             : {stats['mean_spearman_rho']}",
                f"    % weeks significant (p<0.05)  : {stats['pct_weeks_significant']}%",
                f"    Hotspot detection rate        : {stats['hotspot_detection_rate']}",
            ]
    except Exception as e:
        lines.append(f"  [not yet generated: {e}]")
    lines.append("")

    # Task 3: Ablation 1
    lines.append("TASK 3: Ablation 1 — Spatial Disaggregation Methodology")
    try:
        df = pd.read_csv(OUT_DIR / "ablation1_summary.csv")
        lines.append(df.to_string(index=False))
    except Exception as e:
        lines.append(f"  [not yet generated: {e}]")
    lines.append("")

    # Task 4: Ablation 2
    lines.append("TASK 4: Ablation 2 — Feature Engineering (Spatial Neighbors)")
    try:
        df = pd.read_csv(OUT_DIR / "ablation2_feature_engineering.csv")
        summary = df.groupby(["disease", "variant"])[["r2", "rmse", "iou"]].mean().round(4)
        lines.append(summary.reset_index().to_string(index=False))
    except Exception as e:
        lines.append(f"  [not yet generated: {e}]")
    lines.append("")

    # Task 5: Ablation 3
    lines.append("TASK 5: Ablation 3 — Environmental Signals (Weather)")
    try:
        df = pd.read_csv(OUT_DIR / "ablation3_weather_signals.csv")
        summary = df.groupby(["disease", "variant"])[["r2", "rmse", "iou"]].mean().round(4)
        lines.append(summary.reset_index().to_string(index=False))
    except Exception as e:
        lines.append(f"  [not yet generated: {e}]")
    lines.append("")

    # Execution log
    lines += ["", "EXECUTION LOG", "-" * 40]
    for task, info in RESULTS.items():
        status = info["status"]
        secs   = info.get("elapsed_s", "?")
        err    = f" — {info['error']}" if info.get("error") else ""
        lines.append(f"  [{status}] {task} ({secs}s){err}")

    report_text = "\n".join(lines)
    report_path = OUT_DIR / "phase11_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n[OK] Phase 11 report saved -> {report_path}")

def main():
    print("=" * 60)
    print("    PHASE 11: Rigorous Evaluation & Ablation Studies      ")
    print("=" * 60)
    print("\nNote: Task 1 (Future Hotspot Verification) is pre-computed (Phase 10).")
    print("      Running Tasks 2–5 now...\n")

    run_task("Task 2: National Spatial Holdout Validation", "spatial_holdout_validation")
    run_task("Task 3: Ablation 1 — Spatial Method",     "ablation_spatial_methodology")
    run_task("Task 4: Ablation 2 — Feature Engineering","ablation_feature_engineering")
    run_task("Task 5: Ablation 3 — Weather Signals",    "ablation_weather_signals")

    write_report()
    print("\n[OK] Phase 11 complete.")

if __name__ == "__main__":
    main()
