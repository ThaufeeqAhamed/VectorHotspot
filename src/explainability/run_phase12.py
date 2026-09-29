"""
Phase 12: SHAP Explainability, Dual-Disease Ecology & Uncertainty Calibration.
Master runner — executes all Phase 12 sub-tasks in order.

Tasks:
  1. TreeSHAP Feature Importance Analysis
  2. Dual-Disease Ecological Divergence
  3. Uncertainty Quantification & Calibration

Produces: outputs/explainability/phase12_report.txt
"""
import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs" / "explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

src = ROOT / "src" / "explainability"
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
    lines = [
        "=" * 60,
        "PHASE 12 EXPLAINABILITY REPORT",
        "VectorHotspot - SHAP, Ecology & Uncertainty",
        "=" * 60,
        "",
    ]

    # Task 1: SHAP
    lines.append("TASK 1: TreeSHAP Feature Importance Analysis")
    lines.append("  SHAP values computed for Dengue & Malaria (horizons 1-4)")
    lines.append("  Outputs: Global summary plots, dependence plots, SHAP values CSV")
    try:
        shap_files = list(OUT_DIR.glob("shap_*.png"))
        lines.append(f"  Generated {len(shap_files)} SHAP plots")
    except Exception as e:
        lines.append(f"  [error: {e}]")
    lines.append("")

    # Task 2: Ecology
    lines.append("TASK 2: Dual-Disease Ecological Divergence")
    try:
        with open(OUT_DIR / "ecological_divergence_summary.json") as f:
            eco = json.load(f)
        lines += [
            f"  Total hex-weeks analyzed      : {eco['total_hex_weeks']:,}",
            f"  Co-hotspots (both diseases)   : {eco['n_cohotspot']:,} ({eco['pct_cohotspot']}%)",
            f"  Dengue-only hotspots          : {eco['n_dengue_only']:,}",
            f"  Malaria-only hotspots         : {eco['n_malaria_only']:,}",
        ]
    except Exception as e:
        lines.append(f"  [not yet generated: {e}]")
    lines.append("")

    # Task 3: Calibration
    lines.append("TASK 3: Uncertainty Quantification & Calibration")
    try:
        with open(OUT_DIR / "uncertainty_calibration_summary.json") as f:
            cal = json.load(f)
        lines.append("  Conformal prediction intervals (90%, 95%) computed")
        lines.append("  Coverage per disease per horizon:")
        for disease in ["dengue", "malaria"]:
            lines.append(f"    [{disease.title()}]")
            for horizon in range(1, 5):
                h = cal[disease][f"lead{horizon}"]
                lines.append(f"      Lead-{horizon}: 90%={h['coverage_90%']:.4f}, 95%={h['coverage_95%']:.4f}")
    except Exception as e:
        lines.append(f"  [not yet generated: {e}]")
    lines.append("")

    # Execution log
    lines += ["", "EXECUTION LOG", "-" * 40]
    for task, info in RESULTS.items():
        status = info["status"]
        secs = info.get("elapsed_s", "?")
        err = f" - {info['error']}" if info.get("error") else ""
        lines.append(f"  [{status}] {task} ({secs}s){err}")

    report_text = "\n".join(lines)
    report_path = OUT_DIR / "phase12_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n[OK] Phase 12 report saved -> {report_path}")


def main():
    print("=" * 60)
    print("    PHASE 12: SHAP, Ecology & Uncertainty Calibration")
    print("=" * 60)
    print("\nNote: Task 1 (SHAP) is computationally expensive (~10-20 min per model).")
    print("      Tasks 2-3 are faster (~1-5 min each).\n")

    run_task("Task 1: TreeSHAP Feature Importance", "shap_feature_importance")
    run_task("Task 2: Dual-Disease Ecology", "dual_disease_ecology")
    run_task("Task 3: Uncertainty Calibration", "uncertainty_calibration")

    write_report()
    print("\n[OK] Phase 12 complete.")


if __name__ == "__main__":
    main()
