"""
Entry point: run ISR gating evaluation on a trained model.

Usage:
    python run_isr.py
    python run_isr.py --model_path ./outputs --output_dir ./outputs
"""

import argparse
import json
import os

from config import H_STAR_HYBRID, H_STAR_ISR, OUTPUT_DIR
from src.models.isr import ISRGating


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=OUTPUT_DIR)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--h_star_isr", type=float, default=H_STAR_ISR)
    p.add_argument("--h_star_hybrid", type=float, default=H_STAR_HYBRID)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    val_csv = os.path.join(args.model_path, "val_prob_matrix.csv")
    test_csv = os.path.join(args.model_path, "test_prob_matrix.csv")

    gating = ISRGating(
        model_path=args.model_path,
        val_csv=val_csv,
        test_csv=test_csv,
        h_star_isr=args.h_star_isr,
        h_star_hybrid=args.h_star_hybrid,
    )

    df_val = gating.compute_table("val")
    df_test = gating.compute_table("test")

    isr_cfg = gating.select_isr_threshold(df_val, h_target=args.h_star_isr)
    if isr_cfg is None:
        isr_cfg = {"threshold": 1.0}

    hybrid_cfg = gating.select_hybrid_params(df_val, isr_cfg, h_target=args.h_star_hybrid)

    with open(os.path.join(args.output_dir, "isr_config.json"), "w") as f:
        json.dump(isr_cfg, f, indent=2)
    if hybrid_cfg:
        with open(os.path.join(args.output_dir, "hybrid_config.json"), "w") as f:
            json.dump(hybrid_cfg, f, indent=2)

    results = gating.evaluate(df_test, isr_cfg, hybrid_cfg)
    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nFinal test results:")
    for mode, stats in results.items():
        print(f"  {mode}: coverage={stats.get('coverage', '-'):.1%}  acc={stats.get('accuracy', '-'):.1%}")

    # Save coverage–accuracy curve
    curve = gating.coverage_curve(df_test)
    curve.to_csv(os.path.join(args.output_dir, "isr_coverage_curve.csv"), index=False)
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
