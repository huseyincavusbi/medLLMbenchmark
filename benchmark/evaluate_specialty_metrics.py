#!/usr/bin/env python3
"""
Evaluate Specialty Predictions with Human Ground Truth
=======================================================
Evaluates specialty prediction accuracy using clinician-validated ground truth (331 cases).

Metrics:
- Top-1 Accuracy: First prediction matches ground truth
- Any-in-Top-3: At least one of top 3 predictions matches ground truth
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


def exact_match(pred, ground_truth):
    """Check if prediction exactly matches ground truth (case-insensitive)"""
    if pd.isna(pred) or pd.isna(ground_truth):
        return False
    return str(pred).strip().lower() == str(ground_truth).strip().lower()


def fuzzy_match(pred, ground_truth):
    """Check if prediction fuzzy-matches ground truth"""
    if pd.isna(pred) or pd.isna(ground_truth):
        return False
    
    pred = str(pred).strip().lower()
    gt = str(ground_truth).strip().lower()
    
    # Exact match
    if pred == gt:
        return True
    
    # Handle common variations
    variations = {
        'orthopedic surgery': ['orthopedics', 'orthopaedics', 'ortho'],
        'general surgery': ['surgery', 'gen surg'],
        'internal medicine': ['medicine', 'im'],
        'emergency medicine': ['emergency', 'em', 'ed'],
        'obstetrics and gynecology': ['obgyn', 'ob/gyn', 'gynecology', 'ob-gyn'],
        'hematology/oncology': ['hematology', 'oncology', 'heme/onc'],
        'gastroenterology': ['gi', 'gastro'],
        'pulmonology': ['pulm', 'respiratory'],
        'nephrology': ['renal'],
        'cardiology': ['cardiac', 'cards'],
    }
    
    for standard, alts in variations.items():
        if gt == standard or gt in alts:
            if pred == standard or pred in alts:
                return True
    
    # Check if one contains the other
    if pred in gt or gt in pred:
        return True
    
    return False


def evaluate_file(filepath):
    """Evaluate a single specialty_human_gt file"""
    df = pd.read_csv(filepath)
    
    total = len(df)
    
    # Count matches
    top1_exact = sum(1 for _, row in df.iterrows() 
                     if exact_match(row.get('pred_1'), row.get('ground_truth')))
    top1_fuzzy = sum(1 for _, row in df.iterrows() 
                     if fuzzy_match(row.get('pred_1'), row.get('ground_truth')))
    
    any_in_top3_exact = sum(1 for _, row in df.iterrows() 
                            if any(exact_match(row.get(f'pred_{i}'), row.get('ground_truth')) 
                                   for i in [1, 2, 3]))
    any_in_top3_fuzzy = sum(1 for _, row in df.iterrows() 
                            if any(fuzzy_match(row.get(f'pred_{i}'), row.get('ground_truth')) 
                                   for i in [1, 2, 3]))
    
    # Count empty predictions
    empty_pred1 = df['pred_1'].isna().sum() if 'pred_1' in df.columns else 0
    
    return {
        'total_cases': total,
        'empty_predictions': int(empty_pred1),
        'top1_exact': top1_exact,
        'top1_exact_pct': top1_exact / total * 100 if total > 0 else 0,
        'top1_fuzzy': top1_fuzzy,
        'top1_fuzzy_pct': top1_fuzzy / total * 100 if total > 0 else 0,
        'any_top3_exact': any_in_top3_exact,
        'any_top3_exact_pct': any_in_top3_exact / total * 100 if total > 0 else 0,
        'any_top3_fuzzy': any_in_top3_fuzzy,
        'any_top3_fuzzy_pct': any_in_top3_fuzzy / total * 100 if total > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate specialty human ground truth predictions")
    parser.add_argument('--results-dir', type=str, default='results_sp_hmn',
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for metrics')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Find all specialty_human_gt files
    files = sorted(results_dir.glob("specialty_human_gt_*.csv"))
    
    if not files:
        print(f"No specialty_human_gt files found in {results_dir}")
        return 1
    
    print("=" * 90)
    print("SPECIALTY PREDICTION EVALUATION (Clinician Ground Truth)")
    print("=" * 90)
    print(f"Found {len(files)} model results\n")
    
    all_metrics = {}
    
    for f in files:
        # Extract model name from filename
        # specialty_human_gt_MODEL_clinical_TIMESTAMP.csv
        name = f.stem
        parts = name.replace('specialty_human_gt_', '').split('_clinical_')
        model_name = parts[0] if parts else name
        
        metrics = evaluate_file(f)
        all_metrics[model_name] = metrics
    
    # Print summary table
    print("-" * 90)
    print(f"{'Model':<35} {'Cases':>6} {'Empty':>6} {'Top1-E':>8} {'Top1-F':>8} {'Top3-E':>8} {'Top3-F':>8}")
    print("-" * 90)
    
    # Sort by top1_fuzzy_pct descending
    for model, m in sorted(all_metrics.items(), key=lambda x: -x[1]['top1_fuzzy_pct']):
        print(f"{model:<35} {m['total_cases']:>6} {m['empty_predictions']:>6} "
              f"{m['top1_exact_pct']:>7.1f}% {m['top1_fuzzy_pct']:>7.1f}% "
              f"{m['any_top3_exact_pct']:>7.1f}% {m['any_top3_fuzzy_pct']:>7.1f}%")
    
    print("-" * 90)
    print("\nLegend: Top1-E = Top-1 Exact Match, Top1-F = Top-1 Fuzzy Match")
    print("        Top3-E = Any-in-Top-3 Exact, Top3-F = Any-in-Top-3 Fuzzy")
    
    # Save metrics
    output_file = args.output or (results_dir / "specialty_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
