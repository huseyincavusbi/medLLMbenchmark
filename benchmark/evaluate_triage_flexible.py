#!/usr/bin/env python3
"""
Flexible Triage Evaluation (Accept All Tag Formats)
====================================================
Evaluates triage predictions by accepting ANY tag format that contains
a valid ESI level (1-5), ignoring exact instruction following.

This shows model accuracy independent of format compliance.
"""

import sys
import re
import json
from pathlib import Path
from collections import Counter

import pandas as pd


def parse_esi_level_flexible(text):
    """
    Extract ESI level from ANY tag format or text.
    
    Accepted formats:
    - <acuity>3</acuity>
    - <esi_level_3>
    - <ESI Level 3>
    - <3>
    - ESI Level: 3
    - Level 3
    - Just digit 1-5 at start of text
    """
    if pd.isna(text) or text == '':
        return None
    
    text = str(text)
    
    # Priority 1: Standard <acuity>X</acuity>
    match = re.search(r'<acuity>\s*([1-5])\s*</acuity>', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Priority 2: <esi_level_X> or <esi_level_X>
    match = re.search(r'<esi[_\s]*level[_\s]*([1-5])>', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Priority 3: <ESI Level X>
    match = re.search(r'<ESI\s+Level\s*([1-5])>', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Priority 4: Bare number tag <X>
    match = re.search(r'^<([1-5])>', text.strip())
    if match:
        return int(match.group(1))
    
    # Priority 5: ESI Level: X in text
    match = re.search(r'ESI\s*(?:Level|level)?\s*[:\-]?\s*([1-5])\b', text)
    if match:
        return int(match.group(1))
    
    # Priority 6: acuity in text
    match = re.search(r'acuity[:\s]*([1-5])\b', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Priority 7: Level X in text
    match = re.search(r'\bLevel\s*([1-5])\b', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Priority 8: Any digit 1-5 at start of cleaned text
    clean = re.sub(r'<[^>]+>', '', text).strip()
    match = re.match(r'^([1-5])\b', clean)
    if match:
        return int(match.group(1))
    
    # Priority 9: Last resort - first standalone digit 1-5
    match = re.search(r'\b([1-5])\b', text[:100])
    if match:
        return int(match.group(1))
    
    return None


def evaluate_triage_flexible(filepath):
    """Evaluate triage predictions with flexible parsing"""
    df = pd.read_csv(filepath)
    
    # Find prediction column and ground truth
    pred_col = [c for c in df.columns if 'triage_' in c and ('_general' in c or '_clinical' in c)][0]
    
    # Ground truth column can be 'triage', 'acuity', or 'triage_level'
    gt_col = None
    for col in ['triage', 'acuity', 'triage_level']:
        if col in df.columns:
            gt_col = col
            break
    
    if gt_col is None:
        raise ValueError(f"No ground truth column found in {df.columns.tolist()}")
    
    total = len(df)
    parsed = 0
    exact_match = 0
    within_1 = 0
    
    results = []
    
    for _, row in df.iterrows():
        pred = parse_esi_level_flexible(row[pred_col])
        gt = int(row[gt_col]) if pd.notna(row[gt_col]) else None
        
        if pred is not None and gt is not None:
            parsed += 1
            if pred == gt:
                exact_match += 1
                within_1 += 1
            elif abs(pred - gt) == 1:
                within_1 += 1
        
        results.append({'pred': pred, 'gt': gt})
    
    return {
        'total_cases': total,
        'parsed': parsed,
        'parsed_pct': parsed / total * 100,
        'unparsed': total - parsed,
        'exact_match': exact_match,
        'exact_accuracy': exact_match / parsed * 100 if parsed > 0 else 0,
        'within_1': within_1,
        'within_1_accuracy': within_1 / parsed * 100 if parsed > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Flexible triage evaluation")
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to results directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print('='*90)
    print('FLEXIBLE TRIAGE EVALUATION (Accept All Tag Formats)')
    print('='*90)
    
    all_metrics = {}
    
    # Evaluate general triage
    print('\n[TRIAGE GENERAL]')
    print('-'*80)
    print(f'{"Model":<25} {"Parsed":>10} {"Unparsed":>10} {"Exact":>12} {"Within-1":>12}')
    print('-'*80)
    
    for triage_file in sorted(results_dir.glob('triage_general_*.csv')):
        model = triage_file.stem.replace('triage_general_', '').rsplit('_', 2)[0]
        metrics = evaluate_triage_flexible(triage_file)
        all_metrics[f'{model}_general'] = metrics
        
        print(f'{model:<25} {metrics["parsed"]:>9} ({metrics["parsed_pct"]:>4.0f}%) '
              f'{metrics["unparsed"]:>6} {metrics["exact_accuracy"]:>11.1f}% '
              f'{metrics["within_1_accuracy"]:>11.1f}%')
    
    # Evaluate clinical triage
    print('\n[TRIAGE CLINICAL]')
    print('-'*80)
    print(f'{"Model":<25} {"Parsed":>10} {"Unparsed":>10} {"Exact":>12} {"Within-1":>12}')
    print('-'*80)
    
    for triage_file in sorted(results_dir.glob('triage_clinical_*.csv')):
        model = triage_file.stem.replace('triage_clinical_', '').rsplit('_', 2)[0]
        metrics = evaluate_triage_flexible(triage_file)
        all_metrics[f'{model}_clinical'] = metrics
        
        print(f'{model:<25} {metrics["parsed"]:>9} ({metrics["parsed_pct"]:>4.0f}%) '
              f'{metrics["unparsed"]:>6} {metrics["exact_accuracy"]:>11.1f}% '
              f'{metrics["within_1_accuracy"]:>11.1f}%')
    
    # Summary comparison
    print('\n' + '='*90)
    print('SUMMARY: EXACT ACCURACY (Flexible vs Strict Parsing)')
    print('='*90)
    print(f'{"Model":<25} {"General Flex":>15} {"Clinical Flex":>15}')
    print('-'*60)
    
    models = set(k.rsplit('_', 1)[0] for k in all_metrics.keys())
    for model in sorted(models):
        gen = all_metrics.get(f'{model}_general', {})
        clin = all_metrics.get(f'{model}_clinical', {})
        print(f'{model:<25} {gen.get("exact_accuracy", 0):>14.1f}% {clin.get("exact_accuracy", 0):>14.1f}%')
    
    # Save results
    output_file = results_dir / 'flexible_triage_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\nMetrics saved: {output_file}')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
