#!/usr/bin/env python3
"""
MIMIC-IV-Ext Benchmark Evaluation Script
=========================================
Evaluates benchmark results against ground truth and computes metrics.

Evaluates:
- Triage: Exact match, Within-1-level accuracy
- Specialty: Top-1, Top-3 accuracy
- Diagnosis: Top-1, Top-3 accuracy (semantic matching)

Usage:
    python evaluate_results.py [RUN_ID]
    python evaluate_results.py --latest
    python evaluate_results.py --compare RUN_ID1 RUN_ID2
"""

import sys
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


class ResultsEvaluator:
    """Evaluates benchmark predictions against ground truth"""
    
    def __init__(self, results_dir=None):
        self.results_dir = results_dir or Path(__file__).parent / "results"
        self.metrics = defaultdict(dict)
    
    def find_latest_run(self):
        """Find the most recent benchmark run"""
        metadata_files = list(self.results_dir.glob("run_metadata_*.json"))
        if not metadata_files:
            print("No benchmark runs found in results directory")
            return None
        
        # Sort by modification time
        latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest) as f:
            metadata = json.load(f)
        
        return metadata['run_id']
    
    def load_results(self, run_id):
        """Load all result files for a benchmark run"""
        results = {}
        
        task_names = [
            'triage_general',
            'triage_clinical',
            'diagnosis_specialty_general',
            'diagnosis_specialty_clinical'
        ]
        
        for task_name in task_names:
            result_file = self.results_dir / f"{task_name}_{run_id}.csv"
            if result_file.exists():
                results[task_name] = pd.read_csv(result_file)
                print(f"  Loaded {task_name}: {len(results[task_name])} cases")
            else:
                print(f"  Not found: {task_name}")
        
        return results
    
    def parse_acuity(self, text):
        """Extract ESI level from LLM output"""
        if pd.isna(text):
            return None
        
        text = str(text)
        
        # Check for errors
        if text.startswith("ERROR:"):
            return None
        
        # Try to extract from <acuity> tag
        match = re.search(r'<acuity>\s*(\d)\s*</acuity>', text, re.IGNORECASE)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 5:
                return level
        
        # Try to find "ESI level X" or "Level X"
        match = re.search(r'(?:ESI\s+)?[Ll]evel\s*[:\s]*(\d)', text)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 5:
                return level
        
        # Try to find any digit 1-5
        match = re.search(r'\b([1-5])\b', text)
        if match:
            return int(match.group(1))
        
        return None
    
    def evaluate_triage(self, df, prediction_col, task_name):
        """Evaluate triage predictions"""
        print(f"\n{'='*70}")
        print(f"Evaluating: {task_name}")
        print(f"{'='*70}")
        
        # Parse predictions
        df['pred_parsed'] = df[prediction_col].apply(self.parse_acuity)
        
        # Get ground truth
        if 'triage' not in df.columns:
            print("No ground truth 'triage' column found")
            return
        
        # Filter valid predictions
        valid_mask = df['pred_parsed'].notna() & df['triage'].notna()
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            print("No valid predictions to evaluate")
            return
        
        # Compute metrics
        total = len(valid_df)
        exact_match = (valid_df['pred_parsed'] == valid_df['triage']).sum()
        within_1 = (abs(valid_df['pred_parsed'] - valid_df['triage']) <= 1).sum()
        
        exact_pct = (exact_match / total) * 100
        within_1_pct = (within_1 / total) * 100
        
        # Store metrics
        self.metrics[task_name] = {
            'total_cases': len(df),
            'valid_predictions': total,
            'unparsed': len(df) - total,
            'exact_match': int(exact_match),
            'exact_accuracy': exact_pct,
            'within_1': int(within_1),
            'within_1_accuracy': within_1_pct
        }
        
        # Print results
        print(f"\nTotal cases: {len(df)}")
        print(f"Valid predictions: {total} ({(total/len(df)*100):.1f}%)")
        print(f"Unparsed/errors: {len(df) - total}")
        print()
        print(f"Exact Match Accuracy: {exact_match}/{total} = {exact_pct:.1f}%")
        print(f"Within-1-Level Accuracy: {within_1}/{total} = {within_1_pct:.1f}%")
        
        # Distribution of predictions
        print("\nPrediction Distribution:")
        pred_dist = valid_df['pred_parsed'].value_counts().sort_index()
        gt_dist = valid_df['triage'].value_counts().sort_index()
        
        print(f"{'ESI Level':<12} {'Ground Truth':<15} {'Predicted':<15}")
        print("-" * 42)
        for level in range(1, 6):
            gt_count = gt_dist.get(level, 0)
            pred_count = pred_dist.get(level, 0)
            print(f"Level {level:<6} {gt_count:<15} {pred_count:<15}")
        
        # Error analysis
        print("\nError Analysis (misclassified cases):")
        errors = valid_df[valid_df['pred_parsed'] != valid_df['triage']]
        if len(errors) > 0:
            error_types = errors.groupby(['triage', 'pred_parsed']).size().reset_index(name='count')
            error_types = error_types.sort_values('count', ascending=False)
            print(f"{'True ESI':<12} {'Predicted':<12} {'Count':<8}")
            print("-" * 32)
            for _, row in error_types.head(10).iterrows():
                print(f"Level {int(row['triage']):<6} Level {int(row['pred_parsed']):<6} {int(row['count']):<8}")
    
    def parse_specialty_diagnosis(self, text, tag_name):
        """Extract specialties or diagnoses from XML tags"""
        if pd.isna(text):
            return []
        
        text = str(text)
        
        # Check for errors
        if text.startswith("ERROR:"):
            return []
        
        results = []
        
        # Extract all tags
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            # Clean up the text
            cleaned = match.strip()
            # Remove numbering like "1.", "2.", etc.
            cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
            if cleaned:
                results.append(cleaned)
        
        # Limit to top 3
        return results[:3]
    
    def evaluate_specialty_diagnosis(self, df, prediction_col, task_name):
        """Evaluate specialty and diagnosis predictions"""
        print(f"\n{'='*70}")
        print(f"Evaluating: {task_name}")
        print(f"{'='*70}")
        
        # Parse specialties
        df['spec_parsed'] = df[prediction_col].apply(
            lambda x: self.parse_specialty_diagnosis(x, 'specialty')
        )
        
        # Parse diagnoses
        df['diag_parsed'] = df[prediction_col].apply(
            lambda x: self.parse_specialty_diagnosis(x, 'diagnosis')
        )
        
        # Check if we have parsed results
        valid_spec = df['spec_parsed'].apply(len) > 0
        valid_diag = df['diag_parsed'].apply(len) > 0
        
        print(f"\nTotal cases: {len(df)}")
        print(f"Valid specialty predictions: {valid_spec.sum()} ({(valid_spec.sum()/len(df)*100):.1f}%)")
        print(f"Valid diagnosis predictions: {valid_diag.sum()} ({(valid_diag.sum()/len(df)*100):.1f}%)")
        
        # Store metrics
        self.metrics[task_name] = {
            'total_cases': len(df),
            'valid_specialty': int(valid_spec.sum()),
            'valid_diagnosis': int(valid_diag.sum()),
            'unparsed': len(df) - max(valid_spec.sum(), valid_diag.sum())
        }
        
        # Note: Full evaluation requires ground truth specialty/diagnosis
        # and potentially LLM-as-judge for semantic matching
        print("\nNote: Full accuracy metrics require:")
        print("   - Ground truth specialty labels")
        print("   - Ground truth diagnosis labels")
        print("   - LLM-as-judge for semantic diagnosis matching")
        print("\nTo compute full metrics, run:")
        print("   python ../postprocess_specialty_prediction.py")
        print("   python ../postprocess_diagnosis_prediction.py")
        print("   python ../diagnosis_evaluation.py")
    
    def evaluate_run(self, run_id):
        """Evaluate all tasks for a benchmark run"""
        print("\n" + "=" * 70)
        print("BENCHMARK EVALUATION")
        print("=" * 70)
        print(f"Run ID: {run_id}\n")
        
        # Load results
        print("Loading results...")
        results = self.load_results(run_id)
        
        if not results:
            print("No results found for this run")
            return
        
        # Extract model name from run_id
        model_name = run_id.rsplit('_', 2)[0]
        
        # Evaluate triage tasks
        if 'triage_general' in results:
            self.evaluate_triage(
                results['triage_general'],
                f'triage_{model_name}_general',
                'triage_general'
            )
        
        if 'triage_clinical' in results:
            self.evaluate_triage(
                results['triage_clinical'],
                f'triage_{model_name}_clinical',
                'triage_clinical'
            )
        
        # Evaluate diagnosis/specialty tasks
        if 'diagnosis_specialty_general' in results:
            self.evaluate_specialty_diagnosis(
                results['diagnosis_specialty_general'],
                f'diag_spec_{model_name}_general',
                'diagnosis_specialty_general'
            )
        
        if 'diagnosis_specialty_clinical' in results:
            self.evaluate_specialty_diagnosis(
                results['diagnosis_specialty_clinical'],
                f'diag_spec_{model_name}_clinical',
                'diagnosis_specialty_clinical'
            )
        
        # Save metrics
        self.save_metrics(run_id)
        
        # Print summary
        self.print_summary()
    
    def save_metrics(self, run_id):
        """Save evaluation metrics to JSON"""
        metrics_file = self.results_dir / f"metrics_{run_id}.json"
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        import numpy as np
        def convert_to_native(obj):
            """Recursively convert numpy/pandas types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        metrics_dict = convert_to_native(dict(self.metrics))
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nMetrics saved: {metrics_file}")
    
    def print_summary(self):
        """Print summary of all metrics"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        # Triage metrics
        triage_tasks = [k for k in self.metrics.keys() if 'triage' in k]
        if triage_tasks:
            print("\nTriage Performance:")
            print("-" * 70)
            print(f"{'Task':<30} {'Exact':<15} {'Within-1':<15}")
            print("-" * 70)
            for task in triage_tasks:
                m = self.metrics[task]
                if 'exact_accuracy' in m:
                    print(f"{task:<30} {m['exact_accuracy']:>6.1f}%        {m['within_1_accuracy']:>6.1f}%")
        
        # Diagnosis/Specialty metrics
        diag_tasks = [k for k in self.metrics.keys() if 'diagnosis' in k]
        if diag_tasks:
            print("\nDiagnosis & Specialty Performance:")
            print("-" * 70)
            print(f"{'Task':<30} {'Valid Predictions':<20}")
            print("-" * 70)
            for task in diag_tasks:
                m = self.metrics[task]
                valid = m.get('valid_specialty', 0)
                total = m.get('total_cases', 0)
                if total > 0:
                    print(f"{task:<30} {valid}/{total} ({(valid/total*100):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIMIC-IV-Ext benchmark results"
    )
    parser.add_argument(
        'run_id',
        nargs='?',
        help='Run ID to evaluate (or use --latest)'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Evaluate the most recent benchmark run'
    )
    
    args = parser.parse_args()
    
    evaluator = ResultsEvaluator()
    
    # Determine which run to evaluate
    if args.latest:
        run_id = evaluator.find_latest_run()
        if not run_id:
            return 1
        print(f"Using latest run: {run_id}")
    elif args.run_id:
        run_id = args.run_id
    else:
        # Default to latest
        run_id = evaluator.find_latest_run()
        if not run_id:
            print("No runs found. Please specify a run_id or use --latest")
            return 1
        print(f"Using latest run: {run_id}")
    
    # Evaluate
    evaluator.evaluate_run(run_id)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
