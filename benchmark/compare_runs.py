#!/usr/bin/env python3
"""
MIMIC-IV-Ext Benchmark Comparison Tool
=======================================
Compare results from multiple benchmark runs.

Usage:
    python compare_runs.py RUN_ID1 RUN_ID2 [RUN_ID3 ...]
    python compare_runs.py --all
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


class BenchmarkComparator:
    """Compare multiple benchmark runs"""
    
    def __init__(self, results_dir=None):
        self.results_dir = results_dir or Path(__file__).parent / "results"
    
    def list_all_runs(self):
        """List all available benchmark runs"""
        metadata_files = list(self.results_dir.glob("run_metadata_*.json"))
        
        runs = []
        for meta_file in metadata_files:
            with open(meta_file) as f:
                metadata = json.load(f)
                runs.append({
                    'run_id': metadata['run_id'],
                    'model_name': metadata['model_name'],
                    'timestamp': metadata['timestamp'],
                    'total_time': metadata['total_time_seconds'],
                    'test_mode': metadata.get('test_mode', False),
                    'num_cases': metadata.get('num_cases', 'all')
                })
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return runs
    
    def load_metrics(self, run_id):
        """Load metrics for a run"""
        metrics_file = self.results_dir / f"metrics_{run_id}.json"
        
        if not metrics_file.exists():
            return None
        
        with open(metrics_file) as f:
            return json.load(f)
    
    def compare_runs(self, run_ids):
        """Compare multiple runs"""
        print("\n" + "=" * 80)
        print("BENCHMARK RUN COMPARISON")
        print("=" * 80)
        
        # Load all metrics
        all_metrics = {}
        for run_id in run_ids:
            metrics = self.load_metrics(run_id)
            if metrics:
                all_metrics[run_id] = metrics
            else:
                print(f"No metrics found for: {run_id}")
        
        if not all_metrics:
            print("No metrics to compare")
            return
        
        # Extract model names from run_ids
        model_names = {run_id: run_id.rsplit('_', 2)[0] for run_id in all_metrics.keys()}
        
        # Compare triage tasks
        self.compare_triage_tasks(all_metrics, model_names)
        
        # Compare diagnosis/specialty tasks
        self.compare_diagnosis_tasks(all_metrics, model_names)
        
        # Overall summary
        self.print_overall_summary(all_metrics, model_names)
    
    def compare_triage_tasks(self, all_metrics, model_names):
        """Compare triage performance across runs"""
        print("\n" + "=" * 80)
        print("TRIAGE PERFORMANCE COMPARISON")
        print("=" * 80)
        
        task_types = ['triage_general', 'triage_clinical']
        
        for task_type in task_types:
            print(f"\n{task_type.replace('_', ' ').title()}:")
            print("-" * 80)
            
            # Prepare comparison data
            comparison = []
            for run_id, metrics in all_metrics.items():
                if task_type in metrics:
                    m = metrics[task_type]
                    if 'exact_accuracy' in m:
                        comparison.append({
                            'model': model_names[run_id],
                            'run_id': run_id[:30] + '...' if len(run_id) > 30 else run_id,
                            'exact': m['exact_accuracy'],
                            'within_1': m['within_1_accuracy'],
                            'valid': m['valid_predictions'],
                            'total': m['total_cases']
                        })
            
            if not comparison:
                print("  No data available")
                continue
            
            # Print comparison table
            print(f"{'Model':<20} {'Exact Accuracy':<18} {'Within-1 Accuracy':<18} {'Valid/Total':<15}")
            print("-" * 80)
            
            # Sort by within-1 accuracy (most important metric)
            comparison.sort(key=lambda x: x['within_1'], reverse=True)
            
            for c in comparison:
                print(f"{c['model']:<20} {c['exact']:>6.1f}%            "
                      f"{c['within_1']:>6.1f}%            {c['valid']}/{c['total']}")
            
            # Highlight best performer
            if comparison:
                best = comparison[0]
                print(f"\n🏆 Best: {best['model']} - {best['within_1']:.1f}% within-1 accuracy")
    
    def compare_diagnosis_tasks(self, all_metrics, model_names):
        """Compare diagnosis/specialty performance across runs"""
        print("\n" + "=" * 80)
        print("DIAGNOSIS & SPECIALTY PERFORMANCE COMPARISON")
        print("=" * 80)
        
        task_types = ['diagnosis_specialty_general', 'diagnosis_specialty_clinical']
        
        for task_type in task_types:
            print(f"\n{task_type.replace('_', ' ').title()}:")
            print("-" * 80)
            
            # Prepare comparison data
            comparison = []
            for run_id, metrics in all_metrics.items():
                if task_type in metrics:
                    m = metrics[task_type]
                    comparison.append({
                        'model': model_names[run_id],
                        'valid_spec': m.get('valid_specialty', 0),
                        'valid_diag': m.get('valid_diagnosis', 0),
                        'total': m['total_cases']
                    })
            
            if not comparison:
                print("  No data available")
                continue
            
            # Print comparison table
            print(f"{'Model':<20} {'Valid Specialty':<20} {'Valid Diagnosis':<20}")
            print("-" * 80)
            
            for c in comparison:
                spec_pct = (c['valid_spec'] / c['total']) * 100 if c['total'] > 0 else 0
                diag_pct = (c['valid_diag'] / c['total']) * 100 if c['total'] > 0 else 0
                
                print(f"{c['model']:<20} {c['valid_spec']}/{c['total']} ({spec_pct:.1f}%)      "
                      f"{c['valid_diag']}/{c['total']} ({diag_pct:.1f}%)")
    
    def print_overall_summary(self, all_metrics, model_names):
        """Print overall summary and recommendations"""
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        # Calculate average performance per model
        model_performance = defaultdict(lambda: {'exact': [], 'within_1': [], 'completion': []})
        
        for run_id, metrics in all_metrics.items():
            model = model_names[run_id]
            
            # Collect triage metrics
            for task in ['triage_general', 'triage_clinical']:
                if task in metrics and 'exact_accuracy' in metrics[task]:
                    model_performance[model]['exact'].append(metrics[task]['exact_accuracy'])
                    model_performance[model]['within_1'].append(metrics[task]['within_1_accuracy'])
            
            # Collect completion rates
            for task in ['diagnosis_specialty_general', 'diagnosis_specialty_clinical']:
                if task in metrics:
                    m = metrics[task]
                    total = m['total_cases']
                    valid = m.get('valid_specialty', 0)
                    if total > 0:
                        model_performance[model]['completion'].append((valid / total) * 100)
        
        # Print averages
        print("\nAverage Performance by Model:")
        print("-" * 80)
        print(f"{'Model':<20} {'Avg Exact':<15} {'Avg Within-1':<15} {'Avg Completion':<15}")
        print("-" * 80)
        
        for model in sorted(model_performance.keys()):
            perf = model_performance[model]
            avg_exact = sum(perf['exact']) / len(perf['exact']) if perf['exact'] else 0
            avg_within_1 = sum(perf['within_1']) / len(perf['within_1']) if perf['within_1'] else 0
            avg_completion = sum(perf['completion']) / len(perf['completion']) if perf['completion'] else 0
            
            print(f"{model:<20} {avg_exact:>6.1f}%         {avg_within_1:>6.1f}%         {avg_completion:>6.1f}%")
        
        # Recommendations
        print("\nRecommendations:")
        print("-" * 80)
        
        # Find best model for triage safety
        best_safety = max(
            model_performance.items(),
            key=lambda x: sum(x[1]['within_1']) / len(x[1]['within_1']) if x[1]['within_1'] else 0
        )
        print(f"🏆 Best for Triage Safety: {best_safety[0]}")
        
        # Find best for completion
        best_completion = max(
            model_performance.items(),
            key=lambda x: sum(x[1]['completion']) / len(x[1]['completion']) if x[1]['completion'] else 0
        )
        print(f"🏆 Best for Completion Rate: {best_completion[0]}")
    
    def print_run_list(self):
        """Print list of all available runs"""
        runs = self.list_all_runs()
        
        if not runs:
            print("No benchmark runs found")
            return
        
        print("\n" + "=" * 80)
        print("AVAILABLE BENCHMARK RUNS")
        print("=" * 80)
        print(f"\nTotal runs: {len(runs)}\n")
        
        print(f"{'Model':<20} {'Run ID':<35} {'Time':<12} {'Cases':<10}")
        print("-" * 80)
        
        for run in runs:
            time_str = f"{run['total_time']/60:.1f} min"
            cases_str = f"{run['num_cases']}" if run['num_cases'] != 'all' else 'all'
            if run['test_mode']:
                cases_str = '10 (test)'
            
            run_id_short = run['run_id']
            if len(run_id_short) > 35:
                run_id_short = run_id_short[:32] + '...'
            
            print(f"{run['model_name']:<20} {run_id_short:<35} {time_str:<12} {cases_str:<10}")
        
        print("\nTo compare runs:")
        print("  python compare_runs.py RUN_ID1 RUN_ID2")
        print("  python compare_runs.py --all")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MIMIC-IV-Ext benchmark runs"
    )
    parser.add_argument(
        'run_ids',
        nargs='*',
        help='Run IDs to compare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all available runs'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available runs'
    )
    
    args = parser.parse_args()
    
    comparator = BenchmarkComparator()
    
    if args.list:
        comparator.print_run_list()
        return 0
    
    if args.all:
        # Get all runs
        runs = comparator.list_all_runs()
        run_ids = [r['run_id'] for r in runs]
        
        if not run_ids:
            print("No runs found to compare")
            return 1
        
        print(f"Comparing {len(run_ids)} runs...")
        comparator.compare_runs(run_ids)
    
    elif args.run_ids:
        if len(args.run_ids) < 2:
            print("Please provide at least 2 run IDs to compare")
            print("\nAvailable runs:")
            comparator.print_run_list()
            return 1
        
        comparator.compare_runs(args.run_ids)
    
    else:
        # Show help
        parser.print_help()
        print()
        comparator.print_run_list()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
