#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation for Diagnosis and Specialty Predictions
Supports both HuggingFace and vLLM backends for local GPU execution.
"""

import sys
import argparse
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from functions.backends import create_backend


class LLMJudge:
    """LLM-as-judge using local GPU backends (HuggingFace or vLLM)"""
    
    def __init__(self, model_path, backend_type="vllm", quantization=None):
        """
        Initialize the LLM judge with a local model.
        
        Args:
            model_path: Path to model (local or HuggingFace ID)
            backend_type: 'hf' or 'vllm'
            quantization: Optional quantization method (vLLM: "bitsandbytes", "awq", etc.)
        """
        self.model_path = model_path
        self.backend_type = backend_type
        self.quantization = quantization
        self._backend = None
    
    def get_backend(self):
        """Lazy load the backend"""
        if self._backend is None:
            print(f"\nInitializing {self.backend_type.upper()} judge backend...")
            print(f"Model: {self.model_path}")
            if self.quantization:
                print(f"Quantization: {self.quantization}")
            self._backend = create_backend(self.backend_type, self.model_path, quantization=self.quantization)
            print("[OK] Judge model loaded\n")
        return self._backend
    
    def evaluate_diagnosis_match(self, real_diagnosis, predicted_diagnosis):
        """
        Evaluate if a predicted diagnosis matches the real diagnosis.
        Returns: True if match, False if no match, None if parse failed
        """
        prompt = f"""You are a medical expert evaluating diagnosis predictions.

Real diagnosis: {real_diagnosis}
Predicted diagnosis: {predicted_diagnosis}

Does the predicted diagnosis match the real diagnosis (same condition, synonym, or closely related)?

Answer with ONLY "Yes" or "No":"""
        
        backend = self.get_backend()
        responses = backend.generate_batch([prompt], max_tokens=10)
        
        if responses and responses[0]:
            response = responses[0].strip().lower()
            first_word = response.split()[0].rstrip('.,!') if response.split() else ""
            if first_word in ['yes', 'true']:
                return True
            elif first_word in ['no', 'false']:
                return False
        
        return None
    
    def evaluate_batch(self, cases):
        """
        Evaluate multiple diagnosis pairs in batch for efficiency.
        
        Args:
            cases: List of (real_diagnosis, predicted_diagnosis) tuples
            Note: real_diagnosis may be a list like "['Diag1', 'Diag2']"
        
        Returns: List of True/False/None results
        """
        # Parse ground truth lists and expand into individual comparisons
        expanded_cases = []
        case_mapping = []  # Maps expanded index back to original case index
        
        for orig_idx, (real_diag, pred_diag) in enumerate(cases):
            # Parse the ground truth - it might be a stringified list
            gt_diagnoses = self._parse_ground_truth(real_diag)
            for gt_diag in gt_diagnoses:
                expanded_cases.append((gt_diag, pred_diag))
                case_mapping.append(orig_idx)
        
        # Build prompts for expanded cases
        prompts = []
        for real_diag, pred_diag in expanded_cases:
            prompt = f"""Real diagnosis: {real_diag}
Predicted diagnosis: {pred_diag}

Does the predicted diagnosis match the real diagnosis (same meaning or broader category)?
Answer: """
            prompts.append(prompt)
        
        backend = self.get_backend()
        responses = backend.generate_batch(prompts, max_tokens=20)
        
        # Parse responses
        expanded_results = []
        for response in responses:
            result = self._parse_yes_no(response)
            expanded_results.append(result)
        
        # Aggregate results: True if ANY ground truth diagnosis matches
        final_results = [None] * len(cases)
        for exp_idx, result in enumerate(expanded_results):
            orig_idx = case_mapping[exp_idx]
            if result is True:
                final_results[orig_idx] = True
            elif final_results[orig_idx] is None and result is False:
                final_results[orig_idx] = False
        
        return final_results
    
    def _parse_ground_truth(self, gt_string):
        """Parse ground truth which may be a stringified list like ['Diag1', 'Diag2']"""
        import ast
        
        gt_string = str(gt_string).strip()
        
        # Try to parse as Python list
        if gt_string.startswith('[') and gt_string.endswith(']'):
            try:
                parsed = ast.literal_eval(gt_string)
                if isinstance(parsed, list):
                    return [str(d).strip() for d in parsed if d]
            except (ValueError, SyntaxError):
                pass
        
        # Return as single-element list
        return [gt_string]
    
    def _parse_yes_no(self, response):
        """Parse Yes/No from model response with multiple fallback strategies"""
        if not response:
            return None
        
        response = str(response).strip().lower()
        
        # Strategy 1: Check first word
        words = response.split()
        if words:
            first_word = words[0].rstrip('.,!:')
            if first_word in ['yes', 'true', 'correct', 'match', 'matches']:
                return True
            elif first_word in ['no', 'false', 'incorrect', 'different', 'not']:
                return False
        
        # Strategy 2: Look for yes/no anywhere in first 50 chars
        first_part = response[:50]
        if 'yes' in first_part or 'match' in first_part:
            return True
        elif 'no' in first_part or 'not' in first_part or 'different' in first_part:
            return False
        
        return None


def parse_diagnoses(text):
    """Extract diagnoses from <diagnosis> tags"""
    if pd.isna(text):
        return [None, None, None]
    
    matches = re.findall(r'<diagnosis>(.*?)</diagnosis>', str(text), re.DOTALL | re.IGNORECASE)
    diagnoses = [m.strip() for m in matches][:3]
    
    while len(diagnoses) < 3:
        diagnoses.append(None)
    
    return diagnoses[:3]


def evaluate_diagnosis_results(csv_path, judge, ground_truth_col='primary_diagnosis', limit=None):
    """
    Evaluate diagnosis predictions using LLM-as-judge
    
    Args:
        csv_path: Path to results CSV
        judge: LLMJudge instance
        ground_truth_col: Column name for ground truth diagnosis
        limit: Optional limit on number of cases to evaluate
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {Path(csv_path).name}")
    print('='*70)
    
    df = pd.read_csv(csv_path)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} cases for testing")
    
    # Find prediction column (more flexible matching)
    pred_cols = [c for c in df.columns if 'diag_spec' in c.lower()]
    if not pred_cols:
        print("No prediction column found!")
        return None
    
    pred_col = pred_cols[0]
    print(f"Prediction column: {pred_col}")
    print(f"Ground truth column: {ground_truth_col}")
    print(f"Total cases: {len(df)}")
    
    # Check if ground truth exists
    if ground_truth_col not in df.columns:
        print(f"\nWARNING: Ground truth column '{ground_truth_col}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Parse predictions
    print("\nParsing predictions...")
    for i in range(3):
        df[f'pred_diag_{i+1}'] = None
    
    for idx, row in df.iterrows():
        preds = parse_diagnoses(row[pred_col])
        for i in range(3):
            df.at[idx, f'pred_diag_{i+1}'] = preds[i]
    
    # Build evaluation cases
    print("Building evaluation cases...")
    eval_cases = []
    case_indices = []
    
    for idx, row in df.iterrows():
        real_diag = row[ground_truth_col]
        if pd.isna(real_diag):
            continue
        
        for i in range(1, 4):
            pred_diag = row[f'pred_diag_{i}']
            if pred_diag:
                eval_cases.append((real_diag, pred_diag))
                case_indices.append((idx, i))
    
    print(f"Total evaluations: {len(eval_cases)}")
    
    # Evaluate in batches
    print("\nEvaluating with LLM-as-judge...")
    df['eval_diag_1'] = None
    df['eval_diag_2'] = None
    df['eval_diag_3'] = None
    
    batch_size = 500
    all_results = []
    
    for i in tqdm(range(0, len(eval_cases), batch_size), desc="Batches"):
        batch = eval_cases[i:i+batch_size]
        results = judge.evaluate_batch(batch)
        all_results.extend(results)
    
    # Map results back to dataframe
    for (idx, diag_num), result in zip(case_indices, all_results):
        df.at[idx, f'eval_diag_{diag_num}'] = result
    
    # Calculate metrics (paper methodology)
    print("\n" + "="*70)
    print("RESULTS (Paper Methodology)")
    print("="*70)
    
    # Per-case metrics
    proportional_scores = []
    any_match_count = 0
    total_cases = 0
    
    import ast
    
    for idx, row in df.iterrows():
        # Get ground truth list
        gt_raw = row[ground_truth_col]
        if pd.isna(gt_raw):
            continue
        
        try:
            gt_list = ast.literal_eval(gt_raw) if isinstance(gt_raw, str) and gt_raw.startswith('[') else [gt_raw]
        except:
            gt_list = [gt_raw]
        
        # Get predictions that were evaluated
        preds_evaluated = []
        for i in range(1, 4):
            if pd.notna(row.get(f'pred_diag_{i}')):
                preds_evaluated.append(row.get(f'pred_diag_{i}'))
        
        if not preds_evaluated:
            continue
        
        total_cases += 1
        
        # Count matches for this case
        matches = 0
        any_matched = False
        for i in range(1, 4):
            eval_result = row.get(f'eval_diag_{i}')
            if eval_result is True:
                matches += 1
                any_matched = True
        
        # Proportional: matches / min(len(predictions), len(GT))
        shorter_len = min(len(preds_evaluated), len(gt_list))
        proportional = matches / shorter_len if shorter_len > 0 else 0
        proportional_scores.append(proportional)
        
        # Any match
        if any_matched:
            any_match_count += 1
    
    # Report metrics
    print(f"\nTotal cases evaluated: {total_cases}")
    
    if total_cases > 0:
        proportional_avg = sum(proportional_scores) / len(proportional_scores) * 100
        any_match_pct = any_match_count / total_cases * 100
        
        print(f"\n--- Paper Metrics ---")
        print(f"Proportional Match: {proportional_avg:.1f}%")
        print(f"Any Match (binary): {any_match_count}/{total_cases} = {any_match_pct:.1f}%")
    
    # Also report per-slot stats for debugging
    print(f"\n--- Per-Slot Statistics ---")
    for i in range(1, 4):
        col = f'eval_diag_{i}'
        valid = df[col].notna().sum()
        matches = df[col].sum() if valid > 0 else 0
        
        print(f"Diagnosis {i}: {int(matches)}/{valid} matches" + (f" ({matches/valid*100:.1f}%)" if valid > 0 else ""))
    
    # Save results
    output_path = str(csv_path).replace('.csv', '_llm_evaluated.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return {
        'total_cases': total_cases,
        'proportional_match': proportional_avg if total_cases > 0 else 0,
        'any_match': any_match_pct if total_cases > 0 else 0,
        'any_match_count': any_match_count
    }


def main():
    parser = argparse.ArgumentParser(description='LLM-as-Judge Diagnosis Evaluation')
    parser.add_argument('--results-dir', default='benchmark/results',
                        help='Directory containing result CSV files')
    parser.add_argument('--model-path', default='prometheus-eval/prometheus-8x7b-v2.0',
                        help='Path to judge model')
    parser.add_argument('--backend', choices=['hf', 'vllm'], default='vllm',
                        help='Backend: hf (HuggingFace) or vllm')
    parser.add_argument('--ground-truth-col', default='primary_diagnosis',
                        help='Column name for ground truth diagnosis')
    parser.add_argument('--latest', action='store_true',
                        help='Evaluate the latest run only')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of cases to evaluate (for testing)')
    parser.add_argument('--file', type=str, default=None,
                        help='Evaluate a specific file only')
    parser.add_argument('--quantization', type=str, default=None,
                        choices=['bitsandbytes', 'awq', 'gptq', 'fp8'],
                        help='Quantization method for vLLM (e.g., bitsandbytes for 8-bit)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLM-AS-JUDGE DIAGNOSIS EVALUATION")
    print("="*70)
    print(f"Backend: {args.backend.upper()}")
    print(f"Judge Model: {args.model_path}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    
    # Find diagnosis result files
    results_dir = Path(args.results_dir)
    
    if args.file:
        diagnosis_files = [Path(args.file)]
    else:
        diagnosis_files = list(results_dir.glob("diagnosis_specialty_*.csv"))
        diagnosis_files = [f for f in diagnosis_files if 'evaluated' not in str(f)]
    
    if not diagnosis_files:
        print(f"\nNo diagnosis files found in {results_dir}")
        return 1
    
    # Sort by modification time (newest first)
    diagnosis_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    if args.latest:
        # Take only the 2 most recent (general + clinical)
        diagnosis_files = diagnosis_files[:2]
    
    print(f"\nFound {len(diagnosis_files)} file(s) to evaluate")
    
    # Initialize judge
    judge = LLMJudge(model_path=args.model_path, backend_type=args.backend, quantization=args.quantization)
    
    # Evaluate each file and collect metrics
    all_metrics = {}
    for csv_file in diagnosis_files:
        try:
            metrics = evaluate_diagnosis_results(csv_file, judge, args.ground_truth_col, limit=args.limit)
            if metrics:
                # Extract model name from filename
                model_name = csv_file.stem.replace('diagnosis_specialty_', '').rsplit('_', 2)[0]
                all_metrics[model_name] = metrics
        except Exception as e:
            print(f"\nError evaluating {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all metrics to JSON
    if all_metrics:
        import json
        metrics_file = results_dir / 'diagnosis_judge_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
