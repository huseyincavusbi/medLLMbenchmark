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
    
    def __init__(self, model_path, backend_type="vllm"):
        """
        Initialize the LLM judge with a local model.
        
        Args:
            model_path: Path to model (local or HuggingFace ID)
            backend_type: 'hf' or 'vllm'
        """
        self.model_path = model_path
        self.backend_type = backend_type
        self._backend = None
    
    def get_backend(self):
        """Lazy load the backend"""
        if self._backend is None:
            print(f"\nInitializing {self.backend_type.upper()} judge backend...")
            print(f"Model: {self.model_path}")
            self._backend = create_backend(self.backend_type, self.model_path)
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
        
        Returns: List of True/False/None results
        """
        prompts = []
        for real_diag, pred_diag in cases:
            prompt = f"""You are a medical expert. Does "{pred_diag}" match "{real_diag}" (same/similar condition)?
Answer: """
            prompts.append(prompt)
        
        backend = self.get_backend()
        responses = backend.generate_batch(prompts, max_tokens=5)
        
        results = []
        for response in responses:
            if response:
                first_word = response.strip().split()[0].lower().rstrip('.,!') if response.strip().split() else ""
                if first_word in ['yes', 'true']:
                    results.append(True)
                elif first_word in ['no', 'false']:
                    results.append(False)
                else:
                    results.append(None)
            else:
                results.append(None)
        
        return results


def parse_diagnoses(text):
    """Extract diagnoses from <diagnosis> tags"""
    if pd.isna(text):
        return [None, None, None]
    
    matches = re.findall(r'<diagnosis>(.*?)</diagnosis>', str(text), re.DOTALL | re.IGNORECASE)
    diagnoses = [m.strip() for m in matches][:3]
    
    while len(diagnoses) < 3:
        diagnoses.append(None)
    
    return diagnoses[:3]


def evaluate_diagnosis_results(csv_path, judge, ground_truth_col='primary_diagnosis'):
    """
    Evaluate diagnosis predictions using LLM-as-judge
    
    Args:
        csv_path: Path to results CSV
        judge: LLMJudge instance
        ground_truth_col: Column name for ground truth diagnosis
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {Path(csv_path).name}")
    print('='*70)
    
    df = pd.read_csv(csv_path)
    
    # Find prediction column
    pred_cols = [c for c in df.columns if 'medgemma' in c.lower() and ('diag' in c.lower() or 'spec' in c.lower())]
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
    
    batch_size = 10
    all_results = []
    
    for i in tqdm(range(0, len(eval_cases), batch_size), desc="Batches"):
        batch = eval_cases[i:i+batch_size]
        results = judge.evaluate_batch(batch)
        all_results.extend(results)
    
    # Map results back to dataframe
    for (idx, diag_num), result in zip(case_indices, all_results):
        df.at[idx, f'eval_diag_{diag_num}'] = result
    
    # Calculate metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    total_predictions = 0
    total_matches = 0
    
    for i in range(1, 4):
        col = f'eval_diag_{i}'
        valid = df[col].notna().sum()
        matches = df[col].sum() if valid > 0 else 0
        
        print(f"\nDiagnosis {i}:")
        print(f"  Valid evaluations: {valid}/{len(df)}")
        if valid > 0:
            print(f"  Matches: {int(matches)}/{valid} ({matches/valid*100:.1f}%)")
        
        total_predictions += valid
        total_matches += matches
    
    if total_predictions > 0:
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {int(total_matches)}/{total_predictions} = {total_matches/total_predictions*100:.1f}%")
        print("="*70)
    
    # Save results
    output_path = str(csv_path).replace('.csv', '_llm_evaluated.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='LLM-as-Judge Diagnosis Evaluation')
    parser.add_argument('--results-dir', default='benchmark/results',
                        help='Directory containing result CSV files')
    parser.add_argument('--model-path', default='google/medgemma-4b-it',
                        help='Path to judge model')
    parser.add_argument('--backend', choices=['hf', 'vllm'], default='vllm',
                        help='Backend: hf (HuggingFace) or vllm')
    parser.add_argument('--ground-truth-col', default='primary_diagnosis',
                        help='Column name for ground truth diagnosis')
    parser.add_argument('--latest', action='store_true',
                        help='Evaluate the latest run only')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLM-AS-JUDGE DIAGNOSIS EVALUATION")
    print("="*70)
    print(f"Backend: {args.backend.upper()}")
    print(f"Judge Model: {args.model_path}")
    
    # Find diagnosis result files
    results_dir = Path(args.results_dir)
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
    judge = LLMJudge(model_path=args.model_path, backend_type=args.backend)
    
    # Evaluate each file
    for csv_file in diagnosis_files:
        try:
            evaluate_diagnosis_results(csv_file, judge, args.ground_truth_col)
        except Exception as e:
            print(f"\nError evaluating {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
