#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation for Diagnosis and Specialty Predictions
Adapted to work with MedGemma-4B's flexible output formats
"""

import pandas as pd
import requests
import re
import json
import time
from tqdm import tqdm
from pathlib import Path


class LLMJudge:
    """Flexible LLM-as-judge that handles multiple output formats"""
    
    def __init__(self, base_url="http://localhost:1234/v1", model_name="medgemma-4b-it-mlx"):
        self.base_url = base_url
        self.model_name = model_name
        self.test_connection()
    
    def test_connection(self):
        """Test if LLM server is running"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                },
                timeout=10
            )
            if response.status_code == 200:
                print(f"Connected to LLM at {self.base_url}")
                return True
            else:
                print(f"LLM server error: {response.status_code}")
                return False
        except Exception as e:
            print(f"Cannot connect to LLM: {e}")
            print(f"   Please start LM Studio with {self.model_name}")
            return False
    
    def query(self, prompt, max_tokens=100):
        """Send query to LLM and get response"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Query error: {e}")
            return None
    
    def parse_evaluation(self, text):
        """
        Flexible parser that handles multiple output formats:
        - <evaluation>True</evaluation>
        - {"Match": "True"}
        - "Answer: True"
        - Just "True" or "False"
        - "Yes" or "No"
        """
        if not text:
            return None
        
        text = str(text)
        
        # Method 1: Look for <evaluation> tags (original format)
        eval_match = re.search(r'<evaluation>\s*(True|False)\s*</evaluation>', text, re.IGNORECASE)
        if eval_match:
            return eval_match.group(1).lower() == 'true'
        
        # Method 2: Look for JSON with "Match" field
        try:
            # Try to find JSON in the text
            json_match = re.search(r'\{[^}]*"Match"\s*:\s*"(True|False)"[^}]*\}', text, re.IGNORECASE)
            if json_match:
                return 'true' in json_match.group(1).lower()
        except:
            pass
        
        # Method 3: Look for "Answer: True/False"
        answer_match = re.search(r'Answer:\s*(True|False)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).lower() == 'true'
        
        # Method 4: Look for standalone True/False/Yes/No
        # Clean the text first
        clean_text = text.strip().split('\n')[0]  # Take first line
        
        # Yes/No mapping
        if re.match(r'^\s*(Yes|True)\b', clean_text, re.IGNORECASE):
            return True
        if re.match(r'^\s*(No|False)\b', clean_text, re.IGNORECASE):
            return False
        
        # Look anywhere in first 50 chars
        first_part = clean_text[:50]
        if re.search(r'\bTrue\b', first_part, re.IGNORECASE):
            return True
        if re.search(r'\bFalse\b', first_part, re.IGNORECASE):
            return False
        
        print(f"Could not parse: {text[:100]}")
        return None
    
    def evaluate_diagnosis_match(self, real_diagnoses, predicted_diagnosis):
        """
        Evaluate if a single predicted diagnosis matches any real diagnosis
        Returns: True if match, False if no match, None if parse failed
        """
        # Yes/No prompt - MedGemma-4B responds cleanly to this
        prompt = f"""Real diagnosis: {real_diagnoses}
Predicted diagnosis: {predicted_diagnosis}

Does the predicted diagnosis match the real diagnosis (same meaning or broader category)?
Answer: """
        
        response = self.query(prompt, max_tokens=5)
        
        # Parse: Look for Yes/No/True/False in first word
        if response:
            first_word = response.strip().split()[0].lower().rstrip(',.')
            if first_word in ['yes', 'true']:
                return True
            elif first_word in ['no', 'false']:
                return False
        
        return None
    
    def evaluate_diagnosis_triple(self, real_diagnoses, pred1, pred2, pred3):
        """
        Evaluate 3 predictions at once (original method)
        Returns: list of 3 booleans or None values
        """
        # Ask 3 separate yes/no questions
        prompt = f"""Real diagnosis: {real_diagnoses}

Question 1: Does "{pred1}" match (same meaning or broader category)? 
Question 2: Does "{pred2}" match (same meaning or broader category)?
Question 3: Does "{pred3}" match (same meaning or broader category)?

Answer each with Yes or No:
1. """
        
        response = self.query(prompt, max_tokens=30)
        
        if not response:
            return [None, None, None]
        
        # Extract Yes/No from response
        results = []
        lines = response.split('\n')
        
        for line in lines[:3]:  # Check first 3 lines
            first_word = line.strip().split()[0].lower().rstrip(',.') if line.strip().split() else ""
            if first_word in ['yes', 'true']:
                results.append(True)
            elif first_word in ['no', 'false']:
                results.append(False)
            elif len(results) > 0:  # Only append None if we've started getting results
                results.append(None)
        
        # Pad with None if we didn't get 3 results
        while len(results) < 3:
            results.append(None)
        
        return results[:3]


def evaluate_diagnosis_results(csv_path, judge, method='one_by_one'):
    """
    Evaluate diagnosis predictions using LLM-as-judge
    
    Args:
        csv_path: Path to results CSV
        judge: LLMJudge instance
        method: 'one_by_one' (reliable) or 'batch' (faster)
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {Path(csv_path).name}")
    print(f"Method: {method}")
    print('='*80)
    
    df = pd.read_csv(csv_path)
    
    # Determine user type from filename
    user_type = 'general' if 'general' in str(csv_path).lower() else 'clinical'
    pred_col = f'diag_spec_medgemma-4b-it-mlx_{user_type}'
    
    print(f"User type: {user_type}")
    print(f"Prediction column: {pred_col}")
    print(f"Total cases: {len(df)}\n")
    
    # Parse predictions to extract diagnoses
    def parse_diagnoses(text):
        """Extract 3 diagnoses from model output"""
        if pd.isna(text):
            return [None, None, None]
        
        # Find all <diagnosis> tags
        matches = re.findall(r'<diagnosis>(.*?)</diagnosis>', str(text), re.DOTALL | re.IGNORECASE)
        
        # Clean and take first 3
        diagnoses = [m.strip() for m in matches][:3]
        
        # Pad with None if less than 3
        while len(diagnoses) < 3:
            diagnoses.append(None)
        
        return diagnoses[:3]
    
    # Extract predictions
    print("Parsing predictions...")
    df['pred_diag_1'] = None
    df['pred_diag_2'] = None
    df['pred_diag_3'] = None
    
    for idx, row in df.iterrows():
        preds = parse_diagnoses(row[pred_col])
        df.at[idx, 'pred_diag_1'] = preds[0]
        df.at[idx, 'pred_diag_2'] = preds[1]
        df.at[idx, 'pred_diag_3'] = preds[2]
    
    # Get ground truth
    ground_truth_col = 'diagnosis_list'  # or use primary_diagnosis
    
    # Evaluate with LLM-as-judge
    print(f"\nEvaluating with LLM-as-judge ({method} method)...")
    
    df['eval_diag_1'] = None
    df['eval_diag_2'] = None
    df['eval_diag_3'] = None
    
    if method == 'one_by_one':
        # Evaluate each prediction separately (more reliable)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            real_diag = row[ground_truth_col]
            
            for i in [1, 2, 3]:
                pred_diag = row[f'pred_diag_{i}']
                if pred_diag:
                    result = judge.evaluate_diagnosis_match(real_diag, pred_diag)
                    df.at[idx, f'eval_diag_{i}'] = result
                    time.sleep(0.1)  # Small delay to avoid overwhelming the server
    
    else:  # batch method
        # Evaluate 3 predictions together (faster but less reliable for small models)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            real_diag = row[ground_truth_col]
            pred1 = row['pred_diag_1']
            pred2 = row['pred_diag_2']
            pred3 = row['pred_diag_3']
            
            if pred1 or pred2 or pred3:
                results = judge.evaluate_diagnosis_triple(real_diag, pred1, pred2, pred3)
                df.at[idx, 'eval_diag_1'] = results[0]
                df.at[idx, 'eval_diag_2'] = results[1]
                df.at[idx, 'eval_diag_3'] = results[2]
                time.sleep(0.1)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Count matches
    total_predictions = 0
    total_matches = 0
    
    for i in [1, 2, 3]:
        valid = df[f'eval_diag_{i}'].notna().sum()
        matches = df[f'eval_diag_{i}'].sum()
        
        print(f"\nDiagnosis {i}:")
        print(f"  Valid evaluations: {valid}/{len(df)}")
        print(f"  Matches: {matches}/{valid} ({matches/valid*100:.1f}%)" if valid > 0 else "  No valid evaluations")
        
        total_predictions += valid
        total_matches += matches
    
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY:")
    print(f"  {total_matches}/{total_predictions} = {total_matches/total_predictions*100:.1f}%")
    print("="*80)
    
    # Save results
    output_path = str(csv_path).replace('.csv', '_llm_evaluated.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df


def main():
    print("="*80)
    print("LLM-AS-JUDGE DIAGNOSIS EVALUATION")
    print("="*80)
    
    # Initialize judge
    judge = LLMJudge(
        base_url="http://localhost:1234/v1",
        model_name="medgemma-4b-it-mlx"
    )
    
    # Find result files
    results_dir = Path("results")
    
    diagnosis_files = [
        results_dir / "diagnosis_specialty_general_medgemma-4b-it-mlx_20251023_211152.csv",
        results_dir / "diagnosis_specialty_clinical_medgemma-4b-it-mlx_20251023_211152.csv"
    ]
    
    # Check which files exist
    existing_files = [f for f in diagnosis_files if f.exists()]
    
    if not existing_files:
        print("\nNo result files found in results/")
        print("Expected files:")
        for f in diagnosis_files:
            print(f"  - {f.name}")
        return
    
    print(f"\nFound {len(existing_files)} result file(s)")
    
    # Choose evaluation method
    print("\nEvaluation methods:")
    print("1. one_by_one - Evaluate each diagnosis separately (slower, more reliable)")
    print("2. batch - Evaluate 3 diagnoses together (faster, may be less reliable)")
    
    method = input("\nChoose method (1 or 2) [default: 1]: ").strip()
    method = 'batch' if method == '2' else 'one_by_one'
    
    # Evaluate each file
    for csv_file in existing_files:
        try:
            evaluate_diagnosis_results(csv_file, judge, method=method)
        except Exception as e:
            print(f"\nError evaluating {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
