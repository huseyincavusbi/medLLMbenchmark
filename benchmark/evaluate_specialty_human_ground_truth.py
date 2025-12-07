#!/usr/bin/env python3
"""
Specialty Evaluation with Human (Clinician-Approved) Ground Truth
Runs predictions on the 331 clinician-verified cases and evaluates accuracy.
Supports both HuggingFace and vLLM backends.
"""

import sys
import argparse
import pandas as pd
import re
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from functions.backends import create_backend


# Default paths
DEFAULT_GROUND_TRUTH = "../dataset/specialty_referral_clinician_approved.csv"
DEFAULT_MODEL = "google/medgemma-27b-it"


def parse_specialties(text):
    """Extract specialties from <specialty> tags"""
    if pd.isna(text):
        return [None, None, None]
    
    matches = re.findall(r'<specialty>(.*?)</specialty>', str(text), re.DOTALL | re.IGNORECASE)
    specialties = [m.strip() for m in matches][:3]
    
    while len(specialties) < 3:
        specialties.append(None)
    
    return specialties[:3]


def normalize_specialty(specialty):
    """Normalize specialty name for comparison"""
    if not specialty:
        return None
    
    s = specialty.lower().strip()
    
    # Common normalizations
    mappings = {
        'internal medicine': 'internal medicine',
        'medicine': 'internal medicine',
        'general medicine': 'internal medicine',
        'hospitalist': 'internal medicine',
        'hospital medicine': 'internal medicine',
        'emergency medicine': 'emergency medicine',
        'emergency': 'emergency medicine',
        'em': 'emergency medicine',
        'cardiology': 'cardiology',
        'cardiovascular': 'cardiology',
        'pulmonology': 'pulmonology',
        'pulmonary': 'pulmonology',
        'pulmonary medicine': 'pulmonology',
        'gastroenterology': 'gastroenterology',
        'gi': 'gastroenterology',
        'nephrology': 'nephrology',
        'renal': 'nephrology',
        'neurology': 'neurology',
        'neuro': 'neurology',
        'infectious disease': 'infectious disease',
        'id': 'infectious disease',
        'general surgery': 'general surgery',
        'surgery': 'general surgery',
        'orthopedics': 'orthopedics',
        'ortho': 'orthopedics',
        'orthopedic surgery': 'orthopedics',
        'neurosurgery': 'neurosurgery',
        'neuro surgery': 'neurosurgery',
        'vascular surgery': 'vascular surgery',
        'urology': 'urology',
    }
    
    return mappings.get(s, s)


def exact_match(pred, ground_truth):
    """Check if prediction matches ground truth"""
    pred_norm = normalize_specialty(pred)
    gt_norm = normalize_specialty(ground_truth)
    
    if pred_norm is None or gt_norm is None:
        return None
    
    return pred_norm == gt_norm


def run_specialty_predictions(df, backend, user_type='clinical'):
    """
    Run specialty predictions on the dataset.
    
    Args:
        df: DataFrame with HPI, patient_info, initial_vitals columns
        backend: Inference backend (HF or vLLM)
        user_type: 'clinical' (with vitals) or 'general' (without vitals)
    """
    # Define prompt template
    if user_type == 'clinical':
        prompt_template = """You are an experienced healthcare professional with expertise in determining the medical specialty based on a patient's history of present illness, personal information and initial vitals.

Identify the three most likely, distinct specialties to manage the condition.

CRITICAL: Respond ONLY in this EXACT XML format (no other text):
<specialty>First Specialty Name</specialty>
<specialty>Second Specialty Name</specialty>
<specialty>Third Specialty Name</specialty>

History of present illness: {hpi}
Personal information: {patient_info}
Initial vitals: {initial_vitals}"""
    else:
        prompt_template = """You are an experienced healthcare professional with expertise in determining the medical specialty based on a patient's history of present illness and personal information.

Identify the three most likely, distinct specialties to manage the condition.

CRITICAL: Respond ONLY in this EXACT XML format (no other text):
<specialty>First Specialty Name</specialty>
<specialty>Second Specialty Name</specialty>
<specialty>Third Specialty Name</specialty>

History of present illness: {hpi}
Personal information: {patient_info}"""
    
    # Format prompts
    prompts = []
    for _, row in df.iterrows():
        if user_type == 'clinical':
            prompt = prompt_template.format(
                hpi=row['HPI'],
                patient_info=row['patient_info'],
                initial_vitals=row['initial_vitals']
            )
        else:
            prompt = prompt_template.format(
                hpi=row['HPI'],
                patient_info=row['patient_info']
            )
        prompts.append(prompt)
    
    # Generate predictions
    print(f"Generating {len(prompts)} specialty predictions...")
    responses = backend.generate_batch(prompts, max_tokens=256)
    
    return responses


def evaluate_predictions(df, predictions, gt_col):
    """Evaluate predictions against ground truth"""
    results = {
        'top1_correct': 0,
        'top1_total': 0,
        'top3_correct': 0,
        'top3_total': 0,
        'predictions': [],
    }
    
    for i, (_, row) in enumerate(df.iterrows()):
        gt_raw = row[gt_col]
        
        # Parse ground truth (may be a list like "['General Surgery']")
        if isinstance(gt_raw, str) and gt_raw.startswith('['):
            gt = eval(gt_raw)[0] if eval(gt_raw) else None
        else:
            gt = gt_raw
        
        if pd.isna(gt):
            continue
        
        preds = parse_specialties(predictions[i])
        
        # Store for analysis
        results['predictions'].append({
            'stay_id': row['stay_id'],
            'ground_truth': gt,
            'pred_1': preds[0],
            'pred_2': preds[1],
            'pred_3': preds[2],
        })
        
        # Top-1 accuracy
        if preds[0]:
            results['top1_total'] += 1
            if exact_match(preds[0], gt):
                results['top1_correct'] += 1
        
        # Top-3 accuracy (any match)
        results['top3_total'] += 1
        for pred in preds:
            if pred and exact_match(pred, gt):
                results['top3_correct'] += 1
                break
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Specialty Evaluation with Clinician-Approved Ground Truth (331 cases)')
    parser.add_argument('--ground-truth', default=DEFAULT_GROUND_TRUTH,
                        help='Path to clinician-approved ground truth CSV')
    parser.add_argument('--model-path', default=DEFAULT_MODEL,
                        help='Path to model')
    parser.add_argument('--backend', choices=['hf', 'vllm'], default='vllm',
                        help='Backend: hf (HuggingFace) or vllm')
    parser.add_argument('--user-type', choices=['clinical', 'general'], default='clinical',
                        help='User type: clinical (with vitals) or general (without)')
    parser.add_argument('--num-cases', type=int, default=None,
                        help='Number of cases to evaluate (default: all 331)')
    parser.add_argument('--test', action='store_true',
                        help='Run on 10 test cases only')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SPECIALTY EVALUATION - HUMAN GROUND TRUTH")
    print("="*70)
    print(f"Backend: {args.backend.upper()}")
    print(f"Model: {args.model_path}")
    print(f"User Type: {args.user_type}")
    
    # Load ground truth dataset
    project_root = Path(__file__).parent.parent
    gt_path = project_root / args.ground_truth
    
    if not gt_path.exists():
        print(f"\nERROR: Ground truth file not found: {gt_path}")
        return 1
    
    df = pd.read_csv(gt_path)
    gt_col = 'specialty clinician approved'
    
    print(f"\nTotal clinician-approved cases: {len(df)}")
    
    # Subset if requested
    if args.test:
        df = df.head(10)
        print(f"Test mode: Using {len(df)} cases")
    elif args.num_cases:
        df = df.head(args.num_cases)
        print(f"Using {len(df)} cases")
    
    # Initialize backend
    print(f"\nInitializing {args.backend.upper()} backend...")
    backend = create_backend(args.backend, args.model_path)
    print("[OK] Model loaded")
    
    # Run predictions
    start_time = time.time()
    predictions = run_specialty_predictions(df, backend, args.user_type)
    elapsed = time.time() - start_time
    
    print(f"\nPredictions completed in {elapsed:.1f}s ({elapsed/len(df):.2f}s per case)")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    results = evaluate_predictions(df, predictions, gt_col)
    
    if results['top1_total'] > 0:
        top1_acc = results['top1_correct'] / results['top1_total'] * 100
        print(f"\nTop-1 Accuracy: {results['top1_correct']}/{results['top1_total']} = {top1_acc:.1f}%")
    
    if results['top3_total'] > 0:
        top3_acc = results['top3_correct'] / results['top3_total'] * 100
        print(f"Top-3 Accuracy: {results['top3_correct']}/{results['top3_total']} = {top3_acc:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    model_name = args.model_path.split('/')[-1]
    output_file = results_dir / f"specialty_human_gt_{model_name}_{args.user_type}_{timestamp}.csv"
    
    results_df = pd.DataFrame(results['predictions'])
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved: {output_file}")
    
    # Show some examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    for i, pred in enumerate(results['predictions'][:5]):
        match = "✓" if exact_match(pred['pred_1'], pred['ground_truth']) else "✗"
        print(f"\n{i+1}. Ground Truth: {pred['ground_truth']}")
        print(f"   Pred 1: {pred['pred_1']} {match}")
        print(f"   Pred 2: {pred['pred_2']}")
        print(f"   Pred 3: {pred['pred_3']}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
