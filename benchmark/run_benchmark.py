#!/usr/bin/env python3
"""
MIMIC-IV-Ext Comprehensive Benchmark Script
============================================
Runs all clinical decision support tasks with local LLM and evaluates results.

Tasks:
1. Triage prediction (General User)
2. Triage prediction (Clinical User)
3. Diagnosis & Specialty prediction (General User)
4. Diagnosis & Specialty prediction (Clinical User)

Usage:
    python run_benchmark.py [--test] [--model-name MODEL_NAME] [--num-cases N]
    
Options:
    --test          Run on 10 test cases only (fast testing)
    --model-name    Name of the model for results tracking (default: medgemma-27b-it)
    --model-path    Path to model directory (default: ./models/medgemma-27b-it)
    --num-cases     Number of cases to run (default: all available)
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.LLM_predictions import (
    create_chain,
    test_gpu_connection,
    get_prediction_GeneralUser,
    get_prediction_ClinicalUser
)


class BenchmarkRunner:
    """Orchestrates benchmark execution and result tracking"""
    
    def __init__(self, model_path="./models/medgemma-27b-it", model_name=None, 
                 num_cases=None, test_mode=False, quantization=None, judge_model_path=None, judge_quantization=None):
        self.model_path = model_path
        self.model_name = model_name or "medgemma-27b-it"
        # quantization: None -> use bfloat16, '4bit' or '8bit' for quantized loading
        self.quantization = quantization
        # Optional judge model settings (local LLM-as-judge)
        self.judge_model_path = judge_model_path
        self.judge_quantization = judge_quantization
        self.num_cases = num_cases
        self.test_mode = test_mode
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        # Optional base URL for remote model or dataset references
        self.base_url = None
        
        # Track timing and errors
        self.task_times = {}
        self.task_errors = {}
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.model_name}_{self.timestamp}"
        
    def test_connection(self):
        """Test GPU connection"""
        print("=" * 70)
        print("MIMIC-IV-Ext Benchmark - GPU Inference (bfloat16 default)")
        print("=" * 70)
        print(f"\nModel: {self.model_name}")
        print(f"Model Path: {self.model_path}")
        print(f"Test Mode: {self.test_mode}")
        if self.num_cases:
            print(f"Cases to run: {self.num_cases}")
        print()
        
        print("Testing GPU connection and model loading...")
        # If a judge model is provided, try loading both models together
        if self.judge_model_path:
            from functions.LLM_predictions import DualModelManager
            # Map quantization settings
            pred_quant = self.quantization
            judge_quant = self.judge_quantization
            try:
                self.dual_manager = DualModelManager(self.model_path, self.judge_model_path,
                                                     predictor_quant=pred_quant, judge_quant=judge_quant)
                print("Both predictor and judge models loaded successfully.")
            except Exception as e:
                print(f"\nERROR: Cannot load predictor+judge models on GPU: {e}")
                print("\nPlease verify model paths and GPU memory")
                return False
        else:
            if not test_gpu_connection(self.model_path):
                print("\nERROR: Cannot load model on GPU")
            print("\nPlease verify:")
            print("  1. GPU is available (nvidia-smi)")
            print("  2. Model is downloaded")
            print(f"  3. Model path is correct: {self.model_path}")
            print("  4. Sufficient GPU memory available")
            return False
        
        print("Connection successful!\n")
        return True
    
    def load_data(self, task_name):
        """Load appropriate dataset for task"""
        data_dir = Path(__file__).parent.parent
        
        # Determine which file to load
        if "triage" in task_name.lower():
            csv_file = data_dir / "MIMIC-IV-Ext-Triage.csv"
        elif "diag" in task_name.lower() or "spec" in task_name.lower():
            csv_file = data_dir / "MIMIC-IV-Ext-Diagnosis-Specialty.csv"
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        if not csv_file.exists():
            print(f"Dataset not found: {csv_file}")
            print("Please run data preparation scripts first:")
            print("  1. MIMIC-IV-Ext-Creation.py")
            print("  2. create_ground_truth_specialty.py")
            return None
        
        df = pd.read_csv(csv_file)
        
        # Subset for test mode or num_cases
        if self.test_mode:
            df = df.head(10)
            print(f"  Test mode: Using {len(df)} cases")
        elif self.num_cases:
            df = df.head(self.num_cases)
            print(f"  Using {len(df)} cases")
        else:
            print(f"  Loaded {len(df)} cases")
        
        return df
    
    def run_triage_general(self):
        """Task 1: Triage prediction - General User (Patient)"""
        task_name = "triage_general"
        print("\n" + "=" * 70)
        print("TASK 1: Triage Prediction - General User (Patient)")
        print("=" * 70)
        print("Input: HPI + Demographics")
        print("Output: ESI level (1-5)")
        print()
        
        # Load data
        df = self.load_data(task_name)
        if df is None:
            return None
        
        # Define prompt
        prompt = """You are a nurse with emergency and triage experience. Using the patient's history of present illness and information, determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity): 1: Assign if the patient requires immediate lifesaving intervention. 2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress)  3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care. 4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG). 5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).
History of present illness: {HPI}, patient info: {patient_info}. Respond with the level in an <acuity> tag."""
        
        # Create chain
        chain = create_chain(prompt, model_path=self.model_path, quantization=self.quantization)
        
        # Run predictions
        start_time = time.time()
        errors = 0
        
        print("Running predictions...")
        tqdm.pandas()
        
        def predict_with_error_tracking(row):
            try:
                return get_prediction_GeneralUser(row, chain)
            except Exception as e:
                nonlocal errors
                errors += 1
                return f"ERROR: {str(e)}"
        
        df[f'triage_{self.model_name}_general'] = df.progress_apply(
            predict_with_error_tracking, axis=1
        )
        
        elapsed = time.time() - start_time
        self.task_times[task_name] = elapsed
        self.task_errors[task_name] = errors
        
        # Save results
        output_file = self.results_dir / f"{task_name}_{self.run_id}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nTask completed in {elapsed:.1f}s ({elapsed/len(df):.2f}s per case)")
        print(f"   Errors: {errors}/{len(df)}")
        print(f"   Results saved: {output_file}")
        
        return df
    
    def run_triage_clinical(self):
        """Task 2: Triage prediction - Clinical User (Provider)"""
        task_name = "triage_clinical"
        print("\n" + "=" * 70)
        print("TASK 2: Triage Prediction - Clinical User (Provider)")
        print("=" * 70)
        print("Input: HPI + Demographics + Vital Signs")
        print("Output: ESI level (1-5)")
        print()
        
        # Load data
        df = self.load_data(task_name)
        if df is None:
            return None
        
        # Define prompt
        prompt = """You are a nurse with emergency and triage experience. Using the patient's history of present illness, his information and initial vitals, determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity): 1: Assign if the patient requires immediate lifesaving intervention. 2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress)  3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care. 4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG). 5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).
History of present illness: {hpi}, patient info: {patient_info} and initial vitals: {initial_vitals}. Respond with the level in an <acuity> tag."""
        
        # Create chain
        chain = create_chain(prompt, model_path=self.model_path, quantization=self.quantization)
        
        # Run predictions
        start_time = time.time()
        errors = 0
        
        print("Running predictions...")
        tqdm.pandas()
        
        def predict_with_error_tracking(row):
            try:
                return get_prediction_ClinicalUser(row, chain)
            except Exception as e:
                nonlocal errors
                errors += 1
                return f"ERROR: {str(e)}"
        
        df[f'triage_{self.model_name}_clinical'] = df.progress_apply(
            predict_with_error_tracking, axis=1
        )
        
        elapsed = time.time() - start_time
        self.task_times[task_name] = elapsed
        self.task_errors[task_name] = errors
        
        # Save results
        output_file = self.results_dir / f"{task_name}_{self.run_id}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nTask completed in {elapsed:.1f}s ({elapsed/len(df):.2f}s per case)")
        print(f"   Errors: {errors}/{len(df)}")
        print(f"   Results saved: {output_file}")
        
        return df
    
    def run_diagnosis_specialty_general(self):
        """Task 3: Diagnosis & Specialty prediction - General User"""
        task_name = "diagnosis_specialty_general"
        print("\n" + "=" * 70)
        print("TASK 3: Diagnosis & Specialty - General User (Patient)")
        print("=" * 70)
        print("Input: HPI + Demographics")
        print("Output: Top 3 Specialties + Top 3 Diagnoses")
        print()
        
        # Load data
        df = self.load_data(task_name)
        if df is None:
            return None
        
        # Define prompt
        prompt = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness and personal information.

Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses.

CRITICAL: Respond ONLY in this EXACT XML format (no other text):
<specialty>First Specialty Name</specialty>
<specialty>Second Specialty Name</specialty>
<specialty>Third Specialty Name</specialty>
<diagnosis>First Diagnosis Name</diagnosis>
<diagnosis>Second Diagnosis Name</diagnosis>
<diagnosis>Third Diagnosis Name</diagnosis>

Do NOT include explanations, preambles, or any text outside the XML tags.

History of present illness: {HPI}
Personal information: {patient_info}"""
        
        # Create chain
        chain = create_chain(prompt, model_path=self.model_path, quantization=self.quantization)
        
        # Run predictions
        start_time = time.time()
        errors = 0
        
        print("Running predictions...")
        tqdm.pandas()
        
        def predict_with_error_tracking(row):
            try:
                return get_prediction_GeneralUser(row, chain)
            except Exception as e:
                nonlocal errors
                errors += 1
                return f"ERROR: {str(e)}"
        
        df[f'diag_spec_{self.model_name}_general'] = df.progress_apply(
            predict_with_error_tracking, axis=1
        )
        
        elapsed = time.time() - start_time
        self.task_times[task_name] = elapsed
        self.task_errors[task_name] = errors
        
        # Save results
        output_file = self.results_dir / f"{task_name}_{self.run_id}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nTask completed in {elapsed:.1f}s ({elapsed/len(df):.2f}s per case)")
        print(f"   Errors: {errors}/{len(df)}")
        print(f"   Results saved: {output_file}")
        
        return df
    
    def run_diagnosis_specialty_clinical(self):
        """Task 4: Diagnosis & Specialty prediction - Clinical User"""
        task_name = "diagnosis_specialty_clinical"
        print("\n" + "=" * 70)
        print("TASK 4: Diagnosis & Specialty - Clinical User (Provider)")
        print("=" * 70)
        print("Input: HPI + Demographics + Vital Signs")
        print("Output: Top 3 Specialties + Top 3 Diagnoses")
        print()
        
        # Load data
        df = self.load_data(task_name)
        if df is None:
            return None
        
        # Define prompt
        prompt = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness, personal information and initial vitals.

Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses.

CRITICAL: Respond ONLY in this EXACT XML format (no other text):
<specialty>First Specialty Name</specialty>
<specialty>Second Specialty Name</specialty>
<specialty>Third Specialty Name</specialty>
<diagnosis>First Diagnosis Name</diagnosis>
<diagnosis>Second Diagnosis Name</diagnosis>
<diagnosis>Third Diagnosis Name</diagnosis>

Do NOT include explanations, preambles, or any text outside the XML tags.

History of present illness: {hpi}
Personal information: {patient_info}
Initial vitals: {initial_vitals}"""
        
        # Create chain
        chain = create_chain(prompt, model_path=self.model_path)
        
        # Run predictions
        start_time = time.time()
        errors = 0
        
        print("Running predictions...")
        tqdm.pandas()
        
        def predict_with_error_tracking(row):
            try:
                return get_prediction_ClinicalUser(row, chain)
            except Exception as e:
                nonlocal errors
                errors += 1
                return f"ERROR: {str(e)}"
        
        df[f'diag_spec_{self.model_name}_clinical'] = df.progress_apply(
            predict_with_error_tracking, axis=1
        )
        
        elapsed = time.time() - start_time
        self.task_times[task_name] = elapsed
        self.task_errors[task_name] = errors
        
        # Save results
        output_file = self.results_dir / f"{task_name}_{self.run_id}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nTask completed in {elapsed:.1f}s ({elapsed/len(df):.2f}s per case)")
        print(f"   Errors: {errors}/{len(df)}")
        print(f"   Results saved: {output_file}")
        
        return df
    
    def run_all_tasks(self):
        """Run all benchmark tasks"""
        if not self.test_connection():
            return False
        
        start_time = time.time()
        
        # Run all tasks
        results = {}
        results['triage_general'] = self.run_triage_general()
        results['triage_clinical'] = self.run_triage_clinical()
        results['diagnosis_specialty_general'] = self.run_diagnosis_specialty_general()
        results['diagnosis_specialty_clinical'] = self.run_diagnosis_specialty_clinical()
        
        total_time = time.time() - start_time
        
        # Print summary
        self.print_summary(total_time)
        
        # Save run metadata
        self.save_metadata(total_time)
        
        return results
    
    def print_summary(self, total_time):
        """Print benchmark summary"""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"\nRun ID: {self.run_id}")
        print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print()
        
        print("Task Performance:")
        print("-" * 70)
        for task_name, task_time in self.task_times.items():
            errors = self.task_errors.get(task_name, 0)
            print(f"  {task_name:35s} {task_time:8.1f}s   Errors: {errors}")
        
        print("\nResults saved to: benchmark/results/")
        print(f"Metadata: benchmark/results/run_metadata_{self.run_id}.json")
    
    def save_metadata(self, total_time):
        """Save benchmark run metadata"""
        metadata = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'base_url': self.base_url,
            'test_mode': self.test_mode,
            'num_cases': self.num_cases,
            'total_time_seconds': total_time,
            'task_times': self.task_times,
            'task_errors': self.task_errors,
            'tasks': list(self.task_times.keys())
        }
        
        metadata_file = self.results_dir / f"run_metadata_{self.run_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved: {metadata_file}")



def main():
    parser = argparse.ArgumentParser(
        description="Run MIMIC-IV-Ext benchmark with GPU inference"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run on 10 test cases only (for quick testing)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for tracking results (default: medgemma-27b-it)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/medgemma-27b-it',
        help='Path to model directory (default: ./models/medgemma-27b-it)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none','4bit','8bit'],
        default='none',
        help="Quantization mode: 'none'->bfloat16 (default), '4bit' or '8bit'"
    )
    parser.add_argument(
        '--num-cases',
        type=int,
        default=None,
        help='Number of cases to run (default: all available)'
    )
    parser.add_argument(
        '--judge-model-path',
        type=str,
        default=None,
        help='Path to local judge model (e.g., ./models/medgemma-4b-it)'
    )
    parser.add_argument(
        '--judge-quantization',
        type=str,
        choices=['none','4bit','8bit'],
        default='none',
        help="Judge quantization: 'none'->bfloat16 (default), '4bit' or '8bit'"
    )
    
    args = parser.parse_args()
    
    # Create and run benchmark
    # Map CLI quantization flag to internal value (None for bfloat16)
    quant = None if args.quantization == 'none' else args.quantization
    judge_quant = None if args.judge_quantization == 'none' else args.judge_quantization

    runner = BenchmarkRunner(
        model_path=args.model_path,
        model_name=args.model_name,
        num_cases=args.num_cases,
        test_mode=args.test,
        quantization=quant,
        judge_model_path=args.judge_model_path,
        judge_quantization=judge_quant
    )
    
    results = runner.run_all_tasks()
    
    if results:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run: python benchmark/evaluate_results.py")
        print("  2. To compare results and compute metrics")
        return 0
    else:
        print("\nBENCHMARK FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

