## LLM TO EVALUATE DIAGNOSIS PREDICTION
## Using Local LLM via LM Studio
## Updated to work with benchmark results

## Import libraries
import pandas as pd
from tqdm import tqdm
import re
from pathlib import Path

## Import Functions
from functions.LLM_predictions import get_evaluation_diagnosis, create_local_chain, test_local_llm_connection


## Configuration
BASE_URL = "http://localhost:1234/v1"  # LM Studio uses whatever model you loaded
MODEL_NAME = "medgemma-4b-it-mlx"  # Model being evaluated

## Test LM Studio connection
print("Testing local LLM connection...")
if not test_local_llm_connection(BASE_URL, model_name=MODEL_NAME):
    print("\nWARNING: Please start LM Studio server before running this script.")
    exit(1)

print(f"✅ Connected to {MODEL_NAME}")

## Find benchmark result files
results_dir = Path("benchmark/results")
result_files = list(results_dir.glob("diagnosis_specialty_*_medgemma-4b-it-mlx_*.csv"))

# Filter out already evaluated files
result_files = [f for f in result_files if '_llm_evaluated' not in f.name]

if not result_files:
    print("\n❌ No benchmark result files found!")
    print(f"Looking in: {results_dir}")
    print("\nTrying original data file...")
    # Fallback to original data file if exists
    if Path("MIMIC-IV-Ext-Diagnosis-prediction.csv").exists():
        df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-prediction.csv")
        result_files = None
    else:
        exit(1)
else:
    print(f"\nFound {len(result_files)} benchmark result file(s):")
    for f in result_files:
        print(f"  - {f.name}")
    df = None

## Function to parse diagnoses from benchmark output
def parse_diagnoses(text):
    """Extract 3 diagnoses from model output with <diagnosis> tags"""
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


## Function to evaluate diagnosis with simple Yes/No prompt (works with MedGemma-4B)
def get_evaluation_diagnosis_simple(row, pred_col, chain):
    """
    Evaluate each of the 3 predicted diagnoses against ground truth
    Returns: List of 3 True/False/None values
    """
    real_diag = row['diagnosis_list']  # Ground truth
    
    # Parse predictions
    pred_text = row[pred_col]
    predictions = parse_diagnoses(pred_text)
    
    results = []
    
    for pred in predictions:
        if pred is None:
            results.append(None)
            continue
        
        # Simple Yes/No prompt that works with MedGemma-4B
        try:
            response = chain.invoke({
                "real_diag": real_diag,
                "predicted_diag": pred
            }).content
            
            # Parse Yes/No from response
            first_word = response.strip().split()[0].lower().rstrip(',.') if response.strip() else ""
            
            if first_word in ['yes', 'true']:
                results.append(True)
            elif first_word in ['no', 'false']:
                results.append(False)
            else:
                # Fallback: look for Yes/No anywhere in first 20 chars
                first_part = response[:20].lower()
                if 'yes' in first_part or 'true' in first_part:
                    results.append(True)
                elif 'no' in first_part or 'false' in first_part:
                    results.append(False)
                else:
                    results.append(None)
        except Exception as e:
            print(f"  ⚠️  Error evaluating: {e}")
            results.append(None)
    
    return results


## Convert the diagnosis rows into lists - data in columns are stored as strings but actually represent lists
if df is not None:
    # Original data file processing
    prediction_columns = [
        'diagnosis_prediction',
        'diagnosis_prediction_clinical'
    ]

    for col in prediction_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
else:
    prediction_columns = None


## Define the prompt template - Simple Yes/No format that works with MedGemma-4B
prompt = """Real diagnosis: {real_diag}
Predicted diagnosis: {predicted_diag}

Does the predicted diagnosis match the real diagnosis (same meaning or broader category)?
Answer: """


## Create local LLM chain
print("\nCreating LLM evaluation chain...")
chain_local = create_local_chain(prompt, base_url=BASE_URL, model_name=MODEL_NAME)


## Run evaluation
if result_files:
    # Process benchmark result files
    for csv_file in result_files:
        print(f"\n{'='*80}")
        print(f"Evaluating: {csv_file.name}")
        print('='*80)
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Determine user type from filename
        user_type = 'general' if 'general' in csv_file.name else 'clinical'
        pred_col = f'diag_spec_{MODEL_NAME}_{user_type}'
        
        print(f"User type: {user_type}")
        print(f"Prediction column: {pred_col}")
        print(f"Total cases: {len(df)}")
        
        # Check if column exists
        if pred_col not in df.columns:
            print(f"❌ Column '{pred_col}' not found in CSV!")
            continue
        
        # Evaluate predictions
        print(f"\nEvaluating {len(df)} cases with LLM-as-judge...")
        tqdm.pandas(desc="Evaluating")
        
        # Get evaluations for all 3 diagnoses per case
        evaluations = df.progress_apply(
            lambda row: get_evaluation_diagnosis_simple(row, pred_col, chain_local), 
            axis=1
        )
        
        # Split into 3 columns
        df['eval_diag_1'] = evaluations.apply(lambda x: x[0])
        df['eval_diag_2'] = evaluations.apply(lambda x: x[1])
        df['eval_diag_3'] = evaluations.apply(lambda x: x[2])
        
        # Calculate metrics
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        total_predictions = 0
        total_matches = 0
        
        for i in [1, 2, 3]:
            col = f'eval_diag_{i}'
            valid = df[col].notna().sum()
            matches = df[col].sum() if valid > 0 else 0
            
            print(f"\nDiagnosis {i}:")
            print(f"  Valid evaluations: {valid}/{len(df)}")
            if valid > 0:
                print(f"  Matches: {matches}/{valid} ({matches/valid*100:.1f}%)")
            else:
                print(f"  Matches: N/A")
            
            total_predictions += valid
            total_matches += matches
        
        if total_predictions > 0:
            overall_accuracy = total_matches / total_predictions * 100
            print(f"\n{'='*80}")
            print(f"OVERALL ACCURACY: {total_matches}/{total_predictions} = {overall_accuracy:.1f}%")
            print('='*80)
        
        # Save results
        output_path = csv_file.parent / csv_file.name.replace('.csv', '_llm_evaluated.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path.name}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)

else:
    # Original processing for MIMIC-IV-Ext-Diagnosis-prediction.csv
    print("\nEvaluating diagnosis predictions with local LLM...")
    tqdm.pandas()

    # Evaluate only the columns that exist in the dataframe
    for key in prediction_columns:
        if key in df.columns:
            print(f"Evaluating {key}...")
            df["eval_"+key] = df.progress_apply(lambda row: get_evaluation_diagnosis(row, key, chain_local), axis=1)
            df.to_csv('MIMIC-IV-Ext-Diagnosis-evaluation.csv', index=False)

    print("\nDiagnosis evaluation saved to MIMIC-IV-Ext-Diagnosis-evaluation.csv")
