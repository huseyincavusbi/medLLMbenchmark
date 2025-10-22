## LLM TO EVALUATE DIAGNOSIS PREDICTION
## Using Local LLM via LM Studio

## Import libraries
import pandas as pd
from tqdm import tqdm

## Import Functions
from functions.LLM_predictions import get_evaluation_diagnosis, create_local_chain, test_local_llm_connection


## Configuration
BASE_URL = "http://localhost:1234/v1"  # LM Studio uses whatever model you loaded

## Test LM Studio connection
print("Testing local LLM connection...")
if not test_local_llm_connection(BASE_URL):
    print("\nWARNING: Please start LM Studio server before running this script.")
    exit(1)

## Load Data from additional_prostprocessing.py
df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-prediction.csv")


## Convert the diagnosis rows into lists - data in columns are stored as strings but actually represent lists
# Prediction columns from the local LLM scripts (using standard column names)
prediction_columns = [
    'diagnosis_prediction',
    'diagnosis_prediction_clinical'
]

for col in prediction_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)


## Define the prompt template
prompt = """You are an experienced healthcare professional with expertise in medical and clinical domains. I will provide a list of real diagnoses for a patient and 3 predicted diagnoses. For each predicted diagnosis, determine if it has the same meaning as one of the real diagnoses or if the prediction falls under a broader category of one of the real diagnoses (e.g., a specific condition falling under a general diagnosis category). If it matches, return 'True'; otherwise, return 'False'. Return only 'True' or 'False' for each predicted diagnosis within <evaluation> tags and nothing else.
Real Diagnoses: {real_diag}, predicted diagnosis 1: {diag1}, predicted diagnosis 2: {diag2}, and predicted diagnosis 3: {diag3}."""


## Create local LLM chain
chain_local = create_local_chain(prompt, base_url=BASE_URL)


## Run evaluation
print("\nEvaluating diagnosis predictions with local LLM...")
tqdm.pandas()

# Evaluate only the columns that exist in the dataframe
for key in prediction_columns:
    if key in df.columns:
        print(f"Evaluating {key}...")
        df["eval_"+key] = df.progress_apply(lambda row: get_evaluation_diagnosis(row, key, chain_local), axis=1)
        df.to_csv('MIMIC-IV-Ext-Diagnosis-evaluation.csv', index=False)

print("\nDiagnosis evaluation saved to MIMIC-IV-Ext-Diagnosis-evaluation.csv")
