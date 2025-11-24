## PREDICT SPECIALTY AND DIAGNOSIS GENERAL USER CASE
## Using GPU inference with HuggingFace Transformers

## Import libraries
import pandas as pd
from tqdm import tqdm

## Import Functions
from functions.LLM_predictions import get_prediction_GeneralUser, create_chain, test_gpu_connection


## Configuration
MODEL_PATH = "./models/medgemma-27b-it"  # Path to downloaded model

## Test GPU connection
print("Testing GPU connection and model loading...")
if not test_gpu_connection(MODEL_PATH):
    print("\nERROR: GPU not available or model failed to load.")
    exit(1)

## Load Data from create_ground_truth_specialty.py
df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-Specialty.csv")


## Define the prompt template
prompt = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness and personal information. Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses. List specialties first, in order of likelihood, then diagnoses. 
Respond with the specialties in <specialty> tags and the diagnoses in <diagnosis> tags.
History of present illness: {hpi} and personal information: {patient_info}."""


## Create GPU chain
chain_local = create_chain(prompt, model_path=MODEL_PATH)


## Run predictions
print("\nRunning diagnosis/specialty predictions with GPU model...")
tqdm.pandas()
df['diag_spec_prediction'] = df.progress_apply(lambda row: get_prediction_GeneralUser(row, chain_local), axis=1)
df.to_csv('MIMIC-IV-Ext-Diagnosis-Specialty.csv', index=False)

print("\nPredictions saved to MIMIC-IV-Ext-Diagnosis-Specialty.csv")

