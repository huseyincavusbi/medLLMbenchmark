## PREDICT SPECIALTY AND DIAGNOSIS CLINICAL USER CASE
## Using Local LLM via LM Studio

## Import libraries
import pandas as pd
from tqdm import tqdm

## Import functions
from functions.LLM_predictions import get_prediction_ClinicalUser, create_local_chain, test_local_llm_connection


## Configuration
BASE_URL = "http://localhost:1234/v1"  # LM Studio uses whatever model you loaded

## Test LM Studio connection
print("Testing local LLM connection...")
if not test_local_llm_connection(BASE_URL):
    print("\n⚠️  Please start LM Studio server before running this script.")
    exit(1)

## Load Data from create_ground_truth_specialty.py
df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-Specialty.csv")


## Define the prompt template
prompt = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness, personal information and initial vitals. Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses. List specialties first, in order of likelihood, then diagnoses.
Respond with the specialties in <specialty> tags and the diagnoses in <diagnosis> tags.
History of present illness: {hpi}, personal information: {patient_info} and initial vitals: {initial_vitals}."""


## Create local LLM chain
chain_local = create_local_chain(prompt, base_url=BASE_URL)


## Run predictions
print("\nRunning clinical diagnosis/specialty predictions with local LLM...")
tqdm.pandas()
df['diag_spec_prediction_clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_local), axis=1)
df.to_csv('MIMIC-IV-Ext-Diagnosis-Specialty.csv', index=False)

print("\n✅ Predictions saved to MIMIC-IV-Ext-Diagnosis-Specialty.csv")