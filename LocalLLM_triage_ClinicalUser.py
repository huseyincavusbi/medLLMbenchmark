## PREDICT TRIAGE/ACUITY CLINICAL USER CASE
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
df = pd.read_csv("MIMIC-IV-Ext-Triage.csv")


## Define the prompt template
prompt = """You are a nurse with emergency and triage experience. Using the patient's history of present illness, his information and initial vitals, determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity): 1: Assign if the patient requires immediate lifesaving intervention. 2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress)  3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care. 4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG). 5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).
History of present illness: {HPI}, patient info: {patient_info} and initial vitals: {initial_vitals}. Respond with the level in an <acuity> tag."""


## Create local LLM chain
chain_local = create_local_chain(prompt, base_url=BASE_URL)


## Run predictions
print("\nRunning clinical triage predictions with local LLM...")
tqdm.pandas()
df['triage_prediction_clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_local), axis=1)
df.to_csv('MIMIC-IV-Ext-Triage.csv', index=False)

print("\n✅ Predictions saved to MIMIC-IV-Ext-Triage.csv")
