#!/usr/bin/env python3
## COMPREHENSIVE TEST - 10 Cases Across All Tasks
## Tests: Triage (General & Clinical), Diagnosis, and Specialty Predictions

import pandas as pd
import time
from functions.LLM_predictions import (
    get_prediction_GeneralUser, 
    get_prediction_ClinicalUser,
    create_local_chain, 
    test_local_llm_connection
)
import re

## Configuration
MODEL_NAME = "MedGemma-4B-IT"
BASE_URL = "http://192.168.1.203:1234/v1"
NUM_CASES = 10

def parse_triage_level(response):
    """Parse triage level from LLM response"""
    if not response or not isinstance(response, str):
        return None
    
    # Try <acuity> tag first
    match = re.search(r'<acuity>(\d+)</acuity>', response, re.IGNORECASE)
    if match:
        level = int(match.group(1))
        if 1 <= level <= 5:
            return level
    
    # Try **ESI Level:** **number**
    match = re.search(r'\*\*ESI Level:\*\*\s*\*\*(\d+)\*\*', response)
    if match:
        level = int(match.group(1))
        if 1 <= level <= 5:
            return level
    
    # Try any standalone digit 1-5
    digits = re.findall(r'\b([1-5])\b', response)
    if digits:
        return int(digits[0])
    
    return None

print("="*80)
print("COMPREHENSIVE LOCAL LLM TEST")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Test Cases: {NUM_CASES}")
print(f"Tasks: Triage (General + Clinical), Diagnosis, Specialty")
print("="*80)

## Test connection
print("\n1. Testing LM Studio connection...")
if not test_local_llm_connection(BASE_URL):
    print("\n⚠️  LM Studio not ready. Please ensure:")
    print("   - LM Studio is open")
    print("   - Model is loaded in the server")
    print("   - Server is running")
    exit(1)

## Load raw clinical data
print(f"\n2. Loading clinical data...")
try:
    clinical_data = pd.read_csv("../physionet.org/files/mimic-iv-ext-cds/1.0.2/clinical_data.csv")
    print(f"   ✓ Loaded {len(clinical_data)} total cases")
except FileNotFoundError:
    print("   ⚠️  Clinical data not found")
    exit(1)

# Load ground truth data for comparison
try:
    triage_gt = pd.read_csv("../physionet.org/files/mimic-iv-ext-cds/1.0.2/triage_level.csv")
    specialty_gt = pd.read_csv("../physionet.org/files/mimic-iv-ext-cds/1.0.2/specialty_referral_clinician_approved.csv")
    diagnosis_gt = pd.read_csv("../physionet.org/files/mimic-iv-ext-cds/1.0.2/diagnosis.csv")
    vitals = pd.read_csv("../physionet.org/files/mimic-iv-ext-cds/1.0.2/vital_signs.csv")
    demographics = pd.read_csv("../physionet.org/files/mimic-iv-ext-cds/1.0.2/patient_demographics.csv")
    
    print(f"   ✓ Loaded ground truth data")
except Exception as e:
    print(f"   ⚠️  Could not load some ground truth files: {e}")
    triage_gt = specialty_gt = diagnosis_gt = vitals = demographics = None

# Select test cases
df_test = clinical_data.head(NUM_CASES).copy()
print(f"   ✓ Selected first {NUM_CASES} cases for testing")

# Merge with ground truth and supporting data
if triage_gt is not None:
    cols_to_merge = ['stay_id']
    if 'triage' in triage_gt.columns:
        cols_to_merge.append('triage')
    if 'patient_info' in triage_gt.columns:
        cols_to_merge.append('patient_info')
    if 'initial_vitals' in triage_gt.columns:
        cols_to_merge.append('initial_vitals')
    df_test = df_test.merge(triage_gt[cols_to_merge], on='stay_id', how='left')

if specialty_gt is not None:
    cols_to_merge = ['stay_id']
    for col in ['specialty', 'specialty_referral', 'specialty_clinician']:
        if col in specialty_gt.columns:
            cols_to_merge.append(col)
            break
    if len(cols_to_merge) > 1:
        df_test = df_test.merge(specialty_gt[cols_to_merge], on='stay_id', how='left')

if diagnosis_gt is not None:
    cols_to_merge = ['stay_id']
    if 'diagnosis' in diagnosis_gt.columns:
        cols_to_merge.append('diagnosis')
    if len(cols_to_merge) > 1:
        df_test = df_test.merge(diagnosis_gt[cols_to_merge], on='stay_id', how='left', suffixes=('', '_gt'))

print("\n" + "="*80)
print("TASK 1: TRIAGE PREDICTION (GENERAL USER)")
print("="*80)

# Triage prompt - General User (patient perspective, no vitals)
triage_prompt_general = """You are a nurse with emergency and triage experience. Using the patient's history of present illness and his information, determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity):
1: Assign if the patient requires immediate lifesaving intervention.
2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress)
3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care.
4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG).
5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).
History of present illness: {HPI} and patient info: {patient_info}. Respond with the level in an <acuity> tag."""

print("\nCreating chain for general user triage...")
chain_triage_general = create_local_chain(triage_prompt_general, base_url=BASE_URL)

print(f"\nRunning predictions on {NUM_CASES} cases...")
start_time = time.time()

triage_general_results = []
for idx, row in df_test.iterrows():
    print(f"  Case {idx+1}/{NUM_CASES}...", end=" ", flush=True)
    try:
        raw_response = get_prediction_GeneralUser(row, chain_triage_general)
        prediction = parse_triage_level(raw_response)
        if prediction is not None:
            triage_general_results.append(prediction)
            print(f"✓ ESI {prediction}")
        else:
            triage_general_results.append(f"ERROR: Could not parse response")
            print(f"✗ Could not parse: {raw_response[:50]}...")
    except Exception as e:
        triage_general_results.append(f"ERROR: {str(e)}")
        print(f"✗ Error: {e}")

df_test[f'triage_{MODEL_NAME}_general'] = triage_general_results
elapsed = time.time() - start_time
print(f"\n✓ General user triage complete in {elapsed:.1f}s ({elapsed/NUM_CASES:.1f}s per case)")

print("\n" + "="*80)
print("TASK 2: TRIAGE PREDICTION (CLINICAL USER)")
print("="*80)

# Triage prompt - Clinical User (provider perspective, with vitals)
triage_prompt_clinical = """You are a nurse with emergency and triage experience. Using the patient's history of present illness, his information and initial vitals, determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity):
1: Assign if the patient requires immediate lifesaving intervention.
2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress)
3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care.
4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG).
5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).
History of present illness: {HPI}, patient info: {patient_info} and initial vitals: {initial_vitals}. Respond with the level in an <acuity> tag."""

print("\nCreating chain for clinical user triage...")
chain_triage_clinical = create_local_chain(triage_prompt_clinical, base_url=BASE_URL)

print(f"\nRunning predictions on {NUM_CASES} cases...")
start_time = time.time()

triage_clinical_results = []
for idx, row in df_test.iterrows():
    print(f"  Case {idx+1}/{NUM_CASES}...", end=" ", flush=True)
    try:
        raw_response = get_prediction_ClinicalUser(row, chain_triage_clinical)
        prediction = parse_triage_level(raw_response)
        if prediction is not None:
            triage_clinical_results.append(prediction)
            print(f"✓ ESI {prediction}")
        else:
            triage_clinical_results.append(f"ERROR: Could not parse response")
            print(f"✗ Could not parse: {raw_response[:50]}...")
    except Exception as e:
        triage_clinical_results.append(f"ERROR: {str(e)}")
        print(f"✗ Error: {e}")

df_test[f'triage_{MODEL_NAME}_clinical'] = triage_clinical_results
elapsed = time.time() - start_time
print(f"\n✓ Clinical user triage complete in {elapsed:.1f}s ({elapsed/NUM_CASES:.1f}s per case)")

print("\n" + "="*80)
print("TASK 3: DIAGNOSIS & SPECIALTY PREDICTION (GENERAL USER)")
print("="*80)

# Diagnosis & Specialty prompt - General User
diag_spec_prompt_general = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness and personal information. Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses. List specialties first, in order of likelihood, then diagnoses. 
Respond with the specialties in <specialty> tags and the diagnoses in <diagnosis> tags.
History of present illness: {hpi} and personal information: {patient_info}."""

print("\nCreating chain for diagnosis/specialty (general user)...")
chain_diag_spec_general = create_local_chain(diag_spec_prompt_general, base_url=BASE_URL)

print(f"\nRunning predictions on {NUM_CASES} cases...")
start_time = time.time()

diag_spec_general_results = []
for idx, row in df_test.iterrows():
    print(f"  Case {idx+1}/{NUM_CASES}...", end=" ", flush=True)
    try:
        # Create a row dict with lowercase keys for the prompt
        row_dict = {'hpi': row.get('hpi', row.get('HPI', '')), 
                   'patient_info': row.get('patient_info', '')}
        response = chain_diag_spec_general.invoke(row_dict)
        prediction = response.content
        diag_spec_general_results.append(prediction)
        print(f"✓")
    except Exception as e:
        diag_spec_general_results.append(f"ERROR: {str(e)}")
        print(f"✗ Error: {e}")

df_test[f'diag_spec_{MODEL_NAME}_general'] = diag_spec_general_results
elapsed = time.time() - start_time
print(f"\n✓ Diagnosis/specialty (general) complete in {elapsed:.1f}s ({elapsed/NUM_CASES:.1f}s per case)")

print("\n" + "="*80)
print("TASK 4: DIAGNOSIS & SPECIALTY PREDICTION (CLINICAL USER)")
print("="*80)

# Diagnosis & Specialty prompt - Clinical User
diag_spec_prompt_clinical = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness, personal information and initial vitals. Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses. List specialties first, in order of likelihood, then diagnoses.
Respond with the specialties in <specialty> tags and the diagnoses in <diagnosis> tags.
History of present illness: {hpi}, personal information: {patient_info} and initial vitals: {initial_vitals}."""

print("\nCreating chain for diagnosis/specialty (clinical user)...")
chain_diag_spec_clinical = create_local_chain(diag_spec_prompt_clinical, base_url=BASE_URL)

print(f"\nRunning predictions on {NUM_CASES} cases...")
start_time = time.time()

diag_spec_clinical_results = []
for idx, row in df_test.iterrows():
    print(f"  Case {idx+1}/{NUM_CASES}...", end=" ", flush=True)
    try:
        prediction = get_prediction_ClinicalUser(row, chain_diag_spec_clinical)
        diag_spec_clinical_results.append(prediction)
        print(f"✓")
    except Exception as e:
        diag_spec_clinical_results.append(f"ERROR: {str(e)}")
        print(f"✗ Error: {e}")

df_test[f'diag_spec_{MODEL_NAME}_clinical'] = diag_spec_clinical_results
elapsed = time.time() - start_time
print(f"\n✓ Diagnosis/specialty (clinical) complete in {elapsed:.1f}s ({elapsed/NUM_CASES:.1f}s per case)")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_file = f"comprehensive_test_{NUM_CASES}_cases.csv"
df_test.to_csv(output_file, index=False)
print(f"✓ Results saved to: {output_file}")

# Generate summary report
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

# Triage accuracy (General User)
if 'triage' in df_test.columns:
    valid_triage = df_test[df_test['triage'].notna()]
    if len(valid_triage) > 0:
        correct = sum(valid_triage['triage'].astype(str) == valid_triage[f'triage_{MODEL_NAME}_general'].astype(str))
        within_1 = sum(abs(valid_triage['triage'].astype(int) - 
                          valid_triage[f'triage_{MODEL_NAME}_general'].astype(int)) <= 1)
        
        print(f"\n📊 TRIAGE (GENERAL USER) - {len(valid_triage)} cases:")
        print(f"   Exact Match: {correct}/{len(valid_triage)} = {100*correct/len(valid_triage):.1f}%")
        print(f"   Within 1 Level: {within_1}/{len(valid_triage)} = {100*within_1/len(valid_triage):.1f}%")

# Triage accuracy (Clinical User)
if 'triage' in df_test.columns:
    valid_triage = df_test[df_test['triage'].notna()]
    if len(valid_triage) > 0:
        correct = sum(valid_triage['triage'].astype(str) == valid_triage[f'triage_{MODEL_NAME}_clinical'].astype(str))
        within_1 = sum(abs(valid_triage['triage'].astype(int) - 
                          valid_triage[f'triage_{MODEL_NAME}_clinical'].astype(int)) <= 1)
        
        print(f"\n📊 TRIAGE (CLINICAL USER) - {len(valid_triage)} cases:")
        print(f"   Exact Match: {correct}/{len(valid_triage)} = {100*correct/len(valid_triage):.1f}%")
        print(f"   Within 1 Level: {within_1}/{len(valid_triage)} = {100*within_1/len(valid_triage):.1f}%")

# Prediction distribution
print(f"\n📈 TRIAGE DISTRIBUTION:")
print(f"   General User:")
print(df_test[f'triage_{MODEL_NAME}_general'].value_counts().to_string())
print(f"\n   Clinical User:")
print(df_test[f'triage_{MODEL_NAME}_clinical'].value_counts().to_string())

print("\n" + "="*80)
print("✅ COMPREHENSIVE TEST COMPLETE!")
print("="*80)
print(f"\nResults saved to: {output_file}")
print(f"\nTasks completed:")
print(f"  ✓ Triage prediction (general user) - {NUM_CASES} cases")
print(f"  ✓ Triage prediction (clinical user) - {NUM_CASES} cases")
print(f"  ✓ Diagnosis/specialty (general user) - {NUM_CASES} cases")
print(f"  ✓ Diagnosis/specialty (clinical user) - {NUM_CASES} cases")
print(f"\nTotal predictions: {NUM_CASES * 4} = {NUM_CASES * 4} predictions")
print("\n" + "="*80)
