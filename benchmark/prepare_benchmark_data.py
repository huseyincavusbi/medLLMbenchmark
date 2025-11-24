#!/usr/bin/env python3
"""
Prepare MIMIC-IV-Ext Benchmark Datasets
========================================
Creates the required CSV files for benchmark from PhysioNet data.

The PhysioNet MIMIC-IV-Ext-CDS dataset already has properly formatted
files, so we just need to copy/rename them to the expected locations.
"""

import pandas as pd
from pathlib import Path
import shutil

# Paths
DATA_DIR = Path("/home/ubuntu/physionet.org/files/mimic-iv-ext-cds/1.0.2")
OUTPUT_DIR = Path(__file__).parent

print("=" * 70)
print("MIMIC-IV-Ext Benchmark Data Preparation")
print("=" * 70)
print()

# Load and process Triage dataset
print("Preparing Triage dataset...")
triage = pd.read_csv(DATA_DIR / "triage_level.csv")
print(f"  ✓ Loaded {len(triage)} triage records")

# Save to benchmark location
triage_path = OUTPUT_DIR / "MIMIC-IV-Ext-Triage.csv"
triage.to_csv(triage_path, index=False)
print(f"  ✓ Saved to: {triage_path}")
print()

# Load and process Diagnosis dataset  
print("Preparing Diagnosis & Specialty dataset...")
diagnosis = pd.read_csv(DATA_DIR / "diagnosis.csv")
specialty = pd.read_csv(DATA_DIR / "specialty_referral_clinician_approved.csv")
print(f"  ✓ Loaded {len(diagnosis)} diagnosis records")
print(f"  ✓ Loaded {len(specialty)} specialty records")

# Merge diagnosis and specialty
# First, let's see what we have
print("\n  Checking data structure...")
print(f"  Diagnosis columns: {diagnosis.columns.tolist()}")
print(f"  Specialty columns: {specialty.columns.tolist()}")

# Merge on stay_id
diag_spec = diagnosis.merge(
    specialty[['stay_id', 'specialty clinician approved']],
    on='stay_id',
    how='inner'
)
print(f"  ✓ Merged {len(diag_spec)} records with both diagnosis and specialty")

# Rename specialty column
diag_spec = diag_spec.rename(columns={'specialty clinician approved': 'specialty_list'})

# The diagnosis column already contains lists, just need to rename
diag_spec = diag_spec.rename(columns={'diagnosis': 'diagnosis_list'})

# Save Diagnosis & Specialty dataset
diag_spec_path = OUTPUT_DIR / "MIMIC-IV-Ext-Diagnosis-Specialty.csv"
diag_spec.to_csv(diag_spec_path, index=False)
print(f"  ✓ Saved to: {diag_spec_path}")
print()

print("=" * 70)
print("Data preparation completed!")
print("=" * 70)
print()
print("Created files:")
print(f"  1. {triage_path} ({len(triage)} cases)")
print(f"  2. {diag_spec_path} ({len(diag_spec)} cases)")
print()
print("You can now run the benchmark:")
print("  python benchmark/run_benchmark.py --test")

