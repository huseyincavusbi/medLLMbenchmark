# Full Benchmark Results Summary

## About This Benchmark

Uses **9,149 emergency department cases** from MIMIC-IV-Ext clinical decision support for referral, triage and diagnosis v1.0.2

## Triage Performance (Strict Parsing)

Uses only `<acuity>X</acuity>` format. Within-1 uses paper methodology (over-triage allowed, under-triage forbidden).

| Model | General Exact | General Within-1 | Clinical Exact | Clinical Within-1 | Valid % |
|-------|-----------|--------|------------|---------|---------|
| **Gemma-3-4B-IT** | **56.9%** | **67.2%** | **56.4%** | 62.1% | 100% |
| Ministral-14B | 53.8% | 55.2% | 53.8% | 55.4% | 99.5% |
| MedGemma-4B-IT | 53.2% | 55.0% | 53.4% | 54.8% | 100% |
| DeepSeek-R1-32B | 50.2% | 69.8% | 51.7% | **70.9%** | 98.5% |
| MediPhi-Instruct | 49.2% | 59.0% | 51.9% | 61.5% | 66.0% |
| MedGemma-27B-IT | 38.7% | 49.9% | 46.1% | 51.4% | 72.3% |

## Triage Performance (Flexible Parsing)

Accepts all tag formats (`<acuity>`, `<esi_level_X>`, `<3>`, etc.)

| Model | General Exact | General Within-1 | Clinical Exact | Clinical Within-1 |
|-------|-----------|--------|------------|---------|
| **Gemma-3-4B-IT** | **56.9%** | 67.1% | **56.4%** | 62.1% |
| MedGemma-4B-IT | 53.8% | 54.6% | 53.8% | 54.5% |
| Ministral-14B | 53.8% | 55.3% | 53.8% | 55.4% |
| DeepSeek-R1-32B | 50.9% | **70.4%** | 52.0% | **71.4%** |
| MediPhi-Instruct | 52.0% | 60.0% | 54.0% | 62.0% |
| MedGemma-27B-IT | 44.3% | 51.6% | 49.9% | 53.2% |

## Specialty Performance (LLM GT - 2,200 cases)

Proportional Match = matches/min(preds,GT), Any Match = binary.

| Model | General Proportional | General Any Match | Clinical Proportional | Clinical Any Match |
|-------|----------|---------|-----------|----------|
| **MedGemma-27B-IT** | **66.7%** | **80.4%** | **67.3%** | **81.0%** |
| Ministral-14B | 64.1% | 75.6% | 64.0% | 76.7% |
| Gemma-3-4B-IT | 63.7% | 77.0% | 64.4% | 77.8% |
| MedGemma-4B-IT | 63.6% | 77.2% | 64.7% | 78.4% |
| DeepSeek-R1-32B | 63.8% | 75.1% | 64.2% | 76.5% |
| MediPhi-Instruct | 52.6% | 64.9% | 58.1% | 69.4% |

## Instruction Following (Format Compliance)

| Model | Triage | Specialty | Overall |
|-------|--------|-----------|---------|
| **Ministral-14B** | **97.1%** | 92.6% | **91.1%** |
| Gemma-3-4B-IT | 75.1% | **100%** | 91.7% |
| DeepSeek-R1-32B | 76.8% | 88.7% | 84.3% |
| MedGemma-4B-IT | 0.3% | 98.4% | 65.6% |
| MedGemma-27B-IT | 6.1% | 95.3% | 65.6% |
| MediPhi-Instruct | 0.4% | 35.0% | 25.6% |

## Key Findings

### 1. Model Size Does Not Equal Performance
4B models (Gemma-3-4B, MedGemma-4B) outperform larger counterparts for triage.

### 2. Medical Fine-tuning Helps Specialty
MedGemma-27B-IT leads specialty prediction (81% Any Match) despite poor triage performance.

### 3. Safety vs Accuracy Trade-off
DeepSeek-R1-32B has highest within-1 safety (71.4%) but lower exact match. It over-triages frequently.