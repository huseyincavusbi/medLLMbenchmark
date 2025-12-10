# Specialty Prediction Evaluation (Clinician Ground Truth)

## About This Benchmark

Uses **331 cases** with **clinician-validated ground truth labels**.

## Results Summary

| Model | Top-1 Fuzzy | Any Match (Top-3) | Empty Preds |
|-------|-------------|-------------------|-------------|
| **Ministral-14B** | **56.8%** | 76.7% | 25 |
| MedGemma-4B-IT | 55.9% | 74.3% | 28 |
| Gemma-3-4B-IT | 55.3% | **81.0%** | 0 |
| DeepSeek-R1-32B | 52.0% | 73.4% | 24 |
| MedGemma-27B-IT | 41.7% | 55.6% | 102 |
| MediPhi-Instruct | 31.1% | 49.8% | 88 |

## Key Findings

1. **Ministral-14B** achieves best Top-1 accuracy (56.8%)
2. **Gemma-3-4B-IT** achieves best Any Match (81.0%) with zero empty predictions
3. **MedGemma-27B-IT** has 31% empty predictions, causing poor accuracy
4. Medical fine-tuning provides marginal Top-1 benefit (+0.6%) but hurts Top-3