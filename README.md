# LLM Benchmark for Clinical Decision Support

A comprehensive benchmark for evaluating Large Language Models on clinical decision-making tasks: **Triage** and **Specialty Referral**.

## Key Findings

### 1. Model Size Does NOT Equal Performance
Smaller 4B models consistently outperform 27B+ models. **Gemma-3-4B-IT** achieves the best triage accuracy (56.9%) and specialty accuracy on clinician-validated data (81.0%).

### 2. Medical Fine-tuning Has Trade-offs
MedGemma models show degraded instruction following (-26%) compared to base Gemma. Medical fine-tuning breaks format compliance by biasing models toward medical terminology.

### 3. LLM-Generated Ground Truth is Biased
Scores on LLM-generated specialty labels can differ by up to 25% from clinician-validated labels. Always prefer human-validated ground truth when available.

### 4. Instruction Following is Critical
Models with poor format compliance (e.g., MediPhi-Instruct at 25.6%) lose up to 48% of predictions to parsing failures.

## Quick Results

| Model | Triage Exact | Triage W1 | Specialty (Clinician GT) |
|-------|--------------|-----------|--------------------------|
| **Gemma-3-4B-IT** | **56.9%** | 67.1% | **81.0%** |
| Ministral-14B | 53.8% | 55.3% | 76.7% |
| MedGemma-4B-IT | 53.8% | 54.6% | 74.3% |
| DeepSeek-R1-32B | 50.9% | **71.4%** | 73.4% |
| MedGemma-27B-IT | 44.3% | 51.6% | 55.6% |

See [results/README.md](results/README.md) for full metrics.

## Dataset

This benchmark uses the **MIMIC-IV-Ext Clinical Decision Support** dataset:
- **9,149 emergency department cases** for triage and diagnosis
- **2,200 cases** with LLM-generated specialty labels
- **331 cases** with clinician-validated specialty labels (gold standard)

## Credits

### Original Paper
This benchmark is based on the methodology from:

> **Evaluating large language model workflows in clinical decision support for triage and referral and diagnosis**  
> Farieda Gaber, Maqsood Shaik, Fabio Allega, Agnes Julia Bilecz, Felix Busch, Kelsey Goon,
Vedran Franke & Altuna Akalin
> https://doi.org/10.1038/s41746-025-01684-1 - https://www.nature.com/articles/s41746-025-01684-1

### Dataset
> **MIMIC-IV-Ext clinical decision support for referral, triage and diagnosis v1.0.2**  
> PhysioNet: [https://physionet.org/content/mimic-iv-ext-cds/1.0.2/](https://physionet.org/content/mimic-iv-ext-cds/1.0.2/)

## Acknowledgments

This project was powered by GPU resources generously provided by [NexGen Cloud](https://www.nexgencloud.com) through their [Hyperstack](https://www.hyperstack.cloud) platform.

## License

This project is licensed under the [MIT License](LICENSE).
