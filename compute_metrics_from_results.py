#!/usr/bin/env python3
"""Compute simple evaluation metrics from comprehensive_test_100_cases.csv

Metrics produced:
- triage exact / within-1 for general and clinical
- specialty/diagnosis top-1 and top-3 (best-effort parser)

Saves metrics to metrics_summary.json and prints a brief report.
"""
import json
import re
from collections import Counter
import pandas as pd

RESULTS_CSV = "comprehensive_test_100_cases.csv"

# Helper parsers
ESI_RE = re.compile(r"ESI[^0-9]*([1-5])", flags=re.IGNORECASE)
DIGIT_RE = re.compile(r"\b([1-5])\b")
TAG_RE = re.compile(r"<([^>]+)>(.*?)</\1>", flags=re.IGNORECASE|re.DOTALL)


def parse_esi(v):
    if pd.isna(v):
        return None
    s = str(v)
    m = DIGIT_RE.search(s)
    if m:
        return int(m.group(1))
    m = ESI_RE.search(s)
    if m:
        return int(m.group(1))
    return None


def extract_tags(s, tagname=None):
    """Return list of values inside tags. If tagname is provided, only that tag is considered."""
    if pd.isna(s):
        return []
    s = str(s)
    results = []
    for m in TAG_RE.finditer(s):
        tag = m.group(1).lower()
        content = m.group(2).strip()
        if tagname is None or tag == tagname.lower():
            # split by commas/newlines/semicolons and strip
            parts = re.split(r"[,;\n]+", content)
            for p in parts:
                t = p.strip()
                if t:
                    results.append(t)
    return results


if __name__ == '__main__':
    df = pd.read_csv(RESULTS_CSV)
    out = {}

    # Show columns and counts
    out['columns'] = df.columns.tolist()
    out['n_rows'] = len(df)

    # Triage metrics
    for role in ['general', 'clinical']:
        pred_col = [c for c in df.columns if c.startswith(f'triage_') and c.endswith(f'_{role}')]
        if not pred_col:
            continue
        pred_col = pred_col[0]
        df[f'{pred_col}_int'] = df[pred_col].apply(parse_esi)

    if 'triage' in df.columns:
        df['triage_int'] = df['triage'].apply(parse_esi)
        gt = df[df['triage_int'].notna()]
        out['triage_total_gt'] = len(gt)
        for role in ['general', 'clinical']:
            pred_col = f'triage_{MODEL_NAME}_' + role if 'MODEL_NAME' in globals() else None
    
    # Instead, compute using detected columns
    triage_pred_cols = [c for c in df.columns if c.startswith('triage_') and c.endswith('_general') or c.endswith('_clinical')]
    triage_metrics = {}
    if 'triage_int' in df.columns:
        for col in [c for c in df.columns if c.startswith('triage_') and c.endswith('_general') or c.endswith('_clinical')]:
            if not col.endswith('_int'):
                int_col = col + '_int'
                if int_col in df.columns:
                    total = df['triage_int'].notna().sum()
                    valid = df[df['triage_int'].notna()]
                    correct = (valid['triage_int'] == valid[int_col]).sum()
                    within1 = ((valid['triage_int'] - valid[int_col]).abs() <= 1).sum()
                    triage_metrics[col] = {
                        'total_gt': int(total),
                        'exact': int(correct),
                        'within_1': int(within1),
                        'exact_pct': 100*correct/total if total>0 else None,
                        'within_1_pct': 100*within1/total if total>0 else None
                    }
    out['triage'] = triage_metrics

    # Specialty/Diagnosis metrics - best effort: compare top-1 match with ground truth columns if present
    def topk_match(gt_values, pred_values, k=3):
        # gt_values: single string or list
        if pd.isna(gt_values):
            return False
        if isinstance(gt_values, str):
            gt_list = [x.strip() for x in re.split(r'[,;\n]+', gt_values) if x.strip()]
        else:
            gt_list = list(gt_values)
        pred_list = pred_values[:k]
        # lowercase compare tokens
        gt_set = set([g.lower() for g in gt_list])
        for p in pred_list:
            if p.lower() in gt_set:
                return True
        return False

    spec_metrics = {}
    diag_metrics = {}
    # find predicted diag/spec cols
    pred_spec_cols = [c for c in df.columns if 'diag_spec' in c and 'general' in c]
    # we'll look for columns named 'specialty' or 'diagnosis' ground truth
    gt_spec_cols = [c for c in df.columns if 'specialty' in c]
    gt_diag_cols = [c for c in df.columns if 'diagnos' in c and 'diag_spec' not in c]

    # For each prediction column, try to extract tags
    for col in [c for c in df.columns if 'diag_spec' in c]:
        tag_spec = []
        tag_diag = []
        preds = df[col].fillna('')
        parsed_specs = preds.apply(lambda s: extract_tags(s, 'specialty'))
        parsed_diags = preds.apply(lambda s: extract_tags(s, 'diagnosis'))
        # fallback: try to split by newlines and take first 3 lines
        def fallback_extract(s):
            s = str(s)
            parts = [p.strip() for p in re.split(r'\n|;|,', s) if p.strip()]
            return parts[:3]
        parsed_specs = parsed_specs.apply(lambda lst, s=None: lst if len(lst)>0 else fallback_extract(s))
        # compute top-1/top-3 against gt if exist
        if gt_spec_cols:
            gtcol = gt_spec_cols[0]
            total = df[gtcol].notna().sum()
            top1 = sum([1 for i,row in df.iterrows() if topk_match(row[gtcol], extract_tags(row[col], 'specialty'), k=1)])
            top3 = sum([1 for i,row in df.iterrows() if topk_match(row[gtcol], extract_tags(row[col], 'specialty'), k=3)])
            spec_metrics[col] = {'total_gt': int(total), 'top1': int(top1), 'top3': int(top3), 'top1_pct': 100*top1/total if total>0 else None, 'top3_pct': 100*top3/total if total>0 else None}
        if gt_diag_cols:
            gtcol = gt_diag_cols[0]
            total = df[gtcol].notna().sum()
            top1 = sum([1 for i,row in df.iterrows() if topk_match(row[gtcol], extract_tags(row[col], 'diagnosis'), k=1)])
            top3 = sum([1 for i,row in df.iterrows() if topk_match(row[gtcol], extract_tags(row[col], 'diagnosis'), k=3)])
            diag_metrics[col] = {'total_gt': int(total), 'top1': int(top1), 'top3': int(top3), 'top1_pct': 100*top1/total if total>0 else None, 'top3_pct': 100*top3/total if total>0 else None}

    out['specialty_metrics'] = spec_metrics
    out['diagnosis_metrics'] = diag_metrics

    # save summary
    with open('metrics_summary.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
