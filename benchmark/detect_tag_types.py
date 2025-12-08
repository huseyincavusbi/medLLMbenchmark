#!/usr/bin/env python3
"""
Detect Tag Types Used by Models
================================
Analyzes model outputs to identify all XML tag patterns used for triage,
specialty, and diagnosis predictions.

This helps identify non-compliant tag formats that can then be parsed.
"""

import sys
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def extract_tags(text):
    """Extract all XML-like tags from text"""
    if pd.isna(text):
        return []
    text = str(text)
    # Match opening tags: <tagname> or <tagname ...>
    tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9_\- ]*)[^>]*>', text)
    return [t.strip().lower() for t in tags]


def extract_tag_patterns(text):
    """Extract full tag patterns (opening + content + closing)"""
    if pd.isna(text):
        return []
    text = str(text)[:500]  # Limit to first 500 chars
    
    patterns = []
    
    # Standard XML patterns: <tag>content</tag>
    standard = re.findall(r'<([a-zA-Z_][a-zA-Z0-9_]*)[^>]*>([^<]*)</\1>', text, re.IGNORECASE)
    for tag, content in standard:
        patterns.append(f'<{tag.lower()}>...</{tag.lower()}>')
    
    # Self-closing or unclosed: <tag>
    unclosed = re.findall(r'<([a-zA-Z_][a-zA-Z0-9_\- ]*[0-9]?)>', text)
    for tag in unclosed:
        if not any(tag.lower() in p for p in patterns):
            patterns.append(f'<{tag.lower()}>')
    
    return patterns


def analyze_triage_tags(filepath):
    """Analyze tag patterns in triage predictions"""
    df = pd.read_csv(filepath)
    pred_col = [c for c in df.columns if 'triage_' in c and ('_general' in c or '_clinical' in c)][0]
    
    tag_counter = Counter()
    first_tag_counter = Counter()
    
    for text in df[pred_col].dropna():
        text = str(text)[:200]
        patterns = extract_tag_patterns(text)
        for p in patterns:
            tag_counter[p] += 1
        
        # First tag pattern
        first = re.match(r'^<([^>]+)>', text)
        if first:
            first_tag_counter[f'<{first.group(1).lower()}>'] += 1
        else:
            # Check for non-tag start
            first_word = text[:20].split()[0] if text.split() else ''
            first_tag_counter[f'starts_with: {first_word[:15]}'] += 1
    
    return {
        'total_predictions': len(df),
        'tag_patterns': dict(tag_counter.most_common(10)),
        'first_tag': dict(first_tag_counter.most_common(10))
    }


def analyze_diag_spec_tags(filepath):
    """Analyze tag patterns in diagnosis/specialty predictions"""
    df = pd.read_csv(filepath)
    pred_col = [c for c in df.columns if 'diag_spec_' in c][0]
    
    specialty_tags = Counter()
    diagnosis_tags = Counter()
    
    for text in df[pred_col].dropna():
        text = str(text)
        
        # Find specialty-related tags
        spec_matches = re.findall(r'<(spec[a-z]*)[^>]*>', text, re.IGNORECASE)
        for tag in spec_matches:
            specialty_tags[f'<{tag.lower()}>'] += 1
        
        # Find diagnosis-related tags
        diag_matches = re.findall(r'<(diag[a-z]*)[^>]*>', text, re.IGNORECASE)
        for tag in diag_matches:
            diagnosis_tags[f'<{tag.lower()}>'] += 1
    
    return {
        'specialty_tags': dict(specialty_tags.most_common(10)),
        'diagnosis_tags': dict(diagnosis_tags.most_common(10))
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect tag types used by models")
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to results directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print('='*80)
    print('TAG TYPE DETECTION ANALYSIS')
    print('='*80)
    
    # Find all triage files
    triage_files = sorted(results_dir.glob('triage_general_*.csv'))
    
    all_results = {}
    
    for triage_file in triage_files:
        # Extract model name
        model_name = triage_file.stem.replace('triage_general_', '').rsplit('_', 2)[0]
        
        print(f'\n{"="*60}')
        print(f'MODEL: {model_name}')
        print('='*60)
        
        # Analyze triage tags
        print('\n[TRIAGE TAGS]')
        triage_analysis = analyze_triage_tags(triage_file)
        print(f'Total predictions: {triage_analysis["total_predictions"]}')
        print('\nFirst tag patterns:')
        for tag, count in triage_analysis['first_tag'].items():
            pct = count / triage_analysis['total_predictions'] * 100
            print(f'  {tag:<35} {count:>5} ({pct:>5.1f}%)')
        
        # Find corresponding diag/spec file
        diag_file = results_dir / triage_file.name.replace('triage_general_', 'diagnosis_specialty_general_')
        if diag_file.exists():
            print('\n[SPECIALTY/DIAGNOSIS TAGS]')
            diag_analysis = analyze_diag_spec_tags(diag_file)
            print('Specialty tags:')
            for tag, count in diag_analysis['specialty_tags'].items():
                print(f'  {tag:<25} {count:>5}')
            print('Diagnosis tags:')
            for tag, count in diag_analysis['diagnosis_tags'].items():
                print(f'  {tag:<25} {count:>5}')
            
            triage_analysis['diag_spec'] = diag_analysis
        
        all_results[model_name] = triage_analysis
    
    # Save results
    output_file = results_dir / 'tag_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n\nResults saved: {output_file}')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
