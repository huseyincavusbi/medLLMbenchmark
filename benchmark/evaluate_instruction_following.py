#!/usr/bin/env python3
"""
Evaluate Instruction Following / Format Compliance
===================================================
Analyzes how well each model follows the expected output format instructions.

Metrics:
- Triage: % using <acuity>X</acuity> format
- Specialty: % using <specialty>X</specialty> format  
- Diagnosis: % using <diagnosis>X</diagnosis> format
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


class InstructionFollowingEvaluator:
    """Evaluates format compliance across models"""
    
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).parent / "results"
        self.metrics = defaultdict(dict)
    
    def analyze_triage_format(self, text):
        """Analyze triage output format"""
        if pd.isna(text) or text == '':
            return {'format': 'empty', 'valid': False}
        
        text = str(text)
        
        # Check for exact expected format: <acuity>X</acuity>
        if re.search(r'<acuity>\s*[1-5]\s*</acuity>', text, re.IGNORECASE):
            return {'format': 'acuity_tag', 'valid': True}
        
        # Check for <acuity> without proper closing
        if re.search(r'<acuity>', text, re.IGNORECASE):
            return {'format': 'acuity_malformed', 'valid': False}
        
        # Check for <esi_level_X> format
        if re.search(r'<esi_level_[1-5]>', text, re.IGNORECASE):
            return {'format': 'esi_level_tag', 'valid': False}
        
        # Check for <ESI Level X> format
        if re.search(r'<ESI\s+Level\s*[1-5]>', text, re.IGNORECASE):
            return {'format': 'esi_level_space', 'valid': False}
        
        # Check for bare <X> format
        if re.search(r'^<[1-5]>$', text.strip()):
            return {'format': 'bare_number_tag', 'valid': False}
        
        # Check for reasoning-first with tag at end
        if re.search(r'<acuity>\s*[1-5]\s*</acuity>', text[-100:], re.IGNORECASE):
            return {'format': 'reasoning_then_tag', 'valid': True}
        
        # Fallback: just has a digit 1-5 somewhere
        if re.search(r'\b[1-5]\b', text):
            return {'format': 'digit_only', 'valid': False}
        
        return {'format': 'unparseable', 'valid': False}
    
    def analyze_specialty_format(self, text):
        """Analyze specialty output format"""
        if pd.isna(text) or text == '':
            return {'format': 'empty', 'count': 0, 'valid': False}
        
        text = str(text)
        
        # Count proper <specialty>X</specialty> tags
        specialty_matches = re.findall(r'<specialty>(.*?)</specialty>', text, re.IGNORECASE | re.DOTALL)
        
        if len(specialty_matches) >= 3:
            return {'format': 'full_compliance', 'count': len(specialty_matches), 'valid': True}
        elif len(specialty_matches) > 0:
            return {'format': 'partial_tags', 'count': len(specialty_matches), 'valid': False}
        elif 'specialty' in text.lower():
            return {'format': 'mentions_specialty', 'count': 0, 'valid': False}
        else:
            return {'format': 'no_tags', 'count': 0, 'valid': False}
    
    def analyze_diagnosis_format(self, text):
        """Analyze diagnosis output format"""
        if pd.isna(text) or text == '':
            return {'format': 'empty', 'count': 0, 'valid': False}
        
        text = str(text)
        
        # Count proper <diagnosis>X</diagnosis> tags
        diagnosis_matches = re.findall(r'<diagnosis>(.*?)</diagnosis>', text, re.IGNORECASE | re.DOTALL)
        
        if len(diagnosis_matches) >= 3:
            return {'format': 'full_compliance', 'count': len(diagnosis_matches), 'valid': True}
        elif len(diagnosis_matches) > 0:
            return {'format': 'partial_tags', 'count': len(diagnosis_matches), 'valid': False}
        elif 'diagnosis' in text.lower():
            return {'format': 'mentions_diagnosis', 'count': 0, 'valid': False}
        else:
            return {'format': 'no_tags', 'count': 0, 'valid': False}
    
    def evaluate_model(self, run_id):
        """Evaluate instruction following for a single model run"""
        model_name = run_id.rsplit('_', 2)[0]
        results = {'model': model_name, 'run_id': run_id}
        
        # Triage General
        triage_gen = self.results_dir / f"triage_general_{run_id}.csv"
        if triage_gen.exists():
            df = pd.read_csv(triage_gen)
            pred_col = [c for c in df.columns if 'triage_' in c and '_general' in c][0]
            
            formats = df[pred_col].apply(self.analyze_triage_format)
            format_counts = pd.Series([f['format'] for f in formats]).value_counts()
            valid_count = sum(1 for f in formats if f['valid'])
            
            results['triage_general'] = {
                'total': len(df),
                'valid_format': valid_count,
                'valid_pct': valid_count / len(df) * 100,
                'format_breakdown': format_counts.to_dict()
            }
        
        # Triage Clinical
        triage_clin = self.results_dir / f"triage_clinical_{run_id}.csv"
        if triage_clin.exists():
            df = pd.read_csv(triage_clin)
            pred_col = [c for c in df.columns if 'triage_' in c and '_clinical' in c][0]
            
            formats = df[pred_col].apply(self.analyze_triage_format)
            format_counts = pd.Series([f['format'] for f in formats]).value_counts()
            valid_count = sum(1 for f in formats if f['valid'])
            
            results['triage_clinical'] = {
                'total': len(df),
                'valid_format': valid_count,
                'valid_pct': valid_count / len(df) * 100,
                'format_breakdown': format_counts.to_dict()
            }
        
        # Diagnosis/Specialty General
        diag_gen = self.results_dir / f"diagnosis_specialty_general_{run_id}.csv"
        if diag_gen.exists():
            df = pd.read_csv(diag_gen)
            pred_col = [c for c in df.columns if 'diag_spec_' in c and '_general' in c][0]
            
            spec_formats = df[pred_col].apply(self.analyze_specialty_format)
            diag_formats = df[pred_col].apply(self.analyze_diagnosis_format)
            
            spec_valid = sum(1 for f in spec_formats if f['valid'])
            diag_valid = sum(1 for f in diag_formats if f['valid'])
            
            results['specialty_general'] = {
                'total': len(df),
                'valid_format': spec_valid,
                'valid_pct': spec_valid / len(df) * 100,
            }
            results['diagnosis_general'] = {
                'total': len(df),
                'valid_format': diag_valid,
                'valid_pct': diag_valid / len(df) * 100,
            }
        
        # Diagnosis/Specialty Clinical
        diag_clin = self.results_dir / f"diagnosis_specialty_clinical_{run_id}.csv"
        if diag_clin.exists():
            df = pd.read_csv(diag_clin)
            pred_col = [c for c in df.columns if 'diag_spec_' in c and '_clinical' in c][0]
            
            spec_formats = df[pred_col].apply(self.analyze_specialty_format)
            diag_formats = df[pred_col].apply(self.analyze_diagnosis_format)
            
            spec_valid = sum(1 for f in spec_formats if f['valid'])
            diag_valid = sum(1 for f in diag_formats if f['valid'])
            
            results['specialty_clinical'] = {
                'total': len(df),
                'valid_format': spec_valid,
                'valid_pct': spec_valid / len(df) * 100,
            }
            results['diagnosis_clinical'] = {
                'total': len(df),
                'valid_format': diag_valid,
                'valid_pct': diag_valid / len(df) * 100,
            }
        
        return results
    
    def evaluate_all(self):
        """Evaluate all models in results directory"""
        metadata_files = list(self.results_dir.glob("run_metadata_*.json"))
        
        all_results = []
        for mf in sorted(metadata_files):
            with open(mf) as f:
                meta = json.load(f)
            
            print(f"Evaluating: {meta['model_name']}...")
            results = self.evaluate_model(meta['run_id'])
            all_results.append(results)
        
        return all_results
    
    def print_summary(self, all_results):
        """Print summary table"""
        print("\n" + "=" * 100)
        print("INSTRUCTION FOLLOWING / FORMAT COMPLIANCE SUMMARY")
        print("=" * 100)
        
        # Triage compliance
        print("\n[TRIAGE] FORMAT COMPLIANCE (<acuity>X</acuity>)")
        print("-" * 80)
        print(f"{'Model':<30} {'General':>15} {'Clinical':>15} {'Average':>15}")
        print("-" * 80)
        
        for r in sorted(all_results, key=lambda x: -(x.get('triage_general', {}).get('valid_pct', 0) + x.get('triage_clinical', {}).get('valid_pct', 0))/2):
            tg = r.get('triage_general', {}).get('valid_pct', 0)
            tc = r.get('triage_clinical', {}).get('valid_pct', 0)
            avg = (tg + tc) / 2
            print(f"{r['model']:<30} {tg:>14.1f}% {tc:>14.1f}% {avg:>14.1f}%")
        
        # Specialty compliance
        print("\n[SPECIALTY] FORMAT COMPLIANCE (<specialty>X</specialty> x3)")
        print("-" * 80)
        print(f"{'Model':<30} {'General':>15} {'Clinical':>15} {'Average':>15}")
        print("-" * 80)
        
        for r in sorted(all_results, key=lambda x: -(x.get('specialty_general', {}).get('valid_pct', 0) + x.get('specialty_clinical', {}).get('valid_pct', 0))/2):
            sg = r.get('specialty_general', {}).get('valid_pct', 0)
            sc = r.get('specialty_clinical', {}).get('valid_pct', 0)
            avg = (sg + sc) / 2
            print(f"{r['model']:<30} {sg:>14.1f}% {sc:>14.1f}% {avg:>14.1f}%")
        
        # Diagnosis compliance
        print("\n[DIAGNOSIS] FORMAT COMPLIANCE (<diagnosis>X</diagnosis> x3)")
        print("-" * 80)
        print(f"{'Model':<30} {'General':>15} {'Clinical':>15} {'Average':>15}")
        print("-" * 80)
        
        for r in sorted(all_results, key=lambda x: -(x.get('diagnosis_general', {}).get('valid_pct', 0) + x.get('diagnosis_clinical', {}).get('valid_pct', 0))/2):
            dg = r.get('diagnosis_general', {}).get('valid_pct', 0)
            dc = r.get('diagnosis_clinical', {}).get('valid_pct', 0)
            avg = (dg + dc) / 2
            print(f"{r['model']:<30} {dg:>14.1f}% {dc:>14.1f}% {avg:>14.1f}%")
        
        # Overall ranking
        print("\n[OVERALL] INSTRUCTION FOLLOWING SCORE")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<30} {'Triage':>12} {'Specialty':>12} {'Diagnosis':>12} {'Overall':>12}")
        print("-" * 80)
        
        scores = []
        for r in all_results:
            triage_avg = (r.get('triage_general', {}).get('valid_pct', 0) + r.get('triage_clinical', {}).get('valid_pct', 0)) / 2
            spec_avg = (r.get('specialty_general', {}).get('valid_pct', 0) + r.get('specialty_clinical', {}).get('valid_pct', 0)) / 2
            diag_avg = (r.get('diagnosis_general', {}).get('valid_pct', 0) + r.get('diagnosis_clinical', {}).get('valid_pct', 0)) / 2
            overall = (triage_avg + spec_avg + diag_avg) / 3
            scores.append((r['model'], triage_avg, spec_avg, diag_avg, overall))
        
        for rank, (model, triage, spec, diag, overall) in enumerate(sorted(scores, key=lambda x: -x[4]), 1):
            rank_str = f"{rank}."
            print(f"{rank_str:<6} {model:<30} {triage:>11.1f}% {spec:>11.1f}% {diag:>11.1f}% {overall:>11.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate instruction following")
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to results directory')
    args = parser.parse_args()
    
    results_dir = args.results_dir or Path(__file__).parent.parent.parent / "results"
    
    evaluator = InstructionFollowingEvaluator(results_dir=results_dir)
    all_results = evaluator.evaluate_all()
    evaluator.print_summary(all_results)
    
    # Save detailed results
    output_file = Path(results_dir) / "instruction_following_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed metrics saved: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
