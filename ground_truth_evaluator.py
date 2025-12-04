"""
Ground Truth Evaluator for Anomaly Detection
=============================================
Creates ground truth datasets and calculates precision/recall metrics.

Usage:
    # Generate ground truth template for manual labeling
    python ground_truth_evaluator.py anomalies.json --generate-template ground_truth.csv
    
    # Generate heuristic-based ground truth (automated guesses)
    python ground_truth_evaluator.py anomalies.json --generate-heuristic ground_truth_heuristic.csv
    
    # Calculate metrics from labeled ground truth
    python ground_truth_evaluator.py anomalies.json --evaluate ground_truth_labeled.csv
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import argparse


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for anomaly detection."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    def __str__(self):
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                 ANOMALY DETECTION EVALUATION                  ║
╠══════════════════════════════════════════════════════════════╣
║  True Positives (correctly flagged real issues):  {self.true_positives:>10}  ║
║  False Positives (incorrectly flagged as issue):  {self.false_positives:>10}  ║
║  False Negatives (missed real issues):            {self.false_negatives:>10}  ║
║  True Negatives (correctly ignored non-issues):   {self.true_negatives:>10}  ║
╠══════════════════════════════════════════════════════════════╣
║  PRECISION (of flagged, how many are real):       {self.precision:>10.1%}  ║
║  RECALL (of real issues, how many were caught):   {self.recall:>10.1%}  ║
║  F1 SCORE (harmonic mean):                        {self.f1_score:>10.1%}  ║
║  ACCURACY (overall correctness):                  {self.accuracy:>10.1%}  ║
╚══════════════════════════════════════════════════════════════╝
"""


class GroundTruthEvaluator:
    """Generate ground truth and evaluate anomaly detection performance."""
    
    # Heuristic rules for classifying anomalies
    HEURISTIC_RULES = {
        # High confidence real issues
        'definitely_real': [
            # Systematic duplication (~100% increase across many countries)
            lambda a: a.get('anomaly_type') == 'feature_count_spike' and 
                      95 <= a.get('percent_change', 0) <= 105,
            # Data regeneration (complete rebuild)
            lambda a: a.get('anomaly_type') == 'data_regeneration',
            # Massive net changes (>1M features)
            lambda a: a.get('anomaly_type') in ['net_feature_loss', 'net_feature_gain'] and
                      abs(a.get('additional_context', {}).get('net_change', 0)) > 1000000,
            # Critical geometry changes (>20%)
            lambda a: a.get('anomaly_type') == 'geometry_length_change' and
                      abs(a.get('percent_change', 0)) > 20,
        ],
        
        # Likely real issues
        'probably_real': [
            # Large feature count changes (>50% but not duplication pattern)
            lambda a: a.get('anomaly_type') in ['feature_count_spike', 'feature_count_drop'] and
                      abs(a.get('percent_change', 0)) > 50 and
                      not (95 <= a.get('percent_change', 0) <= 105),
            # High removal rate (>5%)
            lambda a: a.get('anomaly_type') == 'high_removal_rate' and
                      abs(a.get('percent_change', 0)) > 5,
            # Coverage drops (>10%)
            lambda a: a.get('anomaly_type') == 'attribute_coverage_drop' and
                      abs(a.get('percent_change', 0)) > 10,
            # Very low coverage on required fields (<5%)
            lambda a: a.get('anomaly_type') == 'low_attribute_coverage' and
                      a.get('current_value', 100) < 5 and
                      a.get('additional_context', {}).get('is_required_field', False),
        ],
        
        # Uncertain - needs manual review
        'uncertain': [
            # Moderate changes (20-50%)
            lambda a: a.get('anomaly_type') in ['feature_count_spike', 'feature_count_drop'] and
                      20 <= abs(a.get('percent_change', 0)) < 50,
            # Low coverage on non-required fields
            lambda a: a.get('anomaly_type') == 'low_attribute_coverage' and
                      a.get('current_value', 100) < 20 and
                      not a.get('additional_context', {}).get('is_required_field', False),
            # Category shifts
            lambda a: a.get('anomaly_type') == 'category_distribution_shift',
        ],
        
        # Likely false positives
        'probably_false_positive': [
            # Small changes (<20%)
            lambda a: abs(a.get('percent_change', 0)) < 20 and
                      a.get('anomaly_type') in ['feature_count_spike', 'feature_count_drop'],
            # Geographic concentration (informational, not necessarily a problem)
            lambda a: a.get('anomaly_type') == 'geographic_concentration',
            # Coverage spikes (usually good, not a problem)
            lambda a: a.get('anomaly_type') == 'attribute_coverage_spike',
            # Low coverage on known-sparse fields
            lambda a: a.get('anomaly_type') == 'low_attribute_coverage' and
                      a.get('column') in ['wikidata', 'wikipedia', 'brand', 'phones', 'websites'],
        ],
    }
    
    def __init__(self, anomalies_json: Path):
        """Load anomalies from JSON."""
        with open(anomalies_json) as f:
            data = json.load(f)
        
        self.anomalies = data.get('anomalies', [])
        self.metadata = {k: v for k, v in data.items() if k != 'anomalies'}
        
        print(f"Loaded {len(self.anomalies)} anomalies")
    
    def generate_template(self, output_path: Path) -> pd.DataFrame:
        """Generate a template CSV for manual ground truth labeling."""
        
        records = []
        for i, a in enumerate(self.anomalies):
            records.append({
                'anomaly_id': i,
                'severity': a.get('severity', ''),
                'anomaly_type': a.get('anomaly_type', ''),
                'theme': a.get('theme', ''),
                'type': a.get('type', ''),
                'subtype': a.get('subtype', ''),
                'country': a.get('country', ''),
                'column': a.get('column', ''),
                'description': a.get('description', ''),
                'percent_change': a.get('percent_change', 0),
                'previous_value': a.get('previous_value', 0),
                'current_value': a.get('current_value', 0),
                # Fields to fill in
                'is_real_issue': '',  # TRUE, FALSE, or UNCERTAIN
                'confidence': '',     # HIGH, MEDIUM, LOW
                'notes': '',          # Any notes about why
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Generated ground truth template: {output_path}")
        print(f"  - {len(df)} anomalies to label")
        print(f"\nInstructions:")
        print(f"  1. Open {output_path} in Excel/Google Sheets")
        print(f"  2. For each row, fill in:")
        print(f"     - is_real_issue: TRUE, FALSE, or UNCERTAIN")
        print(f"     - confidence: HIGH, MEDIUM, or LOW")
        print(f"     - notes: (optional) why you made this decision")
        print(f"  3. Save and run: python ground_truth_evaluator.py anomalies.json --evaluate {output_path}")
        
        return df
    
    def generate_heuristic_ground_truth(self, output_path: Path) -> pd.DataFrame:
        """Generate ground truth using heuristic rules (automated guesses)."""
        
        records = []
        for i, a in enumerate(self.anomalies):
            # Classify using heuristics
            classification = self._classify_anomaly(a)
            
            records.append({
                'anomaly_id': i,
                'severity': a.get('severity', ''),
                'anomaly_type': a.get('anomaly_type', ''),
                'theme': a.get('theme', ''),
                'type': a.get('type', ''),
                'subtype': a.get('subtype', ''),
                'country': a.get('country', ''),
                'column': a.get('column', ''),
                'description': a.get('description', ''),
                'percent_change': a.get('percent_change', 0),
                'previous_value': a.get('previous_value', 0),
                'current_value': a.get('current_value', 0),
                # Heuristic classification
                'is_real_issue': classification['is_real'],
                'confidence': classification['confidence'],
                'classification_reason': classification['reason'],
                'notes': 'AUTO-GENERATED - Please verify',
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        
        # Summary
        real_count = len(df[df['is_real_issue'] == 'TRUE'])
        false_count = len(df[df['is_real_issue'] == 'FALSE'])
        uncertain_count = len(df[df['is_real_issue'] == 'UNCERTAIN'])
        
        print(f"\n✓ Generated heuristic ground truth: {output_path}")
        print(f"\nClassification Summary:")
        print(f"  - Real issues (TRUE):      {real_count:>5} ({real_count/len(df)*100:.1f}%)")
        print(f"  - False positives (FALSE): {false_count:>5} ({false_count/len(df)*100:.1f}%)")
        print(f"  - Uncertain:               {uncertain_count:>5} ({uncertain_count/len(df)*100:.1f}%)")
        print(f"\n⚠️  WARNING: This is AUTO-GENERATED based on heuristics.")
        print(f"   You should manually verify, especially the UNCERTAIN ones.")
        
        return df
    
    def _classify_anomaly(self, anomaly: dict) -> dict:
        """Classify a single anomaly using heuristics."""
        
        # Check definitely real
        for rule in self.HEURISTIC_RULES['definitely_real']:
            try:
                if rule(anomaly):
                    return {
                        'is_real': 'TRUE',
                        'confidence': 'HIGH',
                        'reason': 'Matches high-confidence real issue pattern'
                    }
            except:
                pass
        
        # Check probably real
        for rule in self.HEURISTIC_RULES['probably_real']:
            try:
                if rule(anomaly):
                    return {
                        'is_real': 'TRUE',
                        'confidence': 'MEDIUM',
                        'reason': 'Matches likely real issue pattern'
                    }
            except:
                pass
        
        # Check probably false positive
        for rule in self.HEURISTIC_RULES['probably_false_positive']:
            try:
                if rule(anomaly):
                    return {
                        'is_real': 'FALSE',
                        'confidence': 'MEDIUM',
                        'reason': 'Matches likely false positive pattern'
                    }
            except:
                pass
        
        # Check uncertain
        for rule in self.HEURISTIC_RULES['uncertain']:
            try:
                if rule(anomaly):
                    return {
                        'is_real': 'UNCERTAIN',
                        'confidence': 'LOW',
                        'reason': 'Needs manual review'
                    }
            except:
                pass
        
        # Default: uncertain
        return {
            'is_real': 'UNCERTAIN',
            'confidence': 'LOW',
            'reason': 'No matching heuristic rule'
        }
    
    def evaluate(self, ground_truth_path: Path) -> EvaluationMetrics:
        """Calculate precision/recall from labeled ground truth."""
        
        gt = pd.read_csv(ground_truth_path)
        
        # Standardize values
        gt['is_real_issue'] = gt['is_real_issue'].astype(str).str.upper().str.strip()
        
        # Count categories
        true_positives = len(gt[gt['is_real_issue'] == 'TRUE'])
        false_positives = len(gt[gt['is_real_issue'] == 'FALSE'])
        uncertain = len(gt[gt['is_real_issue'] == 'UNCERTAIN'])
        
        # For recall, we need to know about false negatives (real issues we missed)
        # Since we don't have a separate "all real issues" list, we estimate:
        # - Assume the detector found most real issues (low false negative rate)
        # - This is a limitation without external validation
        
        # Conservative estimate: assume 5% false negative rate
        estimated_fn = int(true_positives * 0.05)
        
        # True negatives are unknown (we don't know all the "normal" data points)
        # For accuracy, we'll use a placeholder
        estimated_tn = 0
        
        total_labeled = true_positives + false_positives
        
        if total_labeled == 0:
            print("ERROR: No labeled data (TRUE or FALSE) found in ground truth")
            print(f"  Found: TRUE={true_positives}, FALSE={false_positives}, UNCERTAIN={uncertain}")
            return None
        
        # Calculate metrics
        precision = true_positives / total_labeled if total_labeled > 0 else 0
        recall = true_positives / (true_positives + estimated_fn) if (true_positives + estimated_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + estimated_tn) / (total_labeled + estimated_fn + estimated_tn) if total_labeled > 0 else 0
        
        metrics = EvaluationMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=estimated_fn,
            true_negatives=estimated_tn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy
        )
        
        print(metrics)
        
        # Breakdown by anomaly type
        print("\nBreakdown by Anomaly Type:")
        print("-" * 60)
        for atype in gt['anomaly_type'].unique():
            subset = gt[gt['anomaly_type'] == atype]
            tp = len(subset[subset['is_real_issue'] == 'TRUE'])
            fp = len(subset[subset['is_real_issue'] == 'FALSE'])
            unc = len(subset[subset['is_real_issue'] == 'UNCERTAIN'])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"  {atype}:")
            print(f"    Real: {tp}, False Positive: {fp}, Uncertain: {unc}")
            print(f"    Precision: {prec:.1%}")
        
        # Warning about limitations
        print("\n" + "="*60)
        print("⚠️  IMPORTANT LIMITATIONS:")
        print("="*60)
        print("1. RECALL is estimated (assumes 5% false negative rate)")
        print("   - True recall requires knowing ALL real issues, including")
        print("     ones the detector missed (which we don't know)")
        print("")
        print("2. PRECISION is based on your ground truth labels")
        print("   - Only as accurate as your labeling")
        print("")
        print("3. To improve accuracy:")
        print("   - Have domain expert verify labels")
        print("   - Cross-reference with Overture release notes")
        print("   - Test on multiple releases")
        
        return metrics
    
    def generate_summary_report(self, ground_truth_path: Path, output_path: Path):
        """Generate a detailed evaluation report."""
        
        gt = pd.read_csv(ground_truth_path)
        gt['is_real_issue'] = gt['is_real_issue'].astype(str).str.upper().str.strip()
        
        # Calculate metrics
        metrics = self.evaluate(ground_truth_path)
        
        # Generate report
        report = f"""# Anomaly Detection Evaluation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| **Precision** | {metrics.precision:.1%} |
| **Recall** (estimated) | {metrics.recall:.1%} |
| **F1 Score** | {metrics.f1_score:.1%} |

## Classification Breakdown

| Classification | Count | Percentage |
|---------------|-------|------------|
| True Positives (Real Issues) | {metrics.true_positives} | {metrics.true_positives/(metrics.true_positives+metrics.false_positives)*100:.1f}% |
| False Positives | {metrics.false_positives} | {metrics.false_positives/(metrics.true_positives+metrics.false_positives)*100:.1f}% |

## By Anomaly Type

"""
        for atype in gt['anomaly_type'].unique():
            subset = gt[gt['anomaly_type'] == atype]
            tp = len(subset[subset['is_real_issue'] == 'TRUE'])
            fp = len(subset[subset['is_real_issue'] == 'FALSE'])
            total = tp + fp
            prec = tp / total if total > 0 else 0
            report += f"### {atype}\n"
            report += f"- Total: {total}\n"
            report += f"- Real Issues: {tp}\n"
            report += f"- False Positives: {fp}\n"
            report += f"- Precision: {prec:.1%}\n\n"
        
        report += """## Limitations

1. **Recall is estimated** - We don't know what issues the detector missed
2. **Ground truth quality** - Results depend on labeling accuracy
3. **Single release** - Should validate across multiple releases

## Recommendations

1. Have domain expert review UNCERTAIN classifications
2. Test on additional releases to validate consistency
3. Compare with Overture's known issues list if available
"""
        
        Path(output_path).write_text(report)
        print(f"\n✓ Generated evaluation report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ground truth evaluation for anomaly detection")
    parser.add_argument("anomalies_json", type=Path, help="Path to anomalies.json")
    parser.add_argument("--generate-template", type=Path, help="Generate blank template for manual labeling")
    parser.add_argument("--generate-heuristic", type=Path, help="Generate heuristic-based ground truth")
    parser.add_argument("--evaluate", type=Path, help="Evaluate against labeled ground truth")
    parser.add_argument("--report", type=Path, help="Generate evaluation report (use with --evaluate)")
    
    args = parser.parse_args()
    
    evaluator = GroundTruthEvaluator(args.anomalies_json)
    
    if args.generate_template:
        evaluator.generate_template(args.generate_template)
    
    if args.generate_heuristic:
        evaluator.generate_heuristic_ground_truth(args.generate_heuristic)
    
    if args.evaluate:
        metrics = evaluator.evaluate(args.evaluate)
        
        if args.report and metrics:
            evaluator.generate_summary_report(args.evaluate, args.report)


if __name__ == "__main__":
    main()
