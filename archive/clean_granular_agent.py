import anthropic
import pandas as pd
import json
import os
import glob
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class GeoAnomaly:
    theme: str
    type_: str
    severity: str
    category: str
    description: str
    reasoning: str
    confidence: float
    detected_by_rules: bool = False
    geographic_breakdown: List[Dict] = None
    category_breakdown: List[Dict] = None
    
    def to_dict(self):
        result = {
            'theme': self.theme,
            'type': self.type_,
            'severity': self.severity,
            'category': self.category,
            'description': self.description,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'detected_by_rules': self.detected_by_rules
        }
        if self.geographic_breakdown:
            result['geographic_breakdown'] = self.geographic_breakdown
        if self.category_breakdown:
            result['category_breakdown'] = self.category_breakdown
        return result


def load_row_counts(base_dir: str, release_date: str) -> Dict[str, pd.DataFrame]:
    """Load row count files with geographic and category data."""
    row_counts = {}
    row_counts_dir = os.path.join(base_dir, 'Metrics', 'metrics', release_date, 'row_counts')
    
    if not os.path.exists(row_counts_dir):
        print(f"Warning: Row counts not found for {release_date}")
        return row_counts
    
    print(f"\nLoading row counts for {release_date}...")
    
    for theme_dir in os.listdir(row_counts_dir):
        theme_path = os.path.join(row_counts_dir, theme_dir)
        if not os.path.isdir(theme_path) or not theme_dir.startswith('theme='):
            continue
        
        theme = theme_dir.split('=')[1]
        
        for type_dir in os.listdir(theme_path):
            type_path = os.path.join(theme_path, type_dir)
            if not os.path.isdir(type_path) or not type_dir.startswith('type='):
                continue
            
            type_ = type_dir.split('=')[1]
            csv_files = glob.glob(os.path.join(type_path, '*.csv'))
            
            if csv_files:
                try:
                    df = pd.read_csv(csv_files[0])
                    key = f"{theme}/{type_}"
                    row_counts[key] = df
                    print(f"  ✓ {key}: {len(df)} rows")
                except Exception as e:
                    print(f"  ✗ Error loading {theme}/{type_}: {e}")
    
    print(f"✓ Loaded {len(row_counts)} datasets with granular data")
    return row_counts


def prepare_geographic_summary(row_counts: Dict[str, pd.DataFrame]) -> str:
    """Prepare geographic data summary for the prompt."""
    geo_data = {}
    
    for key, df in row_counts.items():
        if 'country' not in df.columns or 'change_type' not in df.columns:
            continue
        
        theme, type_ = key.split('/')
        
        # Get top countries by change type
        summary = df.groupby(['country', 'change_type'])['total_count'].sum().reset_index()
        
        # Top removals
        removed = summary[summary['change_type'] == 'removed'].nlargest(5, 'total_count')
        # Top additions  
        added = summary[summary['change_type'] == 'added'].nlargest(5, 'total_count')
        
        geo_data[key] = {
            'theme': theme,
            'type': type_,
            'top_removals': removed[['country', 'total_count']].to_dict('records'),
            'top_additions': added[['country', 'total_count']].to_dict('records')
        }
    
    return json.dumps(geo_data, indent=2)


def create_prompt(current_stats: List[Dict], 
                  geo_summary: str,
                  rule_findings: List[Dict]) -> str:
    """Create the analysis prompt."""
    
    prompt_parts = []
    
    prompt_parts.append("You are an expert geospatial data analyst reviewing Overture Maps release data.")
    prompt_parts.append("")
    prompt_parts.append("# Current Release Statistics")
    prompt_parts.append(json.dumps(current_stats, indent=2))
    prompt_parts.append("")
    prompt_parts.append("# Geographic Data Available")
    prompt_parts.append("You have country-level breakdowns showing which countries have the most changes.")
    prompt_parts.append(geo_summary)
    prompt_parts.append("")
    prompt_parts.append("# Rule-Based Findings")
    prompt_parts.append(json.dumps(rule_findings, indent=2))
    prompt_parts.append("")
    prompt_parts.append("# Your Task")
    prompt_parts.append("Analyze the data and identify anomalies. For themes with geographic data, ")
    prompt_parts.append("include specific countries in your findings.")
    prompt_parts.append("")
    prompt_parts.append("Return a JSON array:")
    prompt_parts.append("[")
    prompt_parts.append("  {")
    prompt_parts.append('    "theme": "theme_name",')
    prompt_parts.append('    "type": "type_name",')
    prompt_parts.append('    "severity": "CRITICAL|WARNING|INFO",')
    prompt_parts.append('    "category": "Data Quality|Geographic Pattern|etc",')
    prompt_parts.append('    "description": "Brief description",')
    prompt_parts.append('    "reasoning": "Why this matters",')
    prompt_parts.append('    "confidence": 0.85,')
    prompt_parts.append('    "detected_by_rules": false,')
    prompt_parts.append('    "geographic_breakdown": [')
    prompt_parts.append('      {"country": "US", "metric": "removed", "value": 1234}')
    prompt_parts.append('    ]')
    prompt_parts.append("  }")
    prompt_parts.append("]")
    
    return "\n".join(prompt_parts)


def run_geographic_agent(api_key: str, 
                        current_stats: pd.DataFrame,
                        row_counts: Dict[str, pd.DataFrame],
                        rule_anomalies: List) -> List[GeoAnomaly]:
    """Run the AI agent with geographic analysis."""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare data
    stats_list = current_stats.to_dict('records') if current_stats is not None else []
    geo_summary = prepare_geographic_summary(row_counts)
    
    rule_findings = []
    if rule_anomalies:
        rule_findings = [
            {
                'theme': a.theme,
                'type': a.type_,
                'rule': a.rule,
                'message': a.message,
                'severity': a.severity.value
            }
            for a in rule_anomalies
        ]
    
    # Create prompt
    prompt = create_prompt(stats_list, geo_summary, rule_findings)
    
    print("\n" + "="*80)
    print("GEOGRAPHIC AI AGENT ANALYSIS")
    print("="*80)
    print(f"Analyzing with geographic data from {len(row_counts)} themes...")
    
    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    anomalies = []
    response_text = response.content[0].text
    
    try:
        # Find JSON array
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        
        if start != -1:
            json_str = response_text[start:end]
            parsed = json.loads(json_str)
            
            for item in parsed:
                anomaly = GeoAnomaly(
                    theme=item.get('theme', 'unknown'),
                    type_=item.get('type', 'unknown'),
                    severity=item.get('severity', 'INFO'),
                    category=item.get('category', 'Unknown'),
                    description=item.get('description', ''),
                    reasoning=item.get('reasoning', ''),
                    confidence=item.get('confidence', 0.5),
                    detected_by_rules=item.get('detected_by_rules', False),
                    geographic_breakdown=item.get('geographic_breakdown'),
                    category_breakdown=item.get('category_breakdown')
                )
                anomalies.append(anomaly)
        
        print(f"✓ Analysis complete: {len(anomalies)} anomalies detected")
        
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    return anomalies


def generate_report(anomalies: List[GeoAnomaly]) -> str:
    """Generate report."""
    if not anomalies:
        return "\n✓ No anomalies detected"
    
    lines = []
    lines.append("\n" + "="*80)
    lines.append("GEOGRAPHIC AI AGENT REPORT")
    lines.append("="*80)
    
    with_geo = sum(1 for a in anomalies if a.geographic_breakdown)
    lines.append(f"\nTotal: {len(anomalies)} anomalies ({with_geo} with geographic detail)")
    
    critical = [a for a in anomalies if a.severity == 'CRITICAL']
    warnings = [a for a in anomalies if a.severity == 'WARNING']
    info = [a for a in anomalies if a.severity == 'INFO']
    
    for severity, items in [('CRITICAL', critical), ('WARNING', warnings), ('INFO', info)]:
        if items:
            lines.append(f"\n{severity} ({len(items)}):")
            lines.append("-"*80)
            
            for a in items:
                lines.append(f"\n  {a.theme}/{a.type_} - {a.category}")
                lines.append(f"  Confidence: {a.confidence:.2f}")
                lines.append(f"  {a.description}")
                
                if a.geographic_breakdown:
                    lines.append("  Geographic breakdown:")
                    for g in a.geographic_breakdown[:5]:
                        lines.append(f"    • {g}")
                
                lines.append(f"  Reasoning: {a.reasoning}")
    
    return "\n".join(lines)


def main():
    """Main execution."""
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        return
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("CLEAN GEOGRAPHIC AI AGENT")
    print("="*80)
    
    # Load data
    from release_validator import EnhancedReleaseValidator
    
    print("\nLoading aggregate data...")
    data = EnhancedReleaseValidator.load_all_data(base_dir)
    
    # Load row counts
    release_date = '2025-09-24.0'
    row_counts = load_row_counts(base_dir, release_date)
    
    # Run rule-based checker
    print("\nRunning rule-based checker...")
    validator = EnhancedReleaseValidator()
    rule_anomalies = validator.validate_release(data)
    
    # Run geographic agent
    agent_anomalies = run_geographic_agent(
        api_key=api_key,
        current_stats=data.get('release_stats'),
        row_counts=row_counts,
        rule_anomalies=rule_anomalies
    )
    
    # Generate report
    report = generate_report(agent_anomalies)
    print(report)
    
    # Export
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'release': release_date,
        'anomalies': [a.to_dict() for a in agent_anomalies],
        'summary': {
            'total': len(agent_anomalies),
            'with_geographic': sum(1 for a in agent_anomalies if a.geographic_breakdown)
        }
    }
    
    output_path = os.path.join(base_dir, 'geographic_agent_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
