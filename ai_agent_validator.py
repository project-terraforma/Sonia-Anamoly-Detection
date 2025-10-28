import anthropic
import pandas as pd
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AgentAnomaly:
    theme: str
    type_: str
    severity: str  # CRITICAL, WARNING, INFO
    category: str  # e.g., "Geospatial Strangeness", "Data Quality", "Unexpected Churn"
    description: str
    reasoning: str
    confidence: float  # 0-1
    detected_by_rules: bool = False
    
    def to_dict(self):
        return {
            'theme': self.theme,
            'type': self.type_,
            'severity': self.severity,
            'category': self.category,
            'description': self.description,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'detected_by_rules': self.detected_by_rules
        }


class AIReleaseValidator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI-powered release validator.
        
        Args:
            api_key: Anthropic API key. If None, will look for ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter. Get a key at: https://console.anthropic.com"
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"  # Latest Sonnet 4.5
    
    def analyze_release(self, 
                       release_stats: pd.DataFrame,
                       column_historical: pd.DataFrame,
                       historical_totals: pd.DataFrame,
                       rule_based_anomalies: List = None) -> List[AgentAnomaly]:
        """
        Use Claude to analyze release data and detect anomalies with reasoning.
        
        Args:
            release_stats: Current release statistics
            column_historical: Historical column coverage data
            historical_totals: Historical total counts
            rule_based_anomalies: Anomalies found by rule-based checker (for comparison)
            
        Returns:
            List of anomalies detected by the AI agent
        """
        # Prepare context for Claude
        context = self._prepare_analysis_context(
            release_stats, 
            column_historical, 
            historical_totals,
            rule_based_anomalies
        )
        
        # Create the analysis prompt
        prompt = self._create_analysis_prompt(context)
        
        print("\n" + "="*80)
        print("AI AGENT ANALYSIS")
        print("="*80)
        print(f"Sending {len(context['current_stats'])} theme/types to Claude for analysis...")
        print(f"Historical context: {len(context['historical_data'])} records")
        
        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            temperature=0,  # Deterministic for consistency
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Parse Claude's response
        anomalies = self._parse_agent_response(response.content[0].text, rule_based_anomalies)
        
        print(f"✓ Agent analysis complete: {len(anomalies)} anomalies detected")
        
        return anomalies
    
    def _prepare_analysis_context(self, 
                                  release_stats: pd.DataFrame,
                                  column_historical: pd.DataFrame,
                                  historical_totals: pd.DataFrame,
                                  rule_based_anomalies: List) -> Dict:
        """Prepare structured context for Claude."""
        context = {
            'current_stats': release_stats.to_dict('records') if release_stats is not None else [],
            'historical_data': [],
            'rule_based_findings': []
        }
        
        # Add column historical data
        if column_historical is not None:
            # Group by theme/type and calculate trends
            for (theme, type_), group in column_historical.groupby(['theme', 'type']):
                group_sorted = group.sort_values('release')
                latest = group_sorted.iloc[-1]
                
                # Calculate coverage percentages for key columns
                coverage_data = {'theme': theme, 'type': type_, 'releases': []}
                
                for _, row in group_sorted.iterrows():
                    release_data = {'release': row['release']}
                    total = row['total_count']
                    
                    # Calculate coverage for non-metadata columns
                    for col in row.index:
                        if col not in ['release', 'theme', 'type', 'total_count']:
                            try:
                                val = float(row[col])
                                coverage_pct = (val / total * 100) if total > 0 else 0
                                release_data[col] = round(coverage_pct, 2)
                            except:
                                pass
                    
                    coverage_data['releases'].append(release_data)
                
                context['historical_data'].append(coverage_data)
        
        # Add rule-based findings for comparison
        if rule_based_anomalies:
            context['rule_based_findings'] = [
                {
                    'theme': a.theme,
                    'type': a.type_,
                    'rule': a.rule,
                    'message': a.message,
                    'severity': a.severity.value
                }
                for a in rule_based_anomalies
            ]
        
        return context
    
    def _create_analysis_prompt(self, context: Dict) -> str:
        """Create a detailed prompt for Claude to analyze the release."""
        
        prompt = f"""You are an expert data quality analyst reviewing a geospatial data release. Your job is to identify anomalies that might indicate data quality issues, systematic problems, or unusual changes.

# Context

You are analyzing release data from Overture Maps, which publishes monthly geospatial datasets including places (POIs), buildings, transportation networks, divisions (administrative boundaries), and base layers.

# Data Provided

## Current Release Statistics
{json.dumps(context['current_stats'], indent=2)}

## Historical Attribute Coverage Trends
{json.dumps(context['historical_data'][:3], indent=2)}  
... and {len(context['historical_data']) - 3} more theme/type combinations

## Rule-Based Checker Findings
The rule-based system detected these anomalies:
{json.dumps(context['rule_based_findings'], indent=2) if context['rule_based_findings'] else "None detected"}

# Your Task

Analyze this release data and identify anomalies in these categories:

1. **Unexpected Data Churn**: Unusual changes between releases
   - Attribute coverage drops (e.g., 10% fewer places have websites)
   - Feature count fluctuations (sudden drops in total counts)
   - High churn rates that might indicate ID instability

2. **Data Quality Issues**: Content and completeness problems
   - Increases in incomplete data (missing critical attributes)
   - Suspicious patterns that might indicate spam/vandalism
   - Unusual attribute coverage spikes

3. **Systematic Issues**: Patterns suggesting pipeline problems
   - Multiple themes affected similarly
   - Unusual correlations between different metrics
   - Deviations from typical monthly patterns

# Important Guidelines

- Focus on issues that a human analyst would find concerning
- Consider the business impact on map data users (developers, analysts)
- Look for subtle patterns the rule-based checker might miss
- Explain your reasoning clearly
- Assign confidence scores (0-1) based on how certain you are
- Note if a finding duplicates what the rule-based checker found

# Output Format

Return a JSON array of anomalies. Each anomaly should have:
```json
{{
  "theme": "theme name",
  "type": "type name",
  "severity": "CRITICAL|WARNING|INFO",
  "category": "Unexpected Data Churn|Data Quality Issues|Systematic Issues",
  "description": "Brief description of the anomaly",
  "reasoning": "Detailed explanation of why this is concerning and what it might indicate",
  "confidence": 0.85,
  "detected_by_rules": false
}}
```

Focus on quality over quantity. Only flag genuinely concerning issues. If the release looks good, say so.

Provide your analysis:"""
        
        return prompt
    
    def _parse_agent_response(self, response_text: str, rule_based_anomalies: List) -> List[AgentAnomaly]:
        """Parse Claude's JSON response into AgentAnomaly objects."""
        anomalies = []
        
        try:
            # Extract JSON from response (Claude might wrap it in markdown)
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1:
                print("Warning: No JSON array found in agent response")
                return anomalies
            
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Convert to AgentAnomaly objects
            for item in parsed:
                anomaly = AgentAnomaly(
                    theme=item.get('theme', 'unknown'),
                    type_=item.get('type', 'unknown'),
                    severity=item.get('severity', 'INFO'),
                    category=item.get('category', 'Unknown'),
                    description=item.get('description', ''),
                    reasoning=item.get('reasoning', ''),
                    confidence=item.get('confidence', 0.5),
                    detected_by_rules=item.get('detected_by_rules', False)
                )
                anomalies.append(anomaly)
        
        except json.JSONDecodeError as e:
            print(f"Error parsing agent response: {e}")
            print(f"Response text: {response_text[:500]}...")
        
        return anomalies
    
    def compare_with_rules(self, 
                          agent_anomalies: List[AgentAnomaly],
                          rule_based_anomalies: List) -> Dict:
        """
        Compare agent findings with rule-based checker to assess added value.
        
        Returns:
            Dictionary with comparison statistics and analysis
        """
        comparison = {
            'agent_total': len(agent_anomalies),
            'rules_total': len(rule_based_anomalies),
            'agent_only': [],
            'rules_only': [],
            'both_detected': [],
            'summary': {}
        }
        
        # Find agent-only detections (novel findings)
        for agent_anom in agent_anomalies:
            if not agent_anom.detected_by_rules:
                comparison['agent_only'].append({
                    'theme': agent_anom.theme,
                    'type': agent_anom.type_,
                    'category': agent_anom.category,
                    'description': agent_anom.description,
                    'confidence': agent_anom.confidence
                })
        
        # Categorize agent findings
        by_category = {}
        for anom in agent_anomalies:
            cat = anom.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(anom)
        
        comparison['summary'] = {
            'novel_detections': len(comparison['agent_only']),
            'confirmed_rules': sum(1 for a in agent_anomalies if a.detected_by_rules),
            'categories': {cat: len(anoms) for cat, anoms in by_category.items()},
            'avg_confidence': sum(a.confidence for a in agent_anomalies) / len(agent_anomalies) if agent_anomalies else 0
        }
        
        return comparison
    
    def generate_report(self, 
                       agent_anomalies: List[AgentAnomaly],
                       comparison: Dict) -> str:
        """Generate a human-readable report of agent findings."""
        
        report = ["\n" + "="*80]
        report.append("AI AGENT VALIDATION REPORT")
        report.append("="*80 + "\n")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-"*80)
        report.append(f"Total anomalies detected: {comparison['agent_total']}")
        report.append(f"Novel detections (not found by rules): {comparison['summary']['novel_detections']}")
        report.append(f"Confirmed rule-based findings: {comparison['summary']['confirmed_rules']}")
        report.append(f"Average confidence: {comparison['summary']['avg_confidence']:.2f}")
        report.append("")
        
        # By category
        report.append("BY CATEGORY:")
        report.append("-"*80)
        for category, count in comparison['summary']['categories'].items():
            report.append(f"  {category}: {count}")
        report.append("")
        
        # Novel findings (agent's added value)
        if comparison['agent_only']:
            report.append(f"🌟 NOVEL DETECTIONS (Agent's Added Value - {len(comparison['agent_only'])}):")
            report.append("-"*80)
            for finding in comparison['agent_only']:
                report.append(f"\n  [{finding['theme']}/{finding['type']}] - Confidence: {finding['confidence']:.2f}")
                report.append(f"  Category: {finding['category']}")
                report.append(f"  {finding['description']}")
            report.append("")
        
        # Detailed findings
        report.append("DETAILED FINDINGS:")
        report.append("-"*80)
        
        # Group by severity
        critical = [a for a in agent_anomalies if a.severity == 'CRITICAL']
        warnings = [a for a in agent_anomalies if a.severity == 'WARNING']
        info = [a for a in agent_anomalies if a.severity == 'INFO']
        
        for severity, anomalies in [('CRITICAL', critical), ('WARNING', warnings), ('INFO', info)]:
            if anomalies:
                emoji = {'CRITICAL': '🔴', 'WARNING': '⚠️', 'INFO': 'ℹ️'}[severity]
                report.append(f"\n{emoji} {severity} ({len(anomalies)}):")
                report.append("-"*80)
                
                for anom in anomalies:
                    detected = " [Also detected by rules]" if anom.detected_by_rules else " [Novel detection]"
                    report.append(f"\n  {anom.theme}/{anom.type_} - {anom.category}{detected}")
                    report.append(f"  Confidence: {anom.confidence:.2f}")
                    report.append(f"  Description: {anom.description}")
                    report.append(f"  Reasoning: {anom.reasoning}")
        
        return "\n".join(report)


def main():
    """
    Main execution: Run AI agent validator and compare with rule-based checker.
    """
    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("="*80)
        print("ERROR: Anthropic API key not found")
        print("="*80)
        print("\nTo use the AI agent validator, you need a Claude API key.")
        print("\nSteps:")
        print("1. Get a key at: https://console.anthropic.com")
        print("2. Set the environment variable:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("3. Run this script again")
        print("\nEstimated cost: ~$0.10-0.50 per release analysis")
        print("="*80)
        return
    
    # Load data (reuse the data loading from rule-based validator)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("AI AGENT RELEASE VALIDATOR")
    print("="*80)
    print("\nLoading release data...")
    
    # Import the data loading functions
    from release_validator import EnhancedReleaseValidator
    
    data = EnhancedReleaseValidator.load_all_data(base_dir)
    
    # Run rule-based checker first for comparison
    print("\nRunning rule-based checker for comparison...")
    rule_validator = EnhancedReleaseValidator()
    rule_anomalies = rule_validator.validate_release(data)
    
    # Run AI agent
    print("\nInitializing AI agent...")
    agent = AIReleaseValidator(api_key=api_key)
    
    agent_anomalies = agent.analyze_release(
        release_stats=data.get('release_stats'),
        column_historical=data.get('column_historical'),
        historical_totals=data.get('historical_totals'),
        rule_based_anomalies=rule_anomalies
    )
    
    # Compare results
    comparison = agent.compare_with_rules(agent_anomalies, rule_anomalies)
    
    # Generate and print report
    report = agent.generate_report(agent_anomalies, comparison)
    print(report)
    
    # Export results
    output_path = os.path.join(base_dir, 'ai_agent_anomalies.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'anomalies': [a.to_dict() for a in agent_anomalies],
            'comparison': comparison
        }, f, indent=2)
    
    print(f"\n✓ AI agent results exported to '{output_path}'")
    
    # Also export to CSV for easy viewing
    if agent_anomalies:
        csv_path = os.path.join(base_dir, 'ai_agent_anomalies.csv')
        pd.DataFrame([a.to_dict() for a in agent_anomalies]).to_csv(csv_path, index=False)
        print(f"✓ CSV export: '{csv_path}'")


if __name__ == "__main__":
    main()