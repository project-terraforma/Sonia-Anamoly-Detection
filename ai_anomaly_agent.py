"""
AI-Powered Anomaly Detection Agent
==================================
Uses rule-based detection + LLM reasoning to find and explain anomalies.
"""

import os
import json
from pathlib import Path
from anthropic import Anthropic

# Import the rule-based detector
from anomaly_detector import OvertureAnomalyDetector, AnomalyThresholds


class AIAnomalyAgent:
    """
    AI Agent that combines rule-based detection with LLM reasoning.
    """
    
    def __init__(self, data_root: Path, api_key: str = None):
        self.data_root = Path(data_root)
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.detector = OvertureAnomalyDetector(data_root)
        self.conversation_history = []
        
    def analyze_release(self, current: str = None, previous: str = None) -> dict:
        """Run rule-based detection and get AI interpretation."""
        
        # Step 1: Run rule-based detection
        print("🔍 Running rule-based anomaly detection...")
        anomalies = self.detector.run_full_analysis(current, previous)
        
        # Step 2: Prepare context for AI
        anomaly_summary = {
            "total": len(anomalies),
            "critical": len([a for a in anomalies if a.severity.value == "critical"]),
            "warnings": len([a for a in anomalies if a.severity.value == "warning"]),
            "by_theme": {},
            "details": [a.to_dict() for a in anomalies[:20]]  # Top 20 for context
        }
        
        # Group by theme
        for a in anomalies:
            theme = a.theme
            if theme not in anomaly_summary["by_theme"]:
                anomaly_summary["by_theme"][theme] = {"critical": 0, "warning": 0}
            anomaly_summary["by_theme"][theme][a.severity.value] += 1
        
        # Step 3: Get AI interpretation
        print("🤖 Getting AI interpretation...")
        interpretation = self._get_ai_interpretation(anomaly_summary)
        
        return {
            "anomalies": anomaly_summary,
            "ai_interpretation": interpretation,
            "raw_anomalies": [a.to_dict() for a in anomalies]
        }
    
    def _get_ai_interpretation(self, anomaly_data: dict) -> str:
        """Use Claude to interpret the anomaly findings."""
        
        prompt = f"""You are a data quality analyst for Overture Maps geospatial data. 
Analyze these anomaly detection results and provide:

1. **Executive Summary** (2-3 sentences on overall health)
2. **Top Concerns** (prioritized list of issues needing attention)
3. **Likely Root Causes** (what might have caused these anomalies)
4. **Recommended Actions** (specific next steps)

ANOMALY DATA:
```json
{json.dumps(anomaly_data, indent=2)}
```

Be specific and actionable. Reference actual numbers from the data."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def chat(self, user_message: str) -> str:
        """Interactive chat about the data."""
        
        # Add context about available data
        system_prompt = f"""You are an AI assistant analyzing Overture Maps release data.
        
Available data location: {self.data_root}
Available releases: {self.detector.get_sorted_releases()}

You can help users:
- Understand anomaly detection results
- Investigate specific themes/types/countries
- Explain what metrics mean
- Suggest follow-up analyses

Be specific and reference actual data when possible."""

        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def investigate(self, theme: str, feature_type: str, country: str = None) -> str:
        """Deep-dive investigation into a specific area."""
        
        # Load relevant data
        releases = self.detector.get_sorted_releases()
        current = releases[-1]
        previous = releases[-2]
        
        df_current = self.detector.load_row_counts(current, theme, feature_type)
        df_previous = self.detector.load_row_counts(previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return f"Could not load data for {theme}/{feature_type}"
        
        # Filter by country if specified
        if country and 'country' in df_current.columns:
            df_current = df_current[df_current['country'] == country]
            df_previous = df_previous[df_previous['country'] == country]
        
        # Prepare summary statistics
        current_stats = {
            "total_features": int(df_current['id_count'].sum()),
            "total_length_km": float(df_current['total_geometry_length_km'].sum()) if 'total_geometry_length_km' in df_current.columns else None,
        }
        previous_stats = {
            "total_features": int(df_previous['id_count'].sum()),
            "total_length_km": float(df_previous['total_geometry_length_km'].sum()) if 'total_geometry_length_km' in df_previous.columns else None,
        }
        
        # Get AI analysis
        prompt = f"""Analyze this specific data slice from Overture Maps:

Theme: {theme}
Type: {feature_type}
Country: {country or 'All'}
Releases compared: {previous} → {current}

Previous release stats:
{json.dumps(previous_stats, indent=2)}

Current release stats:
{json.dumps(current_stats, indent=2)}

Top subtypes in current release:
{df_current.groupby('subtype')['id_count'].sum().sort_values(ascending=False).head(10).to_dict() if 'subtype' in df_current.columns else 'N/A'}

Provide insights on:
1. What changed and by how much?
2. Is this change expected or concerning?
3. What should be investigated further?"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text


def main():
    """Interactive CLI for the AI agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-powered anomaly detection agent")
    parser.add_argument("data_root", type=Path, help="Path to data directory")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument("--investigate", nargs=2, metavar=("THEME", "TYPE"), help="Investigate specific theme/type")
    parser.add_argument("--country", type=str, help="Filter investigation by country")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY environment variable to use AI features")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("\nFalling back to rule-based only...")
        
        detector = OvertureAnomalyDetector(args.data_root)
        anomalies = detector.run_full_analysis()
        print(detector.generate_report())
        return
    
    agent = AIAnomalyAgent(args.data_root)
    
    if args.analyze:
        results = agent.analyze_release()
        print("\n" + "="*70)
        print("AI INTERPRETATION")
        print("="*70)
        print(results["ai_interpretation"])
        
    elif args.investigate:
        theme, ftype = args.investigate
        print(agent.investigate(theme, ftype, args.country))
        
    elif args.chat:
        print("🤖 AI Anomaly Agent - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
                
            response = agent.chat(user_input)
            print(f"\nAgent: {response}\n")
    
    else:
        # Default: run analysis
        results = agent.analyze_release()
        print("\n" + "="*70)
        print("AI INTERPRETATION")
        print("="*70)
        print(results["ai_interpretation"])


if __name__ == "__main__":
    main()