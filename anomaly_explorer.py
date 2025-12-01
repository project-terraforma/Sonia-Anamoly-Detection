"""
Anomaly Explorer - Interactive filtering and grouping tool
==========================================================
Provides multiple ways to slice, filter, and analyze the raw anomalies.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Optional
import argparse


class AnomalyExplorer:
    """Interactive tool for exploring and filtering anomalies."""
    
    def __init__(self, json_path: str):
        """Load anomalies from JSON file."""
        with open(json_path) as f:
            data = json.load(f)
        
        self.raw_data = data
        self.df = pd.DataFrame(data['anomalies'])
        self.original_count = len(self.df)
        
        # Clean up data types
        if 'percent_change' in self.df.columns:
            self.df['percent_change'] = pd.to_numeric(self.df['percent_change'], errors='coerce')
        if 'previous_value' in self.df.columns:
            self.df['previous_value'] = pd.to_numeric(self.df['previous_value'], errors='coerce')
        if 'current_value' in self.df.columns:
            self.df['current_value'] = pd.to_numeric(self.df['current_value'], errors='coerce')
        
        # Add helper columns
        self.df['is_duplication'] = self.df['percent_change'].between(95, 105)
        self.df['is_near_total_loss'] = self.df['percent_change'].between(-105, -95)
        self.df['abs_change'] = self.df['percent_change'].abs()
        
        print(f"Loaded {self.original_count} anomalies")
        print(f"  - Critical: {len(self.df[self.df['severity'] == 'critical'])}")
        print(f"  - Warning: {len(self.df[self.df['severity'] == 'warning'])}")
        print()
    
    # ==================== FILTERING ====================
    
    def filter_severity(self, severity: str) -> 'AnomalyExplorer':
        """Filter by severity level ('critical' or 'warning')."""
        self.df = self.df[self.df['severity'] == severity]
        print(f"Filtered to {severity}: {len(self.df)} anomalies")
        return self
    
    def filter_theme(self, theme: str) -> 'AnomalyExplorer':
        """Filter by theme (e.g., 'buildings', 'divisions', 'places')."""
        self.df = self.df[self.df['theme'] == theme]
        print(f"Filtered to theme={theme}: {len(self.df)} anomalies")
        return self
    
    def filter_type(self, feature_type: str) -> 'AnomalyExplorer':
        """Filter by feature type (e.g., 'building', 'division', 'place')."""
        self.df = self.df[self.df['type'] == feature_type]
        print(f"Filtered to type={feature_type}: {len(self.df)} anomalies")
        return self
    
    def filter_country(self, country: str) -> 'AnomalyExplorer':
        """Filter by country code (e.g., 'US', 'RU', 'CN')."""
        self.df = self.df[self.df['country'] == country]
        print(f"Filtered to country={country}: {len(self.df)} anomalies")
        return self
    
    def filter_anomaly_type(self, anomaly_type: str) -> 'AnomalyExplorer':
        """Filter by anomaly type (e.g., 'feature_count_spike', 'feature_count_drop')."""
        self.df = self.df[self.df['anomaly_type'] == anomaly_type]
        print(f"Filtered to anomaly_type={anomaly_type}: {len(self.df)} anomalies")
        return self
    
    def filter_percent_change(self, min_pct: Optional[float] = None, max_pct: Optional[float] = None) -> 'AnomalyExplorer':
        """Filter by percent change range."""
        if min_pct is not None:
            self.df = self.df[self.df['percent_change'] >= min_pct]
        if max_pct is not None:
            self.df = self.df[self.df['percent_change'] <= max_pct]
        print(f"Filtered to percent_change [{min_pct}, {max_pct}]: {len(self.df)} anomalies")
        return self
    
    def exclude_duplication_bug(self) -> 'AnomalyExplorer':
        """Exclude the ~100% duplication anomalies to see other issues."""
        self.df = self.df[~self.df['is_duplication']]
        print(f"Excluded duplication bug: {len(self.df)} anomalies remaining")
        return self
    
    def only_duplication_bug(self) -> 'AnomalyExplorer':
        """Show only the ~100% duplication anomalies."""
        self.df = self.df[self.df['is_duplication']]
        print(f"Only duplication bug: {len(self.df)} anomalies")
        return self
    
    def filter_min_previous_value(self, min_val: float) -> 'AnomalyExplorer':
        """Filter to anomalies with significant sample size."""
        self.df = self.df[self.df['previous_value'] >= min_val]
        print(f"Filtered to previous_value >= {min_val}: {len(self.df)} anomalies")
        return self
    
    def reset(self) -> 'AnomalyExplorer':
        """Reset all filters."""
        self.df = pd.DataFrame(self.raw_data['anomalies'])
        self.df['is_duplication'] = self.df['percent_change'].between(95, 105)
        self.df['is_near_total_loss'] = self.df['percent_change'].between(-105, -95)
        self.df['abs_change'] = self.df['percent_change'].abs()
        print(f"Reset: {len(self.df)} anomalies")
        return self
    
    # ==================== GROUPING ====================
    
    def group_by(self, *columns) -> pd.DataFrame:
        """Group anomalies by specified columns and show counts."""
        cols = list(columns)
        grouped = self.df.groupby(cols).agg(
            count=('anomaly_type', 'size'),
            avg_percent_change=('percent_change', 'mean'),
            max_percent_change=('percent_change', 'max'),
            total_previous=('previous_value', 'sum'),
            total_current=('current_value', 'sum')
        ).round(2)
        grouped = grouped.sort_values('count', ascending=False)
        return grouped
    
    def group_by_theme(self) -> pd.DataFrame:
        """Group by theme."""
        return self.group_by('theme')
    
    def group_by_type(self) -> pd.DataFrame:
        """Group by theme and type."""
        return self.group_by('theme', 'type')
    
    def group_by_country(self) -> pd.DataFrame:
        """Group by country."""
        return self.group_by('country')
    
    def group_by_anomaly_type(self) -> pd.DataFrame:
        """Group by anomaly type."""
        return self.group_by('anomaly_type')
    
    def group_by_severity(self) -> pd.DataFrame:
        """Group by severity."""
        return self.group_by('severity')
    
    def group_by_pattern(self) -> pd.DataFrame:
        """Group by detected pattern (duplication vs other)."""
        self.df['pattern'] = 'other'
        self.df.loc[self.df['is_duplication'], 'pattern'] = 'duplication_bug (~100%)'
        self.df.loc[self.df['is_near_total_loss'], 'pattern'] = 'near_total_loss (~-100%)'
        return self.group_by('pattern')
    
    # ==================== ANALYSIS ====================
    
    def summary(self) -> dict:
        """Get summary statistics of current filtered data."""
        return {
            'total_anomalies': len(self.df),
            'by_severity': self.df['severity'].value_counts().to_dict(),
            'by_anomaly_type': self.df['anomaly_type'].value_counts().to_dict(),
            'by_theme': self.df['theme'].value_counts().to_dict(),
            'duplication_bug_count': self.df['is_duplication'].sum(),
            'other_anomalies': (~self.df['is_duplication']).sum(),
            'unique_countries': self.df['country'].nunique(),
            'countries_affected': self.df['country'].dropna().unique().tolist()[:20],
        }
    
    def top_countries(self, n: int = 20) -> pd.DataFrame:
        """Show top N countries by anomaly count."""
        return self.df['country'].value_counts().head(n).to_frame('anomaly_count')
    
    def top_anomalies(self, n: int = 20, by: str = 'abs_change') -> pd.DataFrame:
        """Show top N anomalies by a metric (abs_change, previous_value, etc.)."""
        cols = ['severity', 'anomaly_type', 'theme', 'type', 'subtype', 'country', 
                'percent_change', 'previous_value', 'current_value']
        available_cols = [c for c in cols if c in self.df.columns]
        return self.df.nlargest(n, by)[available_cols]
    
    def unique_patterns(self) -> pd.DataFrame:
        """Find unique anomaly patterns (excluding duplication bug)."""
        non_dup = self.df[~self.df['is_duplication']].copy()
        return non_dup.groupby(['anomaly_type', 'theme', 'type']).agg(
            count=('severity', 'size'),
            countries=('country', lambda x: ', '.join(sorted(set(x.dropna().astype(str)))[:5])),
            avg_change=('percent_change', 'mean')
        ).round(2).sort_values('count', ascending=False)
    
    # ==================== EXPORT ====================
    
    def to_csv(self, path: str) -> None:
        """Export current filtered data to CSV."""
        self.df.to_csv(path, index=False)
        print(f"Exported {len(self.df)} anomalies to {path}")
    
    def to_excel(self, path: str) -> None:
        """Export to Excel with multiple sheets."""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # All anomalies
            self.df.to_excel(writer, sheet_name='All Anomalies', index=False)
            
            # Summary sheets
            self.group_by_severity().to_excel(writer, sheet_name='By Severity')
            self.group_by_theme().to_excel(writer, sheet_name='By Theme')
            self.group_by_anomaly_type().to_excel(writer, sheet_name='By Anomaly Type')
            self.group_by_country().to_excel(writer, sheet_name='By Country')
            self.group_by_pattern().to_excel(writer, sheet_name='By Pattern')
            
            # Top anomalies
            self.top_anomalies(50).to_excel(writer, sheet_name='Top 50 by Change', index=False)
            
            # Unique patterns (non-duplication)
            self.unique_patterns().to_excel(writer, sheet_name='Unique Patterns')
        
        print(f"Exported to {path} with multiple analysis sheets")
    
    def to_html_report(self, path: str) -> None:
        """Export interactive HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Explorer Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #1e3a5f; }}
        h2 {{ color: #16a085; border-bottom: 2px solid #16a085; padding-bottom: 8px; }}
        .summary-box {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; color: #1e3a5f; }}
        .stat-label {{ font-size: 14px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; background: white; margin-bottom: 20px; }}
        th {{ background: #1e3a5f; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f0f4f8; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; }}
        .filter-section {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        input, select {{ padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }}
        button {{ background: #16a085; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }}
        button:hover {{ background: #138d75; }}
    </style>
</head>
<body>
    <h1>🔍 Anomaly Explorer Report</h1>
    
    <div class="summary-box">
        <div class="stat">
            <div class="stat-value">{len(self.df)}</div>
            <div class="stat-label">Total Anomalies</div>
        </div>
        <div class="stat">
            <div class="stat-value critical">{len(self.df[self.df['severity'] == 'critical'])}</div>
            <div class="stat-label">Critical</div>
        </div>
        <div class="stat">
            <div class="stat-value warning">{len(self.df[self.df['severity'] == 'warning'])}</div>
            <div class="stat-label">Warning</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self.df['is_duplication'].sum()}</div>
            <div class="stat-label">Duplication Bug</div>
        </div>
        <div class="stat">
            <div class="stat-value">{(~self.df['is_duplication']).sum()}</div>
            <div class="stat-label">Other Issues</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self.df['country'].nunique()}</div>
            <div class="stat-label">Countries Affected</div>
        </div>
    </div>
    
    <h2>By Pattern</h2>
    {self.group_by_pattern().to_html(classes='table')}
    
    <h2>By Theme</h2>
    {self.group_by_theme().to_html(classes='table')}
    
    <h2>By Anomaly Type</h2>
    {self.group_by_anomaly_type().to_html(classes='table')}
    
    <h2>🌍 Top 20 Countries</h2>
    {self.top_countries(20).to_html(classes='table')}
    
    <h2>🔥 Top 30 Anomalies by Change</h2>
    {self.top_anomalies(30).to_html(classes='table', index=False)}
    
    <h2>🎯 Unique Patterns (Excluding Duplication Bug)</h2>
    {self.unique_patterns().to_html(classes='table')}
    
    <h2>📋 All Anomalies</h2>
    <div class="filter-section">
        <input type="text" id="search" placeholder="Search..." onkeyup="filterTable()">
        <select id="severityFilter" onchange="filterTable()">
            <option value="">All Severities</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
        </select>
        <select id="themeFilter" onchange="filterTable()">
            <option value="">All Themes</option>
            {' '.join(f'<option value="{t}">{t}</option>' for t in self.df['theme'].unique())}
        </select>
        <button onclick="resetFilters()">Reset</button>
    </div>
    
    <table id="anomalyTable">
        <thead>
            <tr>
                <th>Severity</th>
                <th>Type</th>
                <th>Theme</th>
                <th>Feature</th>
                <th>Country</th>
                <th>% Change</th>
                <th>Previous</th>
                <th>Current</th>
            </tr>
        </thead>
        <tbody>
            {''.join(f'''<tr class="{row.get('severity', '')}">
                <td class="{row.get('severity', '')}">{row.get('severity', '').upper()}</td>
                <td>{row.get('anomaly_type', '')}</td>
                <td>{row.get('theme', '')}</td>
                <td>{row.get('type', '')}</td>
                <td>{row.get('country', '')}</td>
                <td>{row.get('percent_change', 0):.1f}%</td>
                <td>{row.get('previous_value', 0):,.0f}</td>
                <td>{row.get('current_value', 0):,.0f}</td>
            </tr>''' for _, row in self.df.iterrows())}
        </tbody>
    </table>
    
    <script>
        function filterTable() {{
            const search = document.getElementById('search').value.toLowerCase();
            const severity = document.getElementById('severityFilter').value;
            const theme = document.getElementById('themeFilter').value;
            const rows = document.querySelectorAll('#anomalyTable tbody tr');
            
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                const rowSeverity = row.cells[0].textContent.toLowerCase();
                const rowTheme = row.cells[2].textContent;
                
                const matchSearch = text.includes(search);
                const matchSeverity = !severity || rowSeverity === severity;
                const matchTheme = !theme || rowTheme === theme;
                
                row.style.display = (matchSearch && matchSeverity && matchTheme) ? '' : 'none';
            }});
        }}
        
        function resetFilters() {{
            document.getElementById('search').value = '';
            document.getElementById('severityFilter').value = '';
            document.getElementById('themeFilter').value = '';
            filterTable();
        }}
    </script>
</body>
</html>
"""
        Path(path).write_text(html)
        print(f"Exported interactive HTML report to {path}")


def print_examples():
    """Print usage examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ANOMALY EXPLORER - USAGE EXAMPLES                  ║
╚══════════════════════════════════════════════════════════════════════╝

# Load anomalies
explorer = AnomalyExplorer('all_anomalies.json')

# ==================== FILTERING ====================

# Filter by severity
explorer.filter_severity('critical')

# Filter by theme
explorer.filter_theme('divisions')

# Filter by country
explorer.filter_country('US')

# Exclude the duplication bug to see OTHER issues
explorer.exclude_duplication_bug()

# Only show duplication bug anomalies
explorer.only_duplication_bug()

# Filter by percent change range
explorer.filter_percent_change(min_pct=50, max_pct=150)

# Chain filters
explorer.reset().filter_severity('critical').filter_theme('buildings')

# ==================== GROUPING ====================

# Group by different dimensions
print(explorer.group_by_theme())
print(explorer.group_by_country())
print(explorer.group_by_anomaly_type())
print(explorer.group_by_pattern())

# Custom grouping
print(explorer.group_by('theme', 'type', 'country'))

# ==================== ANALYSIS ====================

# Summary stats
print(explorer.summary())

# Top countries
print(explorer.top_countries(20))

# Top anomalies by change
print(explorer.top_anomalies(20))

# Unique patterns (excluding duplication)
print(explorer.unique_patterns())

# ==================== EXPORT ====================

# Export to CSV
explorer.to_csv('filtered_anomalies.csv')

# Export to Excel with multiple analysis sheets
explorer.to_excel('anomaly_analysis.xlsx')

# Export interactive HTML report
explorer.to_html_report('anomaly_report.html')
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore and filter anomalies")
    parser.add_argument("json_file", type=str, help="Path to all_anomalies.json")
    parser.add_argument("--export-excel", type=str, help="Export to Excel file")
    parser.add_argument("--export-html", type=str, help="Export to HTML report")
    parser.add_argument("--export-csv", type=str, help="Export to CSV")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    
    args = parser.parse_args()
    
    if args.examples:
        print_examples()
        exit()
    
    explorer = AnomalyExplorer(args.json_file)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = explorer.summary()
    print(f"Total: {summary['total_anomalies']}")
    print(f"By Severity: {summary['by_severity']}")
    print(f"Duplication Bug: {summary['duplication_bug_count']}")
    print(f"Other Anomalies: {summary['other_anomalies']}")
    print(f"Countries Affected: {summary['unique_countries']}")
    
    print("\n" + "="*60)
    print("BY PATTERN")
    print("="*60)
    print(explorer.group_by_pattern())
    
    print("\n" + "="*60)
    print("BY THEME")
    print("="*60)
    print(explorer.group_by_theme())
    
    print("\n" + "="*60)
    print("UNIQUE PATTERNS (NON-DUPLICATION)")
    print("="*60)
    print(explorer.unique_patterns())
    
    # Export if requested
    if args.export_excel:
        explorer.reset()
        explorer.to_excel(args.export_excel)
    
    if args.export_html:
        explorer.reset()
        explorer.to_html_report(args.export_html)
    
    if args.export_csv:
        explorer.reset()
        explorer.to_csv(args.export_csv)
