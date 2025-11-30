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
    attribute_breakdown: List[Dict] = None
    
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
        if self.attribute_breakdown:
            result['attribute_breakdown'] = self.attribute_breakdown
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
                    # Read with error handling for malformed rows
                    df = pd.read_csv(csv_files[0], on_bad_lines='skip', engine='python')
                    key = f"{theme}/{type_}"
                    row_counts[key] = df
                    print(f"  ✓ {key}: {len(df)} rows")
                except Exception as e:
                    print(f"  ✗ Error loading {theme}/{type_}: {e}")
    
    print(f"✓ Loaded {len(row_counts)} datasets with granular data")
    return row_counts


def load_attribute_stats(base_dir: str, release_date: str) -> Dict[str, pd.DataFrame]:
    """Load attribute-level statistics from theme_column_summary_stats."""
    attr_stats = {}
    attr_dir = os.path.join(base_dir, 'Metrics', 'theme_column_summary_stats')
    
    if not os.path.exists(attr_dir):
        print(f"Warning: Attribute stats not found")
        return attr_stats
    
    print(f"\nLoading attribute statistics for {release_date}...")
    
    # Find files matching the release date
    pattern = f"{release_date}.theme=*.csv"
    for csv_file in glob.glob(os.path.join(attr_dir, pattern)):
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            # Parse: 2025-09-24.0.theme=places.type=place.csv
            # Split only on first 2 dots to handle release version numbers
            parts = filename.replace('.csv', '').split('.', 2)
            if len(parts) < 3:
                print(f"  ✗ Skipping {filename}: unexpected format")
                continue
            
            # Parse theme and type from the last part
            metadata = parts[2]  # e.g., "theme=places.type=place"
            meta_parts = metadata.split('.')
            theme = None
            type_ = None
            
            for part in meta_parts:
                if part.startswith('theme='):
                    theme = part.split('=')[1]
                elif part.startswith('type='):
                    type_ = part.split('=')[1]
            
            if theme and type_:
                key = f"{theme}/{type_}"
                attr_stats[key] = df
                print(f"  ✓ {key}: {len(df.columns)} attributes tracked")
            else:
                print(f"  ✗ Could not parse theme/type from {filename}")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")
    
    print(f"✓ Loaded attribute stats for {len(attr_stats)} datasets")
    return attr_stats


def load_attribute_comparisons(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Load release-to-release attribute comparisons."""
    comparisons = {}
    comp_dir = os.path.join(base_dir, 'Metrics', 'theme_column_summary_stats', 
                            'release_to_release_comparisons')
    
    if not os.path.exists(comp_dir):
        print(f"Warning: Comparison stats not found")
        return comparisons
    
    print(f"\nLoading release-to-release comparisons...")
    
    for csv_file in glob.glob(os.path.join(comp_dir, 'theme=*.csv')):
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            # Parse: theme=places.type=place.csv
            parts = filename.replace('.csv', '').split('.')
            
            theme = None
            type_ = None
            for part in parts:
                if part.startswith('theme='):
                    theme = part.split('=')[1]
                elif part.startswith('type='):
                    type_ = part.split('=')[1]
            
            if theme and type_:
                key = f"{theme}/{type_}"
                comparisons[key] = df
                print(f"  ✓ {key}: {len(df)} rows across releases")
            else:
                print(f"  ✗ Could not parse theme/type from {filename}")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")
    
    print(f"✓ Loaded comparisons for {len(comparisons)} datasets")
    return comparisons


def analyze_attribute_coverage(attr_stats: Dict[str, pd.DataFrame],
                               comparisons: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Analyze attribute coverage changes."""
    findings = []
    
    # Key attributes to monitor by theme
    key_attributes = {
        'places': ['names', 'websites', 'phones', 'categories', 'confidence', 'emails', 'socials', 'brand'],
        'buildings': ['names', 'height', 'level', 'class', 'subtype'],
        'divisions': ['names', 'wikidata', 'country'],
        'addresses': ['address_level_1', 'address_level_2', 'address_level_3']
    }
    
    for key, df in attr_stats.items():
        theme, type_ = key.split('/')
        
        if theme not in key_attributes or len(df) == 0:
            continue
        
        # Get the first (and usually only) row with counts
        row = df.iloc[0]
        
        # Get total count
        total_records = row.get('total_count', row.get('id', None))
        
        if total_records is None or total_records == 0:
            continue
        
        total_records = int(total_records)
        
        # Check each key attribute
        for attr_name in key_attributes[theme]:
            if attr_name in df.columns:
                count = row[attr_name]
                if pd.notna(count) and count > 0:
                    coverage = (count / total_records) * 100
                    
                    findings.append({
                        'theme': theme,
                        'type': type_,
                        'attribute': attr_name,
                        'coverage_pct': round(coverage, 2),
                        'count': int(count),
                        'total': int(total_records)
                    })
    
    return findings


def detect_attribute_drops(comparisons: Dict[str, pd.DataFrame],
                          attr_stats: Dict[str, pd.DataFrame],
                          current_release: str,
                          previous_release: str) -> List[GeoAnomaly]:
    """Detect significant drops in attribute coverage."""
    anomalies = []
    threshold = 5.0  # Flag drops > 5%
    
    # Key attributes to monitor
    key_attributes = {
        'places': ['names', 'websites', 'phones', 'categories', 'emails', 'socials'],
        'buildings': ['names', 'height', 'level'],
        'divisions': ['names', 'wikidata'],
        'addresses': ['address_level_1', 'address_level_2', 'address_level_3']
    }
    
    # Use comparisons files (release-to-release) for historical comparison
    for key, df in comparisons.items():
        theme, type_ = key.split('/')
        
        if theme not in key_attributes:
            continue
        
        # Check if we have both releases in columns
        if current_release not in df.columns or previous_release not in df.columns:
            continue
        
        # The comparison files have attribute names, not _count suffix
        # Each row represents a different metric/attribute
        for attr_name in key_attributes[theme]:
            # Try to find rows matching this attribute
            matching_rows = df[df.apply(lambda row: attr_name in str(row.get('type', '')).lower() or 
                                                     attr_name in str(row.get('subtype', '')).lower() or
                                                     attr_name in str(row.get('class', '')).lower(), axis=1)]
            
            if len(matching_rows) == 0:
                continue
            
            for _, row in matching_rows.iterrows():
                current_val = row.get(current_release, 0)
                previous_val = row.get(previous_release, 0)
                
                if pd.notna(current_val) and pd.notna(previous_val) and previous_val > 0:
                    pct_change = ((current_val - previous_val) / previous_val) * 100
                    
                    if pct_change < -threshold:
                        anomaly = GeoAnomaly(
                            theme=theme,
                            type_=type_,
                            severity='WARNING',
                            category='Attribute Coverage Drop',
                            description=f"{attr_name} coverage dropped {abs(pct_change):.1f}%",
                            reasoning=f"Attribute completeness decreased from {previous_val:,.0f} to {current_val:,.0f} records. This may indicate data source issues or processing problems.",
                            confidence=0.85,
                            detected_by_rules=True,
                            attribute_breakdown=[{
                                'attribute': attr_name,
                                'previous': int(previous_val),
                                'current': int(current_val),
                                'change_pct': round(pct_change, 2)
                            }]
                        )
                        anomalies.append(anomaly)
    
    # Also check current vs previous in attr_stats if we can get previous stats
    # For now, we'll rely on comparison files above
    
    return anomalies


def analyze_incomplete_data(row_counts: Dict[str, pd.DataFrame]) -> List[GeoAnomaly]:
    """Detect increases in incomplete/missing data."""
    anomalies = []
    
    for key, df in row_counts.items():
        theme, type_ = key.split('/')
        
        # Check addresses for missing levels
        if theme == 'addresses' and type_ == 'address':
            required_fields = ['address_level_1', 'address_level_2', 'address_level_3']
            
            for field in required_fields:
                if field in df.columns:
                    # Count nulls or 0 values
                    total = df['total_count'].sum()
                    missing = df[df[field].isna() | (df[field] == 0)]['total_count'].sum()
                    
                    if total > 0:
                        missing_pct = (missing / total) * 100
                        
                        if missing_pct > 30:  # Flag if >30% missing
                            anomaly = GeoAnomaly(
                                theme=theme,
                                type_=type_,
                                severity='WARNING',
                                category='Incomplete Data',
                                description=f"{missing_pct:.1f}% of addresses missing {field}",
                                reasoning=f"High percentage of incomplete addresses may impact geocoding quality and address validation.",
                                confidence=0.80,
                                detected_by_rules=True
                            )
                            anomalies.append(anomaly)
    
    return anomalies


def analyze_category_fluctuations(row_counts: Dict[str, pd.DataFrame]) -> List[GeoAnomaly]:
    """Detect unusual category/class changes."""
    anomalies = []
    
    for key, df in row_counts.items():
        theme, type_ = key.split('/')
        
        # Check for subtype/class columns
        if 'subtype' not in df.columns or 'change_type' not in df.columns:
            continue
        
        # Group by subtype and change type
        changes = df.groupby(['subtype', 'change_type'])['total_count'].sum().reset_index()
        
        # Find categories with high removals
        removed = changes[changes['change_type'] == 'removed']
        added = changes[changes['change_type'] == 'added']
        
        for _, row in removed.iterrows():
            subtype = row['subtype']
            removed_count = row['total_count']
            
            # Find corresponding additions
            added_row = added[added['subtype'] == subtype]
            added_count = added_row['total_count'].iloc[0] if len(added_row) > 0 else 0
            
            # Flag if removals >> additions
            if removed_count > added_count * 2 and removed_count > 1000:
                anomaly = GeoAnomaly(
                    theme=theme,
                    type_=type_,
                    severity='WARNING',
                    category='Category Fluctuation',
                    description=f"{subtype} shows high removal rate: {removed_count:,} removed vs {added_count:,} added",
                    reasoning="Disproportionate removals vs additions may indicate data source changes or quality issues in this category.",
                    confidence=0.75,
                    detected_by_rules=True,
                    category_breakdown=[{
                        'subtype': subtype,
                        'removed': int(removed_count),
                        'added': int(added_count),
                        'net_change': int(added_count - removed_count)
                    }]
                )
                anomalies.append(anomaly)
    
    return anomalies


def analyze_geometry_anomalies(row_counts: Dict[str, pd.DataFrame]) -> List[GeoAnomaly]:
    """Detect statistical geometry anomalies."""
    anomalies = []
    
    for key, df in row_counts.items():
        theme, type_ = key.split('/')
        
        # Check for geometry columns
        if 'average_geometry_area_km2' not in df.columns:
            continue
        
        # Buildings with 0 or very small areas
        if theme == 'buildings':
            zero_area = df[df['average_geometry_area_km2'] == 0]
            if len(zero_area) > 0:
                total_zero = zero_area['total_count'].sum()
                
                if total_zero > 100:
                    anomaly = GeoAnomaly(
                        theme=theme,
                        type_=type_,
                        severity='INFO',
                        category='Geometry Quality',
                        description=f"{total_zero:,} buildings with zero area detected",
                        reasoning="Buildings with zero area may indicate flat geometries, data errors, or point representations that should be polygons.",
                        confidence=0.70,
                        detected_by_rules=True
                    )
                    anomalies.append(anomaly)
        
        # Sudden changes in average geometry size
        if 'change_type' in df.columns:
            added = df[df['change_type'] == 'added']
            unchanged = df[df['change_type'] == 'unchanged']
            
            if len(added) > 0 and len(unchanged) > 0:
                added_avg = added['average_geometry_area_km2'].mean()
                unchanged_avg = unchanged['average_geometry_area_km2'].mean()
                
                if unchanged_avg > 0:
                    pct_diff = abs((added_avg - unchanged_avg) / unchanged_avg) * 100
                    
                    if pct_diff > 50:  # 50% difference
                        anomaly = GeoAnomaly(
                            theme=theme,
                            type_=type_,
                            severity='WARNING',
                            category='Geometry Quality',
                            description=f"New {type_} geometries {pct_diff:.0f}% {'larger' if added_avg > unchanged_avg else 'smaller'} than existing",
                            reasoning=f"Average area changed from {unchanged_avg:.2f} to {added_avg:.2f} km². This may indicate unit conversion errors, data source changes, or processing issues.",
                            confidence=0.75,
                            detected_by_rules=True
                        )
                        anomalies.append(anomaly)
    
    return anomalies


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


def create_enhanced_prompt(current_stats: List[Dict], 
                           geo_summary: str,
                           rule_findings: List[Dict],
                           rule_anomalies: List[GeoAnomaly],
                           attr_findings: List[Dict]) -> str:
    """Create the enhanced analysis prompt."""
    
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
    prompt_parts.append("# Attribute Coverage Analysis")
    prompt_parts.append("Current attribute completeness by theme:")
    prompt_parts.append(json.dumps(attr_findings[:20], indent=2))  # Limit to top 20
    prompt_parts.append("")
    prompt_parts.append("# Rule-Based Findings")
    prompt_parts.append(json.dumps(rule_findings, indent=2))
    prompt_parts.append("")
    prompt_parts.append("# Automated Anomaly Detections")
    prompt_parts.append(f"The system has pre-detected {len(rule_anomalies)} anomalies:")
    prompt_parts.append(json.dumps([a.to_dict() for a in rule_anomalies[:10]], indent=2))
    prompt_parts.append("")
    prompt_parts.append("# Your Task")
    prompt_parts.append("Review ALL the data and identify ADDITIONAL anomalies not already detected.")
    prompt_parts.append("Focus on:")
    prompt_parts.append("- Patterns across multiple themes")
    prompt_parts.append("- Subtle data quality issues")
    prompt_parts.append("- Geographic patterns that need investigation")
    prompt_parts.append("- Category-level trends that may indicate problems")
    prompt_parts.append("")
    prompt_parts.append("IMPORTANT: When geographic data is available, ALWAYS include geographic_breakdown with specific countries.")
    prompt_parts.append("IMPORTANT: Include category_breakdown when you notice category/class patterns.")
    prompt_parts.append("")
    prompt_parts.append("Return ONLY a JSON array (no other text):")
    prompt_parts.append("[")
    prompt_parts.append("  {")
    prompt_parts.append('    "theme": "theme_name",')
    prompt_parts.append('    "type": "type_name",')
    prompt_parts.append('    "severity": "CRITICAL|WARNING|INFO",')
    prompt_parts.append('    "category": "Data Quality|Geographic Pattern|Attribute Coverage|etc",')
    prompt_parts.append('    "description": "Brief description",')
    prompt_parts.append('    "reasoning": "Why this matters and what it might indicate",')
    prompt_parts.append('    "confidence": 0.85,')
    prompt_parts.append('    "detected_by_rules": false,')
    prompt_parts.append('    "geographic_breakdown": [')
    prompt_parts.append('      {"country": "US", "metric": "removed", "value": 1234}')
    prompt_parts.append('    ],')
    prompt_parts.append('    "category_breakdown": [')
    prompt_parts.append('      {"subtype": "restaurant", "change": -500}')
    prompt_parts.append('    ]')
    prompt_parts.append("  }")
    prompt_parts.append("]")
    
    return "\n".join(prompt_parts)


def run_enhanced_agent(api_key: str, 
                      current_stats: pd.DataFrame,
                      row_counts: Dict[str, pd.DataFrame],
                      attr_stats: Dict[str, pd.DataFrame],
                      comparisons: Dict[str, pd.DataFrame],
                      rule_anomalies: List) -> List[GeoAnomaly]:
    """Run the enhanced AI agent with all detections."""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Run all detection analyses
    print("\n" + "="*80)
    print("RUNNING AUTOMATED ANOMALY DETECTIONS")
    print("="*80)
    
    detected_anomalies = []
    
    # 1. Attribute coverage drops
    print("\n1. Analyzing attribute coverage changes...")
    # Note: Need to pass actual release dates
    current_release = '2025-09-24.0'
    previous_release = '2025-08-20.1'  # You'll need to determine this
    attr_drop_anomalies = detect_attribute_drops(comparisons, attr_stats, current_release, previous_release)
    detected_anomalies.extend(attr_drop_anomalies)
    print(f"   ✓ Found {len(attr_drop_anomalies)} attribute coverage anomalies")
    
    # 2. Incomplete data
    print("\n2. Analyzing incomplete data patterns...")
    incomplete_anomalies = analyze_incomplete_data(row_counts)
    detected_anomalies.extend(incomplete_anomalies)
    print(f"   ✓ Found {len(incomplete_anomalies)} incomplete data anomalies")
    
    # 3. Category fluctuations
    print("\n3. Analyzing category fluctuations...")
    category_anomalies = analyze_category_fluctuations(row_counts)
    detected_anomalies.extend(category_anomalies)
    print(f"   ✓ Found {len(category_anomalies)} category anomalies")
    
    # 4. Geometry anomalies
    print("\n4. Analyzing geometry patterns...")
    geometry_anomalies = analyze_geometry_anomalies(row_counts)
    detected_anomalies.extend(geometry_anomalies)
    print(f"   ✓ Found {len(geometry_anomalies)} geometry anomalies")
    
    print(f"\n✓ Total automated detections: {len(detected_anomalies)}")
    
    # Prepare data for AI analysis
    stats_list = current_stats.to_dict('records') if current_stats is not None else []
    geo_summary = prepare_geographic_summary(row_counts)
    attr_findings = analyze_attribute_coverage(attr_stats, comparisons)
    
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
    
    # Create enhanced prompt
    prompt = create_enhanced_prompt(stats_list, geo_summary, rule_findings, 
                                   detected_anomalies, attr_findings)
    
    print("\n" + "="*80)
    print("AI AGENT DEEP ANALYSIS")
    print("="*80)
    print(f"Analyzing with data from {len(row_counts)} themes...")
    print(f"AI will look for additional patterns beyond {len(detected_anomalies)} pre-detected anomalies...")
    
    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    ai_anomalies = []
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
                    category_breakdown=item.get('category_breakdown'),
                    attribute_breakdown=item.get('attribute_breakdown')
                )
                ai_anomalies.append(anomaly)
        
        print(f"✓ AI analysis complete: {len(ai_anomalies)} additional anomalies found")
        
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        print(f"Response text: {response_text[:500]}")
    
    # Combine all anomalies
    all_anomalies = detected_anomalies + ai_anomalies
    
    return all_anomalies


def generate_enhanced_report(anomalies: List[GeoAnomaly]) -> str:
    """Generate enhanced report."""
    if not anomalies:
        return "\n✓ No anomalies detected"
    
    lines = []
    lines.append("\n" + "="*80)
    lines.append("ENHANCED ANOMALY DETECTION REPORT")
    lines.append("="*80)
    
    rule_based = sum(1 for a in anomalies if a.detected_by_rules)
    ai_detected = sum(1 for a in anomalies if not a.detected_by_rules)
    with_geo = sum(1 for a in anomalies if a.geographic_breakdown)
    with_attr = sum(1 for a in anomalies if a.attribute_breakdown)
    
    lines.append(f"\nTotal: {len(anomalies)} anomalies")
    lines.append(f"  • Rule-based detections: {rule_based}")
    lines.append(f"  • AI-discovered patterns: {ai_detected}")
    lines.append(f"  • With geographic detail: {with_geo}")
    lines.append(f"  • With attribute detail: {with_attr}")
    
    critical = [a for a in anomalies if a.severity == 'CRITICAL']
    warnings = [a for a in anomalies if a.severity == 'WARNING']
    info = [a for a in anomalies if a.severity == 'INFO']
    
    for severity, items in [('CRITICAL', critical), ('WARNING', warnings), ('INFO', info)]:
        if items:
            lines.append(f"\n{severity} ({len(items)}):")
            lines.append("-"*80)
            
            for a in items:
                detection_type = " AI" if not a.detected_by_rules else " Rule"
                lines.append(f"\n  {detection_type} | {a.theme}/{a.type_} - {a.category}")
                lines.append(f"  Confidence: {a.confidence:.2f}")
                lines.append(f"  {a.description}")
                
                if a.geographic_breakdown:
                    lines.append("  Geographic breakdown:")
                    for g in a.geographic_breakdown[:5]:
                        lines.append(f"    • {g}")
                
                if a.category_breakdown:
                    lines.append("  Category breakdown:")
                    for c in a.category_breakdown[:5]:
                        lines.append(f"    • {c}")
                
                if a.attribute_breakdown:
                    lines.append("  Attribute breakdown:")
                    for attr in a.attribute_breakdown[:5]:
                        lines.append(f"    • {attr}")
                
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
    print("ENHANCED GEOGRAPHIC AI AGENT v2.0")
    print("="*80)
    print("Features:")
    print("  ✓ Attribute coverage analysis")
    print("  ✓ Incomplete data detection")
    print("  ✓ Category fluctuation tracking")
    print("  ✓ Geometry anomaly detection")
    print("  ✓ Geographic pattern analysis")
    print("  ✓ AI-powered deep analysis")
    print("="*80)
    
    # Load data
    from release_validator import EnhancedReleaseValidator
    
    print("\nLoading aggregate data...")
    data = EnhancedReleaseValidator.load_all_data(base_dir)
    
    # Load all metrics
    release_date = '2025-09-24.0'
    row_counts = load_row_counts(base_dir, release_date)
    attr_stats = load_attribute_stats(base_dir, release_date)
    comparisons = load_attribute_comparisons(base_dir)
    
    # Run rule-based checker
    print("\nRunning rule-based checker...")
    validator = EnhancedReleaseValidator()
    rule_anomalies = validator.validate_release(data)
    
    # Run enhanced agent with all detections
    all_anomalies = run_enhanced_agent(
        api_key=api_key,
        current_stats=data.get('release_stats'),
        row_counts=row_counts,
        attr_stats=attr_stats,
        comparisons=comparisons,
        rule_anomalies=rule_anomalies
    )
    
    # Generate enhanced report
    report = generate_enhanced_report(all_anomalies)
    print(report)
    
    # Export
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'release': release_date,
        'anomalies': [a.to_dict() for a in all_anomalies],
        'summary': {
            'total': len(all_anomalies),
            'rule_based': sum(1 for a in all_anomalies if a.detected_by_rules),
            'ai_discovered': sum(1 for a in all_anomalies if not a.detected_by_rules),
            'with_geographic': sum(1 for a in all_anomalies if a.geographic_breakdown),
            'with_attribute': sum(1 for a in all_anomalies if a.attribute_breakdown),
            'by_severity': {
                'CRITICAL': sum(1 for a in all_anomalies if a.severity == 'CRITICAL'),
                'WARNING': sum(1 for a in all_anomalies if a.severity == 'WARNING'),
                'INFO': sum(1 for a in all_anomalies if a.severity == 'INFO')
            },
            'by_category': {}
        }
    }
    
    # Add category breakdown
    for anomaly in all_anomalies:
        cat = anomaly.category
        if cat not in output['summary']['by_category']:
            output['summary']['by_category'][cat] = 0
        output['summary']['by_category'][cat] += 1
    
    output_path = os.path.join(base_dir, 'enhanced_agent_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Generate comparison stats for OKR tracking
    print("\n" + "="*80)
    print("OKR METRICS")
    print("="*80)
    print(f"Total anomalies detected: {len(all_anomalies)}")
    print(f"Detection breakdown:")
    print(f"  • Automated rules: {output['summary']['rule_based']}")
    print(f"  • AI discovered: {output['summary']['ai_discovered']}")
    print(f"\nCoverage:")
    print(f"  • With detailed reasoning: {len(all_anomalies)} (100%)")
    print(f"  • With geographic context: {output['summary']['with_geographic']}")
    print(f"  • With attribute analysis: {output['summary']['with_attribute']}")
    print(f"\nSeverity distribution:")
    for severity, count in output['summary']['by_severity'].items():
        print(f"  • {severity}: {count}")


if __name__ == "__main__":
    main()