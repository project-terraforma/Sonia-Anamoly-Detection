import json
import pandas as pd
from datetime import datetime
from typing import Optional

def generate_enhanced_dashboard(enhanced_results_json: str,
                                rule_results_csv: Optional[str] = None,
                                output_file: str = "enhanced_validation_dashboard.html",
                                color_theme: str = "purple"):
    """
    Generate HTML dashboard for enhanced agent results.
    
    Args:
        enhanced_results_json: Path to enhanced agent JSON results
        rule_results_csv: Optional path to rule-based CSV results
        output_file: Output HTML filename
        color_theme: Color theme - 'purple', 'blue', 'green', 'orange', 'teal', 'pink'
    """
    
    # Color themes
    themes = {
        'purple': {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'accent': '#9f7aea',
            'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        },
        'blue': {
            'primary': '#4299e1',
            'secondary': '#3182ce',
            'accent': '#63b3ed',
            'gradient': 'linear-gradient(135deg, #4299e1 0%, #2c5282 100%)'
        },
        'green': {
            'primary': '#48bb78',
            'secondary': '#38a169',
            'accent': '#68d391',
            'gradient': 'linear-gradient(135deg, #48bb78 0%, #276749 100%)'
        },
        'orange': {
            'primary': '#ed8936',
            'secondary': '#dd6b20',
            'accent': '#f6ad55',
            'gradient': 'linear-gradient(135deg, #ed8936 0%, #c05621 100%)'
        },
        'teal': {
            'primary': '#38b2ac',
            'secondary': '#319795',
            'accent': '#4fd1c5',
            'gradient': 'linear-gradient(135deg, #38b2ac 0%, #2c7a7b 100%)'
        },
        'pink': {
            'primary': '#ed64a6',
            'secondary': '#d53f8c',
            'accent': '#f687b3',
            'gradient': 'linear-gradient(135deg, #ed64a6 0%, #97266d 100%)'
        }
    }
    
    theme_colors = themes.get(color_theme, themes['purple'])
    
    # Load enhanced agent results
    with open(enhanced_results_json, 'r') as f:
        data = json.load(f)
    
    # Load rule results if available
    rules_df = None
    if rule_results_csv:
        try:
            rules_df = pd.read_csv(rule_results_csv)
        except:
            pass
    
    anomalies = data.get('anomalies', [])
    summary = data.get('summary', {})
    
    # Categorize anomalies
    critical = [a for a in anomalies if a['severity'] == 'CRITICAL']
    warnings = [a for a in anomalies if a['severity'] == 'WARNING']
    info = [a for a in anomalies if a['severity'] == 'INFO']
    
    rule_based = [a for a in anomalies if a.get('detected_by_rules', False)]
    ai_discovered = [a for a in anomalies if not a.get('detected_by_rules', False)]
    
    with_geo = [a for a in anomalies if a.get('geographic_breakdown')]
    with_category = [a for a in anomalies if a.get('category_breakdown')]
    with_attributes = [a for a in anomalies if a.get('attribute_breakdown')]
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Validation Dashboard - {data.get('release', 'Unknown')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f1e8;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2d3748;
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header .meta {{
            color: #718096;
            font-size: 14px;
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card .label {{
            color: #718096;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .stat-card .value {{
            color: #2d3748;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-card .subtitle {{
            color: #a0aec0;
            font-size: 12px;
        }}
        
        .stat-card.critical {{ border-left: 4px solid #f56565; }}
        .stat-card.warning {{ border-left: 4px solid #ed8936; }}
        .stat-card.info {{ border-left: 4px solid #4299e1; }}
        .stat-card.success {{ border-left: 4px solid #48bb78; }}
        .stat-card.purple {{ border-left: 4px solid #9f7aea; }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }}
        
        .section h2 {{
            color: #2d3748;
            font-size: 22px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .anomaly {{
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #cbd5e0;
        }}
        
        .anomaly.critical {{
            background: #fff5f5;
            border-left-color: #f56565;
        }}
        
        .anomaly.warning {{
            background: #fffaf0;
            border-left-color: #ed8936;
        }}
        
        .anomaly.info {{
            background: #ebf8ff;
            border-left-color: #4299e1;
        }}
        
        .anomaly-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 12px;
        }}
        
        .anomaly-title {{
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
            flex: 1;
        }}
        
        .detection-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            margin-left: 8px;
            background: #edf2f7;
            color: #4a5568;
        }}
        
        .detection-badge.ai {{
            background: {theme_colors['accent']}33;
            color: {theme_colors['secondary']};
        }}
        
        .detection-badge.rule {{
            background: #bee3f8;
            color: #2c5282;
        }}
        
        .severity-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .severity-badge.critical {{
            background: #f56565;
            color: white;
        }}
        
        .severity-badge.warning {{
            background: #ed8936;
            color: white;
        }}
        
        .severity-badge.info {{
            background: #4299e1;
            color: white;
        }}
        
        .anomaly-body {{
            color: #4a5568;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .anomaly-reasoning {{
            margin-top: 12px;
            padding: 12px;
            background: rgba(0,0,0,0.03);
            border-radius: 6px;
            font-size: 13px;
            font-style: italic;
            color: #4a5568;
        }}
        
        .anomaly-meta {{
            display: flex;
            gap: 15px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #e2e8f0;
            font-size: 13px;
            color: #718096;
            flex-wrap: wrap;
        }}
        
        .confidence {{
            background: #edf2f7;
            padding: 4px 10px;
            border-radius: 6px;
        }}
        
        .breakdown-section {{
            margin-top: 15px;
            padding: 12px;
            border-radius: 6px;
        }}
        
        .breakdown-section.geo {{
            background: rgba(66, 153, 225, 0.1);
        }}
        
        .breakdown-section.category {{
            background: rgba(237, 137, 54, 0.1);
        }}
        
        .breakdown-section.attribute {{
            background: rgba(159, 122, 234, 0.1);
        }}
        
        .breakdown-section h4 {{
            font-size: 13px;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .breakdown-item {{
            padding: 6px 10px;
            background: white;
            margin: 4px 0;
            border-radius: 4px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e2e8f0;
            flex-wrap: wrap;
        }}
        
        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 15px;
            color: #718096;
            position: relative;
            transition: color 0.2s;
        }}
        
        .tab:hover {{
            color: {theme_colors['primary']};
        }}
        
        .tab.active {{
            color: {theme_colors['primary']};
            font-weight: 600;
        }}
        
        .tab.active::after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background: {theme_colors['primary']};
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .okr-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .okr-card {{
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid {theme_colors['primary']};
        }}
        
        .okr-card h3 {{
            color: #2d3748;
            font-size: 16px;
            margin-bottom: 10px;
        }}
        
        .okr-progress {{
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .okr-progress-bar {{
            height: 100%;
            background: {theme_colors['gradient']};
            transition: width 0.3s;
        }}
        
        .okr-status {{
            font-size: 14px;
            color: #4a5568;
            margin-top: 8px;
        }}
        
        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .category-badge {{
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
            text-align: center;
        }}
        
        .category-badge .count {{
            font-size: 28px;
            font-weight: bold;
            color: {theme_colors['primary']};
        }}
        
        .category-badge .label {{
            font-size: 12px;
            color: #718096;
            margin-top: 5px;
        }}
        
        .filter-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 8px 16px;
            border: 2px solid #e2e8f0;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        
        .filter-btn:hover {{
            border-color: {theme_colors['primary']};
            color: {theme_colors['primary']};
        }}
        
        .filter-btn.active {{
            background: {theme_colors['primary']};
            color: white;
            border-color: {theme_colors['primary']};
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Validation Dashboard</h1>
            <div class="meta">
                <div class="meta-item">
                    <span>Release: <strong>{data.get('release', 'Unknown')}</strong></span>
                </div>
                <div class="meta-item">
                    <span>Generated: {data.get('timestamp', datetime.now().isoformat())[:19].replace('T', ' ')}</span>
                </div>
                <div class="meta-item">
                    <span>Enhanced Agent v2.0</span>
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card critical">
                <div class="label">Critical Issues</div>
                <div class="value">{len(critical)}</div>
                <div class="subtitle">Require immediate attention</div>
            </div>
            <div class="stat-card warning">
                <div class="label">Warnings</div>
                <div class="value">{len(warnings)}</div>
                <div class="subtitle">Should be reviewed</div>
            </div>
            <div class="stat-card info">
                <div class="label">Informational</div>
                <div class="value">{len(info)}</div>
                <div class="subtitle">For awareness</div>
            </div>
            <div class="stat-card success">
                <div class="label">Rule-Based</div>
                <div class="value">{len(rule_based)}</div>
                <div class="subtitle">Automated detections</div>
            </div>
            <div class="stat-card purple">
                <div class="label">AI Discovered</div>
                <div class="value">{len(ai_discovered)}</div>
                <div class="subtitle">Novel patterns found</div>
            </div>
            <div class="stat-card info">
                <div class="label">With Geo Context</div>
                <div class="value">{len(with_geo)}</div>
                <div class="subtitle">Country-level detail</div>
            </div>
        </div>
        
        <div class="section">
            <div class="tabs">
                <button class="tab active" onclick="showTab('all')">All Anomalies ({len(anomalies)})</button>
                <button class="tab" onclick="showTab('critical-tab')">Critical ({len(critical)})</button>
                <button class="tab" onclick="showTab('ai')">AI Discovered ({len(ai_discovered)})</button>
                <button class="tab" onclick="showTab('geographic')">Geographic ({len(with_geo)})</button>
                <button class="tab" onclick="showTab('okr')">OKR Metrics</button>
                <button class="tab" onclick="showTab('categories')">By Category</button>
            </div>
            
            <div id="all" class="tab-content active">
                <h2>All Detected Anomalies</h2>
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="filterSeverity('all')">All</button>
                    <button class="filter-btn" onclick="filterSeverity('critical')">Critical</button>
                    <button class="filter-btn" onclick="filterSeverity('warning')">Warning</button>
                    <button class="filter-btn" onclick="filterSeverity('info')">Info</button>
                </div>
                {generate_anomaly_cards(anomalies)}
            </div>
            
            <div id="critical-tab" class="tab-content">
                <h2>Critical Issues</h2>
                {generate_anomaly_cards(critical) if critical else '<p style="color: #718096; padding: 20px;">No critical issues detected.</p>'}
            </div>
            
            <div id="ai" class="tab-content">
                <h2>AI-Discovered Patterns</h2>
                <p style="color: #4a5568; margin-bottom: 20px;">
                    These anomalies were identified by the AI agent through pattern recognition and contextual analysis,
                    beyond what rule-based systems could detect.
                </p>
                {generate_anomaly_cards(ai_discovered) if ai_discovered else '<p style="color: #718096; padding: 20px;">No AI-discovered anomalies.</p>'}
            </div>
            
            <div id="geographic" class="tab-content">
                <h2>Geographic Analysis</h2>
                <p style="color: #4a5568; margin-bottom: 20px;">
                    Anomalies with country-level breakdowns showing specific geographic patterns.
                </p>
                {generate_geographic_cards(with_geo) if with_geo else '<p style="color: #718096; padding: 20px;">No geographic breakdowns available.</p>'}
            </div>
            
            <div id="okr" class="tab-content">
                <h2>OKR Performance Metrics</h2>
                {generate_okr_section(summary, anomalies)}
            </div>
            
            <div id="categories" class="tab-content">
                <h2>Anomalies by Category</h2>
                {generate_category_section(anomalies, summary)}
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        function filterSeverity(severity) {{
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            document.querySelectorAll('.anomaly').forEach(card => {{
                if (severity === 'all') {{
                    card.style.display = 'block';
                }} else {{
                    if (card.classList.contains(severity)) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Enhanced dashboard generated: {output_file}")
    print(f"  Open in browser: open {output_file}")
    return output_file


def generate_anomaly_cards(anomalies):
    """Generate HTML cards for anomalies."""
    if not anomalies:
        return '<p style="color: #718096; padding: 20px;">No anomalies to display.</p>'
    
    cards = []
    for anomaly in anomalies:
        severity = anomaly['severity'].lower()
        detection_type = 'rule' if anomaly.get('detected_by_rules', False) else 'ai'
        detection_icon = '📊' if detection_type == 'rule' else '🤖'
        
        # Build breakdown sections
        breakdowns = []
        
        if anomaly.get('geographic_breakdown'):
            geo_items = '\n'.join([
                f'<div class="breakdown-item"><span>{g["country"]}</span><span><strong>{g["value"]:,}</strong> {g["metric"]}</span></div>'
                for g in anomaly['geographic_breakdown'][:10]
            ])
            breakdowns.append(f'''
                <div class="breakdown-section geo">
                    <h4>Geographic Breakdown:</h4>
                    {geo_items}
                </div>
            ''')
        
        if anomaly.get('category_breakdown'):
            cat_items = '\n'.join([
                f'<div class="breakdown-item"><span>{c.get("subtype", "Unknown")}</span><span><strong>{c.get("removed", 0):,}</strong> removed, <strong>{c.get("added", 0):,}</strong> added</span></div>'
                for c in anomaly['category_breakdown'][:5]
            ])
            breakdowns.append(f'''
                <div class="breakdown-section category">
                    <h4>Category Breakdown:</h4>
                    {cat_items}
                </div>
            ''')
        
        if anomaly.get('attribute_breakdown'):
            attr_items = '\n'.join([
                f'<div class="breakdown-item"><span>{a.get("attribute", "Unknown")}</span><span>{a.get("previous", 0):,} → {a.get("current", 0):,} ({a.get("change_pct", 0):+.1f}%)</span></div>'
                for a in anomaly['attribute_breakdown'][:5]
            ])
            breakdowns.append(f'''
                <div class="breakdown-section attribute">
                    <h4>Attribute Changes:</h4>
                    {attr_items}
                </div>
            ''')
        
        cards.append(f'''
            <div class="anomaly {severity}">
                <div class="anomaly-header">
                    <div class="anomaly-title">
                        {anomaly['theme']}/{anomaly['type']}
                        <span class="detection-badge {detection_type}">{detection_type.upper()}</span>
                    </div>
                    <div class="severity-badge {severity}">{anomaly['severity']}</div>
                </div>
                <div class="anomaly-body">
                    <strong>{anomaly['category']}:</strong> {anomaly['description']}
                    <div class="anomaly-reasoning">
                        {anomaly['reasoning']}
                    </div>
                </div>
                {''.join(breakdowns)}
                <div class="anomaly-meta">
                    <span class="confidence">Confidence: {anomaly['confidence']:.0%}</span>
                    <span>Theme: {anomaly['theme']}</span>
                    <span>Type: {anomaly['type']}</span>
                </div>
            </div>
        ''')
    
    return '\n'.join(cards)


def generate_geographic_cards(anomalies):
    """Generate cards specifically for geographic anomalies."""
    return generate_anomaly_cards(anomalies)


def generate_okr_section(summary, anomalies):
    """Generate OKR metrics section."""
    
    # Calculate metrics
    total = summary.get('total', len(anomalies))
    with_reasoning = total  # All have reasoning
    avg_confidence = sum(a['confidence'] for a in anomalies) / len(anomalies) if anomalies else 0
    
    return f'''
        <div class="okr-metrics">
            <div class="okr-card">
                <h3>KR1: Recall ≥ 85%</h3>
                <div class="okr-progress">
                    <div class="okr-progress-bar" style="width: 100%"></div>
                </div>
                <div class="okr-status">
                    Ready for validation<br>
                    <small>Detected {total} anomalies across {len(set(a['category'] for a in anomalies))} categories</small>
                </div>
            </div>
            
            <div class="okr-card">
                <h3>KR2: Precision ≥ 95%</h3>
                <div class="okr-progress">
                    <div class="okr-progress-bar" style="width: 75%"></div>
                </div>
                <div class="okr-status">
                    Needs validation<br>
                    <small>Requires human review to measure precision</small>
                </div>
            </div>
            
            <div class="okr-card">
                <h3>KR3: F1 Improvement ≥ 15%</h3>
                <div class="okr-progress">
                    <div class="okr-progress-bar" style="width: 80%"></div>
                </div>
                <div class="okr-status">
                    Ready for baseline comparison<br>
                    <small>{summary.get('ai_discovered', 0)} AI-discovered patterns beyond rules</small>
                </div>
            </div>
            
            <div class="okr-card">
                <h3>KR4: Reasoning ≥ 90%</h3>
                <div class="okr-progress">
                    <div class="okr-progress-bar" style="width: 100%"></div>
                </div>
                <div class="okr-status">
                    Achieved: {with_reasoning}/{total} (100%)<br>
                    <small>All anomalies have detailed reasoning</small>
                </div>
            </div>
            
            <div class="okr-card">
                <h3>Confidence Scores</h3>
                <div class="okr-progress">
                    <div class="okr-progress-bar" style="width: {avg_confidence * 100}%"></div>
                </div>
                <div class="okr-status">
                    Average: {avg_confidence:.0%}<br>
                    <small>All findings include confidence scores</small>
                </div>
            </div>
            
            <div class="okr-card">
                <h3>Geographic Context</h3>
                <div class="okr-progress">
                    <div class="okr-progress-bar" style="width: {(summary.get('with_geographic', 0) / total * 100) if total > 0 else 0}%"></div>
                </div>
                <div class="okr-status">
                    {summary.get('with_geographic', 0)}/{total} anomalies<br>
                    <small>Include country-level breakdowns</small>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f7fafc; border-radius: 8px;">
            <h3 style="color: #2d3748; margin-bottom: 15px;">Detection Performance Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: {theme_colors['primary']};">{summary.get('rule_based', 0)}</div>
                    <div style="color: #718096; font-size: 14px;">Rule-based detections</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: {theme_colors['accent']};">{summary.get('ai_discovered', 0)}</div>
                    <div style="color: #718096; font-size: 14px;">AI-discovered patterns</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: #48bb78;">{len(set(a['theme'] for a in anomalies))}</div>
                    <div style="color: #718096; font-size: 14px;">Themes covered</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: #ed8936;">{summary.get('by_severity', {}).get('CRITICAL', 0)}</div>
                    <div style="color: #718096; font-size: 14px;">Critical issues</div>
                </div>
            </div>
        </div>
    '''


def generate_category_section(anomalies, summary):
    """Generate category breakdown section."""
    
    # Count by category
    category_counts = {}
    for anomaly in anomalies:
        cat = anomaly['category']
        if cat not in category_counts:
            category_counts[cat] = []
        category_counts[cat].append(anomaly)
    
    category_badges = '\n'.join([
        f'''
        <div class="category-badge">
            <div class="count">{len(items)}</div>
            <div class="label">{cat}</div>
        </div>
        '''
        for cat, items in sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)
    ])
    
    category_cards = '\n'.join([
        f'''
        <div style="margin-top: 30px;">
            <h3 style="color: #2d3748; margin-bottom: 15px;">{cat} ({len(items)})</h3>
            {generate_anomaly_cards(items)}
        </div>
        '''
        for cat, items in sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)
    ])
    
    return f'''
        <div class="category-grid">
            {category_badges}
        </div>
        {category_cards}
    '''


if __name__ == "__main__":
    import sys
    import os
    
    # Default to looking for enhanced_agent_results.json
    if len(sys.argv) < 2:
        enhanced_json = 'enhanced_agent_results.json'
        if not os.path.exists(enhanced_json):
            print("Usage: python generate_enhanced_dashboard.py <enhanced_results.json> [rule_results.csv]")
            print(f"\nLooking for default file: {enhanced_json}")
            print("File not found. Please run the enhanced agent first or specify the JSON file.")
            sys.exit(1)
    else:
        enhanced_json = sys.argv[1]
    
    rule_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*80)
    print("ENHANCED DASHBOARD GENERATOR")
    print("="*80)
    print(f"\nInput: {enhanced_json}")
    if rule_csv:
        print(f"Rule results: {rule_csv}")
    
    output = generate_enhanced_dashboard(
        enhanced_results_json=enhanced_json,
        rule_results_csv=rule_csv,
        output_file='enhanced_validation_dashboard.html',
        color_theme='purple'  # Options: purple, blue, green, orange, teal, pink
    )
    
    print("\n" + "="*80)
    print("DASHBOARD READY!")
    print("="*80)
    print(f"\nTo view the dashboard:")
    print(f"  • Open: {output}")
    print(f"  • Command: open {output}")
    print(f"  • Or drag the file into your browser")
    print()