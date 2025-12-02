# Overture Maps Release Anomaly Detection System

An AI-powered system for detecting anomalies in Overture Maps monthly release data. It compares metrics across releases to catch data quality issues, unexpected churn, and systemic problems before they impact downstream users.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key (required for AI insights)
export ANTHROPIC_API_KEY='your-api-key-here'

# 3. Run detector
python anomaly_detector.py Metrics/ --json anomalies.json

# 4. View in dashboard
python comparison_dashboard.py --input anomalies.json -o dashboard.html
open dashboard.html
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Anthropic API key (required for AI-powered dashboards)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up API Key

The `comparison_dashboard` requires an Anthropic API key for AI agent analysis.

**Option 1: Environment Variable (Recommended)**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'

# To make permanent, add to your shell profile:
echo "export ANTHROPIC_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Option 2: Configuration File**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

**Getting an API Key:**
1. Sign up at https://console.anthropic.com
2. Navigate to API Keys section
3. Generate a new key

> **Note:** The basic `anomaly_detector` and `anomaly_explorer` do not require an API key. Only the `comparison_dashboard` with AI insights needs it.

---

## System Architecture

```
Metrics/
    │
    ▼
┌─────────────────────┐
│  anomaly_detector   │  ← Rule-based detection (fast, deterministic)
│                     │
│  Detects threshold  │
│  violations         │
└──────────┬──────────┘
           │
           ▼
     anomalies.json     ← Single source of truth
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────────┐  ┌─────────────────────┐
│  anomaly    │  │ comparison_dashboard │
│  explorer   │  │                      │
│             │  │  + AI Agent layer    │
│  Filter,    │  │  + Pattern synthesis │
│  group,     │  │  + Root cause ID     │
│  analyze    │  │  + Recommendations   │
└─────────────┘  └─────────────────────┘
       │                    │
       ▼                    ▼
  HTML report         Full dashboard
  with search         with AI insights
```

### Three Tools

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `anomaly_detector.py` | Finds anomalies (rule-based) | Metrics folder | JSON with all anomalies |
| `anomaly_explorer.py` | Filter, group, analyze | JSON from detector | HTML report, Excel |
| `comparison_dashboard.py` | AI synthesis + visualization | JSON from detector | Interactive dashboard |

---

## What This System Detects

### 1. Feature Count Fluctuation
**Data source:** `metrics/{release}/row_counts/theme=*/type=*/*.csv`

- Sudden drops in feature counts (default: -10% warning, -20% critical)
- Unexpected spikes in feature counts (+20% warning, +50% critical)
- Breakdowns by theme/type, subtype/class, and country

```
[CRITICAL] feature_count_drop: transportation/segment (road/primary) in DE
  Feature count dropped significantly
  id_count: 245,000 → 198,000 (-19.2%)
```

### 2. GERS ID Instability (Churn)
**Data source:** `metrics/{release}/changelog_stats/*.csv`

- High add/remove rates between releases
- Calculates churn rate = (added + removed) / baseline

```
[WARNING] churn_spike: places/place
  High ID churn detected: 45,000 added, 38,000 removed
  churn_rate: 12.3%
```

### 3. Geometry Length Changes
**Data source:** `metrics/{release}/row_counts/` (column: `total_geometry_length_km`)

- Detects significant changes in total road length, boundary length, etc.
- Useful for transportation and division boundaries

```
[CRITICAL] geometry_length_change: divisions/division_boundary in FR
  Total geometry length changed significantly
  total_geometry_length_km: 125,000 → 98,000 (-21.6%)
```

### 4. Category Distribution Shifts
**Data source:** Row counts with subtype/class columns

- Detects when proportions of subtypes change unexpectedly
- Catches new categories appearing or existing ones disappearing

```
[WARNING] category_distribution_shift: buildings/building
  Subtype 'commercial' proportion decreased by 8.2 percentage points
  subtype_proportion_pct: 15.3% → 7.1%
```

### 5. Geographic Concentration
**Data source:** Row counts with country breakdown

- Flags when changes are concentrated in specific countries
- Helps identify regional data source issues

```
Geographic concentration detected: 78% of all addresses/address changes are in US
```

### 6. Incomplete Data Detection
**Data source:** `theme_column_summary_stats/*.csv`

- Detects when null rates increase for required fields
- Monitors: names, addresses, phones, categories, etc.
- Configurable thresholds (default: 2% warning, 5% critical)

```
[WARNING] incomplete_data_increase: places/place
  Missing data increased for 'addresses': 12.3% → 15.8% null
```

### 7. Confidence Score Monitoring
**Data source:** `theme_column_summary_stats/*.csv` (places only)

- Tracks confidence score coverage
- Alerts when coverage drops

```
[CRITICAL] confidence_score_drop: places/place
  Confidence score coverage dropped: 98.2% → 91.5%
```

---

## Monitored Fields for Incomplete Data

| Theme/Type | Required Fields |
|------------|-----------------|
| places/place | names, categories, addresses, phones, websites |
| addresses/address | street, postcode, country, number |
| buildings/building | names, class, height |
| buildings/building_part | class, height |
| transportation/segment | names, class, subtype |
| divisions/division | names, subtype |
| divisions/division_area | names, subtype, class |
| base/water | names, subtype, class |
| base/land | subtype, class |
| base/land_use | subtype, class |
| base/land_cover | subtype |
| base/infrastructure | subtype, class |

---

## Usage

### Command Line

```bash
# Basic analysis (uses two most recent releases)
python anomaly_detector.py Metrics/ --json anomalies.json

# Specific releases
python anomaly_detector.py Metrics/ \
    --current 2025-09-24.0 \
    --previous 2025-08-20.1 \
    --json anomalies.json

# With custom thresholds
python anomaly_detector.py Metrics/ \
    --count-drop-warning -5.0 \
    --count-drop-critical -15.0 \
    --count-spike-warning 20.0 \
    --count-spike-critical 50.0 \
    --incomplete-data-warning 2.0 \
    --incomplete-data-critical 5.0 \
    --json anomalies.json
```

### Dashboard Usage

**Step 1: Generate anomalies**
```bash
python anomaly_detector.py Metrics/ --current 2025-09-24.0 --previous 2025-08-20.1 --json anomalies.json
```

**Step 2: View in dashboard (choose one)**

Option A: **Comparison Dashboard** (with AI insights)
```bash
python comparison_dashboard.py --input anomalies.json -o dashboard.html
open dashboard.html
```

Option B: **Anomaly Explorer** (filtering/grouping)
```bash
python anomaly_explorer.py anomalies.json --export-html report.html
open report.html
```

Option C: **Export to Excel** (for detailed analysis)
```bash
python anomaly_explorer.py anomalies.json --export-excel analysis.xlsx
```

### Python API

```python
from anomaly_detector import OvertureAnomalyDetector, AnomalyThresholds
from pathlib import Path

# Initialize with custom thresholds
thresholds = AnomalyThresholds(
    count_drop_critical=-15.0,
    count_spike_critical=50.0,
    incomplete_data_warning=2.0,
    incomplete_data_critical=5.0
)
detector = OvertureAnomalyDetector(Path("Metrics/"), thresholds)

# Run analysis
anomalies = detector.run_full_analysis("2025-09-24.0", "2025-08-20.1")

# Get report
print(detector.generate_report())

# Export for dashboards
detector.export_json(Path("anomalies.json"))

# Filter by severity
critical = [a for a in anomalies if a.severity.value == "critical"]
print(f"Found {len(critical)} critical issues")

# Filter by type
incomplete = [a for a in anomalies if a.anomaly_type.value == "incomplete_data_increase"]
```

---

## Configuration

### All Threshold Options

```python
thresholds = AnomalyThresholds(
    # Feature count thresholds (percentage)
    count_drop_warning=-10.0,       # % drop for warning
    count_drop_critical=-20.0,      # % drop for critical
    count_spike_warning=20.0,       # % increase for warning
    count_spike_critical=50.0,      # % increase for critical
    
    # Churn thresholds (percentage)
    churn_warning=10.0,             # % churn for warning
    churn_critical=25.0,            # % churn for critical
    
    # Geometry length thresholds (percentage)
    length_drop_warning=-10.0,
    length_drop_critical=-20.0,
    
    # Category distribution shift (percentage points)
    category_shift_warning=5.0,
    category_shift_critical=15.0,
    
    # Incomplete data thresholds (percentage point increase in nulls)
    incomplete_data_warning=2.0,    # 2pp increase = warning
    incomplete_data_critical=5.0,   # 5pp increase = critical
    
    # Confidence score thresholds (percentage point drop)
    confidence_drop_warning=-2.0,
    confidence_drop_critical=-5.0,
    
    # Minimum samples (avoid noise from small datasets)
    min_feature_count=100,          # Ignore groups smaller than this
    min_country_count=50,           # Ignore countries with fewer features
)
```

---

## Sample Output

### Text Report

```
======================================================================
OVERTURE MAPS RELEASE ANOMALY REPORT
======================================================================
Generated: 2025-09-25T10:30:00
Total anomalies detected: 983

CRITICAL: 978
WARNING: 5

By Anomaly Type:
  feature_count_spike: 970
  geometry_length_change: 8
  incomplete_data_increase: 3
  confidence_score_drop: 1
  churn_spike: 1

----------------------------------------------------------------------
CRITICAL ISSUES
----------------------------------------------------------------------

[CRITICAL] feature_count_spike: divisions/division_area (neighborhood/land) in US
  Feature count spiked unexpectedly
  id_count: 20,688 → 41,862 (+102.3%)

[CRITICAL] feature_count_spike: transportation/segment (rail/standard_gauge)
  Feature count spiked unexpectedly
  id_count: 1,429,962 → 2,865,484 (+100.4%)

[CRITICAL] incomplete_data_increase: places/place
  Missing data increased for 'addresses': 12.3% → 17.8% null
  addresses_null_rate: 12.30 → 17.80 (+5.5%)
```

### AI Agent Summary (from comparison_dashboard)

```
## Executive Summary
The Overture Maps dataset exhibits severe data integrity issues with 978 
critical anomalies. A systematic duplication pattern has nearly doubled 
feature counts across 47+ countries, indicating a major data pipeline failure.

## Key Patterns Identified
1. Systematic ~100% duplication in division_area (47+ countries)
2. Rail infrastructure duplication (all subtypes)
3. Road network duplication (footway, cycleway, etc.)
4. Water geometry reduction (isolated)

## Root Cause Hypotheses
1. Data pipeline duplication bug - source data processed twice
2. Merge/conflation error - multiple data sources incorrectly combined

## Recommended Actions
1. IMMEDIATE: Halt distribution of release 2025-09-24.0
2. URGENT: Rollback to 2025-08-20.1
3. INVESTIGATE: Audit data ingestion pipeline
4. IMPLEMENT: Add automated duplicate detection

## Risk Assessment: CRITICAL
```

---

## Data Directory Structure

The system expects the following directory structure:

```
Metrics/
├── metrics/
│   ├── 2025-08-20.1/
│   │   ├── row_counts/
│   │   │   ├── theme=places/type=place/*.csv
│   │   │   ├── theme=buildings/type=building/*.csv
│   │   │   └── ...
│   │   └── changelog_stats/*.csv
│   ├── 2025-09-24.0/
│   │   └── ...
│   └── ...
├── theme_column_summary_stats/
│   ├── 2025-08-20.1.theme=places.type=place.csv
│   ├── 2025-09-24.0.theme=places.type=place.csv
│   └── ...
└── theme_class_summary_stats/
    └── release_to_release_comparisons/
        └── theme=places.type=place.csv
```

---

## License

Apache-2.0