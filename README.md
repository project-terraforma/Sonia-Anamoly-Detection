# Overture Maps Release Anomaly Detection System

## Overview

This system detects anomalies in Overture Maps monthly release data by comparing metrics across releases. It's designed to catch data quality issues, unexpected churn, and systemic problems before they impact downstream users.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (required for AI-powered dashboards)

### Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your Anthropic API key:**

The `comparison_dashboard` requires an Anthropic API key to run the AI agent analysis. You can set this up in one of two ways:

**Option 1: Environment Variable (Recommended)**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

To make this permanent, add it to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
```bash
echo "export ANTHROPIC_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Option 2: Configuration File**
Create a `.env` file in the project root:
```bash
ANTHROPIC_API_KEY=your-api-key-here
```

**Getting an API Key:**
- Sign up at https://console.anthropic.com
- Navigate to API Keys section
- Generate a new key

**Note:** The basic `anomaly_detector` and `anomaly_explorer` do not require an API key. Only the `comparison_dashboard` with AI insights needs it.

## What This System CAN Detect

Based on your available data (row_counts, changelog_stats, theme_column_summary_stats, release_to_release_comparisons), the system can detect:

### 1. Feature Count Fluctuation
**Data source:** `metrics/{release}/row_counts/theme=*/type=*/*.csv`

- Sudden drops in feature counts (configurable thresholds: -5% warning, -15% critical)
- Unexpected spikes in feature counts (+20% warning, +50% critical)
- Breakdowns by:
  - Theme/Type (buildings, transportation, places, etc.)
  - Subtype/Class (residential, commercial, highway, etc.)
  - Country (for divisions, addresses)

**Example alert:**
```
[CRITICAL] feature_count_drop: transportation/segment (road/primary) in DE
  Feature count dropped significantly
  id_count: 245,000 → 198,000 (-19.2%)
```

### 2. GERS ID Instability (Churn)
**Data source:** `metrics/{release}/changelog_stats/*.csv`

- High add/remove rates between releases
- Tracks: added, removed, data_changed counts
- Calculates churn rate = (added + removed) / baseline

**Example alert:**
```
[WARNING] churn_spike: places/place
  High ID churn detected: 45,000 added, 38,000 removed
  churn_rate: 12.3%
```

### 3. Geometry Length Changes
**Data source:** `metrics/{release}/row_counts/theme=*/type=*/*.csv` (column: `total_geometry_length_km`)

- Detects significant changes in total road length, boundary length, etc.
- Useful for transportation and division boundaries

**Example alert:**
```
[CRITICAL] geometry_length_change: divisions/division_boundary in FR
  Total geometry length changed significantly
  total_geometry_length_km: 125,000 → 98,000 (-21.6%)
```

### 4. Category Distribution Shifts
**Data source:** Row counts with subtype/class columns

- Detects when proportions of subtypes change unexpectedly
- Catches new categories appearing or existing ones disappearing

**Example alert:**
```
[WARNING] category_distribution_shift: buildings/building
  Subtype 'commercial' proportion decreased by 8.2 percentage points
  subtype_proportion_pct: 15.3% → 7.1%
```

### 5. Geographic Concentration
**Data source:** Row counts with country breakdown

- Flags when changes are concentrated in specific countries
- Helps identify regional data source issues

**Example alert:**
```
Geographic concentration detected: 78% of all addresses/address changes are in US
```

### 6. Multi-Release Trends
**Data source:** `theme_class_summary_stats/release_to_release_comparisons/*.csv`

- Detects gradual degradation over multiple releases
- Catches slow-moving issues that single-release comparisons miss

### 7. Data Quality Degradation

| Type | What It Detects |
|------|----------------|
| INCOMPLETE_DATA_INCREASE | When null rates increase for required fields (e.g., places missing names, addresses missing street) |
| CONFIDENCE_SCORE_DROP | When confidence score coverage drops (places only) |

---


## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OvertureAnomalyDetector                      │
├─────────────────────────────────────────────────────────────────┤
│  Main orchestrator class                                        │
│  - Loads data from metrics/                                     │
│  - Runs all detection modules                                   │
│  - Generates reports (text, JSON)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Feature Count    │ │ Churn Detection  │ │ Category Shift   │
│ Detection        │ │                  │ │ Detection        │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ - Count drops    │ │ - Add/remove     │ │ - Subtype props  │
│ - Count spikes   │ │   rates          │ │ - New categories │
│ - By geo/class   │ │ - ID instability │ │ - Missing cats   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Anomaly Objects  │
                    ├──────────────────┤
                    │ - Type           │
                    │ - Severity       │
                    │ - Theme/Type     │
                    │ - Geography      │
                    │ - Metrics        │
                    └──────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────┐                    ┌──────────────────┐
│ Text Report      │                    │ JSON Export      │
└──────────────────┘                    └──────────────────┘
```

---

## Dashboards

The system includes interactive dashboards for exploring and analyzing detected anomalies:

```
Metrics/
                       │
                       ▼
            ┌─────────────────────┐
            │  anomaly_detector   │  ← Detects threshold violations
            │                     │
            │  "981 anomalies"    │
            └──────────┬──────────┘
                       │
                       ▼
              all_anomalies.json
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│  anomaly_explorer   │   │ comparison_dashboard │
│                     │   │                      │
│  Filter, group,     │   │  Runs detector +     │
│  analyze the 981    │   │  AI agent, shows     │
│                     │   │  side-by-side        │
└─────────────────────┘   └──────────────────────┘
          │                         │
          ▼                         ▼
   HTML report with          Full dashboard with
   searchable table          charts & AI insights
```

### Dashboard Types

**anomaly_explorer**
- Filter and group detected anomalies
- Searchable table interface
- Export to HTML reports
- Interactive analysis of all detected issues

**comparison_dashboard**
- Runs anomaly detector with AI agent analysis
- Side-by-side comparison views
- Charts and visualizations
- AI-generated insights and recommendations

---

## Configuration

### Thresholds

All thresholds are configurable via `AnomalyThresholds`:

```python
thresholds = AnomalyThresholds(
    # Feature count thresholds
    count_drop_warning=-5.0,      # % drop for warning
    count_drop_critical=-15.0,    # % drop for critical
    count_spike_warning=20.0,     # % increase for warning
    count_spike_critical=50.0,    # % increase for critical
    
    # Churn thresholds
    churn_warning=10.0,           # % churn for warning
    churn_critical=25.0,          # % churn for critical
    
    # Minimum samples (avoid noise)
    min_feature_count=100,        # Ignore groups smaller than this
    min_country_count=50,         # Ignore countries with fewer features
)
```

---

## Usage

### Command Line

```bash
# Basic analysis (uses two most recent releases)
python anomaly_detector.py /path/to/data

# Specific releases
python anomaly_detector.py /path/to/data --current 2025-09-24.0 --previous 2025-08-20.1

# With custom thresholds and output
python anomaly_detector.py /path/to/data \
    --count-drop-critical -10.0 \
    --output report.txt \
    --json anomalies.json
```

### Python API

```python
from anomaly_detector import OvertureAnomalyDetector, AnomalyThresholds
from pathlib import Path

# Initialize
thresholds = AnomalyThresholds(count_drop_critical=-10.0)
detector = OvertureAnomalyDetector(Path("/data"), thresholds)

# Run analysis
anomalies = detector.run_full_analysis("2025-09-24.0", "2025-08-20.1")

# Get report
print(detector.generate_report())

# Export for dashboards
detector.export_json(Path("anomalies.json"))

# Filter critical only
critical = [a for a in anomalies if a.severity.value == "critical"]
```

---

## Sample Output

```
======================================================================
OVERTURE MAPS RELEASE ANOMALY REPORT
======================================================================
Generated: 2025-09-25T10:30:00
Total anomalies detected: 7

CRITICAL: 2
WARNING: 5

----------------------------------------------------------------------
CRITICAL ISSUES
----------------------------------------------------------------------

[CRITICAL] feature_count_drop: addresses/address in BR
  Feature count dropped significantly
  id_count: 12,450,000 → 9,876,000 (-20.7%)

[CRITICAL] churn_spike: places/place
  High ID churn detected: 892,000 added, 756,000 removed
  churn_rate: 28.4%

----------------------------------------------------------------------
WARNINGS
----------------------------------------------------------------------

[WARNING] geometry_length_change: transportation/segment (road/motorway)
  Total geometry length changed significantly
  total_geometry_length_km: 456,000 → 423,000 (-7.2%)

[WARNING] category_distribution_shift: buildings/building
  Subtype 'industrial' proportion decreased by 5.8 percentage points
```