Overture Maps Release Anomaly Detection System
Overview

This system detects anomalies in Overture Maps monthly release data by comparing metrics across releases. It identifies data quality issues, unexpected churn, and systemic problems before they impact downstream users.

It produces both a JSON export and two dashboards:

anomaly_explorer — searchable table, filters, grouping

comparison_dashboard — runs detector + AI, shows side-by-side metrics, charts, and insights

Metrics/
                       │
                       ▼
            ┌─────────────────────┐
            │  anomaly_detector   │
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
   searchable table          charts and AI insights

What This System CAN Detect

Based on your input metrics (row_counts, changelog_stats, theme_column_summary_stats, release_to_release_comparisons), the system can detect:

1. Feature Count Fluctuation

Data source: metrics/{release}/row_counts/theme=*/type=*/*.csv

Sudden drops in feature counts

Unexpected spikes

Breakdowns by theme, type, class, and country

2. GERS ID Instability (Churn)

Data source: metrics/{release}/changelog_stats/*.csv

High add/remove rates between releases

Churn rate computation

Helps identify unstable data sources

3. Geometry Length Changes

Data source: metrics/.../row_counts/.../*.csv (total_geometry_length_km)

Large shifts in road or boundary lengths

Identifies geometry regressions

4. Category Distribution Shifts

Unexpected changes in subtype proportions

New categories appearing, old categories disappearing

5. Geographic Concentration

Changes concentrated in specific countries

Early signal for region-specific failures

6. Multi-Release Trends

Data source: theme_class_summary_stats/release_to_release_comparisons/*.csv

Gradual degradation across multiple consecutive releases

7. Incomplete Data Increase (New)

Type: INCOMPLETE_DATA_INCREASE
Detects rising null rates for required fields, such as:

Places missing names

Addresses missing streets

Divisions missing class or subtype

8. Confidence Score Coverage Drop (New)

Type: CONFIDENCE_SCORE_DROP
Detects decreases in confidence score availability for places, signaling a degradation in upstream scoring or metadata pipelines.

What This System CANNOT Detect
Geospatial Issues That Require Raw Geometry

Cannot detect:

POIs in water

Invalid or self-intersecting geometries

Out-of-bounds coordinates

These require geometry-level validation.

Content-Based Quality Issues

Cannot detect:

Spam names

Junk records

Semantic nonsense in attributes

These require raw attribute text or precomputed quality flags.

Attribute Coverage (Partial)

Coverage detection is possible only if metrics include non-null counts for each column.

Architecture
┌─────────────────────────────────────────────────────────────────┐
│                    OvertureAnomalyDetector                      │
├─────────────────────────────────────────────────────────────────┤
│  Loads data, runs detection modules, produces reports           │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   Feature Count        Churn Detection     Category Shift
     Detection                                  Detection
                              │
                              ▼
                     Anomaly Objects
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
       Text Report                             JSON Export

Configuration

Thresholds are configurable:

thresholds = AnomalyThresholds(
    count_drop_warning=-5.0,
    count_drop_critical=-15.0,
    count_spike_warning=20.0,
    count_spike_critical=50.0,
    churn_warning=10.0,
    churn_critical=25.0,
    min_feature_count=100,
    min_country_count=50,
)

Usage
Command Line
python anomaly_detector.py /path/to/data
python anomaly_detector.py /path/to/data --current 2025-09-24.0 --previous 2025-08-20.1
python anomaly_detector.py /path/to/data --json anomalies.json

Python API
thresholds = AnomalyThresholds(count_drop_critical=-10.0)
detector = OvertureAnomalyDetector(Path("/data"), thresholds)
anomalies = detector.run_full_analysis("2025-09-24.0", "2025-08-20.1")
detector.export_json(Path("anomalies.json"))

Files
File	Description
anomaly_detector.py	Main detection engine
coverage_detector.py	Attribute coverage and trend analysis
README.md	Documentation
