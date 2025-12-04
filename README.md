# Overture Maps Release Anomaly Detection System

Detects anomalies in Overture Maps monthly releases by comparing metrics across releases to catch data quality issues before they impact downstream users.

## Quick Start

```bash
pip install pandas anthropic
export ANTHROPIC_API_KEY='your-api-key-here'

python anomaly_detector.py Metrics/ --json anomalies.json
python comparison_dashboard.py --input anomalies.json -o dashboard.html
```

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Precision | 95.7% |
| Recall | 95.3% (estimated) |
| F1 Score | 95.5% |

## Architecture

```
Metrics/ --> anomaly_detector.py --> anomalies.json
                                          |
                    +---------------------+---------------------+
                    |                     |                     |
            anomaly_explorer    comparison_dashboard    ground_truth_evaluator
            (filter/group)      (AI synthesis)          (precision/recall)
```

| Tool | Purpose | Requires API Key? |
|------|---------|-------------------|
| `anomaly_detector.py` | Rule-based anomaly detection | No |
| `anomaly_explorer.py` | Filter, group, drill-down into anomalies | No |
| `comparison_dashboard.py` | AI synthesizes patterns, identifies root causes, recommends actions | Yes |
| `ground_truth_evaluator.py` | Calculate precision/recall metrics | No |

## What It Detects

- **Feature count drops/spikes** - Sudden changes in feature counts (~100% spikes indicate duplication bugs)
- **High removal/addition rates** - Unusual churn between releases
- **Net feature change** - Large net gains or losses
- **Data regeneration** - When <5% of records unchanged (complete rebuild)
- **Churn spikes** - High ID instability
- **Geometry length changes** - Significant changes in total geometry
- **Low attribute coverage** - Important fields with low fill rates
- **Category distribution shifts** - Unexpected changes in subtype proportions
- **Geographic concentration** - Changes concentrated in specific countries
- **Historical deviations** - Statistical outliers vs historical patterns

## Usage

### Basic Usage

```bash
# Detect anomalies (uses two most recent releases)
python anomaly_detector.py Metrics/ --json anomalies.json

# Export to Excel directly
python anomaly_detector.py Metrics/ --json anomalies.json --excel anomalies.xlsx

# View dashboard with AI insights
python comparison_dashboard.py --input anomalies.json -o dashboard.html

# Explore/filter anomalies (alternative to Excel)
python anomaly_explorer.py anomalies.json --export-html explorer.html
```

### Comparing Specific Releases

```bash
# Compare specific releases
python anomaly_detector.py Metrics/ \
    --current 2025-09-24.0 \
    --previous 2025-08-20.1 \
    --json anomalies.json

# Compare older releases
python anomaly_detector.py Metrics/ \
    --current 2025-08-20.1 \
    --previous 2025-07-23.0 \
    --json anomalies_aug.json
```

### Chat Feature (Interactive Analysis)

The comparison dashboard includes an AI chat panel for interactive analysis:

```bash
python comparison_dashboard.py --input anomalies.json -o dashboard.html
```

Click the chat bubble in the bottom-right corner to ask questions like:
- "Which countries are most affected by the duplication bug?"
- "What's the root cause of the geometry changes?"
- "Which anomalies should I prioritize fixing first?"
- "What files should I check to debug the divisions issue?"

The chat will prompt for your API key on first use (stored in browser localStorage).

### Evaluate Precision/Recall

```bash
python ground_truth_evaluator.py anomalies.json --generate-heuristic ground_truth.csv
python ground_truth_evaluator.py anomalies.json --evaluate ground_truth.csv
```

## Data Directory Structure

```
Metrics/
├── metrics/
│   ├── 2025-08-20.1/
│   │   ├── row_counts/theme=*/type=*/*.csv
│   │   └── changelog_stats/*.csv
│   └── 2025-09-24.0/
├── theme_column_summary_stats/*.csv
└── theme_class_summary_stats/release_to_release_comparisons/*.csv
```

## License

Apache-2.0