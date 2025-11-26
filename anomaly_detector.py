"""
Overture Maps Release Anomaly Detector
======================================
Detects unexpected changes, data quality issues, and potential problems
in monthly release data.

Anomaly Categories Covered:
1. Feature Count Fluctuation - sudden drops/spikes in counts
2. Attribute Coverage Drop - missing data increasing  
3. GERS ID Instability - unusual churn in adds/removes
4. Geographic Concentration - changes concentrated in specific regions
5. Category Distribution Shift - subtype/class ratios changing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import json
from datetime import datetime


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyType(Enum):
    FEATURE_COUNT_DROP = "feature_count_drop"
    FEATURE_COUNT_SPIKE = "feature_count_spike"
    ATTRIBUTE_COVERAGE_DROP = "attribute_coverage_drop"
    CHURN_SPIKE = "churn_spike"
    GEOMETRY_LENGTH_CHANGE = "geometry_length_change"
    GEOGRAPHIC_CONCENTRATION = "geographic_concentration"
    CATEGORY_DISTRIBUTION_SHIFT = "category_distribution_shift"
    NEW_CATEGORY_APPEARED = "new_category_appeared"
    CATEGORY_DISAPPEARED = "category_disappeared"


@dataclass
class Anomaly:
    """Represents a detected anomaly in the data."""
    anomaly_type: AnomalyType
    severity: Severity
    theme: str
    feature_type: str
    description: str
    metric_name: str
    previous_value: float
    current_value: float
    percent_change: float
    release_current: str
    release_previous: str
    subtype: Optional[str] = None
    class_name: Optional[str] = None
    country: Optional[str] = None
    additional_context: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "theme": self.theme,
            "type": self.feature_type,
            "subtype": self.subtype,
            "class": self.class_name,
            "country": self.country,
            "description": self.description,
            "metric": self.metric_name,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "percent_change": round(self.percent_change, 2),
            "release_current": self.release_current,
            "release_previous": self.release_previous,
            "additional_context": self.additional_context
        }
    
    def __str__(self) -> str:
        location = ""
        if self.country:
            location = f" in {self.country}"
        category = ""
        if self.subtype or self.class_name:
            parts = [p for p in [self.subtype, self.class_name] if p]
            category = f" ({'/'.join(parts)})"
        
        return (
            f"[{self.severity.value.upper()}] {self.anomaly_type.value}: "
            f"{self.theme}/{self.feature_type}{category}{location}\n"
            f"  {self.description}\n"
            f"  {self.metric_name}: {self.previous_value:,.0f} → {self.current_value:,.0f} "
            f"({self.percent_change:+.1f}%)"
        )


class AnomalyThresholds:
    """Configurable thresholds for anomaly detection."""
    
    def __init__(
        self,
        # Feature count thresholds
        count_drop_warning: float = -10.0,     # % drop to trigger warning
        count_drop_critical: float = -20.0,    # % drop to trigger critical
        count_spike_warning: float = 20.0,     # % increase to trigger warning
        count_spike_critical: float = 50.0,    # % increase to trigger critical
        
        # Churn thresholds (from changelog_stats)
        churn_warning: float = 10.0,           # % added+removed to trigger warning
        churn_critical: float = 25.0,          # % added+removed to trigger critical
        
        # Attribute coverage thresholds
        coverage_drop_warning: float = -3.0,   # % drop in attribute fill rate
        coverage_drop_critical: float = -10.0,
        
        # Geometry length thresholds (for roads, boundaries)
        length_drop_warning: float = -10.0,
        length_drop_critical: float = -20.0,
        
        # Minimum counts to consider (avoid noise from small samples)
        min_feature_count: int = 100,
        min_country_count: int = 50,
        
        # Category distribution shift
        category_shift_warning: float = 5.0,   # percentage point shift
        category_shift_critical: float = 15.0,
    ):
        self.count_drop_warning = count_drop_warning
        self.count_drop_critical = count_drop_critical
        self.count_spike_warning = count_spike_warning
        self.count_spike_critical = count_spike_critical
        self.churn_warning = churn_warning
        self.churn_critical = churn_critical
        self.coverage_drop_warning = coverage_drop_warning
        self.coverage_drop_critical = coverage_drop_critical
        self.length_drop_warning = length_drop_warning
        self.length_drop_critical = length_drop_critical
        self.min_feature_count = min_feature_count
        self.min_country_count = min_country_count
        self.category_shift_warning = category_shift_warning
        self.category_shift_critical = category_shift_critical


class OvertureAnomalyDetector:
    """
    Main anomaly detection engine for Overture Maps releases.
    """
    
    def __init__(self, data_root: Path, thresholds: Optional[AnomalyThresholds] = None):
        self.data_root = Path(data_root)
        self.thresholds = thresholds or AnomalyThresholds()
        self.anomalies: list[Anomaly] = []
        
    def get_sorted_releases(self) -> list[str]:
        """Get list of releases sorted chronologically."""
        metrics_dir = self.data_root / "metrics"
        if not metrics_dir.exists():
            return []
        
        releases = [d.name for d in metrics_dir.iterdir() if d.is_dir()]
        # Sort by date (format: YYYY-MM-DD.N)
        return sorted(releases, key=lambda x: (x.split('.')[0], int(x.split('.')[-1])))
    
    def load_row_counts(self, release: str, theme: str, feature_type: str) -> Optional[pd.DataFrame]:
        """Load row counts CSV for a specific release/theme/type."""
        pattern = f"metrics/{release}/row_counts/theme={theme}/type={feature_type}/*.csv"
        files = list(self.data_root.glob(pattern))
        if not files:
            return None
        try:
            return pd.read_csv(files[0], on_bad_lines='skip')
        except Exception as e:
            print(f"  Warning: Could not load {files[0]}: {e}")
            return None
    
    def load_changelog_stats(self, release: str) -> Optional[pd.DataFrame]:
        """Load changelog statistics for a release."""
        pattern = f"metrics/{release}/changelog_stats/*.csv"
        files = list(self.data_root.glob(pattern))
        if not files:
            return None
        try:
            return pd.read_csv(files[0], on_bad_lines='skip')
        except Exception as e:
            print(f"  Warning: Could not load changelog stats: {e}")
            return None
    
    def load_release_comparison(self, theme: str, feature_type: str) -> Optional[pd.DataFrame]:
        """Load release-to-release comparison file."""
        path = self.data_root / f"theme_class_summary_stats/release_to_release_comparisons/theme={theme}.type={feature_type}.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, on_bad_lines='skip')
        except Exception as e:
            print(f"  Warning: Could not load comparison file: {e}")
            return None
    
    def load_column_stats(self, release: str, theme: str, feature_type: str) -> Optional[pd.DataFrame]:
        """Load column summary stats for attribute coverage analysis."""
        pattern = f"theme_column_summary_stats/{release}.theme={theme}.type={feature_type}.csv"
        files = list(self.data_root.glob(pattern))
        if not files:
            return None
        try:
            return pd.read_csv(files[0], on_bad_lines='skip')
        except Exception as e:
            print(f"  Warning: Could not load column stats: {e}")
            return None
    
    def _calculate_percent_change(self, old: float, new: float) -> float:
        """Calculate percent change, handling edge cases."""
        if old == 0:
            return 100.0 if new > 0 else 0.0
        return ((new - old) / old) * 100
    
    def _determine_severity(self, pct_change: float, warning_threshold: float, critical_threshold: float) -> Optional[Severity]:
        """Determine severity based on thresholds."""
        # For negative thresholds (drops)
        if warning_threshold < 0:
            if pct_change <= critical_threshold:
                return Severity.CRITICAL
            elif pct_change <= warning_threshold:
                return Severity.WARNING
        # For positive thresholds (spikes)
        else:
            if pct_change >= critical_threshold:
                return Severity.CRITICAL
            elif pct_change >= warning_threshold:
                return Severity.WARNING
        return None
    
    def detect_feature_count_anomalies(
        self, 
        release_current: str, 
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> list[Anomaly]:
        """Detect anomalies in feature counts between releases."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        # Determine grouping columns available
        group_cols = []
        for col in ['subtype', 'class', 'country']:
            if col in df_current.columns and col in df_previous.columns:
                group_cols.append(col)
        
        if not group_cols:
            # Aggregate total
            group_cols = []
        
        # Aggregate by available dimensions
        if group_cols:
            current_agg = df_current.groupby(group_cols)['id_count'].sum().reset_index()
            previous_agg = df_previous.groupby(group_cols)['id_count'].sum().reset_index()
            
            # Merge and compare
            merged = current_agg.merge(
                previous_agg, 
                on=group_cols, 
                how='outer', 
                suffixes=('_current', '_previous')
            ).fillna(0)
            
            for _, row in merged.iterrows():
                current_count = row['id_count_current']
                previous_count = row['id_count_previous']
                
                # Skip small samples
                if previous_count < self.thresholds.min_feature_count:
                    continue
                
                pct_change = self._calculate_percent_change(previous_count, current_count)
                
                # Check for drops
                severity = self._determine_severity(
                    pct_change,
                    self.thresholds.count_drop_warning,
                    self.thresholds.count_drop_critical
                )
                if severity:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FEATURE_COUNT_DROP,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        subtype=row.get('subtype'),
                        class_name=row.get('class'),
                        country=row.get('country'),
                        description=f"Feature count dropped significantly",
                        metric_name="id_count",
                        previous_value=previous_count,
                        current_value=current_count,
                        percent_change=pct_change,
                        release_current=release_current,
                        release_previous=release_previous
                    ))
                
                # Check for spikes
                severity = self._determine_severity(
                    pct_change,
                    self.thresholds.count_spike_warning,
                    self.thresholds.count_spike_critical
                )
                if severity:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FEATURE_COUNT_SPIKE,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        subtype=row.get('subtype'),
                        class_name=row.get('class'),
                        country=row.get('country'),
                        description=f"Feature count spiked unexpectedly",
                        metric_name="id_count",
                        previous_value=previous_count,
                        current_value=current_count,
                        percent_change=pct_change,
                        release_current=release_current,
                        release_previous=release_previous
                    ))
        
        return anomalies
    
    def detect_geometry_length_anomalies(
        self,
        release_current: str,
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> list[Anomaly]:
        """Detect anomalies in total geometry length (useful for roads, boundaries)."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        if 'total_geometry_length_km' not in df_current.columns:
            return anomalies
        
        # Determine grouping columns
        group_cols = []
        for col in ['subtype', 'class', 'country']:
            if col in df_current.columns and col in df_previous.columns:
                group_cols.append(col)
        
        if group_cols:
            current_agg = df_current.groupby(group_cols)['total_geometry_length_km'].sum().reset_index()
            previous_agg = df_previous.groupby(group_cols)['total_geometry_length_km'].sum().reset_index()
            
            merged = current_agg.merge(
                previous_agg,
                on=group_cols,
                how='outer',
                suffixes=('_current', '_previous')
            ).fillna(0)
            
            for _, row in merged.iterrows():
                current_len = row['total_geometry_length_km_current']
                previous_len = row['total_geometry_length_km_previous']
                
                if previous_len < 100:  # Skip if less than 100km total
                    continue
                
                pct_change = self._calculate_percent_change(previous_len, current_len)
                
                severity = self._determine_severity(
                    pct_change,
                    self.thresholds.length_drop_warning,
                    self.thresholds.length_drop_critical
                )
                
                if severity:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.GEOMETRY_LENGTH_CHANGE,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        subtype=row.get('subtype'),
                        class_name=row.get('class'),
                        country=row.get('country'),
                        description=f"Total geometry length changed significantly",
                        metric_name="total_geometry_length_km",
                        previous_value=previous_len,
                        current_value=current_len,
                        percent_change=pct_change,
                        release_current=release_current,
                        release_previous=release_previous
                    ))
        
        return anomalies
    
    def detect_churn_anomalies(self, release: str) -> list[Anomaly]:
        """Detect high churn rates from changelog stats."""
        anomalies = []
        
        df = self.load_changelog_stats(release)
        if df is None:
            return anomalies
        
        for _, row in df.iterrows():
            theme = row.get('theme', 'unknown')
            feature_type = row.get('type', 'unknown')
            
            # Calculate churn rate
            total_baseline = row.get('total_baseline', 0)
            added = row.get('added', 0)
            removed = row.get('removed', 0)
            
            if total_baseline < self.thresholds.min_feature_count:
                continue
            
            churn_rate = ((added + removed) / total_baseline) * 100 if total_baseline > 0 else 0
            
            if churn_rate >= self.thresholds.churn_critical:
                severity = Severity.CRITICAL
            elif churn_rate >= self.thresholds.churn_warning:
                severity = Severity.WARNING
            else:
                continue
            
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.CHURN_SPIKE,
                severity=severity,
                theme=theme,
                feature_type=feature_type,
                description=f"High ID churn detected: {added:,} added, {removed:,} removed",
                metric_name="churn_rate",
                previous_value=total_baseline,
                current_value=total_baseline + added - removed,
                percent_change=churn_rate,
                release_current=release,
                release_previous="baseline",
                additional_context={
                    "added": added,
                    "removed": removed,
                    "data_changed": row.get('data_changed', 0)
                }
            ))
        
        return anomalies
    
    def detect_category_distribution_shift(
        self,
        release_current: str,
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> list[Anomaly]:
        """Detect shifts in category distributions (subtype/class proportions)."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        if 'subtype' not in df_current.columns:
            return anomalies
        
        # Calculate proportion by subtype
        current_total = df_current['id_count'].sum()
        previous_total = df_previous['id_count'].sum()
        
        if current_total == 0 or previous_total == 0:
            return anomalies
        
        current_props = df_current.groupby('subtype')['id_count'].sum() / current_total * 100
        previous_props = df_previous.groupby('subtype')['id_count'].sum() / previous_total * 100
        
        # Compare proportions
        all_subtypes = set(current_props.index) | set(previous_props.index)
        
        for subtype in all_subtypes:
            current_pct = current_props.get(subtype, 0)
            previous_pct = previous_props.get(subtype, 0)
            
            shift = current_pct - previous_pct
            
            if abs(shift) >= self.thresholds.category_shift_critical:
                severity = Severity.CRITICAL
            elif abs(shift) >= self.thresholds.category_shift_warning:
                severity = Severity.WARNING
            else:
                continue
            
            # Check for new/disappeared categories
            if previous_pct == 0 and current_pct > 0:
                anomaly_type = AnomalyType.NEW_CATEGORY_APPEARED
                description = f"New subtype '{subtype}' appeared with {current_pct:.1f}% of features"
            elif current_pct == 0 and previous_pct > 0:
                anomaly_type = AnomalyType.CATEGORY_DISAPPEARED
                description = f"Subtype '{subtype}' disappeared (was {previous_pct:.1f}% of features)"
            else:
                anomaly_type = AnomalyType.CATEGORY_DISTRIBUTION_SHIFT
                direction = "increased" if shift > 0 else "decreased"
                description = f"Subtype '{subtype}' proportion {direction} by {abs(shift):.1f} percentage points"
            
            anomalies.append(Anomaly(
                anomaly_type=anomaly_type,
                severity=severity,
                theme=theme,
                feature_type=feature_type,
                subtype=subtype,
                description=description,
                metric_name="subtype_proportion_pct",
                previous_value=previous_pct,
                current_value=current_pct,
                percent_change=shift,
                release_current=release_current,
                release_previous=release_previous
            ))
        
        return anomalies
    
    def run_full_analysis(
        self,
        release_current: Optional[str] = None,
        release_previous: Optional[str] = None
    ) -> list[Anomaly]:
        """
        Run complete anomaly detection between two releases.
        If releases not specified, uses the two most recent.
        """
        self.anomalies = []
        
        releases = self.get_sorted_releases()
        if len(releases) < 2:
            print("Need at least 2 releases for comparison")
            return []
        
        if release_current is None:
            release_current = releases[-1]
        if release_previous is None:
            release_previous = releases[-2]
        
        print(f"Analyzing: {release_previous} → {release_current}")
        print("=" * 60)
        
        # Define themes and types to check
        checks = [
            ("buildings", "building"),
            ("buildings", "building_part"),
            ("base", "land_cover"),
            ("base", "infrastructure"),
            ("base", "land"),
            ("base", "water"),
            ("base", "land_use"),
            ("places", "place"),
            ("divisions", "division"),
            ("divisions", "division_boundary"),
            ("divisions", "division_area"),
            ("transportation", "segment"),
            ("transportation", "connector"),
            ("addresses", "address"),
        ]
        
        for theme, feature_type in checks:
            print(f"\nChecking {theme}/{feature_type}...")
            
            # Feature count anomalies
            self.anomalies.extend(
                self.detect_feature_count_anomalies(
                    release_current, release_previous, theme, feature_type
                )
            )
            
            # Geometry length anomalies
            self.anomalies.extend(
                self.detect_geometry_length_anomalies(
                    release_current, release_previous, theme, feature_type
                )
            )
            
            # Category distribution shifts
            self.anomalies.extend(
                self.detect_category_distribution_shift(
                    release_current, release_previous, theme, feature_type
                )
            )
        
        # Churn anomalies from changelog
        self.anomalies.extend(self.detect_churn_anomalies(release_current))
        
        # Sort by severity
        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        self.anomalies.sort(key=lambda a: severity_order[a.severity])
        
        return self.anomalies
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a human-readable report of detected anomalies."""
        lines = []
        lines.append("=" * 70)
        lines.append("OVERTURE MAPS RELEASE ANOMALY REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total anomalies detected: {len(self.anomalies)}")
        lines.append("")
        
        # Summary by severity
        critical = [a for a in self.anomalies if a.severity == Severity.CRITICAL]
        warnings = [a for a in self.anomalies if a.severity == Severity.WARNING]
        
        lines.append(f"🔴 CRITICAL: {len(critical)}")
        lines.append(f"🟡 WARNING: {len(warnings)}")
        lines.append("")
        
        if critical:
            lines.append("-" * 70)
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 70)
            for anomaly in critical:
                lines.append("")
                lines.append(str(anomaly))
        
        if warnings:
            lines.append("")
            lines.append("-" * 70)
            lines.append("WARNINGS")
            lines.append("-" * 70)
            for anomaly in warnings:
                lines.append("")
                lines.append(str(anomaly))
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
        
        return report
    
    def export_json(self, output_path: Path) -> None:
        """Export anomalies to JSON for further processing."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_anomalies": len(self.anomalies),
            "summary": {
                "critical": len([a for a in self.anomalies if a.severity == Severity.CRITICAL]),
                "warning": len([a for a in self.anomalies if a.severity == Severity.WARNING]),
            },
            "anomalies": [a.to_dict() for a in self.anomalies]
        }
        output_path.write_text(json.dumps(data, indent=2))


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect anomalies in Overture Maps releases")
    parser.add_argument("data_root", type=Path, help="Root directory containing metrics/")
    parser.add_argument("--current", type=str, help="Current release (e.g., 2025-09-24.0)")
    parser.add_argument("--previous", type=str, help="Previous release to compare against")
    parser.add_argument("--output", type=Path, help="Output file for report")
    parser.add_argument("--json", type=Path, help="Output JSON file")
    parser.add_argument("--count-drop-warning", type=float, default=-5.0)
    parser.add_argument("--count-drop-critical", type=float, default=-15.0)
    
    args = parser.parse_args()
    
    thresholds = AnomalyThresholds(
        count_drop_warning=args.count_drop_warning,
        count_drop_critical=args.count_drop_critical,
    )
    
    detector = OvertureAnomalyDetector(args.data_root, thresholds)
    anomalies = detector.run_full_analysis(args.current, args.previous)
    
    report = detector.generate_report(args.output)
    print(report)
    
    if args.json:
        detector.export_json(args.json)
        print(f"\nJSON exported to: {args.json}")