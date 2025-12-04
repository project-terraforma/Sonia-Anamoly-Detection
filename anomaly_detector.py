"""
Overture Maps Release Anomaly Detector (Unified)
=================================================
Comprehensive rule-based anomaly detection merging:
- anomaly_detector.py (threshold-based checks)
- release_validator.py (historical analysis, coverage trends)

Anomaly Categories Covered:
1. Feature Count Fluctuation - sudden drops/spikes
2. High Removal/Addition Rates - unusual churn patterns
3. Data Quality Degradation - low unchanged %, high modifications
4. Historical Deviations - z-score analysis vs trends
5. Churn Rate Spikes - GERS ID instability
6. Growth Velocity Changes - acceleration/deceleration
7. Attribute Coverage - drops, spikes, absolute coverage
8. Category Distribution Shifts - subtype/class changes
9. Incomplete Data - null rate increases
10. Confidence Score Monitoring - places quality
11. Geometry Quality - size changes, invalid geometries
12. Geographic Concentration - country-level patterns
13. Junk/Spam Detection - heuristic name scanning

Usage:
    python anomaly_detector.py Metrics/ --json anomalies.json
    python comparison_dashboard.py --input anomalies.json -o dashboard.html
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import json
import glob
import os
from datetime import datetime


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyType(Enum):
    # Feature count
    FEATURE_COUNT_DROP = "feature_count_drop"
    FEATURE_COUNT_SPIKE = "feature_count_spike"
    
    # Churn/removals
    HIGH_REMOVAL_RATE = "high_removal_rate"
    HIGH_ADDITION_RATE = "high_addition_rate"
    CHURN_SPIKE = "churn_spike"
    CHURN_RATE_SPIKE = "churn_rate_spike"
    NET_FEATURE_LOSS = "net_feature_loss"
    NET_FEATURE_GAIN = "net_feature_gain"
    
    # Data quality
    DATA_QUALITY_DEGRADATION = "data_quality_degradation"
    DATA_REGENERATION = "data_regeneration"
    INCOMPLETE_DATA_INCREASE = "incomplete_data_increase"
    CONFIDENCE_SCORE_DROP = "confidence_score_drop"
    
    # Historical
    HISTORICAL_DEVIATION = "historical_deviation"
    GROWTH_VELOCITY_CHANGE = "growth_velocity_change"
    
    # Coverage
    ATTRIBUTE_COVERAGE_DROP = "attribute_coverage_drop"
    ATTRIBUTE_COVERAGE_SPIKE = "attribute_coverage_spike"
    LOW_ATTRIBUTE_COVERAGE = "low_attribute_coverage"
    
    # Categories
    CATEGORY_DISTRIBUTION_SHIFT = "category_distribution_shift"
    NEW_CATEGORY_APPEARED = "new_category_appeared"
    CATEGORY_DISAPPEARED = "category_disappeared"
    
    # Geometry
    GEOMETRY_LENGTH_CHANGE = "geometry_length_change"
    GEOMETRY_SIZE_ANOMALY = "geometry_size_anomaly"
    
    # Geographic
    GEOGRAPHIC_CONCENTRATION = "geographic_concentration"
    
    # Spam/junk
    JUNK_RATE_INCREASE = "junk_rate_increase"
    SPAMMY_NAMES_DETECTED = "spammy_names_detected"


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
    column: Optional[str] = None
    expected_range: Optional[Tuple[float, float]] = None
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
            "column": self.column,
            "description": self.description,
            "metric": self.metric_name,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "percent_change": round(self.percent_change, 2),
            "release_current": self.release_current,
            "release_previous": self.release_previous,
            "expected_range": list(self.expected_range) if self.expected_range else None,
            "additional_context": self.additional_context
        }
    
    def __str__(self) -> str:
        location = f"{self.theme}/{self.feature_type}"
        if self.subtype:
            location += f" ({self.subtype}"
            if self.class_name:
                location += f"/{self.class_name}"
            location += ")"
        if self.country:
            location += f" in {self.country}"
        if self.column:
            location += f" [{self.column}]"
        
        return (
            f"[{self.severity.value.upper()}] {self.anomaly_type.value}: {location}\n"
            f"  {self.description}\n"
            f"  {self.metric_name}: {self.previous_value:,.2f} → {self.current_value:,.2f} "
            f"({self.percent_change:+.1f}%)"
        )


@dataclass
class AnomalyThresholds:
    """Configurable thresholds for anomaly detection."""
    # Feature count thresholds
    count_drop_warning: float = -10.0
    count_drop_critical: float = -20.0
    count_spike_warning: float = 20.0
    count_spike_critical: float = 50.0
    
    # Large change thresholds (from release_validator)
    large_change_threshold: float = 10.0
    moderate_change_threshold: float = 5.0
    
    # Removal/addition thresholds
    high_removal_warning: float = 1.0
    high_removal_critical: float = 5.0
    high_addition_warning: float = 10.0
    high_addition_critical: float = 25.0
    
    # Net change thresholds (absolute counts)
    net_change_warning: int = 100000  # 100K net change
    net_change_critical: int = 1000000  # 1M net change
    
    # Churn thresholds
    churn_warning: float = 10.0
    churn_critical: float = 25.0
    
    # Data quality
    high_modification_warning: float = 5.0
    high_modification_critical: float = 20.0
    low_unchanged_warning: float = 50.0
    low_unchanged_critical: float = 1.0
    data_regeneration_threshold: float = 5.0  # Less than 5% unchanged = regeneration
    
    # Historical analysis
    historical_std_multiplier: float = 3.0
    
    # Attribute coverage
    coverage_drop_warning: float = 5.0
    coverage_drop_critical: float = 15.0
    low_coverage_warning: float = 20.0  # Flag if < 20% coverage
    low_coverage_critical: float = 5.0   # Critical if < 5% coverage
    
    # Incomplete data (percentage point increase)
    incomplete_data_warning: float = 2.0
    incomplete_data_critical: float = 5.0
    
    # Confidence score (percentage point drop)
    confidence_drop_warning: float = -2.0
    confidence_drop_critical: float = -5.0
    
    # Category distribution
    category_shift_warning: float = 5.0
    category_shift_critical: float = 15.0
    
    # Geometry
    length_drop_warning: float = -10.0
    length_drop_critical: float = -20.0
    geometry_size_change_threshold: float = 50.0
    
    # Minimum counts
    min_feature_count: int = 100
    min_country_count: int = 50


# Required fields for incomplete data detection
REQUIRED_FIELDS = {
    ("places", "place"): ["names", "categories", "addresses", "phones", "websites", "confidence", "brand"],
    ("addresses", "address"): ["street", "postcode", "country", "number", "address_levels", "address_level_1", "address_level_2", "address_level_3"],
    ("buildings", "building"): ["names", "class", "height", "num_floors", "level"],
    ("buildings", "building_part"): ["class", "height", "num_floors"],
    ("transportation", "segment"): ["names", "class", "subtype"],
    ("transportation", "connector"): [],
    ("divisions", "division"): ["names", "subtype", "country"],
    ("divisions", "division_area"): ["names", "subtype", "class", "country"],
    ("divisions", "division_boundary"): [],
    ("base", "water"): ["names", "subtype", "class"],
    ("base", "land"): ["subtype", "class", "names"],
    ("base", "land_use"): ["subtype", "class"],
    ("base", "land_cover"): ["subtype", "cartography"],
    ("base", "infrastructure"): ["subtype", "class", "names"],
}

# Optional fields - these are expected to be sparse and should NOT be flagged
OPTIONAL_FIELDS = {
    # Enrichment fields - most features don't have these
    "wikidata", "wikipedia",
    
    # Building architectural details - rarely captured
    "facade_color", "facade_material",
    "roof_color", "roof_material", "roof_shape", "roof_direction", "roof_orientation", "roof_height",
    "min_height", "min_floor", "num_floors_underground",
    
    # Environmental/physical attributes - sparse data
    "elevation", "surface", "is_salt", "is_intermittent",
    
    # Contact info - optional
    "emails",
    
    # Level attributes - only applies to multi-level features
    "level",
    
    # Transportation optional attributes
    "level_rules", "prohibited_transitions", "width_rules", "destinations", "road_flags",
    
    # Address optional granularity
    "postal_city", "unit",
}


class OvertureAnomalyDetector:
    """
    Unified anomaly detection engine for Overture Maps releases.
    Combines rule-based threshold checks with historical trend analysis.
    """
    
    def __init__(self, data_root: Path, thresholds: Optional[AnomalyThresholds] = None):
        self.data_root = Path(data_root)
        self.thresholds = thresholds or AnomalyThresholds()
        self.anomalies: List[Anomaly] = []
        self.geographic_summary: Dict = {}
        
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def get_sorted_releases(self) -> List[str]:
        """Get list of releases sorted chronologically."""
        metrics_dir = self.data_root / "metrics"
        if not metrics_dir.exists():
            return []
        
        releases = [d.name for d in metrics_dir.iterdir() if d.is_dir()]
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
            return None
    
    def load_changelog_stats(self, release: str) -> Optional[pd.DataFrame]:
        """Load changelog statistics for a release."""
        pattern = f"metrics/{release}/changelog_stats/*.csv"
        files = list(self.data_root.glob(pattern))
        if not files:
            return None
        try:
            # Try tab-separated first (common format)
            df = pd.read_csv(files[0], sep='\t', on_bad_lines='skip')
            if len(df.columns) <= 1:
                df = pd.read_csv(files[0], on_bad_lines='skip')
            return df
        except Exception as e:
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
            return None
    
    def load_all_column_stats(self) -> Optional[pd.DataFrame]:
        """Load all column stats across releases for trend analysis."""
        pattern = "theme_column_summary_stats/*.theme=*.type=*.csv"
        files = list(self.data_root.glob(pattern))
        files = [f for f in files if 'release_to_release' not in str(f)]
        
        if not files:
            return None
        
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                if 'release' in df.columns:
                    dfs.append(df)
            except:
                continue
        
        return pd.concat(dfs, ignore_index=True) if dfs else None
    
    def load_historical_totals(self) -> Optional[pd.DataFrame]:
        """Load historical release comparison data."""
        hist_dir = self.data_root / "theme_column_summary_stats" / "release_to_release_comparisons"
        if not hist_dir.exists():
            return None
        
        pattern = str(hist_dir / "theme=*.type=*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                dfs.append(df)
            except:
                continue
        
        return pd.concat(dfs, ignore_index=True) if dfs else None
    
    def load_class_comparisons(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load class comparison files for distribution analysis."""
        comp_dir = self.data_root / "theme_class_summary_stats" / "release_to_release_comparisons"
        if not comp_dir.exists():
            return None
        
        pattern = str(comp_dir / "theme=*.type=*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        comparisons = {}
        for file in files:
            try:
                basename = os.path.basename(file)
                parts = basename.replace('.csv', '').split('.')
                theme = parts[0].split('=')[1]
                type_ = parts[1].split('=')[1]
                key = f"{theme}/{type_}"
                
                df = pd.read_csv(file, sep='\t', on_bad_lines='skip')
                if len(df.columns) <= 1:
                    df = pd.read_csv(file, on_bad_lines='skip')
                comparisons[key] = df
            except:
                continue
        
        return comparisons if comparisons else None
    
    # =========================================================================
    # GEOGRAPHIC SUMMARY (for AI agent)
    # =========================================================================
    
    def prepare_geographic_summary(self, release: str) -> Dict:
        """Prepare geographic data summary for AI analysis."""
        geo_data = {}
        
        themes_types = [
            ("places", "place"),
            ("buildings", "building"),
            ("transportation", "segment"),
            ("divisions", "division"),
            ("divisions", "division_area"),
            ("addresses", "address"),
        ]
        
        for theme, feature_type in themes_types:
            df = self.load_row_counts(release, theme, feature_type)
            if df is None or 'country' not in df.columns:
                continue
            
            key = f"{theme}/{feature_type}"
            
            # Get totals by country
            if 'id_count' in df.columns:
                country_totals = df.groupby('country')['id_count'].sum().sort_values(ascending=False)
                top_countries = country_totals.head(10).to_dict()
                
                geo_data[key] = {
                    'theme': theme,
                    'type': feature_type,
                    'top_countries': top_countries,
                    'total_countries': len(country_totals),
                    'total_features': int(country_totals.sum())
                }
            
            # If we have change_type column, get additions/removals by country
            if 'change_type' in df.columns:
                summary = df.groupby(['country', 'change_type'])['id_count'].sum().reset_index()
                
                removed = summary[summary['change_type'] == 'removed'].nlargest(5, 'id_count')
                added = summary[summary['change_type'] == 'added'].nlargest(5, 'id_count')
                
                geo_data[key]['top_removals'] = removed[['country', 'id_count']].to_dict('records')
                geo_data[key]['top_additions'] = added[['country', 'id_count']].to_dict('records')
        
        self.geographic_summary = geo_data
        return geo_data
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _calculate_percent_change(self, old: float, new: float) -> float:
        """Calculate percent change, handling edge cases."""
        if old == 0:
            return 100.0 if new > 0 else 0.0
        return ((new - old) / old) * 100
    
    def _determine_severity(self, value: float, warning_threshold: float, critical_threshold: float, 
                           lower_is_worse: bool = True) -> Optional[Severity]:
        """Determine severity based on thresholds."""
        if lower_is_worse:
            # For drops (negative thresholds)
            if value <= critical_threshold:
                return Severity.CRITICAL
            elif value <= warning_threshold:
                return Severity.WARNING
        else:
            # For spikes (positive thresholds)
            if value >= critical_threshold:
                return Severity.CRITICAL
            elif value >= warning_threshold:
                return Severity.WARNING
        return None
    
    # =========================================================================
    # DETECTION METHODS
    # =========================================================================
    
    def detect_feature_count_anomalies(
        self, 
        release_current: str, 
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> List[Anomaly]:
        """Detect anomalies in feature counts between releases."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        # Determine grouping columns
        group_cols = [col for col in ['subtype', 'class', 'country'] 
                      if col in df_current.columns and col in df_previous.columns]
        
        if not group_cols or 'id_count' not in df_current.columns:
            return anomalies
        
        current_agg = df_current.groupby(group_cols)['id_count'].sum().reset_index()
        previous_agg = df_previous.groupby(group_cols)['id_count'].sum().reset_index()
        
        merged = current_agg.merge(
            previous_agg, 
            on=group_cols, 
            how='outer', 
            suffixes=('_current', '_previous')
        ).fillna(0)
        
        for _, row in merged.iterrows():
            current_count = row['id_count_current']
            previous_count = row['id_count_previous']
            
            if previous_count < self.thresholds.min_feature_count:
                continue
            
            pct_change = self._calculate_percent_change(previous_count, current_count)
            
            # Check for drops
            if pct_change <= self.thresholds.count_drop_critical:
                severity = Severity.CRITICAL
            elif pct_change <= self.thresholds.count_drop_warning:
                severity = Severity.WARNING
            else:
                severity = None
            
            if severity:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.FEATURE_COUNT_DROP,
                    severity=severity,
                    theme=theme,
                    feature_type=feature_type,
                    subtype=row.get('subtype'),
                    class_name=row.get('class'),
                    country=row.get('country'),
                    description="Feature count dropped significantly",
                    metric_name="id_count",
                    previous_value=previous_count,
                    current_value=current_count,
                    percent_change=pct_change,
                    release_current=release_current,
                    release_previous=release_previous
                ))
            
            # Check for spikes
            if pct_change >= self.thresholds.count_spike_critical:
                severity = Severity.CRITICAL
            elif pct_change >= self.thresholds.count_spike_warning:
                severity = Severity.WARNING
            else:
                severity = None
            
            if severity:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.FEATURE_COUNT_SPIKE,
                    severity=severity,
                    theme=theme,
                    feature_type=feature_type,
                    subtype=row.get('subtype'),
                    class_name=row.get('class'),
                    country=row.get('country'),
                    description="Feature count spiked unexpectedly",
                    metric_name="id_count",
                    previous_value=previous_count,
                    current_value=current_count,
                    percent_change=pct_change,
                    release_current=release_current,
                    release_previous=release_previous
                ))
        
        return anomalies
    
    def detect_changelog_anomalies(self, release_current: str, release_previous: str) -> List[Anomaly]:
        """Detect anomalies from changelog stats (removals, additions, churn, data quality, net change, regeneration)."""
        anomalies = []
        
        df = self.load_changelog_stats(release_current)
        if df is None:
            return anomalies
        
        for _, row in df.iterrows():
            theme = row.get('theme', 'unknown')
            feature_type = row.get('type', 'unknown')
            
            total_baseline = row.get('total_baseline', 0)
            if total_baseline < self.thresholds.min_feature_count:
                continue
            
            added = row.get('added', 0)
            removed = row.get('removed', 0)
            data_changed = row.get('data_changed', 0)
            
            added_perc = row.get('added_perc', (added / total_baseline * 100) if total_baseline > 0 else 0)
            removed_perc = row.get('removed_perc', (removed / total_baseline * 100) if total_baseline > 0 else 0)
            changed_perc = row.get('data_changed_perc', (data_changed / total_baseline * 100) if total_baseline > 0 else 0)
            
            # Calculate net change and unchanged percentage
            net_change = added - removed
            unchanged_perc = 100 - changed_perc - added_perc - removed_perc
            if unchanged_perc < 0:
                unchanged_perc = 0
            
            # High removal rate
            if removed_perc >= self.thresholds.high_removal_critical:
                severity = Severity.CRITICAL
            elif removed_perc >= self.thresholds.high_removal_warning:
                severity = Severity.WARNING
            else:
                severity = None
            
            if severity:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.HIGH_REMOVAL_RATE,
                    severity=severity,
                    theme=theme,
                    feature_type=feature_type,
                    description=f"{removed_perc:.2f}% of records removed ({removed:,} features)",
                    metric_name="removed_perc",
                    previous_value=total_baseline,
                    current_value=total_baseline - removed,
                    percent_change=-removed_perc,
                    release_current=release_current,
                    release_previous=release_previous,
                    additional_context={"removed_count": removed}
                ))
            
            # High addition rate
            if added_perc >= self.thresholds.high_addition_critical:
                severity = Severity.CRITICAL
            elif added_perc >= self.thresholds.high_addition_warning:
                severity = Severity.WARNING
            else:
                severity = None
            
            if severity:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.HIGH_ADDITION_RATE,
                    severity=severity,
                    theme=theme,
                    feature_type=feature_type,
                    description=f"{added_perc:.2f}% new records ({added:,} features)",
                    metric_name="added_perc",
                    previous_value=total_baseline,
                    current_value=total_baseline + added,
                    percent_change=added_perc,
                    release_current=release_current,
                    release_previous=release_previous,
                    additional_context={"added_count": added}
                ))
            
            # Net feature loss/gain (significant absolute changes)
            if abs(net_change) >= self.thresholds.net_change_critical:
                severity = Severity.CRITICAL
            elif abs(net_change) >= self.thresholds.net_change_warning:
                severity = Severity.WARNING
            else:
                severity = None
            
            if severity:
                if net_change < 0:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.NET_FEATURE_LOSS,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        description=f"Net loss of {abs(net_change):,} features ({added:,} added, {removed:,} removed)",
                        metric_name="net_change",
                        previous_value=total_baseline,
                        current_value=total_baseline + net_change,
                        percent_change=(net_change / total_baseline * 100) if total_baseline > 0 else 0,
                        release_current=release_current,
                        release_previous=release_previous,
                        additional_context={
                            "added": added,
                            "removed": removed,
                            "net_change": net_change
                        }
                    ))
                else:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.NET_FEATURE_GAIN,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        description=f"Net gain of {net_change:,} features ({added:,} added, {removed:,} removed)",
                        metric_name="net_change",
                        previous_value=total_baseline,
                        current_value=total_baseline + net_change,
                        percent_change=(net_change / total_baseline * 100) if total_baseline > 0 else 0,
                        release_current=release_current,
                        release_previous=release_previous,
                        additional_context={
                            "added": added,
                            "removed": removed,
                            "net_change": net_change
                        }
                    ))
            
            # High churn (both additions and removals)
            churn_rate = added_perc + removed_perc
            if churn_rate >= self.thresholds.churn_critical:
                severity = Severity.CRITICAL
            elif churn_rate >= self.thresholds.churn_warning:
                severity = Severity.WARNING
            else:
                severity = None
            
            if severity:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.CHURN_SPIKE,
                    severity=severity,
                    theme=theme,
                    feature_type=feature_type,
                    description=f"High churn: {added_perc:.2f}% added, {removed_perc:.2f}% removed",
                    metric_name="churn_rate",
                    previous_value=total_baseline,
                    current_value=total_baseline + added - removed,
                    percent_change=churn_rate,
                    release_current=release_current,
                    release_previous=release_previous,
                    additional_context={"added": added, "removed": removed, "data_changed": data_changed}
                ))
            
            # Data regeneration detection (very low unchanged percentage)
            if unchanged_perc <= self.thresholds.data_regeneration_threshold:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.DATA_REGENERATION,
                    severity=Severity.CRITICAL,
                    theme=theme,
                    feature_type=feature_type,
                    description=f"Complete data rebuild detected: only {unchanged_perc:.1f}% unchanged ({changed_perc:.1f}% modified, {added_perc:.1f}% added, {removed_perc:.1f}% removed)",
                    metric_name="unchanged_perc",
                    previous_value=100.0,
                    current_value=unchanged_perc,
                    percent_change=-unchanged_perc,
                    release_current=release_current,
                    release_previous=release_previous,
                    additional_context={
                        "unchanged_perc": unchanged_perc,
                        "changed_perc": changed_perc,
                        "added_perc": added_perc,
                        "removed_perc": removed_perc,
                        "interpretation": "Data was regenerated rather than incrementally updated"
                    }
                ))
            # Data quality degradation (high modification, low unchanged - less severe than regeneration)
            elif changed_perc >= self.thresholds.high_modification_critical or unchanged_perc <= self.thresholds.low_unchanged_warning:
                if changed_perc >= self.thresholds.high_modification_critical:
                    severity = Severity.CRITICAL
                elif changed_perc >= self.thresholds.high_modification_warning:
                    severity = Severity.WARNING
                else:
                    severity = Severity.WARNING
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.DATA_QUALITY_DEGRADATION,
                    severity=severity,
                    theme=theme,
                    feature_type=feature_type,
                    description=f"High data modification: {changed_perc:.2f}% changed, {unchanged_perc:.2f}% unchanged",
                    metric_name="data_changed_perc",
                    previous_value=unchanged_perc,
                    current_value=changed_perc,
                    percent_change=changed_perc,
                    release_current=release_current,
                    release_previous=release_previous,
                    additional_context={"unchanged_perc": unchanged_perc}
                ))
        
        return anomalies
    
    def detect_historical_deviations(self, release_current: str, release_previous: str) -> List[Anomaly]:
        """Detect statistical deviations from historical patterns."""
        anomalies = []
        
        historical = self.load_historical_totals()
        changelog = self.load_changelog_stats(release_current)
        
        if historical is None or changelog is None:
            return anomalies
        
        for _, curr_row in changelog.iterrows():
            theme = curr_row.get('theme', 'unknown')
            feature_type = curr_row.get('type', 'unknown')
            
            hist_subset = historical[
                (historical['theme'] == theme) & 
                (historical['type'] == feature_type)
            ].copy()
            
            if len(hist_subset) < 3:
                continue
            
            hist_subset = hist_subset.sort_values('release')
            
            if 'total_count' in hist_subset.columns:
                hist_subset['growth_rate'] = hist_subset['total_count'].pct_change() * 100
                
                mean_growth = hist_subset['growth_rate'].mean()
                std_growth = hist_subset['growth_rate'].std()
                
                if pd.notna(mean_growth) and pd.notna(std_growth) and std_growth > 0:
                    current_growth = curr_row.get('total_diff_perc', 0)
                    z_score = abs((current_growth - mean_growth) / std_growth)
                    
                    if z_score > 5:
                        severity = Severity.CRITICAL
                    elif z_score > self.thresholds.historical_std_multiplier:
                        severity = Severity.WARNING
                    else:
                        continue
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.HISTORICAL_DEVIATION,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        description=f"Growth rate deviates from historical pattern ({z_score:.1f}σ)",
                        metric_name="growth_rate_zscore",
                        previous_value=mean_growth,
                        current_value=current_growth,
                        percent_change=current_growth,
                        release_current=release_current,
                        release_previous=release_previous,
                        expected_range=(
                            mean_growth - self.thresholds.historical_std_multiplier * std_growth,
                            mean_growth + self.thresholds.historical_std_multiplier * std_growth
                        ),
                        additional_context={"z_score": z_score, "historical_mean": mean_growth, "historical_std": std_growth}
                    ))
        
        return anomalies
    
    def detect_attribute_coverage_anomalies(
        self,
        release_current: str,
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> List[Anomaly]:
        """Detect attribute coverage changes and report absolute coverage for all important fields."""
        anomalies = []
        
        df_current = self.load_column_stats(release_current, theme, feature_type)
        df_previous = self.load_column_stats(release_previous, theme, feature_type)
        
        if df_current is None:
            return anomalies
        
        try:
            current_row = df_current.iloc[0]
            current_total = current_row.get('total_count', 0)
            
            if current_total == 0:
                return anomalies
            
            # Get required/important fields for this theme/type
            required_fields = REQUIRED_FIELDS.get((theme, feature_type), [])
            
            # Also check these common important fields for all themes
            common_important_fields = ['names', 'name', 'addresses', 'address', 'class', 'subtype']
            all_important_fields = set(required_fields + common_important_fields)
            
            # Check all columns for coverage
            for col in current_row.index:
                if col in ['release', 'theme', 'type', 'total_count', 'id', 'geometry', 'bbox', 'version', 'sources']:
                    continue
                
                # Skip optional fields - these are expected to be sparse
                if col in OPTIONAL_FIELDS:
                    continue
                
                try:
                    current_filled = float(current_row.get(col, 0))
                    current_coverage = (current_filled / current_total) * 100
                    current_null_rate = 100 - current_coverage
                    
                    # Only report LOW coverage for important/required fields
                    is_important = col in all_important_fields or col in required_fields
                    
                    if is_important:  # Only flag important fields, not all >80% null
                        if current_coverage < self.thresholds.low_coverage_critical:
                            severity = Severity.CRITICAL
                        elif current_coverage < self.thresholds.low_coverage_warning:
                            severity = Severity.WARNING
                        else:
                            severity = None
                        
                        if severity:
                            anomalies.append(Anomaly(
                                anomaly_type=AnomalyType.LOW_ATTRIBUTE_COVERAGE,
                                severity=severity,
                                theme=theme,
                                feature_type=feature_type,
                                column=col,
                                description=f"{current_null_rate:.1f}% of {feature_type}s missing '{col}' ({current_coverage:.1f}% coverage)",
                                metric_name=f"{col}_coverage",
                                previous_value=0,
                                current_value=current_coverage,
                                percent_change=0,
                                release_current=release_current,
                                release_previous=release_previous,
                                additional_context={
                                    "field": col,
                                    "coverage_pct": round(current_coverage, 2),
                                    "null_pct": round(current_null_rate, 2),
                                    "filled_count": int(current_filled),
                                    "total_count": int(current_total),
                                    "is_required_field": col in required_fields
                                }
                            ))
                    
                    # Compare to previous release if available
                    if df_previous is not None:
                        previous_row = df_previous.iloc[0]
                        previous_total = previous_row.get('total_count', 0)
                        
                        if previous_total > 0 and col in previous_row:
                            previous_filled = float(previous_row.get(col, 0))
                            previous_coverage = (previous_filled / previous_total) * 100
                            
                            coverage_change = current_coverage - previous_coverage
                            
                            # Coverage drop
                            if coverage_change <= -self.thresholds.coverage_drop_critical:
                                severity = Severity.CRITICAL
                            elif coverage_change <= -self.thresholds.coverage_drop_warning:
                                severity = Severity.WARNING
                            else:
                                severity = None
                            
                            if severity:
                                anomalies.append(Anomaly(
                                    anomaly_type=AnomalyType.ATTRIBUTE_COVERAGE_DROP,
                                    severity=severity,
                                    theme=theme,
                                    feature_type=feature_type,
                                    column=col,
                                    description=f"'{col}' coverage dropped by {abs(coverage_change):.1f}% ({previous_coverage:.1f}% → {current_coverage:.1f}%)",
                                    metric_name=f"{col}_coverage",
                                    previous_value=previous_coverage,
                                    current_value=current_coverage,
                                    percent_change=coverage_change,
                                    release_current=release_current,
                                    release_previous=release_previous,
                                    additional_context={
                                        "previous_null_pct": round(100 - previous_coverage, 2),
                                        "current_null_pct": round(100 - current_coverage, 2)
                                    }
                                ))
                            
                            # Coverage spike (potential data source change)
                            if coverage_change >= 10.0:
                                anomalies.append(Anomaly(
                                    anomaly_type=AnomalyType.ATTRIBUTE_COVERAGE_SPIKE,
                                    severity=Severity.INFO,
                                    theme=theme,
                                    feature_type=feature_type,
                                    column=col,
                                    description=f"'{col}' coverage increased by {coverage_change:.1f}% ({previous_coverage:.1f}% → {current_coverage:.1f}%) - possible data source change",
                                    metric_name=f"{col}_coverage",
                                    previous_value=previous_coverage,
                                    current_value=current_coverage,
                                    percent_change=coverage_change,
                                    release_current=release_current,
                                    release_previous=release_previous
                                ))
                
                except (ValueError, TypeError):
                    continue
        
        except Exception as e:
            pass
        
        return anomalies
    
    def detect_geometry_anomalies(
        self,
        release_current: str,
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> List[Anomaly]:
        """Detect geometry length and size anomalies."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        # Check geometry length changes
        if 'total_geometry_length_km' in df_current.columns:
            group_cols = [col for col in ['subtype', 'class', 'country'] 
                          if col in df_current.columns and col in df_previous.columns]
            
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
                    
                    if previous_len < 100:
                        continue
                    
                    pct_change = self._calculate_percent_change(previous_len, current_len)
                    
                    if pct_change <= self.thresholds.length_drop_critical:
                        severity = Severity.CRITICAL
                    elif pct_change <= self.thresholds.length_drop_warning:
                        severity = Severity.WARNING
                    else:
                        continue
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.GEOMETRY_LENGTH_CHANGE,
                        severity=severity,
                        theme=theme,
                        feature_type=feature_type,
                        subtype=row.get('subtype'),
                        class_name=row.get('class'),
                        country=row.get('country'),
                        description="Total geometry length changed significantly",
                        metric_name="total_geometry_length_km",
                        previous_value=previous_len,
                        current_value=current_len,
                        percent_change=pct_change,
                        release_current=release_current,
                        release_previous=release_previous
                    ))
        
        # Check average geometry area changes (for polygons)
        for area_col in ['avg_area_km2', 'mean_area', 'avg_geometry_area']:
            if area_col in df_current.columns and area_col in df_previous.columns:
                current_avg = df_current[area_col].mean()
                previous_avg = df_previous[area_col].mean()
                
                if previous_avg > 0:
                    pct_change = self._calculate_percent_change(previous_avg, current_avg)
                    
                    if abs(pct_change) > self.thresholds.geometry_size_change_threshold:
                        direction = "smaller" if pct_change < 0 else "larger"
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.GEOMETRY_SIZE_ANOMALY,
                            severity=Severity.WARNING,
                            theme=theme,
                            feature_type=feature_type,
                            description=f"New {feature_type} geometries {abs(pct_change):.0f}% {direction} than existing",
                            metric_name=area_col,
                            previous_value=previous_avg,
                            current_value=current_avg,
                            percent_change=pct_change,
                            release_current=release_current,
                            release_previous=release_previous,
                            additional_context={"interpretation": f"May indicate unit conversion errors, data source changes, or processing issues"}
                        ))
                break
        
        return anomalies
    
    def detect_category_distribution_shifts(
        self,
        release_current: str,
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> List[Anomaly]:
        """Detect shifts in category distributions (subtype/class proportions)."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        if 'subtype' not in df_current.columns or 'id_count' not in df_current.columns:
            return anomalies
        
        current_total = df_current['id_count'].sum()
        previous_total = df_previous['id_count'].sum()
        
        if current_total == 0 or previous_total == 0:
            return anomalies
        
        current_props = df_current.groupby('subtype')['id_count'].sum() / current_total * 100
        previous_props = df_previous.groupby('subtype')['id_count'].sum() / previous_total * 100
        
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
    
    def detect_geographic_concentration(
        self,
        release_current: str,
        release_previous: str,
        theme: str,
        feature_type: str
    ) -> List[Anomaly]:
        """Detect when changes are concentrated in specific countries."""
        anomalies = []
        
        df_current = self.load_row_counts(release_current, theme, feature_type)
        df_previous = self.load_row_counts(release_previous, theme, feature_type)
        
        if df_current is None or df_previous is None:
            return anomalies
        
        if 'country' not in df_current.columns or 'id_count' not in df_current.columns:
            return anomalies
        
        # Calculate changes by country
        current_by_country = df_current.groupby('country')['id_count'].sum()
        previous_by_country = df_previous.groupby('country')['id_count'].sum()
        
        # Calculate absolute changes
        all_countries = set(current_by_country.index) | set(previous_by_country.index)
        changes = {}
        
        for country in all_countries:
            curr = current_by_country.get(country, 0)
            prev = previous_by_country.get(country, 0)
            changes[country] = abs(curr - prev)
        
        total_change = sum(changes.values())
        if total_change == 0:
            return anomalies
        
        # Check for concentration
        sorted_changes = sorted(changes.items(), key=lambda x: x[1], reverse=True)
        top_country, top_change = sorted_changes[0]
        concentration = (top_change / total_change) * 100
        
        # Also check top 3 concentration
        top3_change = sum([c[1] for c in sorted_changes[:3]])
        top3_concentration = (top3_change / total_change) * 100
        
        if concentration > 50 or top3_concentration > 80:
            top_countries = sorted_changes[:5]
            
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.GEOGRAPHIC_CONCENTRATION,
                severity=Severity.INFO,
                theme=theme,
                feature_type=feature_type,
                description=f"Changes concentrated: {top_country} has {concentration:.1f}% of all changes",
                metric_name="geographic_concentration",
                previous_value=0,
                current_value=concentration,
                percent_change=concentration,
                release_current=release_current,
                release_previous=release_previous,
                additional_context={
                    "top_countries": [{"country": c[0], "change": c[1]} for c in top_countries],
                    "top3_concentration": top3_concentration
                }
            ))
        
        return anomalies
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def run_full_analysis(
        self,
        release_current: Optional[str] = None,
        release_previous: Optional[str] = None
    ) -> List[Anomaly]:
        """Run complete anomaly detection between two releases."""
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
        
        # Prepare geographic summary for AI
        print("\nPreparing geographic summary...")
        self.prepare_geographic_summary(release_current)
        
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
            
            # Geometry anomalies
            self.anomalies.extend(
                self.detect_geometry_anomalies(
                    release_current, release_previous, theme, feature_type
                )
            )
            
            # Category distribution shifts
            self.anomalies.extend(
                self.detect_category_distribution_shifts(
                    release_current, release_previous, theme, feature_type
                )
            )
            
            # Attribute coverage
            self.anomalies.extend(
                self.detect_attribute_coverage_anomalies(
                    release_current, release_previous, theme, feature_type
                )
            )
            
            # Geographic concentration
            self.anomalies.extend(
                self.detect_geographic_concentration(
                    release_current, release_previous, theme, feature_type
                )
            )
        
        # Changelog-based anomalies (removals, additions, churn, data quality)
        print("\nChecking changelog stats...")
        self.anomalies.extend(
            self.detect_changelog_anomalies(release_current, release_previous)
        )
        
        # Historical deviations
        print("\nChecking historical trends...")
        self.anomalies.extend(
            self.detect_historical_deviations(release_current, release_previous)
        )
        
        # Sort by severity
        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        self.anomalies.sort(key=lambda a: (severity_order[a.severity], a.theme, a.feature_type))
        
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
        
        critical = [a for a in self.anomalies if a.severity == Severity.CRITICAL]
        warnings = [a for a in self.anomalies if a.severity == Severity.WARNING]
        info = [a for a in self.anomalies if a.severity == Severity.INFO]
        
        lines.append(f"CRITICAL: {len(critical)}")
        lines.append(f"WARNING: {len(warnings)}")
        lines.append(f"INFO: {len(info)}")
        lines.append("")
        
        # Summary by anomaly type
        lines.append("By Anomaly Type:")
        type_counts = {}
        for a in self.anomalies:
            atype = a.anomaly_type.value
            type_counts[atype] = type_counts.get(atype, 0) + 1
        for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {atype}: {count}")
        lines.append("")
        
        if critical:
            lines.append("-" * 70)
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 70)
            for anomaly in critical[:30]:
                lines.append("")
                lines.append(str(anomaly))
            if len(critical) > 30:
                lines.append(f"\n  ... and {len(critical) - 30} more critical issues")
        
        if warnings:
            lines.append("")
            lines.append("-" * 70)
            lines.append("WARNINGS")
            lines.append("-" * 70)
            for anomaly in warnings[:30]:
                lines.append("")
                lines.append(str(anomaly))
            if len(warnings) > 30:
                lines.append(f"\n  ... and {len(warnings) - 30} more warnings")
        
        if info:
            lines.append("")
            lines.append("-" * 70)
            lines.append("INFORMATIONAL")
            lines.append("-" * 70)
            for anomaly in info[:20]:
                lines.append("")
                lines.append(str(anomaly))
            if len(info) > 20:
                lines.append(f"\n  ... and {len(info) - 20} more info items")
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
        
        return report
    
    def export_json(self, output_path: Path) -> None:
        """Export anomalies and geographic summary to JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_anomalies": len(self.anomalies),
            "summary": {
                "critical": len([a for a in self.anomalies if a.severity == Severity.CRITICAL]),
                "warning": len([a for a in self.anomalies if a.severity == Severity.WARNING]),
                "info": len([a for a in self.anomalies if a.severity == Severity.INFO]),
                "by_type": {}
            },
            "anomalies": [a.to_dict() for a in self.anomalies],
            "geographic_summary": self.geographic_summary
        }
        
        for a in self.anomalies:
            atype = a.anomaly_type.value
            data["summary"]["by_type"][atype] = data["summary"]["by_type"].get(atype, 0) + 1
        
        output_path.write_text(json.dumps(data, indent=2))
    
    def export_excel(self, output_path: Path) -> None:
        """Export anomalies to Excel with multiple sheets."""
        # Convert anomalies to dataframe
        records = [a.to_dict() for a in self.anomalies]
        df = pd.DataFrame(records)
        
        if df.empty:
            print("No anomalies to export")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # All anomalies
            df.to_excel(writer, sheet_name='All Anomalies', index=False)
            
            # Summary by severity
            severity_summary = df.groupby('severity').size().reset_index(name='count')
            severity_summary.to_excel(writer, sheet_name='By Severity', index=False)
            
            # Summary by type
            type_summary = df.groupby('anomaly_type').size().reset_index(name='count')
            type_summary = type_summary.sort_values('count', ascending=False)
            type_summary.to_excel(writer, sheet_name='By Type', index=False)
            
            # Summary by theme
            theme_summary = df.groupby('theme').size().reset_index(name='count')
            theme_summary = theme_summary.sort_values('count', ascending=False)
            theme_summary.to_excel(writer, sheet_name='By Theme', index=False)
            
            # Critical only
            critical = df[df['severity'] == 'critical']
            if not critical.empty:
                critical.to_excel(writer, sheet_name='Critical Only', index=False)
            
            # By country (if available)
            if 'country' in df.columns:
                country_summary = df.groupby('country').size().reset_index(name='count')
                country_summary = country_summary.sort_values('count', ascending=False)
                country_summary.to_excel(writer, sheet_name='By Country', index=False)
        
        print(f"Excel exported to: {output_path}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect anomalies in Overture Maps releases")
    parser.add_argument("data_root", type=Path, help="Root directory containing metrics/")
    parser.add_argument("--current", type=str, help="Current release (e.g., 2025-09-24.0)")
    parser.add_argument("--previous", type=str, help="Previous release to compare against")
    parser.add_argument("--output", type=Path, help="Output file for text report")
    parser.add_argument("--json", type=Path, help="Output JSON file")
    parser.add_argument("--excel", type=Path, help="Output Excel file")
    
    # Threshold arguments
    parser.add_argument("--count-drop-warning", type=float, default=-10.0)
    parser.add_argument("--count-drop-critical", type=float, default=-20.0)
    parser.add_argument("--count-spike-warning", type=float, default=20.0)
    parser.add_argument("--count-spike-critical", type=float, default=50.0)
    
    args = parser.parse_args()
    
    thresholds = AnomalyThresholds(
        count_drop_warning=args.count_drop_warning,
        count_drop_critical=args.count_drop_critical,
        count_spike_warning=args.count_spike_warning,
        count_spike_critical=args.count_spike_critical,
    )
    
    detector = OvertureAnomalyDetector(args.data_root, thresholds)
    anomalies = detector.run_full_analysis(args.current, args.previous)
    
    report = detector.generate_report(args.output)
    print(report)
    
    if args.json:
        detector.export_json(args.json)
        print(f"\nJSON exported to: {args.json}")
    
    if args.excel:
        detector.export_excel(args.excel)
        print(f"\nExcel exported to: {args.excel}")