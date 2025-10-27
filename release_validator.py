import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import glob
import os

class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class Anomaly:
    theme: str
    type_: str
    rule: str
    severity: Severity
    message: str
    actual_value: float
    expected_range: Tuple[float, float] = None
    subtype: str = None
    class_: str = None
    column: str = None
    
    def __str__(self):
        location = f"{self.theme}/{self.type_}"
        if self.subtype:
            location += f"/{self.subtype}"
        if self.class_:
            location += f"/{self.class_}"
        if self.column:
            location += f" [{self.column}]"
            
        if self.expected_range:
            return f"[{self.severity.value}] {location}: {self.message} (actual: {self.actual_value:.2f}, expected: {self.expected_range[0]:.2f}-{self.expected_range[1]:.2f})"
        return f"[{self.severity.value}] {location}: {self.message} (value: {self.actual_value:.2f})"


class EnhancedReleaseValidator:
    def __init__(self, 
                 large_change_threshold: float = 10.0,
                 moderate_change_threshold: float = 5.0,
                 high_removal_threshold: float = 1.0,
                 high_addition_threshold: float = 10.0,
                 historical_std_multiplier: float = 3.0,
                 attribute_coverage_drop_threshold: float = 10.0,
                 class_shift_threshold: float = 20.0):
        """
        Initialize the enhanced validator with configurable thresholds.
        """
        self.large_change_threshold = large_change_threshold
        self.moderate_change_threshold = moderate_change_threshold
        self.high_removal_threshold = high_removal_threshold
        self.high_addition_threshold = high_addition_threshold
        self.historical_std_multiplier = historical_std_multiplier
        self.attribute_coverage_drop_threshold = attribute_coverage_drop_threshold
        self.class_shift_threshold = class_shift_threshold
    
    @staticmethod
    def load_all_data(base_dir: str = '.') -> Dict[str, pd.DataFrame]:
        """
        Load all available data sources for comprehensive validation.
        
        Returns:
            Dictionary with keys: 'release_stats', 'historical_totals', 
            'column_stats', 'class_stats', 'class_comparisons'
        """
        data = {}
        
        # 1. Current release statistics (release-to-release comparison)
        stats_file = os.path.join(base_dir, 'part-00000-e026a40b-54d9-40e1-88da-120888a6928a-c000.csv')
        if os.path.exists(stats_file):
            data['release_stats'] = pd.read_csv(stats_file, sep='\t')
            print(f"✓ Loaded release stats: {len(data['release_stats'])} theme/type combinations")
        
        # 2. Historical total counts
        hist_dir = os.path.join(base_dir, 'Metrics', 'theme_column_summary_stats', 'release_to_release_comparisons')
        if os.path.exists(hist_dir):
            data['historical_totals'] = EnhancedReleaseValidator._load_historical_files(hist_dir)
        
        # 3. Column-level statistics (for attribute coverage)
        col_stats_dir = os.path.join(base_dir, 'Metrics', 'theme_column_summary_stats')
        if os.path.exists(col_stats_dir):
            data['column_stats'] = EnhancedReleaseValidator._load_column_stats(col_stats_dir)
        
        # 3b. Historical column statistics (for attribute coverage trends)
        col_hist_dir = os.path.join(base_dir, 'Metrics', 'theme_column_summary_stats', 'release_to_release_comparisons')
        if os.path.exists(col_hist_dir):
            data['column_historical'] = EnhancedReleaseValidator._load_column_historical(col_hist_dir)
        
        # 4. Class-level statistics
        class_stats_dir = os.path.join(base_dir, 'Metrics', 'theme_class_summary_stats')
        if os.path.exists(class_stats_dir):
            data['class_stats'] = EnhancedReleaseValidator._load_class_stats(class_stats_dir)
        
        # 5. Class comparison files (historical class trends)
        class_comp_dir = os.path.join(base_dir, 'Metrics', 'theme_class_summary_stats', 'release_to_release_comparisons')
        if os.path.exists(class_comp_dir):
            data['class_comparisons'] = EnhancedReleaseValidator._load_class_comparisons(class_comp_dir)
        
        return data
    
    @staticmethod
    def _load_historical_files(directory: str) -> pd.DataFrame:
        """Load historical theme/type total counts."""
        pattern = os.path.join(directory, 'theme=*.type=*.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No historical files found in {directory}")
            return None
        
        print(f"\nLoading {len(files)} historical total count files...")
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"✓ Loaded {len(combined)} historical records")
            return combined
        return None
    
    @staticmethod
    def _load_column_stats(directory: str) -> pd.DataFrame:
        """Load column-level statistics for attribute coverage analysis."""
        pattern = os.path.join(directory, '*.theme=*.type=*.csv')
        files = glob.glob(pattern)
        # Exclude subdirectories
        files = [f for f in files if 'release_to_release_comparisons' not in f and os.path.isfile(f)]
        
        if not files:
            print(f"Info: No single-release column stats files found in {directory}")
            return None
        
        print(f"\nLoading {len(files)} column statistics files...")
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, sep='\t')
                # Verify it has the expected structure
                if 'release' not in df.columns:
                    continue  # Skip files without release column
                dfs.append(df)
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"✓ Loaded column stats: {len(combined)} records across {combined['release'].nunique()} releases")
            return combined
        
        print(f"Info: No valid column stats files with 'release' column found")
        return None
    
    @staticmethod
    def _load_column_historical(directory: str) -> pd.DataFrame:
        """Load historical column-level statistics for trend analysis."""
        pattern = os.path.join(directory, 'theme=*.type=*.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No historical column stats found in {directory}")
            return None
        
        print(f"\nLoading {len(files)} historical column statistics files...")
        dfs = []
        for file in files:
            try:
                # Try comma-separated first, then tab-separated
                df = pd.read_csv(file)
                
                # Check if this file has the expected structure
                if 'release' not in df.columns:
                    # Try tab-separated
                    df = pd.read_csv(file, sep='\t')
                    if 'release' not in df.columns:
                        print(f"  ⊘ Skipping {os.path.basename(file)}: no 'release' column")
                        continue
                
                # Extract theme and type from filename if not in data
                if 'theme' not in df.columns or 'type' not in df.columns:
                    basename = os.path.basename(file)
                    parts = basename.replace('.csv', '').split('.')
                    theme = parts[0].split('=')[1]
                    type_ = parts[1].split('=')[1]
                    df['theme'] = theme
                    df['type'] = type_
                
                dfs.append(df)
                print(f"  ✓ Loaded {os.path.basename(file)}: {len(df)} records")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"✓ Loaded historical column stats: {len(combined)} records across {combined['release'].nunique()} releases")
            return combined
        
        print(f"Info: No valid historical column stats files found")
        return None
    
    @staticmethod
    def _load_class_stats(directory: str) -> pd.DataFrame:
        """Load class/subtype statistics."""
        pattern = os.path.join(directory, '*.theme=*.type=*.csv')
        files = glob.glob(pattern)
        files = [f for f in files if 'release_to_release_comparisons' not in f]
        
        if not files:
            print(f"Warning: No class stats files found in {directory}")
            return None
        
        print(f"\nLoading {len(files)} class statistics files...")
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, sep='\t')
                dfs.append(df)
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"✓ Loaded class stats: {len(combined)} records")
            return combined
        return None
    
    @staticmethod
    def _load_class_comparisons(directory: str) -> Dict[str, pd.DataFrame]:
        """Load class comparison files."""
        pattern = os.path.join(directory, 'theme=*.type=*.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No class comparison files found in {directory}")
            return None
        
        print(f"\nLoading {len(files)} class comparison files...")
        comparisons = {}
        for file in files:
            try:
                # Extract theme and type from filename
                basename = os.path.basename(file)
                parts = basename.replace('.csv', '').split('.')
                theme = parts[0].split('=')[1]
                type_ = parts[1].split('=')[1]
                key = f"{theme}/{type_}"
                
                df = pd.read_csv(file, sep='\t')
                comparisons[key] = df
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if comparisons:
            print(f"✓ Loaded {len(comparisons)} class comparison datasets")
        return comparisons
    
    def validate_release(self, data: Dict[str, pd.DataFrame]) -> List[Anomaly]:
        """
        Run comprehensive validation using all available data sources.
        """
        anomalies = []
        
        release_stats = data.get('release_stats')
        historical_totals = data.get('historical_totals')
        column_stats = data.get('column_stats')
        column_historical = data.get('column_historical')
        class_stats = data.get('class_stats')
        class_comparisons = data.get('class_comparisons')
        
        if release_stats is not None:
            print("\n" + "="*80)
            print("Running basic threshold checks...")
            print("="*80)
            anomalies.extend(self._check_large_changes(release_stats))
            anomalies.extend(self._check_high_removals(release_stats))
            anomalies.extend(self._check_high_additions(release_stats))
            anomalies.extend(self._check_high_churn(release_stats))
            anomalies.extend(self._check_data_quality_degradation(release_stats))
        
        if historical_totals is not None and release_stats is not None:
            print("\nRunning historical trend analysis...")
            anomalies.extend(self._check_historical_deviations(release_stats, historical_totals))
            anomalies.extend(self._check_churn_rate_spikes(release_stats, historical_totals))
            anomalies.extend(self._check_feature_count_velocity(release_stats, historical_totals))
        
        # Use historical column data if available, otherwise fall back to column_stats
        if column_historical is not None:
            print("\nRunning attribute coverage analysis (historical trends)...")
            anomalies.extend(self._check_attribute_coverage_trends(column_historical))
        elif column_stats is not None:
            print("\nRunning attribute coverage analysis...")
            anomalies.extend(self._check_attribute_coverage(column_stats))
        
        if class_comparisons is not None:
            print("\nRunning class distribution analysis...")
            anomalies.extend(self._check_class_shifts(class_comparisons))
        
        return sorted(anomalies, key=lambda x: (x.severity.value, x.theme, x.type_))
    
    def _check_large_changes(self, df: pd.DataFrame) -> List[Anomaly]:
        """Flag large percentage changes in total counts."""
        anomalies = []
        for _, row in df.iterrows():
            pct_change = abs(row['total_diff_perc'])
            if pct_change > self.large_change_threshold:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="Large Total Change",
                    severity=Severity.CRITICAL,
                    message=f"Total count changed by {row['total_diff_perc']:.2f}%",
                    actual_value=pct_change
                ))
            elif pct_change > self.moderate_change_threshold:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="Moderate Total Change",
                    severity=Severity.WARNING,
                    message=f"Total count changed by {row['total_diff_perc']:.2f}%",
                    actual_value=pct_change
                ))
        return anomalies
    
    def _check_high_removals(self, df: pd.DataFrame) -> List[Anomaly]:
        """Flag high removal rates."""
        anomalies = []
        for _, row in df.iterrows():
            if row['removed_perc'] > self.high_removal_threshold:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="High Removal Rate",
                    severity=Severity.WARNING,
                    message=f"{row['removed_perc']:.2f}% of records removed ({row['removed']:,})",
                    actual_value=row['removed_perc']
                ))
        return anomalies
    
    def _check_high_additions(self, df: pd.DataFrame) -> List[Anomaly]:
        """Flag unusually high addition rates."""
        anomalies = []
        for _, row in df.iterrows():
            if row['added_perc'] > self.high_addition_threshold:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="High Addition Rate",
                    severity=Severity.WARNING,
                    message=f"{row['added_perc']:.2f}% new records ({row['added']:,})",
                    actual_value=row['added_perc']
                ))
        return anomalies
    
    def _check_high_churn(self, df: pd.DataFrame) -> List[Anomaly]:
        """Flag high churn (both high additions and removals)."""
        anomalies = []
        for _, row in df.iterrows():
            if row['added_perc'] > 2.0 and row['removed_perc'] > 0.5:
                churn_rate = row['added_perc'] + row['removed_perc']
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="High Churn Rate",
                    severity=Severity.WARNING,
                    message=f"High churn: {row['added_perc']:.2f}% added, {row['removed_perc']:.2f}% removed",
                    actual_value=churn_rate
                ))
        return anomalies
    
    def _check_data_quality_degradation(self, df: pd.DataFrame) -> List[Anomaly]:
        """Check for data quality issues based on change patterns."""
        anomalies = []
        for _, row in df.iterrows():
            if row['data_changed_perc'] > 5.0:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="High Data Modification Rate",
                    severity=Severity.WARNING,
                    message=f"{row['data_changed_perc']:.2f}% of records modified (potential quality issue)",
                    actual_value=row['data_changed_perc']
                ))
        return anomalies
    
    def _check_historical_deviations(self, current: pd.DataFrame, historical: pd.DataFrame) -> List[Anomaly]:
        """Check for statistical deviations from historical patterns."""
        anomalies = []
        
        for _, curr_row in current.iterrows():
            theme, type_ = curr_row['theme'], curr_row['type']
            
            hist_subset = historical[
                (historical['theme'] == theme) & 
                (historical['type'] == type_)
            ].copy()
            
            if len(hist_subset) < 3:
                continue
            
            hist_subset = hist_subset.sort_values('release')
            hist_subset['growth_rate'] = hist_subset['total_count'].pct_change() * 100
            
            mean_growth = hist_subset['growth_rate'].mean()
            std_growth = hist_subset['growth_rate'].std()
            
            if pd.notna(mean_growth) and pd.notna(std_growth) and std_growth > 0:
                current_growth = curr_row['total_diff_perc']
                z_score = abs((current_growth - mean_growth) / std_growth)
                
                if z_score > self.historical_std_multiplier:
                    anomalies.append(Anomaly(
                        theme=theme,
                        type_=type_,
                        rule="Historical Deviation",
                        severity=Severity.CRITICAL if z_score > 5 else Severity.WARNING,
                        message=f"Growth rate deviates from historical pattern ({z_score:.1f}σ)",
                        actual_value=current_growth,
                        expected_range=(
                            mean_growth - self.historical_std_multiplier * std_growth,
                            mean_growth + self.historical_std_multiplier * std_growth
                        )
                    ))
        
        return anomalies
    
    def _check_churn_rate_spikes(self, current: pd.DataFrame, historical: pd.DataFrame) -> List[Anomaly]:
        """Detect unusual spikes in churn rate (GERS ID instability indicator)."""
        anomalies = []
        
        for _, curr_row in current.iterrows():
            theme, type_ = curr_row['theme'], curr_row['type']
            current_churn = curr_row['added_perc'] + curr_row['removed_perc']
            
            hist_subset = historical[
                (historical['theme'] == theme) & 
                (historical['type'] == type_)
            ]
            
            if len(hist_subset) < 3:
                continue
            
            # Calculate historical churn if we have the data
            if 'added' in hist_subset.columns and 'removed' in hist_subset.columns:
                hist_subset = hist_subset.copy()
                hist_subset['churn_rate'] = (
                    (hist_subset['added'] + hist_subset['removed']) / 
                    hist_subset['total_count'] * 100
                )
                
                recent_churn = hist_subset['churn_rate'].tail(3)
                avg_churn = recent_churn.mean()
                std_churn = recent_churn.std()
                
                if pd.notna(avg_churn) and pd.notna(std_churn) and std_churn > 0:
                    z_score = (current_churn - avg_churn) / std_churn
                    
                    if z_score > 3.0:
                        anomalies.append(Anomaly(
                            theme=theme,
                            type_=type_,
                            rule="Churn Rate Spike",
                            severity=Severity.CRITICAL if z_score > 5 else Severity.WARNING,
                            message=f"Unusual churn spike ({current_churn:.2f}%) may indicate GERS ID instability",
                            actual_value=current_churn,
                            expected_range=(avg_churn - 2*std_churn, avg_churn + 2*std_churn)
                        ))
        
        return anomalies
    
    def _check_feature_count_velocity(self, current: pd.DataFrame, historical: pd.DataFrame) -> List[Anomaly]:
        """Detect unexpected changes in growth velocity."""
        anomalies = []
        
        for _, curr_row in current.iterrows():
            theme, type_ = curr_row['theme'], curr_row['type']
            
            hist_subset = historical[
                (historical['theme'] == theme) & 
                (historical['type'] == type_)
            ].sort_values('release')
            
            if len(hist_subset) < 4:
                continue
            
            hist_subset = hist_subset.copy()
            hist_subset['growth_rate'] = hist_subset['total_count'].pct_change() * 100
            recent_velocities = hist_subset['growth_rate'].tail(3).dropna()
            
            if len(recent_velocities) >= 2:
                avg_velocity = recent_velocities.mean()
                std_velocity = recent_velocities.std()
                current_change = curr_row['total_diff_perc']
                
                if pd.notna(std_velocity) and std_velocity > 0:
                    velocity_deviation = abs(current_change - avg_velocity) / std_velocity
                    
                    if velocity_deviation > 3.0:
                        direction = "acceleration" if current_change > avg_velocity else "deceleration"
                        anomalies.append(Anomaly(
                            theme=theme,
                            type_=type_,
                            rule="Growth Velocity Change",
                            severity=Severity.WARNING,
                            message=f"Unexpected {direction} (current: {current_change:.2f}%, avg: {avg_velocity:.2f}%)",
                            actual_value=velocity_deviation,
                            expected_range=(avg_velocity - 2*std_velocity, avg_velocity + 2*std_velocity)
                        ))
        
        return anomalies
    
    def _check_attribute_coverage_trends(self, column_historical: pd.DataFrame) -> List[Anomaly]:
        """
        Detect attribute coverage drops using historical trend data.
        This is more robust than simple two-release comparison.
        """
        anomalies = []
        
        # Group by theme/type
        for (theme, type_), group in column_historical.groupby(['theme', 'type']):
            # Sort by release
            group = group.sort_values('release')
            
            if len(group) < 3:
                continue
            
            # Get the latest release
            latest = group.iloc[-1]
            total_count = latest['total_count']
            
            if total_count == 0:
                continue
            
            # Check each column for coverage trends
            for col in group.columns:
                if col in ['release', 'theme', 'type', 'total_count']:
                    continue
                
                try:
                    # Calculate coverage percentage over time
                    coverage_series = []
                    for _, row in group.iterrows():
                        val = float(row[col])
                        total = row['total_count']
                        coverage_pct = (val / total * 100) if total > 0 else 0
                        coverage_series.append(coverage_pct)
                    
                    if len(coverage_series) < 3:
                        continue
                    
                    # Calculate historical average and std (excluding current)
                    historical_coverage = coverage_series[:-1]
                    current_coverage = coverage_series[-1]
                    
                    # Skip if all historical values are 0 (new attribute)
                    if max(historical_coverage) == 0:
                        continue
                    
                    hist_mean = np.mean(historical_coverage)
                    hist_std = np.std(historical_coverage)
                    
                    # Check for significant drop
                    if hist_std > 0.1 and hist_mean > 1.0:  # Only check if there's variation and meaningful coverage
                        z_score = (hist_mean - current_coverage) / hist_std
                        
                        # Flag if current is significantly below historical
                        if z_score > 3.0:
                            anomalies.append(Anomaly(
                                theme=theme,
                                type_=type_,
                                column=col,
                                rule="Attribute Coverage Drop (Trend)",
                                severity=Severity.CRITICAL if z_score > 5 else Severity.WARNING,
                                message=f"'{col}' coverage dropped significantly ({current_coverage:.1f}% vs historical {hist_mean:.1f}%, {z_score:.1f}σ)",
                                actual_value=current_coverage,
                                expected_range=(hist_mean - 2*hist_std, hist_mean + 2*hist_std)
                            ))
                    
                    # Also check for absolute drop (more reliable for stable attributes)
                    if len(historical_coverage) >= 2:
                        recent_coverage = historical_coverage[-1]
                        coverage_drop = recent_coverage - current_coverage
                        
                        # Only flag if the attribute had meaningful coverage before
                        if recent_coverage > 5.0 and coverage_drop > self.attribute_coverage_drop_threshold:
                            anomalies.append(Anomaly(
                                theme=theme,
                                type_=type_,
                                column=col,
                                rule="Attribute Coverage Drop",
                                severity=Severity.CRITICAL if coverage_drop > 20 else Severity.WARNING,
                                message=f"'{col}' coverage dropped by {coverage_drop:.1f}% ({recent_coverage:.1f}% → {current_coverage:.1f}%)",
                                actual_value=current_coverage,
                                expected_range=(recent_coverage * 0.9, recent_coverage * 1.1)
                            ))
                    
                    # Check for sudden spike (potential data quality issue or spam)
                    if len(historical_coverage) >= 2:
                        recent_coverage = historical_coverage[-1]
                        coverage_increase = current_coverage - recent_coverage
                        
                        # Flag significant increases - could indicate methodology change or data source change
                        # Two scenarios: 
                        # 1. Large absolute increase (>10%) from any baseline
                        # 2. Large relative increase (>5x) from low baseline
                        if coverage_increase > 10.0:
                            severity = Severity.WARNING if coverage_increase > 20.0 else Severity.INFO
                            anomalies.append(Anomaly(
                                theme=theme,
                                type_=type_,
                                column=col,
                                rule="Attribute Coverage Spike",
                                severity=severity,
                                message=f"'{col}' coverage increased by {coverage_increase:.1f}% ({recent_coverage:.1f}% → {current_coverage:.1f}%) - possible data source/methodology change",
                                actual_value=current_coverage
                            ))
                        elif recent_coverage > 0 and current_coverage / recent_coverage > 5.0:
                            anomalies.append(Anomaly(
                                theme=theme,
                                type_=type_,
                                column=col,
                                rule="Attribute Coverage Spike (Relative)",
                                severity=Severity.INFO,
                                message=f"'{col}' coverage increased {current_coverage/recent_coverage:.1f}x ({recent_coverage:.1f}% → {current_coverage:.1f}%)",
                                actual_value=current_coverage
                            ))
                
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    continue
        
        return anomalies
    
    def _check_attribute_coverage(self, column_stats: pd.DataFrame) -> List[Anomaly]:
        """
        Detect drops in attribute coverage (e.g., fewer places with websites).
        This is a key data quality indicator.
        """
        anomalies = []
        
        # Get the latest two releases
        releases = sorted(column_stats['release'].unique(), reverse=True)
        if len(releases) < 2:
            return anomalies
        
        current_release = releases[0]
        previous_release = releases[1]
        
        current_data = column_stats[column_stats['release'] == current_release]
        previous_data = column_stats[column_stats['release'] == previous_release]
        
        for _, curr_row in current_data.iterrows():
            theme = curr_row['theme']
            type_ = curr_row['type']
            
            # Find matching previous record
            prev_row = previous_data[
                (previous_data['theme'] == theme) & 
                (previous_data['type'] == type_)
            ]
            
            if len(prev_row) == 0:
                continue
            
            prev_row = prev_row.iloc[0]
            total_curr = curr_row['total_count']
            total_prev = prev_row['total_count']
            
            if total_prev == 0:
                continue
            
            # Check each column for coverage drops
            for col in curr_row.index:
                if col in ['release', 'theme', 'type', 'total_count']:
                    continue
                
                try:
                    curr_val = float(curr_row[col])
                    prev_val = float(prev_row[col])
                    
                    # Calculate coverage percentages
                    curr_coverage = (curr_val / total_curr * 100) if total_curr > 0 else 0
                    prev_coverage = (prev_val / total_prev * 100) if total_prev > 0 else 0
                    
                    coverage_drop = prev_coverage - curr_coverage
                    
                    if coverage_drop > self.attribute_coverage_drop_threshold:
                        anomalies.append(Anomaly(
                            theme=theme,
                            type_=type_,
                            column=col,
                            rule="Attribute Coverage Drop",
                            severity=Severity.CRITICAL if coverage_drop > 20 else Severity.WARNING,
                            message=f"Attribute '{col}' coverage dropped by {coverage_drop:.1f}% ({prev_coverage:.1f}% → {curr_coverage:.1f}%)",
                            actual_value=curr_coverage,
                            expected_range=(prev_coverage * 0.9, prev_coverage * 1.1)
                        ))
                except (ValueError, TypeError):
                    continue
        
        return anomalies
    
    def _check_class_shifts(self, class_comparisons: Dict[str, pd.DataFrame]) -> List[Anomaly]:
        """
        Detect unusual shifts in class/subtype distributions.
        E.g., sudden drop in 'village' but increase in 'hamlet'.
        """
        anomalies = []
        
        for key, df in class_comparisons.items():
            theme, type_ = key.split('/')
            
            # Get last two releases
            release_cols = [col for col in df.columns if col not in ['type', 'subtype', 'class']]
            if len(release_cols) < 2:
                continue
            
            current_release = release_cols[-1]
            previous_release = release_cols[-2]
            
            for _, row in df.iterrows():
                subtype = row.get('subtype', '')
                class_ = row.get('class', '')
                
                try:
                    curr_count = float(row[current_release])
                    prev_count = float(row[previous_release])
                    
                    if prev_count == 0:
                        continue
                    
                    pct_change = ((curr_count - prev_count) / prev_count) * 100
                    
                    if abs(pct_change) > self.class_shift_threshold:
                        anomalies.append(Anomaly(
                            theme=theme,
                            type_=type_,
                            subtype=subtype,
                            class_=class_,
                            rule="Class Distribution Shift",
                            severity=Severity.WARNING,
                            message=f"Count changed by {pct_change:.1f}% ({prev_count:,.0f} → {curr_count:,.0f})",
                            actual_value=pct_change
                        ))
                except (ValueError, TypeError, KeyError):
                    continue
        
        return anomalies
    
    def generate_report(self, anomalies: List[Anomaly]) -> str:
        """Generate comprehensive validation report."""
        if not anomalies:
            return "\n" + "="*80 + "\n✓ No anomalies detected. Release looks good!\n" + "="*80
        
        report = ["\n" + "="*80]
        report.append(f"RELEASE VALIDATION REPORT - {len(anomalies)} anomalies detected")
        report.append("="*80 + "\n")
        
        # Summary by category
        report.append("SUMMARY BY RULE:")
        report.append("-"*80)
        rule_counts = {}
        for a in anomalies:
            rule_counts[a.rule] = rule_counts.get(a.rule, 0) + 1
        
        for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            report.append(f"  {rule}: {count}")
        report.append("")
        
        # Summary by severity
        critical = [a for a in anomalies if a.severity == Severity.CRITICAL]
        warnings = [a for a in anomalies if a.severity == Severity.WARNING]
        info = [a for a in anomalies if a.severity == Severity.INFO]
        
        if critical:
            report.append(f"🔴 CRITICAL ISSUES ({len(critical)}):")
            report.append("-"*80)
            for anomaly in critical[:20]:  # Limit output
                report.append(f"  {anomaly}")
            if len(critical) > 20:
                report.append(f"  ... and {len(critical) - 20} more")
            report.append("")
        
        if warnings:
            report.append(f"⚠️  WARNINGS ({len(warnings)}):")
            report.append("-"*80)
            for anomaly in warnings[:30]:
                report.append(f"  {anomaly}")
            if len(warnings) > 30:
                report.append(f"  ... and {len(warnings) - 30} more")
            report.append("")
        
        if info:
            report.append(f"ℹ️  INFORMATIONAL ({len(info)}):")
            report.append("-"*80)
            for anomaly in info[:20]:
                report.append(f"  {anomaly}")
            if len(info) > 20:
                report.append(f"  ... and {len(info) - 20} more")
            report.append("")
        
        return "\n".join(report)


# Main execution
if __name__ == "__main__":
    import os
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("ENHANCED RELEASE VALIDATOR")
    print("="*80)
    
    # Load all available data sources
    print("\nLoading data sources...")
    data = EnhancedReleaseValidator.load_all_data(base_dir)
    
    # Initialize validator
    validator = EnhancedReleaseValidator(
        large_change_threshold=10.0,
        moderate_change_threshold=5.0,
        high_removal_threshold=1.0,
        high_addition_threshold=10.0,
        historical_std_multiplier=3.0,
        attribute_coverage_drop_threshold=10.0,
        class_shift_threshold=20.0
    )
    
    # Run validation
    anomalies = validator.validate_release(data)
    
    # Generate and print report
    report = validator.generate_report(anomalies)
    print(report)
    
    # Export detailed results
    if anomalies:
        anomaly_data = []
        for a in anomalies:
            record = {
                'theme': a.theme,
                'type': a.type_,
                'subtype': a.subtype,
                'class': a.class_,
                'column': a.column,
                'severity': a.severity.value,
                'rule': a.rule,
                'message': a.message,
                'actual_value': a.actual_value,
                'expected_min': a.expected_range[0] if a.expected_range else None,
                'expected_max': a.expected_range[1] if a.expected_range else None
            }
            anomaly_data.append(record)
        
        output_path = os.path.join(base_dir, 'detected_anomalies.csv')
        pd.DataFrame(anomaly_data).to_csv(output_path, index=False)
        print(f"\n✓ Full anomaly details exported to '{output_path}'")
        print(f"✓ Total anomalies: {len(anomalies)} ({len([a for a in anomalies if a.severity == Severity.CRITICAL])} critical, {len([a for a in anomalies if a.severity == Severity.WARNING])} warnings)")