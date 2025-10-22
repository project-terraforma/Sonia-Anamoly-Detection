import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

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
    
    def __str__(self):
        if self.expected_range:
            return f"[{self.severity.value}] {self.theme}/{self.type_}: {self.message} (actual: {self.actual_value:.2f}, expected: {self.expected_range[0]:.2f}-{self.expected_range[1]:.2f})"
        return f"[{self.severity.value}] {self.theme}/{self.type_}: {self.message} (value: {self.actual_value:.2f})"


class ReleaseValidator:
    def __init__(self, 
                 large_change_threshold: float = 10.0,
                 moderate_change_threshold: float = 5.0,
                 high_removal_threshold: float = 1.0,
                 high_addition_threshold: float = 10.0,
                 historical_std_multiplier: float = 3.0):
        """
        Initialize the validator with configurable thresholds.
        
        Args:
            large_change_threshold: % change to flag as critical (default 10%)
            moderate_change_threshold: % change to flag as warning (default 5%)
            high_removal_threshold: % removed to flag as warning (default 1%)
            high_addition_threshold: % added to flag as warning (default 10%)
            historical_std_multiplier: Number of std deviations for anomaly detection (default 3)
        """
        self.large_change_threshold = large_change_threshold
        self.moderate_change_threshold = moderate_change_threshold
        self.high_removal_threshold = high_removal_threshold
        self.high_addition_threshold = high_addition_threshold
        self.historical_std_multiplier = historical_std_multiplier
    
    @staticmethod
    def load_historical_data(directory: str = '.') -> pd.DataFrame:
        """
        Load and concatenate all historical theme files.
        
        Args:
            directory: Directory containing the historical CSV files (default: current directory)
            
        Returns:
            Concatenated DataFrame with all historical data
        """
        import glob
        import os
        
        # Pattern to match historical files
        pattern = os.path.join(directory, 'theme=*.type=*.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No historical files found matching pattern '{pattern}'")
            return None
        
        print(f"Loading {len(files)} historical files...")
        historical_dfs = []
        
        for file in files:
            try:
                df = pd.read_csv(file)
                # Extract theme and type from filename for validation
                filename = os.path.basename(file)
                # e.g., "theme=places.type=place.csv" -> theme=places, type=place
                parts = filename.replace('.csv', '').split('.')
                expected_theme = parts[0].split('=')[1] if len(parts) > 0 else None
                expected_type = parts[1].split('=')[1] if len(parts) > 1 else None
                
                historical_dfs.append(df)
                print(f"  ✓ Loaded {filename}: {len(df)} records")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if not historical_dfs:
            return None
        
        # Concatenate all dataframes
        combined = pd.concat(historical_dfs, ignore_index=True)
        print(f"\n✓ Total historical records loaded: {len(combined)}")
        print(f"✓ Unique themes: {combined['theme'].nunique()}")
        print(f"✓ Unique types: {combined['type'].nunique()}")
        print(f"✓ Release range: {combined['release'].min()} to {combined['release'].max()}")
        
        return combined
        
    def validate_release(self, 
                        current_stats: pd.DataFrame,
                        historical_data: pd.DataFrame = None) -> List[Anomaly]:
        """
        Validate a release using rule-based checks.
        
        Args:
            current_stats: DataFrame with current release statistics
            historical_data: Optional DataFrame with historical release data for context
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Rule 1: Large total changes
        anomalies.extend(self._check_large_changes(current_stats))
        
        # Rule 2: High removal rates
        anomalies.extend(self._check_high_removals(current_stats))
        
        # Rule 3: High addition rates
        anomalies.extend(self._check_high_additions(current_stats))
        
        # Rule 4: Suspicious patterns (high churn)
        anomalies.extend(self._check_high_churn(current_stats))
        
        # Rule 5: Unexpected stability (0 changes when changes expected)
        anomalies.extend(self._check_unexpected_stability(current_stats))
        
        # Rule 6: Negative growth with additions
        anomalies.extend(self._check_negative_growth_paradox(current_stats))
        
        # Rule 7: Historical comparison (if data available)
        if historical_data is not None:
            anomalies.extend(self._check_historical_anomalies(current_stats, historical_data))
        
        return sorted(anomalies, key=lambda x: (x.severity.value, x.theme))
    
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
                    message=f"{row['removed_perc']:.2f}% of records removed ({row['removed']:,} records)",
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
                    message=f"{row['added_perc']:.2f}% new records added ({row['added']:,} records)",
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
                    message=f"High churn detected: {row['added_perc']:.2f}% added, {row['removed_perc']:.2f}% removed",
                    actual_value=churn_rate
                ))
        return anomalies
    
    def _check_unexpected_stability(self, df: pd.DataFrame) -> List[Anomaly]:
        """Flag themes with 0% change when some change is typical."""
        anomalies = []
        # Themes that typically never change
        stable_themes = {'addresses', 'bathymetry', 'land_cover'}
        
        for _, row in df.iterrows():
            if row['theme'] not in stable_themes and row['total_diff_perc'] == 0.0:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="Unexpected Stability",
                    severity=Severity.INFO,
                    message="No changes detected, but changes are typically expected for this theme",
                    actual_value=0.0
                ))
        return anomalies
    
    def _check_negative_growth_paradox(self, df: pd.DataFrame) -> List[Anomaly]:
        """Flag cases where there are additions but overall negative growth."""
        anomalies = []
        for _, row in df.iterrows():
            if row['added'] > 0 and row['total_diff_perc'] < -1.0:
                anomalies.append(Anomaly(
                    theme=row['theme'],
                    type_=row['type'],
                    rule="Negative Growth Paradox",
                    severity=Severity.WARNING,
                    message=f"Added {row['added']:,} records but total decreased by {abs(row['total_diff_perc']):.2f}%",
                    actual_value=row['total_diff_perc']
                ))
        return anomalies
    
    def _check_historical_anomalies(self, 
                                   current_stats: pd.DataFrame,
                                   historical_data: pd.DataFrame) -> List[Anomaly]:
        """Check current release against historical patterns."""
        anomalies = []
        
        # Get the latest release data for comparison
        latest_release = historical_data['release'].max()
        
        for _, current_row in current_stats.iterrows():
            theme = current_row['theme']
            type_ = current_row['type']
            
            # Get historical data for this theme/type
            hist_subset = historical_data[
                (historical_data['theme'] == theme) & 
                (historical_data['type'] == type_)
            ].copy()
            
            if len(hist_subset) < 3:  # Need at least 3 historical points
                continue
            
            # Calculate historical growth rates
            hist_subset = hist_subset.sort_values('release')
            hist_subset['growth_rate'] = hist_subset['total_count'].pct_change() * 100
            
            # Statistical comparison
            mean_growth = hist_subset['growth_rate'].mean()
            std_growth = hist_subset['growth_rate'].std()
            
            if pd.notna(mean_growth) and pd.notna(std_growth) and std_growth > 0:
                current_growth = current_row['total_diff_perc']
                z_score = abs((current_growth - mean_growth) / std_growth)
                
                if z_score > self.historical_std_multiplier:
                    anomalies.append(Anomaly(
                        theme=theme,
                        type_=type_,
                        rule="Historical Deviation",
                        severity=Severity.CRITICAL if z_score > 5 else Severity.WARNING,
                        message=f"Growth rate deviates significantly from historical pattern ({z_score:.1f}σ)",
                        actual_value=current_growth,
                        expected_range=(
                            mean_growth - self.historical_std_multiplier * std_growth,
                            mean_growth + self.historical_std_multiplier * std_growth
                        )
                    ))
            
            # Check for sudden drops
            latest_historical = hist_subset[hist_subset['release'] == latest_release]['total_count'].values
            if len(latest_historical) > 0:
                latest_count = latest_historical[0]
                current_count = current_row['total_current']
                
                if current_count < latest_count * 0.5:  # More than 50% drop
                    anomalies.append(Anomaly(
                        theme=theme,
                        type_=type_,
                        rule="Sudden Drop",
                        severity=Severity.CRITICAL,
                        message=f"Total count dropped by more than 50% from previous release",
                        actual_value=(current_count - latest_count) / latest_count * 100
                    ))
        
        return anomalies
    
    def generate_report(self, anomalies: List[Anomaly]) -> str:
        """Generate a human-readable report of anomalies."""
        if not anomalies:
            return "✓ No anomalies detected. Release looks good!"
        
        report = ["=" * 80]
        report.append(f"RELEASE VALIDATION REPORT - {len(anomalies)} anomalies detected")
        report.append("=" * 80)
        report.append("")
        
        # Group by severity
        critical = [a for a in anomalies if a.severity == Severity.CRITICAL]
        warnings = [a for a in anomalies if a.severity == Severity.WARNING]
        info = [a for a in anomalies if a.severity == Severity.INFO]
        
        if critical:
            report.append(f"🔴 CRITICAL ISSUES ({len(critical)}):")
            report.append("-" * 80)
            for anomaly in critical:
                report.append(f"  {anomaly}")
            report.append("")
        
        if warnings:
            report.append(f"⚠️  WARNINGS ({len(warnings)}):")
            report.append("-" * 80)
            for anomaly in warnings:
                report.append(f"  {anomaly}")
            report.append("")
        
        if info:
            report.append(f"ℹ️  INFORMATIONAL ({len(info)}):")
            report.append("-" * 80)
            for anomaly in info:
                report.append(f"  {anomaly}")
            report.append("")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    import os
    
    # Define base directory (where the script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the current release stats (in the main directory) - it's tab-separated!
    current_stats_path = os.path.join(base_dir, 'part-00000-e026a40b-54d9-40e1-88da-120888a6928a-c000.csv')
    current_stats = pd.read_csv(current_stats_path, sep='\t')
    print(f"✓ Loaded current stats: {len(current_stats)} theme/type combinations\n")
    
    # Automatically load and concatenate all historical files (in subdirectory)
    historical_dir = os.path.join(base_dir, 'Metrics', 'theme_column_summary_stats', 'release_to_release_comparisons')
    historical_data = ReleaseValidator.load_historical_data(historical_dir)
    
    # Initialize validator with custom thresholds if needed
    validator = ReleaseValidator(
        large_change_threshold=10.0,
        moderate_change_threshold=5.0,
        high_removal_threshold=1.0,
        high_addition_threshold=10.0,
        historical_std_multiplier=3.0
    )
    
    # Run validation
    print("\n" + "=" * 80)
    print("Running validation...")
    print("=" * 80)
    anomalies = validator.validate_release(current_stats, historical_data)
    
    # Generate report
    report = validator.generate_report(anomalies)
    print(report)
    
    # Export anomalies to CSV for further analysis
    if anomalies:
        anomaly_data = [{
            'theme': a.theme,
            'type': a.type_,
            'severity': a.severity.value,
            'rule': a.rule,
            'message': a.message,
            'actual_value': a.actual_value,
            'expected_min': a.expected_range[0] if a.expected_range else None,
            'expected_max': a.expected_range[1] if a.expected_range else None
        } for a in anomalies]
        
        anomaly_df = pd.DataFrame(anomaly_data)
        output_path = os.path.join(base_dir, 'detected_anomalies.csv')
        anomaly_df.to_csv(output_path, index=False)
        print(f"\n✓ Anomaly details exported to '{output_path}'")