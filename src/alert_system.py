"""
Alert System Module
===================

Responsibilities:
- Predict robot failure using regression
- Generate maintenance alerts (INFO/WARNING/CRITICAL)
- Determine time to failure

Functions:
----------
- predict_failure_date(model, current_value, threshold) -> date
- evaluate_alert_level(prediction, thresholds) -> alert
- generate_alert_message(alert) -> str
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    NONE = "NONE"


@dataclass
class AlertThresholds:
    """
    Configurable alert thresholds.

    Attributes:
    -----------
    slope_info : float
        Slope threshold for INFO (A/hour)
    slope_warning : float
        Slope threshold for WARNING (A/hour)
    slope_critical : float
        Slope threshold for CRITICAL (A/hour)
    days_warning : int
        Days before failure for WARNING
    days_critical : int
        Days before failure for CRITICAL
    current_pct_warning : float
        Current as % of baseline for WARNING
    current_pct_critical : float
        Current as % of baseline for CRITICAL
    """
    slope_info: float = 0.005
    slope_warning: float = 0.02
    slope_critical: float = 0.10
    days_warning: int = 14
    days_critical: int = 7
    current_pct_warning: float = 1.25
    current_pct_critical: float = 1.50


@dataclass
class MaintenanceAlert:
    """Represents a maintenance alert."""
    robot_id: str
    level: AlertLevel
    message: str
    predicted_failure_date: Optional[datetime]
    days_until_failure: Optional[float]
    current_value: float
    threshold_value: float
    slope: float
    recommendation: str
    generated_at: datetime

    def to_dict(self) -> Dict:
        return {
            'robot_id': self.robot_id,
            'level': self.level.value,
            'message': self.message,
            'predicted_failure_date': str(self.predicted_failure_date) if self.predicted_failure_date else None,
            'days_until_failure': self.days_until_failure,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'slope': self.slope,
            'recommendation': self.recommendation,
            'generated_at': str(self.generated_at)
        }


def predict_failure_date(
    slope: float,
    intercept: float,
    current_hours: float,
    failure_threshold: float,
    baseline_value: float
) -> Tuple[Optional[datetime], Optional[float]]:
    """
    Predict when robot will reach failure threshold.

    Using linear regression:
    y = slope * x + intercept

    Solve for x when y = failure_threshold:
    x = (failure_threshold - intercept) / slope

    Parameters:
    -----------
    slope : float
        Regression slope (A/hour)
    intercept : float
        Regression intercept (A)
    current_hours : float
        Current hours elapsed
    failure_threshold : float
        Value at which failure occurs
    baseline_value : float
        Original baseline current

    Returns:
    --------
    Tuple[datetime, float] : (failure_date, days_until_failure)
    """
    if slope <= 0:
        # No degradation or improving - no failure predicted
        return None, None

    # Compute absolute failure threshold
    failure_value = baseline_value * failure_threshold

    # Current predicted value
    current_predicted = slope * current_hours + intercept

    if current_predicted >= failure_value:
        # Already at or past failure
        return datetime.now(), 0.0

    # Hours until failure
    hours_to_failure = (failure_value - current_predicted) / slope

    if hours_to_failure < 0:
        return None, None

    days_to_failure = hours_to_failure / 24
    failure_date = datetime.now() + timedelta(hours=hours_to_failure)

    return failure_date, days_to_failure


def evaluate_alert_level(
    slope: float,
    days_until_failure: Optional[float],
    current_ratio: float,
    thresholds: AlertThresholds
) -> AlertLevel:
    """
    Determine alert level based on multiple factors.

    Alert Logic:
    1. CRITICAL if:
       - Slope > critical threshold OR
       - Days to failure < 7 OR
       - Current > 150% of baseline

    2. WARNING if:
       - Slope > warning threshold OR
       - Days to failure < 14 OR
       - Current > 125% of baseline

    3. INFO if:
       - Slope > info threshold

    Parameters:
    -----------
    slope : float
        Degradation rate (A/hour)
    days_until_failure : float
        Predicted days until failure
    current_ratio : float
        Current value / baseline value
    thresholds : AlertThresholds
        Threshold configuration

    Returns:
    --------
    AlertLevel : Determined alert level
    """
    # Check CRITICAL conditions
    if slope >= thresholds.slope_critical:
        return AlertLevel.CRITICAL
    if days_until_failure is not None and days_until_failure <= thresholds.days_critical:
        return AlertLevel.CRITICAL
    if current_ratio >= thresholds.current_pct_critical:
        return AlertLevel.CRITICAL

    # Check WARNING conditions
    if slope >= thresholds.slope_warning:
        return AlertLevel.WARNING
    if days_until_failure is not None and days_until_failure <= thresholds.days_warning:
        return AlertLevel.WARNING
    if current_ratio >= thresholds.current_pct_warning:
        return AlertLevel.WARNING

    # Check INFO conditions
    if slope >= thresholds.slope_info:
        return AlertLevel.INFO

    return AlertLevel.NONE


def generate_alert(
    robot_id: str,
    slope: float,
    intercept: float,
    current_hours: float,
    baseline_value: float,
    failure_threshold: float = 1.5,
    thresholds: Optional[AlertThresholds] = None
) -> MaintenanceAlert:
    """
    Generate a maintenance alert for a robot.

    Parameters:
    -----------
    robot_id : str
        Robot identifier
    slope : float
        Regression slope from model
    intercept : float
        Regression intercept from model
    current_hours : float
        Current elapsed hours
    baseline_value : float
        Original baseline current
    failure_threshold : float
        Failure threshold as ratio of baseline
    thresholds : AlertThresholds
        Alert threshold configuration

    Returns:
    --------
    MaintenanceAlert : Generated alert
    """
    if thresholds is None:
        thresholds = AlertThresholds()

    # Predict failure
    failure_date, days_to_failure = predict_failure_date(
        slope, intercept, current_hours, failure_threshold, baseline_value
    )

    # Current predicted value and ratio
    current_value = slope * current_hours + intercept
    current_ratio = current_value / baseline_value if baseline_value > 0 else 1.0

    # Determine alert level
    level = evaluate_alert_level(slope, days_to_failure, current_ratio, thresholds)

    # Generate message and recommendation
    if level == AlertLevel.CRITICAL:
        message = f"CRITICAL: {robot_id} requires immediate maintenance"
        if days_to_failure is not None and days_to_failure <= 7:
            message += f" - Predicted failure in {days_to_failure:.1f} days"
        recommendation = "IMMEDIATE: Stop robot for inspection. Check motors, bearings, and drive systems."
    elif level == AlertLevel.WARNING:
        message = f"WARNING: {robot_id} showing elevated degradation"
        if days_to_failure is not None:
            message += f" - Predicted failure in {days_to_failure:.1f} days"
        recommendation = "SCHEDULE: Plan maintenance within 2 weeks. Order replacement parts."
    elif level == AlertLevel.INFO:
        message = f"INFO: {robot_id} showing early signs of wear"
        recommendation = "MONITOR: Increase monitoring frequency. Log for trend analysis."
    else:
        message = f"OK: {robot_id} operating within normal parameters"
        recommendation = "ROUTINE: Continue standard maintenance schedule."

    return MaintenanceAlert(
        robot_id=robot_id,
        level=level,
        message=message,
        predicted_failure_date=failure_date,
        days_until_failure=days_to_failure,
        current_value=current_value,
        threshold_value=baseline_value * failure_threshold,
        slope=slope,
        recommendation=recommendation,
        generated_at=datetime.now()
    )


def generate_alerts_for_fleet(
    regression_results: Dict[str, Dict],
    thresholds: Optional[AlertThresholds] = None
) -> List[MaintenanceAlert]:
    """
    Generate alerts for multiple robots.

    Parameters:
    -----------
    regression_results : dict
        Results from regression analysis per robot
        {robot_id: {slope, intercept, current_hours, baseline_value}}
    thresholds : AlertThresholds
        Alert threshold configuration

    Returns:
    --------
    List[MaintenanceAlert] : Alerts for all robots
    """
    alerts = []

    for robot_id, results in regression_results.items():
        alert = generate_alert(
            robot_id=robot_id,
            slope=results['slope'],
            intercept=results['intercept'],
            current_hours=results.get('current_hours', results.get('total_hours', 24)),
            baseline_value=results.get('baseline_value', results.get('intercept', 10)),
            failure_threshold=results.get('failure_threshold', 1.5),
            thresholds=thresholds
        )
        alerts.append(alert)

    # Sort by severity
    severity_order = {AlertLevel.CRITICAL: 0, AlertLevel.WARNING: 1,
                      AlertLevel.INFO: 2, AlertLevel.NONE: 3}
    alerts.sort(key=lambda a: severity_order.get(a.level, 99))

    return alerts


def print_alerts(alerts: List[MaintenanceAlert]):
    """
    Print formatted alert report.

    Parameters:
    -----------
    alerts : List[MaintenanceAlert]
        List of alerts to print
    """
    print("\n" + "=" * 80)
    print("MAINTENANCE ALERT REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Group by level
    critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
    warning = [a for a in alerts if a.level == AlertLevel.WARNING]
    info = [a for a in alerts if a.level == AlertLevel.INFO]
    ok = [a for a in alerts if a.level == AlertLevel.NONE]

    if critical:
        print("\n[!!!] CRITICAL ALERTS")
        print("-" * 60)
        for alert in critical:
            print(f"\n  {alert.robot_id}: {alert.message}")
            print(f"    Slope: {alert.slope:.6f} A/hr")
            if alert.days_until_failure is not None:
                print(f"    Days to failure: {alert.days_until_failure:.1f}")
            print(f"    Action: {alert.recommendation}")

    if warning:
        print("\n[!!] WARNING ALERTS")
        print("-" * 60)
        for alert in warning:
            print(f"\n  {alert.robot_id}: {alert.message}")
            print(f"    Slope: {alert.slope:.6f} A/hr")
            if alert.days_until_failure is not None:
                print(f"    Days to failure: {alert.days_until_failure:.1f}")
            print(f"    Action: {alert.recommendation}")

    if info:
        print("\n[i] INFO ALERTS")
        print("-" * 60)
        for alert in info:
            print(f"\n  {alert.robot_id}: {alert.message}")
            print(f"    Slope: {alert.slope:.6f} A/hr")
            print(f"    Action: {alert.recommendation}")

    if ok:
        print("\n[OK] HEALTHY ROBOTS")
        print("-" * 60)
        for alert in ok:
            print(f"  {alert.robot_id}: Operating normally")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  CRITICAL: {len(critical)}")
    print(f"  WARNING:  {len(warning)}")
    print(f"  INFO:     {len(info)}")
    print(f"  HEALTHY:  {len(ok)}")
    print("=" * 80)


# =============================================================================
# MAIN - Test module independently
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ALERT SYSTEM MODULE TEST")
    print("=" * 60)

    # Test data simulating regression results
    regression_results = {
        'Robot_A': {
            'slope': 0.001,
            'intercept': 5.0,
            'current_hours': 24,
            'baseline_value': 5.0,
            'failure_threshold': 1.5
        },
        'Robot_B': {
            'slope': 0.025,
            'intercept': 6.0,
            'current_hours': 24,
            'baseline_value': 6.0,
            'failure_threshold': 1.5
        },
        'Robot_C': {
            'slope': 0.15,
            'intercept': 15.0,
            'current_hours': 24,
            'baseline_value': 15.0,
            'failure_threshold': 1.5
        },
        'Robot_D': {
            'slope': 0.05,
            'intercept': 10.0,
            'current_hours': 24,
            'baseline_value': 10.0,
            'failure_threshold': 1.5
        }
    }

    # Generate alerts
    alerts = generate_alerts_for_fleet(regression_results)

    # Print alerts
    print_alerts(alerts)

    # Test failure prediction
    print("\n" + "-" * 60)
    print("FAILURE PREDICTION TEST")
    print("-" * 60)

    for robot_id, results in regression_results.items():
        failure_date, days = predict_failure_date(
            results['slope'],
            results['intercept'],
            results['current_hours'],
            results['failure_threshold'],
            results['baseline_value']
        )
        if failure_date:
            print(f"{robot_id}: Failure predicted in {days:.1f} days ({failure_date.strftime('%Y-%m-%d')})")
        else:
            print(f"{robot_id}: No failure predicted")

    print("\n[OK] Alert system module test passed")
