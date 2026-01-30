"""
Evaluation Module
=================

Responsibilities:
- Compute regression metrics (RMSE, MAE, R2)
- Visualize actual vs predicted
- Generate evaluation reports

Functions:
----------
- compute_metrics(y_true, y_pred) -> dict
- plot_regression(X, y, model) -> figure
- plot_residuals(y_true, y_pred) -> figure
- generate_evaluation_report(metrics, model_name) -> str
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for regression evaluation metrics."""
    rmse: float
    mae: float
    r2: float
    mse: float
    n_samples: int


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
    """
    Compute regression evaluation metrics.

    Metrics Explained:
    ------------------

    1. RMSE (Root Mean Squared Error):
       RMSE = sqrt(mean((y_true - y_pred)^2))
       - Same units as target variable
       - Penalizes large errors more than MAE
       - Lower is better

    2. MAE (Mean Absolute Error):
       MAE = mean(|y_true - y_pred|)
       - Same units as target variable
       - More robust to outliers than RMSE
       - Lower is better

    3. R2 (Coefficient of Determination):
       R2 = 1 - (SS_res / SS_tot)
       where:
         SS_res = sum((y_true - y_pred)^2)
         SS_tot = sum((y_true - mean(y_true))^2)
       - Range: (-inf, 1], 1 is perfect
       - Proportion of variance explained
       - Higher is better

    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    EvaluationMetrics : Computed metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n = len(y_true)

    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # R-squared (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return EvaluationMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        mse=mse,
        n_samples=n
    )


def print_metrics(metrics: EvaluationMetrics, model_name: str = "Model"):
    """
    Print evaluation metrics in formatted table.

    Parameters:
    -----------
    metrics : EvaluationMetrics
        Computed metrics
    model_name : str
        Name of the model for display
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION METRICS: {model_name}")
    print(f"{'='*50}")
    print(f"  RMSE:      {metrics.rmse:.6f}")
    print(f"  MAE:       {metrics.mae:.6f}")
    print(f"  R2 Score:  {metrics.r2:.6f}")
    print(f"  MSE:       {metrics.mse:.6f}")
    print(f"  Samples:   {metrics.n_samples}")
    print(f"{'='*50}")


def compare_metrics(
    metrics1: EvaluationMetrics,
    metrics2: EvaluationMetrics,
    name1: str = "Model 1",
    name2: str = "Model 2"
) -> Dict:
    """
    Compare metrics between two models.

    Parameters:
    -----------
    metrics1, metrics2 : EvaluationMetrics
        Metrics from two models
    name1, name2 : str
        Model names

    Returns:
    --------
    dict : Comparison results
    """
    comparison = {
        'rmse_diff': metrics1.rmse - metrics2.rmse,
        'mae_diff': metrics1.mae - metrics2.mae,
        'r2_diff': metrics1.r2 - metrics2.r2,
        'better_rmse': name1 if metrics1.rmse < metrics2.rmse else name2,
        'better_mae': name1 if metrics1.mae < metrics2.mae else name2,
        'better_r2': name1 if metrics1.r2 > metrics2.r2 else name2
    }

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {name1:<15} {name2:<15} {'Better':<15}")
    print("-" * 60)
    print(f"{'RMSE':<12} {metrics1.rmse:<15.6f} {metrics2.rmse:<15.6f} {comparison['better_rmse']:<15}")
    print(f"{'MAE':<12} {metrics1.mae:<15.6f} {metrics2.mae:<15.6f} {comparison['better_mae']:<15}")
    print(f"{'R2':<12} {metrics1.r2:<15.6f} {metrics2.r2:<15.6f} {comparison['better_r2']:<15}")
    print(f"{'='*60}")

    return comparison


def plot_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    model_name: str = "Linear Regression",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize regression results with train/test distinction.

    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    y_pred_train, y_pred_test : np.ndarray
        Predictions
    model_name : str
        Model name for title
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Scatter with regression line
    ax1 = axes[0]

    # Training data
    ax1.scatter(X_train, y_train, alpha=0.5, color='blue', label='Train Data', s=20)

    # Test data
    ax1.scatter(X_test, y_test, alpha=0.5, color='green', label='Test Data', s=20)

    # Regression line (using all X values)
    X_all = np.concatenate([X_train, X_test])
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])
    sort_idx = np.argsort(X_all)
    ax1.plot(X_all[sort_idx], y_pred_all[sort_idx], 'r-', linewidth=2, label='Regression Line')

    ax1.set_xlabel('X (Feature)')
    ax1.set_ylabel('y (Target)')
    ax1.set_title(f'{model_name}: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted vs Actual
    ax2 = axes[1]

    ax2.scatter(y_train, y_pred_train, alpha=0.5, color='blue', label='Train', s=20)
    ax2.scatter(y_test, y_pred_test, alpha=0.5, color='green', label='Test', s=20)

    # Perfect prediction line
    y_all = np.concatenate([y_train, y_test])
    min_val, max_val = y_all.min(), y_all.max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Predicted vs Actual Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot residual analysis.

    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name for title
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The figure object
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{model_name}: Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residual histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{model_name}: Residual Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_cost_history(
    costs: list,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training cost history (for gradient descent).

    Parameters:
    -----------
    costs : list
        Cost values per iteration
    model_name : str
        Model name for title
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(costs, color='blue', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (MSE)')
    ax.set_title(f'{model_name}: Training Cost History')
    ax.grid(True, alpha=0.3)

    # Add annotation for final cost
    ax.annotate(f'Final: {costs[-1]:.6f}',
                xy=(len(costs)-1, costs[-1]),
                xytext=(len(costs)*0.7, costs[0]*0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def generate_evaluation_report(
    metrics: EvaluationMetrics,
    model_name: str,
    feature_name: str,
    target_name: str
) -> str:
    """
    Generate text evaluation report.

    Parameters:
    -----------
    metrics : EvaluationMetrics
        Computed metrics
    model_name : str
        Model name
    feature_name : str
        Feature (X) name
    target_name : str
        Target (y) name

    Returns:
    --------
    str : Formatted report
    """
    report = f"""
================================================================================
                        MODEL EVALUATION REPORT
================================================================================

Model: {model_name}
Feature (X): {feature_name}
Target (y): {target_name}
Samples: {metrics.n_samples}

--------------------------------------------------------------------------------
                              METRICS
--------------------------------------------------------------------------------

RMSE (Root Mean Squared Error): {metrics.rmse:.6f}
  - Interpretation: Average prediction error is {metrics.rmse:.4f} units
  - Same units as target variable
  - Penalizes large errors heavily

MAE (Mean Absolute Error): {metrics.mae:.6f}
  - Interpretation: Average absolute error is {metrics.mae:.4f} units
  - More robust to outliers than RMSE

R2 Score (Coefficient of Determination): {metrics.r2:.6f}
  - Interpretation: Model explains {metrics.r2*100:.2f}% of variance
  - Range: (-inf, 1], where 1 is perfect prediction
  - {"GOOD" if metrics.r2 > 0.7 else "MODERATE" if metrics.r2 > 0.3 else "POOR"} fit

--------------------------------------------------------------------------------
                        MAINTENANCE PREDICTION UTILITY
--------------------------------------------------------------------------------

{"This model shows GOOD predictive capability for maintenance planning." if metrics.r2 > 0.5 else "This model shows LIMITED predictive capability. Consider:"}
{"" if metrics.r2 > 0.5 else "  - Adding more features (multivariate regression)"}
{"" if metrics.r2 > 0.5 else "  - Using non-linear models"}
{"" if metrics.r2 > 0.5 else "  - Collecting more training data"}

================================================================================
"""
    return report


# =============================================================================
# MAIN - Test module independently
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("EVALUATION MODULE TEST")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    y_true = np.array([3, 5, 2.5, 7, 4.5, 6, 3.5, 5.5, 4, 6.5])
    y_pred = np.array([2.8, 5.2, 2.3, 6.8, 4.7, 5.8, 3.7, 5.3, 4.2, 6.3])

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Model")

    # Generate report
    report = generate_evaluation_report(metrics, "Test Model", "elapsed_time", "mean_current")
    print(report)

    print("\n[OK] Evaluation module test passed")
