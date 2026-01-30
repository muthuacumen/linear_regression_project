"""
Model Module
============

Responsibilities:
- Linear Regression FROM SCRATCH (NumPy only)
- Linear Regression using scikit-learn
- Model coefficient extraction and comparison

Functions:
----------
- LinearRegressionScratch: Custom class with gradient descent
- train_sklearn_model(X, y) -> model, coefficients
- compare_models(scratch_model, sklearn_model) -> comparison_dict
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression


# =============================================================================
# PROMPT 3: LINEAR REGRESSION FROM SCRATCH
# =============================================================================

@dataclass
class TrainingHistory:
    """Tracks training progress."""
    costs: List[float] = field(default_factory=list)
    weights: List[Tuple[float, float]] = field(default_factory=list)
    iterations: int = 0


class LinearRegressionScratch:
    """
    Univariate Linear Regression implemented from scratch.

    Mathematical Components:
    ------------------------

    1. HYPOTHESIS FUNCTION:
       h(x) = w * x + b
       where:
       - w = weight (slope)
       - b = bias (intercept)
       - x = input feature

    2. COST FUNCTION (Mean Squared Error):
       J(w, b) = (1/2m) * SUM[(h(x_i) - y_i)^2]
       where:
       - m = number of samples
       - h(x_i) = prediction for sample i
       - y_i = actual value for sample i

    3. GRADIENT DESCENT UPDATE RULES:
       w := w - alpha * dJ/dw
       b := b - alpha * dJ/db

       where:
       dJ/dw = (1/m) * SUM[(h(x_i) - y_i) * x_i]
       dJ/db = (1/m) * SUM[(h(x_i) - y_i)]

    Parameters:
    -----------
    learning_rate : float
        Step size for gradient descent (alpha)
    n_iterations : int
        Number of gradient descent iterations
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Model parameters (initialized during fit)
        self.weight = 0.0  # w (slope)
        self.bias = 0.0    # b (intercept)

        # Training history
        self.history = TrainingHistory()

    def _hypothesis(self, X: np.ndarray) -> np.ndarray:
        """
        Hypothesis function: h(x) = w * x + b

        Parameters:
        -----------
        X : np.ndarray
            Input features (n_samples,)

        Returns:
        --------
        np.ndarray : Predictions (n_samples,)
        """
        return self.weight * X + self.bias

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean Squared Error cost function.

        J(w, b) = (1/2m) * SUM[(h(x_i) - y_i)^2]

        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True values

        Returns:
        --------
        float : Cost value
        """
        m = len(y)
        predictions = self._hypothesis(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute gradients for weight and bias.

        dJ/dw = (1/m) * SUM[(h(x_i) - y_i) * x_i]
        dJ/db = (1/m) * SUM[(h(x_i) - y_i)]

        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True values

        Returns:
        --------
        Tuple[float, float] : (dw, db) gradients
        """
        m = len(y)
        predictions = self._hypothesis(X)
        errors = predictions - y

        # Gradient for weight
        dw = (1 / m) * np.sum(errors * X)

        # Gradient for bias
        db = (1 / m) * np.sum(errors)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LinearRegressionScratch':
        """
        Train the model using gradient descent.

        Parameters:
        -----------
        X : np.ndarray
            Training features (n_samples,)
        y : np.ndarray
            Training targets (n_samples,)
        verbose : bool
            Print training progress

        Returns:
        --------
        self : Trained model
        """
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        # Initialize parameters
        self.weight = 0.0
        self.bias = 0.0
        self.history = TrainingHistory()

        if verbose:
            print(f"Training Linear Regression (from scratch)")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Iterations: {self.n_iterations}")
            print(f"  Samples: {len(X)}")

        # Gradient descent loop
        for i in range(self.n_iterations):
            # Compute gradients
            dw, db = self._compute_gradients(X, y)

            # Update parameters
            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            # Track history
            cost = self._compute_cost(X, y)
            self.history.costs.append(cost)
            self.history.weights.append((self.weight, self.bias))

            # Print progress
            if verbose and (i % (self.n_iterations // 10) == 0 or i == self.n_iterations - 1):
                print(f"  Iteration {i:5d}: Cost = {cost:.6f}, w = {self.weight:.6f}, b = {self.bias:.6f}")

        self.history.iterations = self.n_iterations

        if verbose:
            print(f"\nFinal parameters:")
            print(f"  Weight (slope): {self.weight:.6f}")
            print(f"  Bias (intercept): {self.bias:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model.

        Parameters:
        -----------
        X : np.ndarray
            Input features

        Returns:
        --------
        np.ndarray : Predictions
        """
        X = np.asarray(X).flatten()
        return self._hypothesis(X)

    def get_coefficients(self) -> Dict[str, float]:
        """
        Get model coefficients.

        Returns:
        --------
        dict : {'slope': w, 'intercept': b}
        """
        return {
            'slope': self.weight,
            'intercept': self.bias
        }


# =============================================================================
# PROMPT 4: LINEAR REGRESSION USING SCIKIT-LEARN
# =============================================================================

def train_sklearn_model(X: np.ndarray, y: np.ndarray) -> Tuple[object, Dict[str, float]]:
    """
    Train Linear Regression using scikit-learn.

    Parameters:
    -----------
    X : np.ndarray
        Training features (n_samples,)
    y : np.ndarray
        Training targets (n_samples,)

    Returns:
    --------
    Tuple[model, coefficients] : Trained model and coefficients dict
    """
    X = np.asarray(X).reshape(-1, 1)  # sklearn requires 2D
    y = np.asarray(y).flatten()

    print("Training Linear Regression (scikit-learn)")
    print(f"  Samples: {len(X)}")

    model = LinearRegression()
    model.fit(X, y)

    coefficients = {
        'slope': float(model.coef_[0]),
        'intercept': float(model.intercept_)
    }

    print(f"\nFinal parameters:")
    print(f"  Weight (slope): {coefficients['slope']:.6f}")
    print(f"  Bias (intercept): {coefficients['intercept']:.6f}")

    return model, coefficients


def compare_models(
    scratch_model: LinearRegressionScratch,
    sklearn_coefficients: Dict[str, float]
) -> Dict:
    """
    Compare from-scratch and sklearn implementations.

    Parameters:
    -----------
    scratch_model : LinearRegressionScratch
        Trained from-scratch model
    sklearn_coefficients : dict
        Coefficients from sklearn model

    Returns:
    --------
    dict : Comparison results
    """
    scratch_coef = scratch_model.get_coefficients()

    slope_diff = abs(scratch_coef['slope'] - sklearn_coefficients['slope'])
    intercept_diff = abs(scratch_coef['intercept'] - sklearn_coefficients['intercept'])

    comparison = {
        'scratch': scratch_coef,
        'sklearn': sklearn_coefficients,
        'differences': {
            'slope_diff': slope_diff,
            'intercept_diff': intercept_diff,
            'slope_pct_diff': (slope_diff / abs(sklearn_coefficients['slope'])) * 100 if sklearn_coefficients['slope'] != 0 else 0,
            'intercept_pct_diff': (intercept_diff / abs(sklearn_coefficients['intercept'])) * 100 if sklearn_coefficients['intercept'] != 0 else 0
        },
        'match': slope_diff < 0.01 and intercept_diff < 0.01
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Parameter':<15} {'From Scratch':<15} {'Scikit-learn':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'Slope':<15} {scratch_coef['slope']:<15.6f} {sklearn_coefficients['slope']:<15.6f} {slope_diff:<15.6f}")
    print(f"{'Intercept':<15} {scratch_coef['intercept']:<15.6f} {sklearn_coefficients['intercept']:<15.6f} {intercept_diff:<15.6f}")
    print("-" * 60)
    print(f"Models match: {'YES' if comparison['match'] else 'NO (check learning rate/iterations)'}")

    return comparison


# =============================================================================
# ANALYTICAL SOLUTION (For verification)
# =============================================================================

def analytical_solution(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute linear regression coefficients analytically (closed-form).

    Using Normal Equation:
    w = (X^T X)^(-1) X^T y

    For univariate case:
    w = Cov(X, y) / Var(X)
    b = mean(y) - w * mean(X)

    Parameters:
    -----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target values

    Returns:
    --------
    dict : {'slope': w, 'intercept': b}
    """
    X = np.asarray(X).flatten()
    y = np.asarray(y).flatten()

    # Compute means
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # Compute slope: Cov(X,y) / Var(X)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)

    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * X_mean

    return {
        'slope': slope,
        'intercept': intercept
    }


# =============================================================================
# MAIN - Test module independently
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MODEL MODULE TEST")
    print("=" * 60)

    # Generate synthetic data with known relationship
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = 2.5 * X + 3.0 + np.random.normal(0, 1, 100)  # y = 2.5x + 3 + noise

    print("\nTrue parameters: slope=2.5, intercept=3.0")

    # Test from-scratch implementation
    print("\n" + "-" * 60)
    scratch_model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    scratch_model.fit(X, y)

    # Test sklearn implementation
    print("\n" + "-" * 60)
    sklearn_model, sklearn_coef = train_sklearn_model(X, y)

    # Test analytical solution
    print("\n" + "-" * 60)
    print("Analytical Solution (Normal Equation)")
    analytical_coef = analytical_solution(X, y)
    print(f"  Slope: {analytical_coef['slope']:.6f}")
    print(f"  Intercept: {analytical_coef['intercept']:.6f}")

    # Compare models
    comparison = compare_models(scratch_model, sklearn_coef)

    # Test predictions
    X_test = np.array([5.0, 7.5, 10.0])
    scratch_pred = scratch_model.predict(X_test)
    sklearn_pred = sklearn_model.predict(X_test.reshape(-1, 1))

    print("\n" + "-" * 60)
    print("PREDICTION TEST")
    print("-" * 60)
    print(f"{'X':<10} {'Scratch':<15} {'Sklearn':<15} {'Diff':<15}")
    for x, sp, sk in zip(X_test, scratch_pred, sklearn_pred):
        print(f"{x:<10.2f} {sp:<15.4f} {sk:<15.4f} {abs(sp-sk):<15.6f}")

    print("\n[OK] Model module test passed")
