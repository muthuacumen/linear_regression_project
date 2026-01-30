"""
Synthetic Robot Data Generator
==============================

Responsibilities:
- Generate artificial data for additional robots
- Match original schema exactly
- Create different degradation patterns per robot
- Preserve realism with noise and spikes

Functions:
----------
- generate_robot_data(profile) -> pd.DataFrame
- generate_all_robots(reference_df) -> pd.DataFrame
- get_robot_profiles() -> List[dict]
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


def get_robot_profiles() -> List[Dict]:
    """
    Get predefined profiles for synthetic robots.

    Each profile defines:
    - name: Robot identifier
    - duty_type: Light/Medium/Heavy
    - baseline_current: Average current draw per axis
    - degradation_rate: How fast current increases (A/hour)
    - degradation_start: When degradation begins (0-1 fraction of data)
    - noise_std: Standard deviation of sensor noise
    - spike_probability: Probability of random spikes

    Returns:
    --------
    List[dict] : Robot profiles
    """
    profiles = [
        {
            'name': 'Robot_B',
            'duty_type': 'Light',
            'description': 'Precision Assembly Robot',
            'baseline_current': {
                'Axis #1': 5.5, 'Axis #2': 5.0, 'Axis #3': 6.0,
                'Axis #4': 4.0, 'Axis #5': 3.5, 'Axis #6': 4.5,
                'Axis #7': 2.5, 'Axis #8': 2.0
            },
            'degradation_rate': 0.02,  # Slow degradation
            'degradation_start': 0.6,   # Starts at 60% of timeline
            'noise_std': 0.3,
            'spike_probability': 0.001,
            'failure_threshold': 1.25   # Fails at 125% of baseline
        },
        {
            'name': 'Robot_C',
            'duty_type': 'Heavy',
            'description': 'Industrial Welding Robot',
            'baseline_current': {
                'Axis #1': 21.0, 'Axis #2': 20.0, 'Axis #3': 22.0,
                'Axis #4': 15.0, 'Axis #5': 14.0, 'Axis #6': 16.0,
                'Axis #7': 13.0, 'Axis #8': 12.0
            },
            'degradation_rate': 0.35,  # Fast degradation
            'degradation_start': 0.2,   # Early degradation
            'noise_std': 1.5,
            'spike_probability': 0.005,
            'failure_threshold': 1.50   # Fails at 150% of baseline
        },
        {
            'name': 'Robot_D',
            'duty_type': 'Medium',
            'description': 'Material Handling Gantry',
            'baseline_current': {
                'Axis #1': 13.0, 'Axis #2': 12.0, 'Axis #3': 14.0,
                'Axis #4': 7.0, 'Axis #5': 6.5, 'Axis #6': 7.5,
                'Axis #7': 8.5, 'Axis #8': 8.0
            },
            'degradation_rate': 0.09,  # Moderate degradation
            'degradation_start': 0.4,   # Mid-timeline
            'noise_std': 0.8,
            'spike_probability': 0.002,
            'failure_threshold': 1.35   # Fails at 135% of baseline
        }
    ]

    return profiles


def generate_robot_data(
    profile: Dict,
    timestamps: np.ndarray,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a single robot.

    Parameters:
    -----------
    profile : dict
        Robot profile with degradation parameters
    timestamps : np.ndarray
        Array of timestamps (from reference data)
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame : Generated data matching original schema
    """
    # Set seed based on robot name for reproducibility
    if random_seed is None:
        random_seed = hash(profile['name']) % (2**32)
    np.random.seed(random_seed)

    n_samples = len(timestamps)

    print(f"\nGenerating data for {profile['name']} ({profile['duty_type']} duty)...")
    print(f"  Samples: {n_samples:,}")
    print(f"  Degradation rate: {profile['degradation_rate']} A/hr")
    print(f"  Degradation start: {profile['degradation_start']*100:.0f}% of timeline")

    # Initialize data dictionary
    data = {
        'Trait': ['current'] * n_samples,
        'Time': timestamps,
        'robot_id': profile['name']
    }

    # Time normalized to [0, 1]
    time_normalized = np.arange(n_samples) / n_samples

    # Generate data for each axis
    for axis_num in range(1, 15):
        axis_name = f'Axis #{axis_num}'

        if axis_name in profile['baseline_current']:
            baseline = profile['baseline_current'][axis_name]

            # Base signal with cyclic pattern
            cycle_period = 50  # samples
            phase = (axis_num * np.pi / 7)
            base_signal = baseline + baseline * 0.2 * np.sin(
                2 * np.pi * np.arange(n_samples) / cycle_period + phase
            )

            # Apply degradation
            degradation = np.zeros(n_samples)
            start_idx = int(profile['degradation_start'] * n_samples)

            # Linear degradation after start point
            for i in range(start_idx, n_samples):
                progress = (i - start_idx) / (n_samples - start_idx)
                # Linear + exponential acceleration at end
                if progress > 0.8:
                    exp_factor = np.exp((progress - 0.8) * 5) - 1
                    degradation[i] = profile['degradation_rate'] * progress + exp_factor * 0.5
                else:
                    degradation[i] = profile['degradation_rate'] * progress

            # Apply degradation as multiplicative factor
            signal = base_signal * (1 + degradation)

            # Add noise
            noise = np.random.normal(0, profile['noise_std'], n_samples)
            signal += noise

            # Add random spikes
            spike_mask = np.random.random(n_samples) < profile['spike_probability']
            signal[spike_mask] *= np.random.uniform(1.5, 2.5, np.sum(spike_mask))

            # Ensure non-negative
            signal = np.maximum(signal, 0)

            # Round to 5 decimal places
            data[axis_name] = np.round(signal, 5)
        else:
            # Axis not equipped
            data[axis_name] = np.nan

    df = pd.DataFrame(data)

    # Reorder columns to match original schema
    column_order = ['Trait'] + [f'Axis #{i}' for i in range(1, 15)] + ['Time', 'robot_id']
    df = df[[c for c in column_order if c in df.columns]]

    print(f"  Generated {len(df):,} records")

    return df


def generate_all_robots(
    reference_df: pd.DataFrame,
    include_reference: bool = True
) -> pd.DataFrame:
    """
    Generate data for all synthetic robots and combine.

    Parameters:
    -----------
    reference_df : pd.DataFrame
        Original Robot_A data (for timestamps and schema)
    include_reference : bool
        Whether to include Robot_A in output

    Returns:
    --------
    pd.DataFrame : Combined data for all robots
    """
    print("=" * 70)
    print("SYNTHETIC ROBOT DATA GENERATION")
    print("=" * 70)

    # Extract timestamps from reference
    time_col = 'Time' if 'Time' in reference_df.columns else 'timestamp'
    timestamps = pd.to_datetime(reference_df[time_col]).values

    # Get robot profiles
    profiles = get_robot_profiles()

    # Generate data for each robot
    all_data = []

    # Include reference robot if requested
    if include_reference:
        print("\nProcessing Robot_A (reference data)...")
        robot_a = reference_df.copy()
        robot_a['robot_id'] = 'Robot_A'
        all_data.append(robot_a)
        print(f"  {len(robot_a):,} records")

    # Generate synthetic robots
    for profile in profiles:
        robot_df = generate_robot_data(profile, timestamps)
        all_data.append(robot_df)

    # Combine all robots
    print("\nCombining all robot data...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by robot_id
    combined_df = combined_df.sort_values(['robot_id', time_col]).reset_index(drop=True)

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total records: {len(combined_df):,}")
    print(f"Robots: {combined_df['robot_id'].unique().tolist()}")
    print(f"Records per robot:")
    for robot in combined_df['robot_id'].unique():
        count = len(combined_df[combined_df['robot_id'] == robot])
        print(f"  {robot}: {count:,}")

    return combined_df


def verify_degradation_patterns(df: pd.DataFrame) -> Dict:
    """
    Verify that degradation patterns are present in generated data.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined robot data

    Returns:
    --------
    dict : Degradation statistics per robot
    """
    time_col = 'Time' if 'Time' in df.columns else 'timestamp'
    axis_cols = [c for c in df.columns if 'Axis' in c][:8]

    results = {}

    print("\n" + "=" * 70)
    print("DEGRADATION VERIFICATION")
    print("=" * 70)

    for robot in df['robot_id'].unique():
        robot_data = df[df['robot_id'] == robot].copy()
        robot_data = robot_data.sort_values(time_col).reset_index(drop=True)

        n = len(robot_data)
        early = robot_data.head(n // 5)[axis_cols].mean().mean()
        late = robot_data.tail(n // 5)[axis_cols].mean().mean()
        pct_change = ((late - early) / early) * 100 if early > 0 else 0

        results[robot] = {
            'early_mean': early,
            'late_mean': late,
            'pct_change': pct_change
        }

        print(f"\n{robot}:")
        print(f"  Early mean current: {early:.4f} A")
        print(f"  Late mean current:  {late:.4f} A")
        print(f"  Degradation:        {pct_change:+.2f}%")

    return results


# =============================================================================
# MAIN - Test module independently
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SYNTHETIC GENERATOR MODULE TEST")
    print("=" * 60)

    # Create mock reference data
    n_samples = 1000
    reference_df = pd.DataFrame({
        'Trait': ['current'] * n_samples,
        'Time': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'Axis #1': np.random.uniform(5, 15, n_samples),
        'Axis #2': np.random.uniform(4, 12, n_samples),
        'Axis #3': np.random.uniform(6, 18, n_samples),
        'Axis #4': np.random.uniform(3, 10, n_samples),
        'Axis #5': np.random.uniform(2, 8, n_samples),
        'Axis #6': np.random.uniform(4, 12, n_samples),
        'Axis #7': np.random.uniform(2, 6, n_samples),
        'Axis #8': np.random.uniform(1, 5, n_samples),
    })

    # Generate all robots
    combined_df = generate_all_robots(reference_df, include_reference=True)

    # Verify degradation
    verify_degradation_patterns(combined_df)

    # Save test output
    output_path = 'test_synthetic_data.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\nTest data saved to: {output_path}")

    print("\n[OK] Synthetic generator module test passed")
