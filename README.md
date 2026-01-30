# Robot Predictive Maintenance - MLOps Workshop

A machine learning project demonstrating **univariate linear regression** for predictive maintenance of industrial robots. This workshop covers the complete ML lifecycle from problem definition to production-ready MLOps practices.

## Problem Definition (Prompt 1)

**Objective:** Predict robot motor degradation over time using univariate linear regression.

| Variable | Name | Description |
|----------|------|-------------|
| **X** (Independent) | `elapsed_hours` | Time elapsed since monitoring began |
| **y** (Dependent) | `mean_current` | Average current draw across motor axes |

**Why Univariate?**
- Single predictor (time) maps directly to single outcome (current)
- Clear interpretation: slope = degradation rate (Amps/hour)
- Foundation for understanding before multivariate expansion

**Maintenance Relevance:**
- Positive slope indicates increasing current draw = motor degradation
- Slope magnitude quantifies degradation speed
- Prediction enables proactive maintenance scheduling

## Features

- **Linear Regression From Scratch** (Prompt 3): Gradient descent implementation using NumPy only
- **Scikit-learn Comparison** (Prompt 4): Side-by-side model comparison
- **Model Evaluation** (Prompt 5): RMSE, MAE, R² metrics with visualizations
- **Fleet-Wide Processing**: Individual models for Robot A, B, C, and D
- **Alert System** (Prompt 12): INFO/WARNING/CRITICAL maintenance alerts
- **Experiment Tracking** (Prompt 14): Auditable results logging per robot
- **Synthetic Data Generation** (Prompt 9): Realistic multi-robot sensor data

## Project Structure (Prompt 10)

```
linear_regression_project/
├── data/                          # Data storage
│   ├── raw/                       # Original CSV files
│   ├── processed/                 # Cleaned, split data
│   └── synthetic/                 # Generated robot data
├── notebooks/
│   └── RobotPM_MLOps.ipynb        # Main workshop notebook
├── src/                           # Modular Python scripts (Prompt 6)
│   ├── __init__.py
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Missing values, scaling, splits
│   ├── model.py                   # LinearRegressionScratch + sklearn
│   ├── evaluation.py              # Metrics computation & plotting
│   ├── synthetic_generator.py     # Multi-robot data generation
│   └── alert_system.py            # Failure prediction & alerts
├── configs/                       # Configuration files (Prompt 11)
│   ├── experiment_config.yaml     # Model hyperparameters
│   ├── db_config.yaml             # Database connection settings
│   └── alert_thresholds.yaml      # Alert threshold definitions
├── experiments/                   # Experiment tracking (Prompt 14)
│   ├── results.csv                # Per-robot experiment logs
│   └── logs/
├── requirements.txt
├── projectexecution.txt           # Execution plan documentation
└── README.md
```

## Design Decisions

### Preprocessing Pipeline (Prompt 2)
1. **Missing Values:** Forward fill (preserves temporal order for time-series)
2. **Feature Scaling:** MinMax normalization to [0,1]
3. **Train/Test Split:** Temporal 80/20 (prevents future data leakage)

### Modular Architecture (Prompts 6 & 7)
Benefits of separating code into modules:
- **Separation of concerns:** Each file has single responsibility
- **Easier debugging:** Isolate issues to specific modules
- **Reusability:** Import functions across notebooks/scripts
- **Scalability:** Add new models without modifying existing code

### Configuration-Driven Experiments (Prompt 11)
YAML configs control:
- Data source (CSV/DB/API)
- Learning rate and iterations
- Train/test split ratio
- Feature and target variables
- Alert thresholds

Change experiments without touching code.

### Fleet-Wide Processing (Prompt 8)
The system processes multiple robots individually:
- **Robot_A, Robot_B, Robot_C, Robot_D** each get separate regression models
- Individual degradation slopes computed per robot
- Fleet-wide alert report generated
- Per-robot metrics logged to `experiments/results.csv`

### Failure Prediction & Alerts (Prompt 12)
Alert levels based on predicted days to failure:
| Level | Condition | Action |
|-------|-----------|--------|
| **CRITICAL** | < 7 days | Immediate maintenance required |
| **WARNING** | 7-14 days | Schedule maintenance soon |
| **INFO** | 14-30 days | Monitor closely |
| **HEALTHY** | > 30 days | Normal operation |

### MLOps Best Practices (Prompt 13)
| Principle | Implementation |
|-----------|----------------|
| Separation of concerns | `src/` modules |
| Configuration-driven | `configs/*.yaml` |
| Experiment tracking | `experiments/results.csv` |
| Reproducibility | Fixed seeds, versioned data |
| Auditability | Full lineage from data to alerts |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/linear_regression_project.git
   cd linear_regression_project
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Main Workshop Notebook
```bash
jupyter notebook notebooks/RobotPM_MLOps.ipynb
```

### Run Modular Scripts
```bash
python src/data_loader.py
python src/preprocessing.py
python src/model.py
python src/evaluation.py
```

### Generate Synthetic Data
```bash
python src/synthetic_generator.py
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0.0 | Data manipulation |
| numpy | >= 1.24.0 | Numerical computing |
| matplotlib | >= 3.7.0 | Visualization |
| seaborn | >= 0.12.0 | Statistical plots |
| scikit-learn | >= 1.3.0 | ML algorithms |
| pyyaml | >= 6.0 | Configuration parsing |
| jupyter | >= 1.0.0 | Interactive notebooks |
| psycopg2-binary | >= 2.9.0 | PostgreSQL connectivity |

## Repository Checklist (Prompt 15)

- [x] Frozen codebase with modular structure
- [x] `requirements.txt` with pinned versions
- [x] `README.md` with design decisions
- [x] `RobotPM_MLOps.ipynb` documenting:
  - [x] Architecture changes
  - [x] Fleet-wide enhancements (Robot A, B, C, D)
  - [x] Per-robot model training
  - [x] Alert system breakdown
- [x] Configuration files in `configs/`
- [x] Experiment tracking in `experiments/`

## License

MIT License
