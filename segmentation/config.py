"""Project-wide constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DATA_PATH = DATA_DIR / "customers.csv"

RANDOM_STATE = 42

# Range of cluster counts scanned by the Elbow / Silhouette analysis.
K_MIN = 2
K_MAX = 10

# Hierarchical clustering needs O(n^2) memory, so it is capped.
MAX_HIERARCHICAL_ROWS = 5000

# Silhouette is O(n^2); above this many rows it is estimated on a random sample.
SILHOUETTE_SAMPLE_SIZE = 5000

# Features pre-selected in the dashboard when they exist in the dataset.
DEFAULT_FEATURES = [
    "age",
    "annual_income_lakh",
    "annual_income_k",
    "spending_score",
    "tenure_months",
    "orders_per_year",
    "avg_order_value_inr",
    "avg_order_value",
    "days_since_last_order",
]
