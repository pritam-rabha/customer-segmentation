"""Regenerate data/customers.csv (reproducible synthetic retail customers).

Usage:  python scripts/generate_dataset.py [--rows 1000] [--seed 42]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.config import SAMPLE_DATA_PATH  # noqa: E402
from segmentation.data_loader import generate_synthetic_customers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=SAMPLE_DATA_PATH)
    args = parser.parse_args()

    df = generate_synthetic_customers(n=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} customers to {args.out}")


if __name__ == "__main__":
    main()
