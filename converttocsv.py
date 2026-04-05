"""
Convert parquet files to CSV.

Usage:
    python converttocsv.py                          # converts all parquet files in data/
    python converttocsv.py path/to/file.parquet     # converts a single file
    python converttocsv.py data/ --output-dir out/  # specify output directory
"""

import sys
import argparse
from pathlib import Path

import pandas as pd


def convert_file(src: Path, output_dir: Path):
    df = pd.read_parquet(src)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / src.with_suffix(".csv").name
    df.to_csv(dest, index=False)
    print(f"  {src} -> {dest}  ({len(df)} rows, {len(df.columns)} cols)")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet files to CSV")
    parser.add_argument(
        "input",
        nargs="?",
        default="data/",
        help="Parquet file or directory (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write CSVs (default: same directory as input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            print(f"No .parquet files found in {input_path}")
            sys.exit(1)
        output_dir = Path(args.output_dir) if args.output_dir else input_path
        for f in files:
            convert_file(f, output_dir)
    elif input_path.is_file():
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        convert_file(input_path, output_dir)
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
