"""
Entry point for the wire-fitting pipeline. Accepts a single parquet file or a directory.

    python scripts/run_pipeline.py --input data/lidar_cable_points_easy.parquet
    python scripts/run_pipeline.py --input data/ --save-plots
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_wire_fitting import (
    load_point_cloud,
    cluster_wires,
    get_clusters,
    fit_catenary,
    sample_catenary_3d,
    plot_results,
    print_fit_summary,
)

DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 8


def process_file(filepath: str, save_plots: bool = False, output_dir: str = "output"):
    """Load, cluster, fit, and plot one parquet file."""
    print(f"\nProcessing: {filepath}")

    points = load_point_cloud(filepath)
    print(f"  Loaded {len(points)} points")

    labels = cluster_wires(points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    clusters = get_clusters(points, labels)

    if not clusters:
        print("  No wire clusters found - try adjusting DBSCAN parameters")
        return

    fits = []
    curves = []
    for i, cluster in enumerate(clusters):
        print(f"  Fitting wire {i+1} ({len(cluster)} points)...")
        fit = fit_catenary(cluster)
        fits.append(fit)

        if fit is not None:
            curve = sample_catenary_3d(fit, cluster)
            curves.append(curve)
        else:
            curves.append(None)

    print_fit_summary(fits, Path(filepath).name)

    file_stem = Path(filepath).stem
    save_path = f"{output_dir}/{file_stem}.png" if save_plots else None
    plot_results(
        points=points,
        labels=labels,
        catenary_curves=curves,
        title=f"Catenary Fit — {file_stem}",
        save_path=save_path
    )


def main():
    parser = argparse.ArgumentParser(description="Fit catenary models to LiDAR wire data")
    parser.add_argument("--input", required=True, help="path to a .parquet file, or a directory of them")
    parser.add_argument("--save-plots", action="store_true", help="save plots to disk instead of popping up a window")
    parser.add_argument("--output-dir", default="output", help="where to save plots (default: output/)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            print(f"No .parquet files found in {input_path}")
            sys.exit(1)
        for f in files:
            process_file(str(f), save_plots=args.save_plots, output_dir=args.output_dir)
    elif input_path.is_file():
        process_file(str(input_path), save_plots=args.save_plots, output_dir=args.output_dir)
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()