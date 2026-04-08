import pandas as pd
import numpy as np
from pathlib import Path


def load_point_cloud(filepath: str) -> np.ndarray:
    """Load a LiDAR point cloud from a parquet file, returns an (N, 3) array."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {filepath}")

    df = pd.read_parquet(path)

    required_cols = ["x", "y", "z"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Columns in file: {list(df.columns)}")

    points = df[required_cols].to_numpy(dtype=np.float64)

    before = len(points)
    points = points[~np.isnan(points).any(axis=1)]
    if len(points) < before:
        print(f"  Warning: dropped {before - len(points)} rows with NaN values")

    return points