import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


WIRE_COLOURS = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#ff7f00", "#a65628"
]


def plot_results(
    points: np.ndarray,
    labels: np.ndarray,
    catenary_curves: list,
    title: str = "Wire Detection Results",
    save_path: str = None
):
    """3D scatter of the point cloud with per-wire colours and fitted curves overlaid."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            points[noise_mask, 0],
            points[noise_mask, 1],
            points[noise_mask, 2],
            c="lightgrey", s=1, alpha=0.3, label="noise"
        )

    cluster_ids = sorted(set(labels) - {-1})
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        colour = WIRE_COLOURS[i % len(WIRE_COLOURS)]
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            c=colour, s=4, alpha=0.6,
            label=f"wire {i+1} points"
        )

    for i, curve in enumerate(catenary_curves):
        if curve is None:
            continue
        colour = WIRE_COLOURS[i % len(WIRE_COLOURS)]
        ax.plot(
            curve[:, 0], curve[:, 1], curve[:, 2],
            c=colour, linewidth=2, linestyle="--",
            label=f"wire {i+1} fit"
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def print_fit_summary(fits: list, file_name: str):
    """Print a quick summary of the fitting results to stdout."""
    print(f"\nResults for {file_name}")
    valid_fits = [f for f in fits if f is not None]
    print(f"Wires detected: {len(valid_fits)}")
    for i, fit in enumerate(valid_fits):
        print(
            f"  Wire {i+1}: c={fit.c:.3f}, "
            f"sag_point=({fit.x0:.2f}, {fit.y0:.2f}), "
            f"RMSE={fit.residual:.4f}"
        )