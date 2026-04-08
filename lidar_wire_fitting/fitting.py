import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass


@dataclass
class CatenaryFit:
    """Result of fitting a catenary curve to a single wire cluster."""
    x0: float
    y0: float
    c: float          # sag parameter: small values mean droopy, large values mean stiff

    origin: np.ndarray
    pca_axes: np.ndarray  # shape (2, 3), used to project back to 3D

    residual: float = None


def catenary_2d(x, x0, y0, c):
    # y = y0 + c*(cosh((x - x0)/c) - 1)
    return y0 + c * (np.cosh((x - x0) / c) - 1)


def project_to_2d(points: np.ndarray):
    """
    Project a 3D wire cluster into a 2D plane for catenary fitting.

    Uses PC1 (wire length) and PC3 (sag direction) from SVD. PC1+PC2 was the
    first attempt, which gave a flat top-down projection with no sag visible at all.
    PC3 is the right axis because Z variance is tiny compared to the X-Y span.
    """
    origin = points.mean(axis=0)
    centered = points - origin

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = np.array([Vt[0], Vt[2]])

    coords_2d = centered @ axes.T
    return coords_2d, origin, axes


def fit_catenary(cluster_points: np.ndarray) -> CatenaryFit | None:
    """Fit a catenary to one wire cluster. Returns None if the fit doesn't converge."""
    if len(cluster_points) < 10:
        print("  Skipping cluster with fewer than 10 points")
        return None

    coords_2d, origin, axes = project_to_2d(cluster_points)
    x_proj = coords_2d[:, 0]  # along wire
    y_proj = coords_2d[:, 1]  # sag direction

    # starting guess: curve_fit is quite sensitive to these, so bad values will give wrong answers
    x0_init = float(np.mean(x_proj))
    y0_init = float(np.min(y_proj))

    sag = np.max(y_proj) - np.min(y_proj)
    span = np.max(x_proj) - np.min(x_proj)
    c_init = max(span / max(sag, 0.1), 1.0)  # a taut wire gives a large c, a droopy one gives a small c

    try:
        popt, _ = curve_fit(
            catenary_2d,
            x_proj,
            y_proj,
            p0=[x0_init, y0_init, c_init],
            maxfev=10000,
            bounds=(
                [-np.inf, -np.inf, 0.01],
                [np.inf, np.inf, np.inf]
            )
        )
        x0, y0, c = popt

        y_pred = catenary_2d(x_proj, x0, y0, c)
        residual = float(np.sqrt(np.mean((y_proj - y_pred) ** 2)))

        return CatenaryFit(
            x0=x0, y0=y0, c=c,
            origin=origin, pca_axes=axes,
            residual=residual
        )

    except RuntimeError as e:
        print(f"  curve_fit failed: {e}")
        return None


def sample_catenary_3d(fit: CatenaryFit, cluster_points: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Sample n_points along the fitted catenary, back in 3D coordinates."""
    centered = cluster_points - fit.origin
    coords_2d = centered @ fit.pca_axes.T
    x_proj = coords_2d[:, 0]

    x_samples = np.linspace(x_proj.min(), x_proj.max(), n_points)
    y_samples = catenary_2d(x_samples, fit.x0, fit.y0, fit.c)

    coords_2d_samples = np.stack([x_samples, y_samples], axis=1)
    points_3d = fit.origin + coords_2d_samples @ fit.pca_axes

    return points_3d