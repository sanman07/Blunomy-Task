import numpy as np
from sklearn.cluster import DBSCAN
from numpy.linalg import svd


def cluster_wires(points: np.ndarray, eps: float = 0.1, min_samples: int = 5) -> np.ndarray:
    """
    Cluster wire points by projecting onto the cross-wire axis first.

    3D DBSCAN kept merging multiple wires together because the wire separation only
    exists in one direction (perpendicular to the wire run), so Z was just adding
    noise to the distance calculation. Instead, we use PCA on X-Y to find that
    cross-wire direction, then run DBSCAN on that single axis.

    Labels of -1 indicate noise or unassigned points.
    """
    # find the wire direction from the X-Y spread
    origin = points[:, :2].mean(axis=0)
    centered_xy = points[:, :2] - origin
    _, _, Vt = svd(centered_xy, full_matrices=False)

    # PC2 is perpendicular to the wire — the axis that actually separates them
    cross_wire = (centered_xy @ Vt[1]).reshape(-1, 1)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cross_wire)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"  Found {n_clusters} wire cluster(s), {n_noise} noise points")

    return labels


def get_clusters(points: np.ndarray, labels: np.ndarray) -> list:
    """Return list of point arrays, one per wire cluster (noise excluded)."""
    cluster_ids = sorted(set(labels) - {-1})
    clusters = []
    for cid in cluster_ids:
        cluster_points = points[labels == cid]
        clusters.append(cluster_points)
    return clusters