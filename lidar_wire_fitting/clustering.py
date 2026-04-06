import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_wires(points: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> np.ndarray:
    """
    Cluster wire points using DBSCAN. Returns label array; -1 = noise.

    DBSCAN works better than k-means
    """
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_scaled)

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