"""Clustering: pose similarity grouping, PCA/UMAP embeddings, dendrograms."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import get_logger

warnings.filterwarnings("ignore")


def build_feature_matrix(all_metrics: List[Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a numeric feature matrix from combined metrics for clustering."""
    feature_cols = [
        "heavy_atom_rmsd", "mcs_rmsd", "ifp_tanimoto", "ifp_dice",
        "pharmacophore_score", "feature_overlap_pct",
        "shape_tanimoto", "usr_similarity", "gaussian_overlap",
        "com_distance", "principal_axis_angle", "orientation_score",
        "consensus_score",
    ]

    compound_names = [m.get("compound", f"mol_{i}") for i, m in enumerate(all_metrics)]
    matrix = []

    for m in all_metrics:
        row = []
        for col in feature_cols:
            val = m.get(col)
            row.append(val if val is not None else np.nan)
        matrix.append(row)

    matrix = np.array(matrix, dtype=np.float64)

    valid_cols = []
    valid_indices = []
    for i, col in enumerate(feature_cols):
        if not np.all(np.isnan(matrix[:, i])):
            valid_cols.append(col)
            valid_indices.append(i)

    matrix = matrix[:, valid_indices]

    for j in range(matrix.shape[1]):
        col_mean = np.nanmean(matrix[:, j])
        mask = np.isnan(matrix[:, j])
        matrix[mask, j] = col_mean

    return matrix, compound_names, valid_cols


def hierarchical_clustering(feature_matrix: np.ndarray,
                             compound_names: List[str],
                             method: str = "ward",
                             n_clusters: int = 0,
                             distance_threshold: float = 0.0) -> Dict:
    """Perform hierarchical clustering on the feature matrix."""
    logger = get_logger()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    Z = linkage(scaled, method=method, metric="euclidean")

    if n_clusters > 0:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    elif distance_threshold > 0:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        n_clusters = min(5, max(2, len(compound_names) // 3))
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = {}
    for name, label in zip(compound_names, labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(name)

    logger.info(f"Hierarchical clustering: {len(clusters)} clusters from {len(compound_names)} compounds")

    return {
        "linkage_matrix": Z,
        "labels": labels.tolist(),
        "compound_names": compound_names,
        "clusters": clusters,
        "n_clusters": len(clusters),
        "scaled_features": scaled,
    }


def compute_pca(feature_matrix: np.ndarray,
                compound_names: List[str],
                n_components: int = 2) -> Dict:
    """PCA dimensionality reduction for visualization."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    n_components = min(n_components, scaled.shape[1], scaled.shape[0])
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(scaled)

    return {
        "coordinates": embedded,
        "compound_names": compound_names,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "n_components": n_components,
    }


def compute_umap(feature_matrix: np.ndarray,
                  compound_names: List[str],
                  n_components: int = 2,
                  n_neighbors: int = 5,
                  min_dist: float = 0.3) -> Optional[Dict]:
    """UMAP embedding for visualization."""
    try:
        import umap
    except ImportError:
        get_logger().warning("UMAP not available, skipping UMAP embedding")
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    n_neighbors = min(n_neighbors, len(compound_names) - 1)
    if n_neighbors < 2:
        return None

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    embedded = reducer.fit_transform(scaled)

    return {
        "coordinates": embedded,
        "compound_names": compound_names,
        "n_components": n_components,
    }


def compute_similarity_matrix(all_metrics: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise similarity matrix from consensus scores and IFP."""
    feature_matrix, names, cols = build_feature_matrix(all_metrics)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    dist = pdist(scaled, metric="euclidean")
    dist_matrix = squareform(dist)

    max_dist = dist_matrix.max() if dist_matrix.max() > 0 else 1.0
    sim_matrix = 1.0 - dist_matrix / max_dist

    return sim_matrix, names


def ifp_clustering(fp_matrix: np.ndarray,
                    compound_names: List[str]) -> Dict:
    """Cluster based on interaction fingerprint similarity."""
    if fp_matrix.shape[0] < 2:
        return {"labels": [1] * len(compound_names), "clusters": {1: compound_names}}

    dist = pdist(fp_matrix.astype(float), metric="jaccard")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")

    n_clusters = min(5, max(2, len(compound_names) // 3))
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = {}
    for name, label in zip(compound_names, labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(name)

    return {
        "linkage_matrix": Z,
        "labels": labels.tolist(),
        "compound_names": compound_names,
        "clusters": clusters,
        "distance_matrix": squareform(dist),
    }
