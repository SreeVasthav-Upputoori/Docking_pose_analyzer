"""METHOD 5: Binding Vector / Orientation Analysis."""

import warnings
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .io_handlers import MoleculeRecord
from .utils import get_logger

warnings.filterwarnings("ignore")


def compute_center_of_mass(record: MoleculeRecord) -> Optional[np.ndarray]:
    """Compute center of mass of heavy atoms."""
    coords = record.get_heavy_atom_coords()
    if coords is None or len(coords) == 0:
        return None

    from rdkit import Chem
    mol = record.mol
    masses = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            masses.append(atom.GetMass())

    masses = np.array(masses[:len(coords)])
    if len(masses) != len(coords):
        return np.mean(coords, axis=0)

    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.mean(coords, axis=0)

    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return com


def compute_principal_axes(record: MoleculeRecord) -> Optional[Dict]:
    """Compute principal axes of inertia for the ligand."""
    coords = record.get_heavy_atom_coords()
    if coords is None or len(coords) < 3:
        return None

    centroid = np.mean(coords, axis=0)
    centered = coords - centroid

    inertia_tensor = np.zeros((3, 3))
    for r in centered:
        inertia_tensor += np.outer(r, r)
    inertia_tensor /= len(centered)

    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return {
        "centroid": centroid,
        "eigenvalues": eigenvalues,
        "principal_axes": eigenvectors,
        "axis1": eigenvectors[:, 0],
        "axis2": eigenvectors[:, 1],
        "axis3": eigenvectors[:, 2],
    }


def compute_ring_planes(record: MoleculeRecord) -> List[Dict]:
    """Compute plane normal vectors for aromatic/ring systems."""
    from rdkit import Chem

    mol = record.mol
    if not mol or not mol.GetNumConformers():
        return []

    conf = mol.GetConformer()
    ring_info = mol.GetRingInfo()
    planes = []

    for ring in ring_info.AtomRings():
        if len(ring) < 3:
            continue

        ring_coords = []
        for idx in ring:
            pos = conf.GetAtomPosition(idx)
            ring_coords.append([pos.x, pos.y, pos.z])
        ring_coords = np.array(ring_coords)

        centroid = np.mean(ring_coords, axis=0)
        centered = ring_coords - centroid

        if len(centered) >= 3:
            _, _, Vt = np.linalg.svd(centered)
            normal = Vt[-1]
            if normal[2] < 0:
                normal = -normal

            is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)

            planes.append({
                "atom_indices": list(ring),
                "centroid": centroid,
                "normal": normal,
                "is_aromatic": is_aromatic,
                "ring_size": len(ring),
            })

    return planes


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees (0-180)."""
    cos_angle = np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10),
        -1.0, 1.0
    )
    return float(np.degrees(np.arccos(abs(cos_angle))))


def compare_orientations(ref: MoleculeRecord, query: MoleculeRecord) -> Dict:
    """Full orientation comparison between reference and query."""
    logger = get_logger()
    result = {
        "com_distance": None,
        "com_ref": None,
        "com_query": None,
        "principal_axis_angle": None,
        "axis2_angle": None,
        "axis3_angle": None,
        "mean_ring_plane_angle": None,
        "is_flipped": False,
        "orientation_score": None,
    }

    ref_com = compute_center_of_mass(ref)
    query_com = compute_center_of_mass(query)

    if ref_com is not None and query_com is not None:
        result["com_distance"] = round(float(np.linalg.norm(ref_com - query_com)), 3)
        result["com_ref"] = ref_com.tolist()
        result["com_query"] = query_com.tolist()

    ref_axes = compute_principal_axes(ref)
    query_axes = compute_principal_axes(query)

    if ref_axes and query_axes:
        angle1 = angle_between_vectors(ref_axes["axis1"], query_axes["axis1"])
        angle2 = angle_between_vectors(ref_axes["axis2"], query_axes["axis2"])
        angle3 = angle_between_vectors(ref_axes["axis3"], query_axes["axis3"])

        result["principal_axis_angle"] = round(angle1, 1)
        result["axis2_angle"] = round(angle2, 1)
        result["axis3_angle"] = round(angle3, 1)

        if angle1 > 120 or (angle1 > 90 and angle2 > 90):
            result["is_flipped"] = True

    ref_planes = compute_ring_planes(ref)
    query_planes = compute_ring_planes(query)

    if ref_planes and query_planes:
        from scipy.spatial.distance import cdist as sp_cdist

        ref_centroids = np.array([p["centroid"] for p in ref_planes])
        query_centroids = np.array([p["centroid"] for p in query_planes])
        dist_matrix = sp_cdist(ref_centroids, query_centroids)

        angles = []
        n = min(len(ref_planes), len(query_planes))
        used_q = set()
        for i in range(len(ref_planes)):
            best_j = None
            best_dist = float("inf")
            for j in range(len(query_planes)):
                if j not in used_q and dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_j = j
            if best_j is not None and best_dist < 5.0:
                used_q.add(best_j)
                angle = angle_between_vectors(
                    ref_planes[i]["normal"], query_planes[best_j]["normal"]
                )
                angles.append(angle)

        if angles:
            result["mean_ring_plane_angle"] = round(float(np.mean(angles)), 1)

    score = 1.0
    if result["com_distance"] is not None:
        score *= np.exp(-0.3 * result["com_distance"])
    if result["principal_axis_angle"] is not None:
        score *= np.exp(-0.01 * result["principal_axis_angle"])
    if result["mean_ring_plane_angle"] is not None:
        score *= np.exp(-0.01 * result["mean_ring_plane_angle"])
    result["orientation_score"] = round(score, 3)

    return result


def batch_orientation_analysis(ref: MoleculeRecord,
                                queries: List[MoleculeRecord]) -> List[Dict]:
    """Run orientation analysis for all queries against reference."""
    results = []
    for q in queries:
        orient_data = compare_orientations(ref, q)
        orient_data["compound"] = q.name
        results.append(orient_data)
    return results
