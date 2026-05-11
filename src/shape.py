"""METHOD 6: Shape Similarity analysis (USR, Gaussian overlap, 3D Tanimoto)."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .io_handlers import MoleculeRecord
from .utils import get_logger

warnings.filterwarnings("ignore")


def _usr_descriptors(coords: np.ndarray) -> np.ndarray:
    """Compute Ultrafast Shape Recognition (USR) descriptors.
    12 descriptors: mean/var/skew of distances from centroid, closest-to-centroid,
    farthest-from-centroid, and farthest-from-farthest.
    """
    if len(coords) < 4:
        return np.zeros(12)

    centroid = np.mean(coords, axis=0)
    d_ctd = np.linalg.norm(coords - centroid, axis=1)

    closest_idx = np.argmin(d_ctd)
    d_cst = np.linalg.norm(coords - coords[closest_idx], axis=1)

    farthest_idx = np.argmax(d_ctd)
    d_fct = np.linalg.norm(coords - coords[farthest_idx], axis=1)

    farthest_from_farthest = np.argmax(d_fct)
    d_ftf = np.linalg.norm(coords - coords[farthest_from_farthest], axis=1)

    descriptors = []
    for d in [d_ctd, d_cst, d_fct, d_ftf]:
        descriptors.extend([np.mean(d), np.var(d), float(
            np.mean(((d - np.mean(d)) / (np.std(d) + 1e-10)) ** 3)
        )])

    return np.array(descriptors)


def _usrcat_descriptors(record: MoleculeRecord) -> np.ndarray:
    """USRCAT: USR with pharmacophoric atom subsets."""
    from rdkit import Chem

    mol = record.mol
    if not mol or not mol.GetNumConformers():
        return np.zeros(60)

    conf = mol.GetConformer()
    all_coords = record.get_heavy_atom_coords()
    if all_coords is None or len(all_coords) < 4:
        return np.zeros(60)

    base_usr = _usr_descriptors(all_coords)

    subsets = {
        "hydrophobic": [],
        "aromatic": [],
        "hb_acceptor": [],
        "hb_donor": [],
    }

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coord = np.array([pos.x, pos.y, pos.z])

        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
            subsets["hydrophobic"].append(coord)
        if atom.GetIsAromatic():
            subsets["aromatic"].append(coord)
        if atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs() == 0:
            subsets["hb_acceptor"].append(coord)
        if atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs() > 0:
            subsets["hb_donor"].append(coord)

    subset_usrs = []
    for name, coords in subsets.items():
        if len(coords) >= 4:
            subset_usrs.append(_usr_descriptors(np.array(coords)))
        else:
            subset_usrs.append(np.zeros(12))

    return np.concatenate([base_usr] + subset_usrs)


def usr_similarity(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """USR similarity score (inverse of normalized Manhattan distance)."""
    n = len(desc1)
    if n == 0:
        return 0.0
    dist = np.sum(np.abs(desc1 - desc2)) / n
    return float(1.0 / (1.0 + dist / 12.0))


def gaussian_shape_overlap(record1: MoleculeRecord, record2: MoleculeRecord,
                            sigma: float = 0.7) -> float:
    """Approximate Gaussian shape overlap score."""
    coords1 = record1.get_heavy_atom_coords()
    coords2 = record2.get_heavy_atom_coords()

    if coords1 is None or coords2 is None:
        return 0.0

    dist_matrix = cdist(coords1, coords2)
    overlap = np.sum(np.exp(-dist_matrix ** 2 / (2 * sigma ** 2)))

    self_overlap1 = 0.0
    d1 = cdist(coords1, coords1)
    self_overlap1 = np.sum(np.exp(-d1 ** 2 / (2 * sigma ** 2)))

    self_overlap2 = 0.0
    d2 = cdist(coords2, coords2)
    self_overlap2 = np.sum(np.exp(-d2 ** 2 / (2 * sigma ** 2)))

    denominator = max(self_overlap1, self_overlap2)
    if denominator == 0:
        return 0.0

    return float(min(overlap / denominator, 1.0))


def volumetric_overlap(record1: MoleculeRecord, record2: MoleculeRecord,
                        grid_spacing: float = 0.5, vdw_scale: float = 1.0) -> float:
    """Grid-based volumetric overlap percentage."""
    coords1 = record1.get_heavy_atom_coords()
    coords2 = record2.get_heavy_atom_coords()

    if coords1 is None or coords2 is None:
        return 0.0

    all_coords = np.vstack([coords1, coords2])
    vdw_radius = 1.7 * vdw_scale

    mins = all_coords.min(axis=0) - vdw_radius - 1.0
    maxs = all_coords.max(axis=0) + vdw_radius + 1.0

    x = np.arange(mins[0], maxs[0], grid_spacing)
    y = np.arange(mins[1], maxs[1], grid_spacing)
    z = np.arange(mins[2], maxs[2], grid_spacing)

    if len(x) * len(y) * len(z) > 500000:
        grid_spacing = 1.0
        x = np.arange(mins[0], maxs[0], grid_spacing)
        y = np.arange(mins[1], maxs[1], grid_spacing)
        z = np.arange(mins[2], maxs[2], grid_spacing)

    grid = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T

    d1 = cdist(grid, coords1)
    mask1 = np.any(d1 < vdw_radius, axis=1)

    d2 = cdist(grid, coords2)
    mask2 = np.any(d2 < vdw_radius, axis=1)

    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    if union == 0:
        return 0.0
    return float(intersection / union * 100)


def compute_shape_similarity(ref: MoleculeRecord, query: MoleculeRecord) -> Dict:
    """Full shape similarity analysis between reference and query."""
    result = {
        "usr_similarity": 0.0,
        "usrcat_similarity": 0.0,
        "gaussian_overlap": 0.0,
        "volumetric_overlap_pct": 0.0,
        "shape_tanimoto": 0.0,
    }

    ref_usr = _usr_descriptors(ref.get_heavy_atom_coords()
                               if ref.get_heavy_atom_coords() is not None
                               else np.zeros((4, 3)))
    query_usr = _usr_descriptors(query.get_heavy_atom_coords()
                                  if query.get_heavy_atom_coords() is not None
                                  else np.zeros((4, 3)))
    result["usr_similarity"] = round(usr_similarity(ref_usr, query_usr), 3)

    ref_usrcat = _usrcat_descriptors(ref)
    query_usrcat = _usrcat_descriptors(query)
    result["usrcat_similarity"] = round(usr_similarity(ref_usrcat, query_usrcat), 3)

    result["gaussian_overlap"] = round(gaussian_shape_overlap(ref, query), 3)

    result["volumetric_overlap_pct"] = round(volumetric_overlap(ref, query), 1)

    shape_scores = [
        result["usr_similarity"],
        result["gaussian_overlap"],
        result["volumetric_overlap_pct"] / 100.0,
    ]
    result["shape_tanimoto"] = round(float(np.mean(shape_scores)), 3)

    return result


def batch_shape_analysis(ref: MoleculeRecord,
                          queries: List[MoleculeRecord]) -> List[Dict]:
    """Run shape similarity for all queries against reference."""
    results = []
    for q in queries:
        shape_data = compute_shape_similarity(ref, q)
        shape_data["compound"] = q.name
        results.append(shape_data)
    return results
