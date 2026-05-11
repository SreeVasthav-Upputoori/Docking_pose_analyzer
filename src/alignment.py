"""METHOD 1 & 2: Heavy Atom RMSD and Maximum Common Substructure alignment."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .io_handlers import MoleculeRecord
from .utils import get_logger

warnings.filterwarnings("ignore")


def _lazy_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS, rdMolAlign, rdmolops
    return Chem, AllChem, rdFMCS, rdMolAlign, rdmolops


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Kabsch algorithm for optimal rigid-body superposition. Returns RMSD, rotation, translation."""
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    p = P - centroid_P
    q = Q - centroid_Q

    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    p_rotated = p @ R.T
    rmsd = np.sqrt(np.mean(np.sum((p_rotated - q) ** 2, axis=1)))
    translation = centroid_Q - R @ centroid_P

    return rmsd, R, translation


def compute_heavy_atom_rmsd(ref: MoleculeRecord, query: MoleculeRecord) -> Dict:
    """Compute heavy-atom RMSD between reference and query after optimal alignment."""
    Chem, AllChem, rdFMCS, rdMolAlign, rdmolops = _lazy_rdkit()
    logger = get_logger()
    result = {
        "heavy_atom_rmsd": None,
        "aligned_atoms": 0,
        "total_ref_heavy": ref.num_heavy_atoms,
        "total_query_heavy": query.num_heavy_atoms,
    }

    if not ref.mol or not query.mol:
        return result

    try:
        rmsd_val = rdMolAlign.GetBestRMS(ref.mol, query.mol)
        result["heavy_atom_rmsd"] = round(rmsd_val, 3)
        result["aligned_atoms"] = min(ref.num_heavy_atoms, query.num_heavy_atoms)
    except Exception:
        ref_coords = ref.get_heavy_atom_coords()
        query_coords = query.get_heavy_atom_coords()
        if ref_coords is not None and query_coords is not None:
            n = min(len(ref_coords), len(query_coords))
            if n >= 3:
                rmsd_val, _, _ = kabsch_rmsd(query_coords[:n], ref_coords[:n])
                result["heavy_atom_rmsd"] = round(rmsd_val, 3)
                result["aligned_atoms"] = n

    return result


def find_mcs(ref: MoleculeRecord, query: MoleculeRecord,
             timeout: int = 30) -> Dict:
    """Find Maximum Common Substructure between reference and query."""
    Chem, AllChem, rdFMCS, rdMolAlign, rdmolops = _lazy_rdkit()
    logger = get_logger()
    result = {
        "mcs_smarts": None,
        "mcs_num_atoms": 0,
        "mcs_num_bonds": 0,
        "mcs_rmsd": None,
        "mcs_fraction_ref": 0.0,
        "mcs_fraction_query": 0.0,
    }

    if not ref.mol or not query.mol:
        return result

    try:
        ref_noH = rdmolops.RemoveHs(ref.mol)
        query_noH = rdmolops.RemoveHs(query.mol)

        # Optimization: if the molecules are likely identical, skip full MCS
        ref_smiles = Chem.MolToSmiles(ref_noH, isomericSmiles=True)
        query_smiles = Chem.MolToSmiles(query_noH, isomericSmiles=True)

        if ref_smiles == query_smiles:
            result["mcs_smarts"] = ref_smiles
            result["mcs_num_atoms"] = ref_noH.GetNumAtoms()
            result["mcs_num_bonds"] = ref_noH.GetNumBonds()
            result["mcs_fraction_ref"] = 1.0
            result["mcs_fraction_query"] = 1.0
            
            # Get correct atom mapping even if ordering differs
            query_match = ref_noH.GetSubstructMatch(query_noH)
            if query_match and len(query_match) == ref_noH.GetNumAtoms():
                ref_match = list(range(ref_noH.GetNumAtoms()))
                
                ref_conf = ref_noH.GetConformer()
                query_conf = query_noH.GetConformer()
                ref_coords = np.array([list(ref_conf.GetAtomPosition(idx)) for idx in ref_match])
                query_coords = np.array([list(query_conf.GetAtomPosition(idx)) for idx in query_match])
                
                rmsd_val, _, _ = kabsch_rmsd(query_coords, ref_coords)
                result["mcs_rmsd"] = round(rmsd_val, 3)
                return result

        mcs_result = rdFMCS.FindMCS(
            [ref_noH, query_noH],
            atomCompare=rdFMCS.AtomCompare.CompareAny,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=timeout,
        )

        if mcs_result.canceled:
            mcs_result = rdFMCS.FindMCS(
                [ref_noH, query_noH],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                timeout=timeout,
            )

        result["mcs_smarts"] = mcs_result.smartsString
        result["mcs_num_atoms"] = mcs_result.numAtoms
        result["mcs_num_bonds"] = mcs_result.numBonds

        ref_heavy = ref_noH.GetNumAtoms()
        query_heavy = query_noH.GetNumAtoms()
        result["mcs_fraction_ref"] = round(mcs_result.numAtoms / max(ref_heavy, 1), 3)
        result["mcs_fraction_query"] = round(mcs_result.numAtoms / max(query_heavy, 1), 3)

        if mcs_result.numAtoms >= 3:
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            if mcs_mol:
                ref_match = ref_noH.GetSubstructMatch(mcs_mol)
                query_match = query_noH.GetSubstructMatch(mcs_mol)

                if ref_match and query_match and len(ref_match) == len(query_match):
                    ref_conf = ref_noH.GetConformer()
                    query_conf = query_noH.GetConformer()

                    ref_coords = np.array([
                        list(ref_conf.GetAtomPosition(idx)) for idx in ref_match
                    ])
                    query_coords = np.array([
                        list(query_conf.GetAtomPosition(idx)) for idx in query_match
                    ])

                    rmsd_val, _, _ = kabsch_rmsd(query_coords, ref_coords)
                    result["mcs_rmsd"] = round(rmsd_val, 3)

    except Exception as e:
        logger.debug(f"MCS computation failed for {query.name}: {e}")

    return result


def compute_symmetry_corrected_rmsd(ref: MoleculeRecord, query: MoleculeRecord) -> Optional[float]:
    """Attempt symmetry-corrected RMSD using RDKit's GetBestRMS with symmetry."""
    Chem, AllChem, rdFMCS, rdMolAlign, rdmolops = _lazy_rdkit()

    if not ref.mol or not query.mol:
        return None

    try:
        ref_noH = rdmolops.RemoveHs(ref.mol)
        query_noH = rdmolops.RemoveHs(query.mol)
        rmsd = rdMolAlign.GetBestRMS(ref_noH, query_noH)
        return round(rmsd, 3)
    except Exception:
        return None


def align_to_reference(ref: MoleculeRecord, query: MoleculeRecord) -> Optional[float]:
    """Align query molecule to reference in-place. Returns RMSD or None."""
    Chem, AllChem, rdFMCS, rdMolAlign, rdmolops = _lazy_rdkit()

    if not ref.mol or not query.mol:
        return None

    try:
        rmsd = rdMolAlign.AlignMol(query.mol, ref.mol)
        return round(rmsd, 3)
    except Exception:
        try:
            ref_noH = rdmolops.RemoveHs(ref.mol)
            query_noH = rdmolops.RemoveHs(query.mol)

            mcs_result = rdFMCS.FindMCS(
                [ref_noH, query_noH],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                timeout=20,
            )

            if mcs_result.numAtoms >= 3:
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                ref_match = ref_noH.GetSubstructMatch(mcs_mol)
                query_match = query_noH.GetSubstructMatch(mcs_mol)

                if ref_match and query_match:
                    atom_map = list(zip(query_match, ref_match))
                    rmsd = rdMolAlign.AlignMol(
                        query.mol, ref.mol, atomMap=atom_map
                    )
                    return round(rmsd, 3)
        except Exception as e:
            get_logger().debug(f"Alignment failed for {query.name}: {e}")

    return None


def batch_rmsd_analysis(ref: MoleculeRecord, queries: List[MoleculeRecord]) -> List[Dict]:
    """Compute RMSD metrics for a batch of query molecules against reference."""
    results = []
    for q in queries:
        rmsd_data = compute_heavy_atom_rmsd(ref, q)
        mcs_data = find_mcs(ref, q)
        sym_rmsd = compute_symmetry_corrected_rmsd(ref, q)

        combined = {
            "compound": q.name,
            **rmsd_data,
            **mcs_data,
            "symmetry_corrected_rmsd": sym_rmsd,
        }
        results.append(combined)

    return results
