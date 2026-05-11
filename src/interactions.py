"""METHOD 3: Interaction Fingerprint Similarity (IFP) analysis."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io_handlers import MoleculeRecord
from .utils import get_logger

warnings.filterwarnings("ignore")


def _tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprints."""
    fp1 = fp1.astype(bool)
    fp2 = fp2.astype(bool)
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return float(intersection / union)


def _dice(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Dice similarity between two binary fingerprints."""
    fp1 = fp1.astype(bool)
    fp2 = fp2.astype(bool)
    intersection = np.sum(fp1 & fp2)
    total = np.sum(fp1) + np.sum(fp2)
    if total == 0:
        return 0.0
    return float(2 * intersection / total)


def _overlap_percentage(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Percentage of reference interactions found in query."""
    fp1 = fp1.astype(bool)
    fp2 = fp2.astype(bool)
    ref_bits = np.sum(fp1)
    if ref_bits == 0:
        return 0.0
    overlap = np.sum(fp1 & fp2)
    return float(overlap / ref_bits * 100)


class InteractionFingerprinter:
    """Generate and compare protein-ligand interaction fingerprints."""

    INTERACTION_TYPES = [
        "HBDonor", "HBAcceptor", "Hydrophobic", "PiStacking",
        "PiCation", "SaltBridge", "MetalCoordination",
    ]

    def __init__(self, protein_path: str, distance_cutoffs: Optional[Dict] = None):
        self.protein_path = protein_path
        self.logger = get_logger()
        self.cutoffs = distance_cutoffs or {
            "HBDonor": 3.5,
            "HBAcceptor": 3.5,
            "Hydrophobic": 4.5,
            "PiStacking": 5.5,
            "PiCation": 6.0,
            "SaltBridge": 4.0,
            "MetalCoordination": 2.8,
        }
        self._prolif_available = False
        self._plip_available = False
        self._protein_mol = None
        self._check_backends()

    def _check_backends(self):
        try:
            import prolif
            self._prolif_available = True
            self.logger.info("ProLIF backend available for IFP")
        except ImportError:
            self._prolif_available = False

        try:
            from plip.structure.preparation import PDBComplex
            self._plip_available = True
            self.logger.info("PLIP backend available for interactions")
        except ImportError:
            self._plip_available = False

    def compute_ifp_prolif(self, ligand_record: MoleculeRecord) -> Optional[np.ndarray]:
        """Compute IFP using ProLIF."""
        if not self._prolif_available:
            return None

        try:
            import prolif
            import MDAnalysis as mda
            from rdkit import Chem

            prot = mda.Universe(self.protein_path)
            prot_mol = prolif.Molecule.from_mda(prot)

            lig_mol = prolif.Molecule.from_rdkit(ligand_record.mol)

            fp = prolif.Fingerprint(
                interactions=[
                    "HBDonor", "HBAcceptor", "Hydrophobic",
                    "PiStacking", "PiCation", "Anionic", "Cationic",
                    "MetalAcceptor",
                ]
            )

            fp.run_from_iterable([lig_mol], prot_mol, progress=False)
            bv = fp.to_bitvectors()
            if bv:
                return np.array(bv[0])
            return None
        except Exception as e:
            self.logger.debug(f"ProLIF IFP failed for {ligand_record.name}: {e}")
            return None

    def compute_ifp_geometric(self, ligand_record: MoleculeRecord,
                               protein_mol=None) -> np.ndarray:
        """Geometric distance-based IFP as universal fallback."""
        from rdkit import Chem

        logger = self.logger
        if protein_mol is None:
            protein_mol = Chem.MolFromPDBFile(self.protein_path, removeHs=False, sanitize=False)
            if protein_mol is None:
                return np.zeros(100, dtype=np.int8)

        lig = ligand_record.mol
        if lig is None or not lig.GetNumConformers():
            return np.zeros(100, dtype=np.int8)

        try:
            prot_conf = protein_mol.GetConformer()
            lig_conf = lig.GetConformer()
        except Exception:
            return np.zeros(100, dtype=np.int8)

        prot_info = []
        for atom in protein_mol.GetAtoms():
            pos = prot_conf.GetAtomPosition(atom.GetIdx())
            res_info = atom.GetPDBResidueInfo()
            res_id = 0
            if res_info:
                res_id = res_info.GetResidueNumber()
            prot_info.append({
                "idx": atom.GetIdx(),
                "num": atom.GetAtomicNum(),
                "pos": np.array([pos.x, pos.y, pos.z]),
                "res_id": res_id,
                "is_aromatic": atom.GetIsAromatic(),
            })

        lig_info = []
        for atom in lig.GetAtoms():
            pos = lig_conf.GetAtomPosition(atom.GetIdx())
            lig_info.append({
                "idx": atom.GetIdx(),
                "num": atom.GetAtomicNum(),
                "pos": np.array([pos.x, pos.y, pos.z]),
                "is_aromatic": atom.GetIsAromatic(),
                "formal_charge": atom.GetFormalCharge(),
            })

        unique_res = sorted(set(p["res_id"] for p in prot_info if p["res_id"] > 0))
        res_to_idx = {r: i for i, r in enumerate(unique_res)}
        n_res = len(unique_res)
        n_types = 5
        fp = np.zeros(n_res * n_types, dtype=np.int8)

        for la in lig_info:
            for pa in prot_info:
                if pa["res_id"] == 0:
                    continue
                dist = np.linalg.norm(la["pos"] - pa["pos"])
                ri = res_to_idx[pa["res_id"]]
                base = ri * n_types

                # HB donor/acceptor (N, O within 3.5 A)
                if dist <= 3.5 and la["num"] in (7, 8) and pa["num"] in (7, 8):
                    fp[base + 0] = 1  # HBond
                # Hydrophobic (C-C within 4.5 A)
                if dist <= 4.5 and la["num"] == 6 and pa["num"] == 6:
                    fp[base + 1] = 1  # Hydrophobic
                # Pi-stacking (aromatic-aromatic within 5.5 A)
                if dist <= 5.5 and la["is_aromatic"] and pa["is_aromatic"]:
                    fp[base + 2] = 1  # PiStack
                # Salt bridge (charged within 4.0 A)
                if dist <= 4.0 and la.get("formal_charge", 0) != 0 and pa["num"] in (7, 8):
                    fp[base + 3] = 1  # SaltBridge
                # Close contact
                if dist <= 3.0:
                    fp[base + 4] = 1

        return fp

    def compute_ifp(self, ligand_record: MoleculeRecord) -> np.ndarray:
        """Compute IFP using best available backend."""
        if self._prolif_available:
            fp = self.compute_ifp_prolif(ligand_record)
            if fp is not None:
                return fp

        return self.compute_ifp_geometric(ligand_record)

    def compute_plip_interactions(self, ligand_record: MoleculeRecord) -> Dict:
        """Extract detailed interactions using PLIP."""
        interactions = {
            "hbonds": [],
            "hydrophobic": [],
            "pi_stacking": [],
            "pi_cation": [],
            "salt_bridges": [],
            "metal_coordination": [],
            "water_bridges": [],
        }

        if not self._plip_available:
            return interactions

        try:
            from plip.structure.preparation import PDBComplex
            import tempfile
            from rdkit import Chem

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
                pdb_block = Chem.MolToPDBBlock(ligand_record.mol)
                prot_text = Path(self.protein_path).read_text()
                combined = prot_text.rstrip() + "\n" + pdb_block
                f.write(combined)
                tmp_path = f.name

            complex_obj = PDBComplex()
            complex_obj.load_pdb(tmp_path)
            complex_obj.analyze()

            for bsite_id, bsite in complex_obj.interaction_sets.items():
                for hb in bsite.hbonds_ldon + bsite.hbonds_pdon:
                    interactions["hbonds"].append({
                        "residue": f"{hb.resnr}{hb.reschain}",
                        "distance": round(hb.distance_ah, 2),
                        "type": hb.type,
                    })
                for hp in bsite.hydrophobic_contacts:
                    interactions["hydrophobic"].append({
                        "residue": f"{hp.resnr}{hp.reschain}",
                        "distance": round(hp.distance, 2),
                    })
                for ps in bsite.pistacking:
                    interactions["pi_stacking"].append({
                        "residue": f"{ps.resnr}{ps.reschain}",
                        "distance": round(ps.distance, 2),
                        "type": ps.type,
                    })
                for sb in bsite.saltbridge_lneg + bsite.saltbridge_pneg:
                    interactions["salt_bridges"].append({
                        "residue": f"{sb.resnr}{sb.reschain}",
                        "distance": round(sb.distance, 2),
                    })
                for wb in bsite.water_bridges:
                    interactions["water_bridges"].append({
                        "residue": f"{wb.resnr}{wb.reschain}",
                        "distance_aw": round(wb.distance_aw, 2),
                        "distance_dw": round(wb.distance_dw, 2),
                    })

            Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.debug(f"PLIP analysis failed for {ligand_record.name}: {e}")

        return interactions

    def compare_ifps(self, ref_fp: np.ndarray, query_fp: np.ndarray) -> Dict:
        """Compare two interaction fingerprints."""
        max_len = max(len(ref_fp), len(query_fp))
        fp1 = np.zeros(max_len, dtype=np.int8)
        fp2 = np.zeros(max_len, dtype=np.int8)
        fp1[:len(ref_fp)] = ref_fp
        fp2[:len(query_fp)] = query_fp

        return {
            "ifp_tanimoto": round(_tanimoto(fp1, fp2), 3),
            "ifp_dice": round(_dice(fp1, fp2), 3),
            "ifp_overlap_pct": round(_overlap_percentage(fp1, fp2), 1),
            "ref_interactions": int(np.sum(ref_fp.astype(bool))),
            "query_interactions": int(np.sum(query_fp.astype(bool))),
            "shared_interactions": int(np.sum(fp1.astype(bool) & fp2.astype(bool))),
        }

    def compute_gap_analysis(self, ref_interactions: Dict, query_interactions: Dict) -> Dict:
        """Analyze conserved, missing (gaps), and new interactions."""
        
        def _get_set(int_dict):
            s = set()
            for key, items in int_dict.items():
                for item in items:
                    residue = item.get("residue", "UNK")
                    s.add((residue, key))
            return s

        ref_set = _get_set(ref_interactions)
        query_set = _get_set(query_interactions)

        conserved = ref_set.intersection(query_set)
        missing = ref_set.difference(query_set)
        new = query_set.difference(ref_set)

        return {
            "conserved_count": len(conserved),
            "missing_count": len(missing),
            "new_count": len(new),
            "conserved_list": sorted([f"{r} ({t})" for r, t in conserved]),
            "missing_list": sorted([f"{r} ({t})" for r, t in missing]),
            "new_list": sorted([f"{r} ({t})" for r, t in new]),
            "gap_score": round(len(conserved) / max(len(ref_set), 1), 3)
        }


def batch_ifp_analysis(protein_path: str, ref: MoleculeRecord,
                        queries: List[MoleculeRecord]) -> Tuple[List[Dict], np.ndarray]:
    """Compute IFP comparison for all query molecules against reference."""
    fingerprinter = InteractionFingerprinter(protein_path)

    ref_fp = fingerprinter.compute_ifp(ref)
    ref_plip = fingerprinter.compute_plip_interactions(ref)
    
    all_fps = [ref_fp]
    results = []

    for q in queries:
        q_fp = fingerprinter.compute_ifp(q)
        all_fps.append(q_fp)
        comparison = fingerprinter.compare_ifps(ref_fp, q_fp)
        comparison["compound"] = q.name

        q_plip = fingerprinter.compute_plip_interactions(q)
        comparison["n_hbonds"] = len(q_plip["hbonds"])
        comparison["n_hydrophobic"] = len(q_plip["hydrophobic"])
        comparison["n_pistacking"] = len(q_plip["pi_stacking"])
        comparison["n_saltbridges"] = len(q_plip["salt_bridges"])

        # Gap Analysis
        gaps = fingerprinter.compute_gap_analysis(ref_plip, q_plip)
        comparison.update(gaps)

        results.append(comparison)

    max_len = max(len(fp) for fp in all_fps)
    fp_matrix = np.zeros((len(all_fps), max_len), dtype=np.int8)
    for i, fp in enumerate(all_fps):
        fp_matrix[i, :len(fp)] = fp

    return results, fp_matrix
