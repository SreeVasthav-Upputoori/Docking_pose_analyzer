"""I/O handlers for molecular file formats: PDB, SDF, MOL2, PDBQT."""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import get_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _lazy_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolfiles, rdmolops
    return Chem, AllChem, rdmolfiles, rdmolops


def _lazy_openbabel():
    try:
        from openbabel import openbabel as ob
        return ob
    except ImportError:
        return None


class MoleculeRecord:
    """Container for a loaded molecule with metadata."""

    def __init__(self, mol, name: str, source_file: str, pose_index: int = 0,
                 compound_id: str = ""):
        self.mol = mol
        self.name = name or f"UNK_{pose_index}"
        self.source_file = source_file
        self.pose_index = pose_index
        self.compound_id = compound_id or self.name

    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumAtoms() if self.mol else 0

    @property
    def num_heavy_atoms(self) -> int:
        Chem = _lazy_rdkit()[0]
        if not self.mol:
            return 0
        return sum(1 for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1)

    def get_coords(self) -> Optional[np.ndarray]:
        if not self.mol or not self.mol.GetNumConformers():
            return None
        conf = self.mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())])

    def get_heavy_atom_coords(self) -> Optional[np.ndarray]:
        if not self.mol or not self.mol.GetNumConformers():
            return None
        Chem = _lazy_rdkit()[0]
        conf = self.mol.GetConformer()
        coords = []
        for i, atom in enumerate(self.mol.GetAtoms()):
            if atom.GetAtomicNum() > 1:
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
        return np.array(coords) if coords else None


def load_protein(protein_path: str) -> Optional[object]:
    """Load protein structure from PDB file."""
    logger = get_logger()
    path = Path(protein_path)
    if not path.exists():
        logger.error(f"Protein file not found: {path}")
        return None

    Chem, AllChem, rdmolfiles, rdmolops = _lazy_rdkit()
    try:
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if mol is None:
            logger.warning(f"RDKit failed to parse protein, trying as-is: {path}")
            mol = Chem.MolFromPDBFile(str(path), removeHs=True, sanitize=False)
        if mol:
            logger.info(f"Loaded protein: {path.name} ({mol.GetNumAtoms()} atoms)")
        return mol
    except Exception as e:
        logger.error(f"Failed to load protein {path}: {e}")
        return None


def load_molecule_sdf(filepath: str) -> List[MoleculeRecord]:
    """Load molecules from SDF file (supports multi-mol SDF).

    When multiple entries share the same _Name, each gets a unique display
    name (e.g. ``DrugA_pose1``, ``DrugA_pose2``) while ``compound_id``
    preserves the original compound identity for grouping.
    """
    Chem = _lazy_rdkit()[0]
    logger = get_logger()
    records = []
    path = Path(filepath)

    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)

        raw_names: List[str] = []
        mols = []
        for i, mol in enumerate(supplier):
            if mol is None:
                logger.warning(f"Failed to parse molecule {i} from {path.name}")
                continue
            base = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            if not base.strip():
                base = f"{path.stem}_{i}"
            raw_names.append(base)
            mols.append((mol, i))

        name_count: Dict[str, int] = {}
        for bn in raw_names:
            name_count[bn] = name_count.get(bn, 0) + 1

        pose_counter: Dict[str, int] = {}
        for (mol, orig_idx), base_name in zip(mols, raw_names):
            compound_id = base_name
            if name_count[base_name] > 1:
                n = pose_counter.get(base_name, 0)
                pose_counter[base_name] = n + 1
                display_name = f"{base_name}_pose{n + 1}"
            else:
                display_name = base_name
            records.append(MoleculeRecord(mol, display_name, str(path), orig_idx,
                                          compound_id=compound_id))

        n_compounds = len(set(raw_names))
        logger.info(
            f"Loaded {len(records)} poses ({n_compounds} compounds) from {path.name}"
        )
    except Exception as e:
        logger.error(f"Error loading SDF {path}: {e}")

    return records


def _pdbqt_to_pdb_block(pdbqt_text: str) -> str:
    """Convert PDBQT text to PDB-compatible block by stripping extra columns."""
    lines = []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            lines.append(line[:66].ljust(66))
        elif line.startswith("END"):
            lines.append("END")
    return "\n".join(lines)


def load_molecule_pdbqt(filepath: str) -> List[MoleculeRecord]:
    """Load molecules from PDBQT file (handles multi-model)."""
    Chem = _lazy_rdkit()[0]
    logger = get_logger()
    path = Path(filepath)
    records = []

    try:
        text = path.read_text()
        models = re.split(r"^MODEL\s+\d+", text, flags=re.MULTILINE)
        if len(models) <= 1:
            models = [text]
        else:
            models = [m for m in models if m.strip()]

        for i, model_text in enumerate(models):
            pdb_block = _pdbqt_to_pdb_block(model_text)
            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
            if mol is None:
                ob = _lazy_openbabel()
                if ob:
                    conv = ob.OBConversion()
                    conv.SetInFormat("pdbqt")
                    conv.SetOutFormat("sdf")
                    obmol = ob.OBMol()
                    conv.ReadString(obmol, model_text)
                    sdf_block = conv.WriteString(obmol)
                    mol = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=False)

            if mol is None:
                logger.warning(f"Failed to parse model {i} from {path.name}")
                continue

            name = f"{path.stem}_model{i}"
            records.append(MoleculeRecord(mol, name, str(path), i))

        logger.info(f"Loaded {len(records)} poses from PDBQT {path.name}")
    except Exception as e:
        logger.error(f"Error loading PDBQT {path}: {e}")

    return records


def load_molecule_mol2(filepath: str) -> List[MoleculeRecord]:
    """Load molecules from MOL2 file."""
    Chem = _lazy_rdkit()[0]
    logger = get_logger()
    path = Path(filepath)
    records = []

    try:
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
        if mol is None:
            ob = _lazy_openbabel()
            if ob:
                conv = ob.OBConversion()
                conv.SetInFormat("mol2")
                conv.SetOutFormat("sdf")
                obmol = ob.OBMol()
                conv.ReadFile(obmol, str(path))
                sdf_block = conv.WriteString(obmol)
                mol = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=False)

        if mol:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else path.stem
            records.append(MoleculeRecord(mol, name, str(path), 0))
            logger.info(f"Loaded molecule from MOL2: {path.name}")
        else:
            logger.error(f"Failed to parse MOL2: {path.name}")
    except Exception as e:
        logger.error(f"Error loading MOL2 {path}: {e}")

    return records


def load_molecule_pdb(filepath: str) -> List[MoleculeRecord]:
    """Load a ligand molecule from PDB file."""
    Chem = _lazy_rdkit()[0]
    logger = get_logger()
    path = Path(filepath)
    records = []

    try:
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if mol:
            name = path.stem
            records.append(MoleculeRecord(mol, name, str(path), 0))
            logger.info(f"Loaded ligand from PDB: {path.name}")
        else:
            logger.warning(f"Failed to parse ligand PDB: {path.name}")
    except Exception as e:
        logger.error(f"Error loading PDB ligand {path}: {e}")

    return records


LOADERS = {
    ".sdf": load_molecule_sdf,
    ".sd": load_molecule_sdf,
    ".mol2": load_molecule_mol2,
    ".pdbqt": load_molecule_pdbqt,
    ".pdb": load_molecule_pdb,
    ".mol": load_molecule_sdf,
}


def load_molecules(path: str) -> List[MoleculeRecord]:
    """Auto-detect format and load molecules from file or directory."""
    logger = get_logger()
    p = Path(path)
    records = []

    if p.is_file():
        ext = p.suffix.lower()
        loader = LOADERS.get(ext)
        if loader:
            records = loader(str(p))
        else:
            logger.warning(f"Unsupported file format: {ext} ({p.name})")

    elif p.is_dir():
        mol_files = sorted(
            f for f in p.rglob("*")
            if f.suffix.lower() in LOADERS and f.is_file()
        )
        logger.info(f"Scanning directory {p}: found {len(mol_files)} molecular files")
        for mf in mol_files:
            loader = LOADERS.get(mf.suffix.lower())
            if loader:
                records.extend(loader(str(mf)))
    else:
        logger.error(f"Path not found: {p}")

    logger.info(f"Total molecules loaded: {len(records)}")
    return records


def load_reference(ref_path: str) -> Optional[MoleculeRecord]:
    """Load the reference docked pose."""
    logger = get_logger()
    records = load_molecules(ref_path)
    if not records:
        logger.error(f"No reference molecule loaded from {ref_path}")
        return None
    if len(records) > 1:
        logger.warning(f"Multiple molecules in reference file, using first: {records[0].name}")
    return records[0]


def write_sdf(records: List[MoleculeRecord], output_path: str):
    """Write molecules to a single SDF file."""
    Chem = _lazy_rdkit()[0]
    writer = Chem.SDWriter(output_path)
    for rec in records:
        if rec.mol:
            rec.mol.SetProp("_Name", rec.name)
            writer.write(rec.mol)
    writer.close()
    get_logger().info(f"Wrote {len(records)} structures to {output_path}")


def select_top_poses(
    records: List[MoleculeRecord],
    metrics: List[Dict],
    top_n: int = 5,
    score_key: str = "consensus_score",
) -> Tuple[List[MoleculeRecord], List[Dict], Dict[str, List[Dict]]]:
    """Select the top-N poses **per compound** ranked by *score_key*.

    Returns
    -------
    top_records : list[MoleculeRecord]
        Filtered molecule records (top N per compound).
    top_metrics : list[dict]
        Corresponding metric dicts.
    per_compound : dict[str, list[dict]]
        {compound_id: [metrics in rank order]} for inspection.
    """
    logger = get_logger()

    name_to_rec = {r.name: r for r in records}
    name_to_cid = {r.name: r.compound_id for r in records}

    groups: Dict[str, List[Dict]] = {}
    for m in metrics:
        pose_name = m.get("compound", "")
        cid = name_to_cid.get(pose_name, pose_name)
        groups.setdefault(cid, []).append(m)

    top_records: List[MoleculeRecord] = []
    top_metrics: List[Dict] = []
    per_compound: Dict[str, List[Dict]] = {}

    for cid, pose_metrics in sorted(groups.items()):
        ranked = sorted(pose_metrics, key=lambda x: x.get(score_key, 0), reverse=True)
        selected = ranked[:top_n]

        for rank, pm in enumerate(selected, 1):
            pm["rank"] = rank
            pm["compound_id"] = cid

        per_compound[cid] = selected
        top_metrics.extend(selected)

        for pm in selected:
            rec = name_to_rec.get(pm["compound"])
            if rec:
                top_records.append(rec)

    n_compounds = len(per_compound)
    logger.info(
        f"Top-{top_n} selection: {len(top_metrics)} poses kept "
        f"from {n_compounds} compounds ({len(metrics)} total input poses)"
    )
    return top_records, top_metrics, per_compound
