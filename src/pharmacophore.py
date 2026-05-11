"""METHOD 4: Pharmacophore Feature Consistency analysis."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .io_handlers import MoleculeRecord
from .utils import get_logger

warnings.filterwarnings("ignore")


FEATURE_DEFINITIONS = {
    "Donor": "[!$([#1])][N,O;!H0]",
    "Acceptor": "[#7,#8;!$([NH2]C(=O));!$([OH]c)]",
    "Aromatic": "a1aaaaa1",
    "Hydrophobic": "[C;!$(C=O);!$(C#N)]",
    "PosIonizable": "[NH2;!$([NH2]C(=O)),$([NH3+]),$(n1ccnc1)]",
    "NegIonizable": "[C;$(C(=O)[OH]),$(C(=O)[O-]),$(S(=O)(=O)[OH]),$(P(=O)([OH])([OH]))]",
}


class PharmacophoreAnalyzer:
    """Extract and compare pharmacophore features from docked poses."""

    def __init__(self):
        self.logger = get_logger()

    def extract_features(self, record: MoleculeRecord) -> List[Dict]:
        """Extract pharmacophore features with 3D coordinates."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        features = []
        mol = record.mol
        if not mol or not mol.GetNumConformers():
            return features

        conf = mol.GetConformer()

        for feat_name, smarts in FEATURE_DEFINITIONS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                coords = []
                for idx in match:
                    pos = conf.GetAtomPosition(idx)
                    coords.append([pos.x, pos.y, pos.z])
                centroid = np.mean(coords, axis=0)
                features.append({
                    "type": feat_name,
                    "atom_indices": list(match),
                    "centroid": centroid,
                    "n_atoms": len(match),
                })

        # RDKit pharmacophore factory as supplement
        try:
            from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
            from rdkit.Chem import ChemicalFeatures
            from rdkit import RDConfig
            import os

            fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
            mol_feats = factory.GetFeaturesForMol(mol)

            rdkit_types = {"Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable"}
            existing_centroids = set()
            for f in features:
                existing_centroids.add(tuple(np.round(f["centroid"], 2)))

            for feat in mol_feats:
                if feat.GetFamily() not in rdkit_types:
                    continue
                pos = feat.GetPos()
                centroid = np.array([pos.x, pos.y, pos.z])
                key = tuple(np.round(centroid, 2))
                if key not in existing_centroids:
                    type_map = {"Hydrophobe": "Hydrophobic"}
                    ftype = type_map.get(feat.GetFamily(), feat.GetFamily())
                    features.append({
                        "type": ftype,
                        "atom_indices": list(feat.GetAtomIds()),
                        "centroid": centroid,
                        "n_atoms": len(feat.GetAtomIds()),
                    })
                    existing_centroids.add(key)
        except Exception:
            pass

        return features

    def match_features(self, ref_features: List[Dict],
                       query_features: List[Dict],
                       distance_threshold: float = 2.0) -> Dict:
        """Match pharmacophore features between reference and query."""
        result = {
            "pharmacophore_score": 0.0,
            "feature_overlap_pct": 0.0,
            "matched_features": 0,
            "total_ref_features": len(ref_features),
            "total_query_features": len(query_features),
            "feature_details": {},
            "mean_feature_distance": None,
        }

        if not ref_features or not query_features:
            return result

        for feat_type in FEATURE_DEFINITIONS:
            ref_of_type = [f for f in ref_features if f["type"] == feat_type]
            query_of_type = [f for f in query_features if f["type"] == feat_type]
            result["feature_details"][feat_type] = {
                "ref_count": len(ref_of_type),
                "query_count": len(query_of_type),
                "matched": 0,
                "avg_distance": None,
            }

        total_matched = 0
        total_distance = 0.0
        distances_list = []

        for feat_type in FEATURE_DEFINITIONS:
            ref_of_type = [f for f in ref_features if f["type"] == feat_type]
            query_of_type = [f for f in query_features if f["type"] == feat_type]

            if not ref_of_type or not query_of_type:
                continue

            ref_coords = np.array([f["centroid"] for f in ref_of_type])
            query_coords = np.array([f["centroid"] for f in query_of_type])
            dist_matrix = cdist(ref_coords, query_coords)

            n = min(len(ref_of_type), len(query_of_type))
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched = 0
            type_distances = []
            for r, c in zip(row_ind, col_ind):
                d = dist_matrix[r, c]
                if d <= distance_threshold:
                    matched += 1
                    type_distances.append(d)

            total_matched += matched
            distances_list.extend(type_distances)
            result["feature_details"][feat_type]["matched"] = matched
            if type_distances:
                result["feature_details"][feat_type]["avg_distance"] = round(
                    np.mean(type_distances), 2
                )

        result["matched_features"] = total_matched
        n_ref = len(ref_features)
        result["feature_overlap_pct"] = round(total_matched / max(n_ref, 1) * 100, 1)

        if distances_list:
            result["mean_feature_distance"] = round(np.mean(distances_list), 3)
            max_score = 1.0
            mean_d = np.mean(distances_list)
            result["pharmacophore_score"] = round(
                max_score * np.exp(-0.5 * mean_d) * (total_matched / max(n_ref, 1)),
                3,
            )
        else:
            result["pharmacophore_score"] = 0.0

        return result


def batch_pharmacophore_analysis(ref: MoleculeRecord,
                                  queries: List[MoleculeRecord]) -> List[Dict]:
    """Run pharmacophore analysis for all queries against reference."""
    analyzer = PharmacophoreAnalyzer()
    ref_features = analyzer.extract_features(ref)

    results = []
    for q in queries:
        q_features = analyzer.extract_features(q)
        match_result = analyzer.match_features(ref_features, q_features)
        match_result["compound"] = q.name
        results.append(match_result)

    return results
