"""Automated pose classification based on multi-method consensus."""

from typing import Dict, List, Optional

import numpy as np

from .utils import get_logger


POSE_CLASSES = {
    "CONSERVED": "CONSERVED BINDING MODE",
    "PARTIAL": "PARTIAL CONSERVATION",
    "DIFFERENT": "DIFFERENT POSE",
    "FLIPPED": "FLIPPED ORIENTATION",
    "OUTLIER": "OUTLIER",
}


class PoseClassifier:
    """Classify docked poses based on multi-method comparison metrics."""

    def __init__(self, thresholds: Optional[Dict] = None):
        self.logger = get_logger()
        self.thresholds = thresholds or self._default_thresholds()

    @staticmethod
    def _default_thresholds() -> Dict:
        return {
            "rmsd_conserved": 2.0,
            "rmsd_partial": 4.0,
            "ifp_conserved": 0.7,
            "ifp_partial": 0.4,
            "pharma_conserved": 0.6,
            "pharma_partial": 0.3,
            "shape_conserved": 0.7,
            "shape_partial": 0.4,
            "com_conserved": 2.0,
            "com_partial": 4.0,
            "angle_flip": 120.0,
        }

    def classify_single(self, metrics: Dict) -> Dict:
        """Classify a single compound's pose based on all metrics."""
        t = self.thresholds

        rmsd = metrics.get("heavy_atom_rmsd")
        mcs_rmsd = metrics.get("mcs_rmsd")
        ifp_sim = metrics.get("ifp_tanimoto")
        pharma_score = metrics.get("pharmacophore_score")
        shape_score = metrics.get("shape_tanimoto")
        com_dist = metrics.get("com_distance")
        axis_angle = metrics.get("principal_axis_angle")
        is_flipped = metrics.get("is_flipped", False)
        orient_score = metrics.get("orientation_score")

        scores = {
            "rmsd_score": 0.0,
            "ifp_score": 0.0,
            "pharma_score_norm": 0.0,
            "shape_score_norm": 0.0,
            "orient_score_norm": 0.0,
        }

        if rmsd is not None:
            if rmsd <= t["rmsd_conserved"]:
                scores["rmsd_score"] = 1.0
            elif rmsd <= t["rmsd_partial"]:
                scores["rmsd_score"] = 0.5
            else:
                scores["rmsd_score"] = 0.0

        if ifp_sim is not None:
            if ifp_sim >= t["ifp_conserved"]:
                scores["ifp_score"] = 1.0
            elif ifp_sim >= t["ifp_partial"]:
                scores["ifp_score"] = 0.5
            else:
                scores["ifp_score"] = 0.0

        if pharma_score is not None:
            if pharma_score >= t["pharma_conserved"]:
                scores["pharma_score_norm"] = 1.0
            elif pharma_score >= t["pharma_partial"]:
                scores["pharma_score_norm"] = 0.5
            else:
                scores["pharma_score_norm"] = 0.0

        if shape_score is not None:
            if shape_score >= t["shape_conserved"]:
                scores["shape_score_norm"] = 1.0
            elif shape_score >= t["shape_partial"]:
                scores["shape_score_norm"] = 0.5
            else:
                scores["shape_score_norm"] = 0.0

        if orient_score is not None:
            scores["orient_score_norm"] = min(orient_score, 1.0)

        weights = {"rmsd_score": 0.25, "ifp_score": 0.25, "pharma_score_norm": 0.2,
                    "shape_score_norm": 0.15, "orient_score_norm": 0.15}

        available_weight = 0.0
        weighted_sum = 0.0
        for key, w in weights.items():
            val = scores[key]
            src_metric = {
                "rmsd_score": rmsd,
                "ifp_score": ifp_sim,
                "pharma_score_norm": pharma_score,
                "shape_score_norm": shape_score,
                "orient_score_norm": orient_score,
            }.get(key)
            if src_metric is not None:
                weighted_sum += val * w
                available_weight += w

        consensus_score = weighted_sum / max(available_weight, 0.01)

        if is_flipped and ifp_sim is not None and ifp_sim >= t["ifp_partial"]:
            classification = POSE_CLASSES["FLIPPED"]
        elif consensus_score >= 0.75:
            classification = POSE_CLASSES["CONSERVED"]
        elif consensus_score >= 0.4:
            classification = POSE_CLASSES["PARTIAL"]
        elif consensus_score >= 0.15:
            classification = POSE_CLASSES["DIFFERENT"]
        else:
            classification = POSE_CLASSES["OUTLIER"]

        return {
            "classification": classification,
            "consensus_score": round(consensus_score, 3),
            "component_scores": {k: round(v, 3) for k, v in scores.items()},
        }

    def classify_batch(self, all_metrics: List[Dict]) -> List[Dict]:
        """Classify all compounds."""
        results = []
        for m in all_metrics:
            cls_result = self.classify_single(m)
            cls_result["compound"] = m.get("compound", "unknown")
            results.append(cls_result)

        class_counts = {}
        for r in results:
            c = r["classification"]
            class_counts[c] = class_counts.get(c, 0) + 1

        self.logger.info(f"Classification summary: {class_counts}")
        return results


def compute_consensus_score(metrics: Dict) -> float:
    """Quick consensus score from combined metrics."""
    classifier = PoseClassifier()
    result = classifier.classify_single(metrics)
    return result["consensus_score"]
