#!/usr/bin/env python3
"""
Docking Pose Consistency Tool
==============================
Automated pipeline for comparing docked ligand poses against a reference pose
and evaluating binding mode conservation.

Usage:
    python pose_consistency.py --protein protein.pdb --reference ref_pose.sdf --poses docked_folder/
    python pose_consistency.py --protein protein.pdb --reference ref_pose.sdf --poses poses.sdf -o results/
"""

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Docking Pose Consistency Tool - Compare docked poses against a reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pose_consistency.py --protein protein.pdb --reference ref_pose.sdf --poses docked/
  python pose_consistency.py --protein protein.pdb --reference ref.sdf --poses ligands.sdf -o results/
  python pose_consistency.py --protein protein.pdb --reference ref.sdf --poses docked/ --skip-ifp --skip-shape
        """,
    )

    parser.add_argument("--protein", required=True, help="Protein structure file (PDB)")
    parser.add_argument("--reference", required=True, help="Reference docked pose (SDF/MOL2/PDBQT/PDB)")
    parser.add_argument("--poses", required=True, help="Docked poses file or directory")
    parser.add_argument("-o", "--output", default="consistency_results", help="Output directory (default: consistency_results)")
    parser.add_argument("--n-clusters", type=int, default=0, help="Number of clusters (0=auto)")
    parser.add_argument("--rmsd-cutoff", type=float, default=2.0, help="RMSD cutoff for conserved classification (default: 2.0)")
    parser.add_argument("--mcs-timeout", type=int, default=30, help="MCS search timeout in seconds (default: 30)")
    parser.add_argument("--skip-ifp", action="store_true", help="Skip interaction fingerprint analysis")
    parser.add_argument("--skip-shape", action="store_true", help="Skip shape similarity analysis")
    parser.add_argument("--skip-pharmacophore", action="store_true", help="Skip pharmacophore analysis")
    parser.add_argument("--top-n", type=int, default=5, help="Keep top N poses per compound (default: 5)")
    parser.add_argument("--skip-plip", action="store_true", help="Skip PLIP detailed interaction analysis")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel workers (default: 1)")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    from src.utils import setup_logging, ProgressTracker
    logger = setup_logging(output_dir, args.verbose)
    logger.info("=" * 70)
    logger.info("DOCKING POSE CONSISTENCY TOOL v1.0")
    logger.info("=" * 70)

    # -------------------------------------------------------------------
    # 1. Load molecules
    # -------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading molecules...")
    from src.io_handlers import load_protein, load_reference, load_molecules, write_sdf, select_top_poses

    protein = load_protein(args.protein)
    if protein is None:
        logger.error("Failed to load protein. Continuing without protein-dependent analyses.")

    ref = load_reference(args.reference)
    if ref is None:
        logger.error("FATAL: Cannot load reference pose. Exiting.")
        sys.exit(1)
    logger.info(f"Reference: {ref.name} ({ref.num_heavy_atoms} heavy atoms)")

    queries = load_molecules(args.poses)
    if not queries:
        logger.error("FATAL: No docked poses loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(queries)} docked poses for comparison")

    # -------------------------------------------------------------------
    # 2. RMSD & MCS Analysis (Methods 1 & 2)
    # -------------------------------------------------------------------
    logger.info("\n[STEP 2] Computing RMSD and MCS analysis (poses kept as-docked)...")
    from src.alignment import batch_rmsd_analysis

    rmsd_results = batch_rmsd_analysis(ref, queries)
    logger.info(f"RMSD analysis complete for {len(rmsd_results)} compounds")

    # Save original (unmodified) structures
    from rdkit import Chem
    ref_sdf = str(output_dir / "reference_pose.sdf")
    w = Chem.SDWriter(ref_sdf)
    ref.mol.SetProp("_Name", ref.name)
    w.write(ref.mol)
    w.close()

    write_sdf(queries, str(output_dir / "docked_poses.sdf"))

    # -------------------------------------------------------------------
    # 3. Interaction Fingerprint Analysis (Method 3)
    # -------------------------------------------------------------------
    ifp_results = []
    fp_matrix = None
    if not args.skip_ifp:
        logger.info("\n[STEP 3] Computing interaction fingerprints...")
        from src.interactions import batch_ifp_analysis
        try:
            ifp_results, fp_matrix = batch_ifp_analysis(args.protein, ref, queries)
            logger.info(f"IFP analysis complete: {len(ifp_results)} compounds")
        except Exception as e:
            logger.warning(f"IFP analysis failed: {e}. Continuing without IFP.")
    else:
        logger.info("\n[STEP 3] Skipping IFP analysis (--skip-ifp)")

    # -------------------------------------------------------------------
    # 4. Pharmacophore Analysis (Method 4)
    # -------------------------------------------------------------------
    pharma_results = []
    if not args.skip_pharmacophore:
        logger.info("\n[STEP 4] Computing pharmacophore consistency...")
        from src.pharmacophore import batch_pharmacophore_analysis
        try:
            pharma_results = batch_pharmacophore_analysis(ref, queries)
            logger.info(f"Pharmacophore analysis complete: {len(pharma_results)} compounds")
        except Exception as e:
            logger.warning(f"Pharmacophore analysis failed: {e}. Continuing.")
    else:
        logger.info("\n[STEP 4] Skipping pharmacophore analysis (--skip-pharmacophore)")

    # -------------------------------------------------------------------
    # 5. Orientation Analysis (Method 5)
    # -------------------------------------------------------------------
    logger.info("\n[STEP 5] Computing binding orientation vectors...")
    from src.orientation import batch_orientation_analysis
    orient_results = batch_orientation_analysis(ref, queries)
    logger.info(f"Orientation analysis complete: {len(orient_results)} compounds")

    # -------------------------------------------------------------------
    # 6. Shape Similarity Analysis (Method 6)
    # -------------------------------------------------------------------
    shape_results = []
    if not args.skip_shape:
        logger.info("\n[STEP 6] Computing shape similarity...")
        from src.shape import batch_shape_analysis
        try:
            shape_results = batch_shape_analysis(ref, queries)
            logger.info(f"Shape analysis complete: {len(shape_results)} compounds")
        except Exception as e:
            logger.warning(f"Shape analysis failed: {e}. Continuing.")
    else:
        logger.info("\n[STEP 6] Skipping shape analysis (--skip-shape)")

    # -------------------------------------------------------------------
    # 7. Merge all metrics
    # -------------------------------------------------------------------
    logger.info("\n[STEP 7] Merging metrics and classifying poses...")

    name_to_cid = {q.name: q.compound_id for q in queries}

    merged = {}
    for r in rmsd_results:
        name = r["compound"]
        merged[name] = dict(r)
        merged[name]["compound_id"] = name_to_cid.get(name, name)

    for dataset in [ifp_results, pharma_results, orient_results, shape_results]:
        for r in dataset:
            name = r.get("compound")
            if name and name in merged:
                for k, v in r.items():
                    if k != "compound":
                        merged[name][k] = v

    all_metrics = list(merged.values())

    # -------------------------------------------------------------------
    # 8. Classification
    # -------------------------------------------------------------------
    from src.classification import PoseClassifier

    thresholds = PoseClassifier._default_thresholds()
    thresholds["rmsd_conserved"] = args.rmsd_cutoff
    classifier = PoseClassifier(thresholds=thresholds)
    classifications = classifier.classify_batch(all_metrics)

    for m, c in zip(all_metrics, classifications):
        m["classification"] = c["classification"]
        m["consensus_score"] = c["consensus_score"]

    # -------------------------------------------------------------------
    # 8b. Top-N pose selection per compound
    # -------------------------------------------------------------------
    logger.info(f"\n[STEP 8b] Selecting top {args.top_n} poses per compound...")
    top_records, top_metrics, per_compound = select_top_poses(
        queries, all_metrics, top_n=args.top_n
    )

    top_n_sdf = str(output_dir / f"top{args.top_n}_poses.sdf")
    from rdkit import Chem as _Chem_top
    _tw = _Chem_top.SDWriter(top_n_sdf)
    for rec, met in zip(top_records, top_metrics):
        if rec.mol:
            rec.mol.SetProp("_Name", rec.name)
            rec.mol.SetProp("compound_id", met.get("compound_id", rec.compound_id))
            rec.mol.SetProp("rank", str(met.get("rank", "")))
            rec.mol.SetProp("consensus_score", f"{met.get('consensus_score', 0):.3f}")
            rec.mol.SetProp("classification", met.get("classification", ""))
            rmsd_val = met.get("heavy_atom_rmsd")
            if rmsd_val is not None:
                rec.mol.SetProp("rmsd_to_reference", f"{rmsd_val:.3f}")
            _tw.write(rec.mol)
    _tw.close()
    logger.info(f"Top-{args.top_n} poses SDF: {top_n_sdf}")

    for cid, poses in per_compound.items():
        ranks = ", ".join(f"#{p['rank']}={p['compound']}" for p in poses)
        logger.info(f"  {cid}: {ranks}")

    # -------------------------------------------------------------------
    # 9. Build results DataFrame and save CSV (all poses)
    # -------------------------------------------------------------------
    results_df = pd.DataFrame(all_metrics)

    csv_cols = [
        "compound", "compound_id", "rank",
        "heavy_atom_rmsd", "symmetry_corrected_rmsd", "mcs_rmsd",
        "mcs_num_atoms", "mcs_fraction_ref",
        "ifp_tanimoto", "ifp_dice", "ifp_overlap_pct",
        "n_hbonds", "n_hydrophobic", "n_pistacking", "n_saltbridges",
        "pharmacophore_score", "feature_overlap_pct", "matched_features",
        "usr_similarity", "usrcat_similarity", "gaussian_overlap",
        "volumetric_overlap_pct", "shape_tanimoto",
        "com_distance", "principal_axis_angle", "orientation_score", "is_flipped",
        "conserved_count", "missing_count", "new_count", "gap_score",
        "consensus_score", "classification",
    ]
    csv_cols = [c for c in csv_cols if c in results_df.columns]
    csv_df = results_df[csv_cols].copy()

    csv_path = str(output_dir / "pose_consistency_results.csv")
    csv_df.to_csv(csv_path, index=False, float_format="%.3f")
    logger.info(f"All-poses CSV: {csv_path}")

    top_df = pd.DataFrame(top_metrics)
    top_csv_cols = [c for c in csv_cols if c in top_df.columns]
    top_csv_path = str(output_dir / f"top{args.top_n}_poses.csv")
    top_df[top_csv_cols].to_csv(top_csv_path, index=False, float_format="%.3f")
    logger.info(f"Top-{args.top_n} CSV: {top_csv_path}")

    # -------------------------------------------------------------------
    # 10. Clustering
    # -------------------------------------------------------------------
    logger.info("\n[STEP 8] Clustering analysis...")
    from src.clustering import (
        build_feature_matrix, hierarchical_clustering, compute_pca,
        compute_umap, compute_similarity_matrix, ifp_clustering,
    )

    feature_matrix, feat_names, feat_cols = build_feature_matrix(all_metrics)
    cluster_result = None
    pca_result = None
    umap_result = None
    sim_matrix = None
    sim_names = None

    if len(all_metrics) >= 3:
        cluster_result = hierarchical_clustering(
            feature_matrix, feat_names, n_clusters=args.n_clusters
        )

        for m, label in zip(all_metrics, cluster_result["labels"]):
            m["cluster"] = label

        pca_result = compute_pca(feature_matrix, feat_names)
        umap_result = compute_umap(feature_matrix, feat_names)
        sim_matrix, sim_names = compute_similarity_matrix(all_metrics)

        if fp_matrix is not None and len(fp_matrix) > 1:
            names_with_ref = [ref.name] + [q.name for q in queries]
            ifp_cluster = ifp_clustering(fp_matrix, names_with_ref)
            logger.info(f"IFP clustering: {len(ifp_cluster['clusters'])} clusters")
    else:
        logger.info("Too few compounds for clustering (need >= 3)")

    # Update CSV with cluster info
    if "cluster" in results_df.columns or any("cluster" in m for m in all_metrics):
        results_df = pd.DataFrame(all_metrics)
        csv_cols_full = csv_cols + (["cluster"] if "cluster" in results_df.columns else [])
        csv_cols_full = [c for c in csv_cols_full if c in results_df.columns]
        results_df[csv_cols_full].to_csv(csv_path, index=False, float_format="%.3f")

    # -------------------------------------------------------------------
    # 11. Visualization
    # -------------------------------------------------------------------
    plot_paths = {}
    if not args.no_plots:
        logger.info("\n[STEP 9] Generating visualizations...")
        from src.visualization import (
            plot_rmsd_histogram, plot_similarity_heatmap, plot_dendrogram,
            plot_pca_embedding, plot_classification_summary,
            plot_interaction_conservation, plot_radar_chart,
            generate_plotly_heatmap, generate_py3dmol_view, generate_pymol_script,
        )

        plot_paths["rmsd_histogram"] = plot_rmsd_histogram(all_metrics, plots_dir)
        plot_paths["classification_summary"] = plot_classification_summary(classifications, plots_dir)

        if ifp_results:
            plot_paths["interaction_conservation"] = plot_interaction_conservation(ifp_results, plots_dir)

        if sim_matrix is not None:
            plot_paths["similarity_heatmap"] = plot_similarity_heatmap(
                sim_matrix, sim_names, plots_dir
            )
            generate_plotly_heatmap(sim_matrix, sim_names, plots_dir)

        if cluster_result:
            plot_paths["dendrogram"] = plot_dendrogram(
                cluster_result["linkage_matrix"],
                cluster_result["compound_names"], plots_dir
            )

        if pca_result:
            cls_labels = [m.get("classification", "") for m in all_metrics]
            cluster_labels = cluster_result["labels"] if cluster_result else None
            plot_paths["pca_embedding"] = plot_pca_embedding(
                pca_result, cluster_labels, cls_labels, plots_dir
            )

        for m in all_metrics[:10]:
            radar_path = plot_radar_chart(m, plots_dir)
            if radar_path:
                plot_paths[f"radar_{m['compound']}"] = radar_path

        viewer_path = generate_py3dmol_view(ref, queries, output_dir)
        plot_paths["3d_viewer"] = viewer_path

        generate_pymol_script(ref, queries, output_dir)
        logger.info("Visualizations generated")
    else:
        viewer_path = ""

    # -------------------------------------------------------------------
    # 12. HTML Report
    # -------------------------------------------------------------------
    if not args.no_html:
        logger.info("\n[STEP 10] Generating HTML report...")
        from src.report import generate_html_report

        report_path = generate_html_report(
            ref_name=ref.name,
            results_df=results_df,
            classifications=classifications,
            plot_paths=plot_paths,
            viewer_html_path=viewer_path,
            output_dir=output_dir,
        )
        logger.info(f"Report: {report_path}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"Compounds analyzed: {len(queries)}")

    class_counts = {}
    for c in classifications:
        cls = c["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, count in sorted(class_counts.items()):
        logger.info(f"  {cls}: {count}")

    # Top poses per compound summary
    logger.info(f"\n*** TOP {args.top_n} POSES PER COMPOUND ***")
    for cid, poses in per_compound.items():
        best_p = poses[0]
        logger.info(
            f"  {cid}: best={best_p['compound']} "
            f"(score={best_p.get('consensus_score',0):.3f}, "
            f"RMSD={best_p.get('heavy_atom_rmsd','N/A')}, "
            f"{best_p.get('classification','N/A')}), "
            f"{len(poses)} poses selected"
        )

    logger.info(f"\nOutputs:")
    logger.info(f"  All poses CSV:   {csv_path}")
    logger.info(f"  Top-{args.top_n} CSV:      {top_csv_path}")
    logger.info(f"  Top-{args.top_n} SDF:      {top_n_sdf}")
    logger.info(f"  Report:          {output_dir / 'report.html'}")
    logger.info(f"  Plots:           {plots_dir}")
    logger.info(f"  All docked SDF:  {output_dir / 'docked_poses.sdf'}")
    logger.info(f"  PyMOL:           {output_dir / 'pose_overlay.pml'}")

    print(f"\nDone. Results in {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
