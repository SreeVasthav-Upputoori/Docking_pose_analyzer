#!/usr/bin/env python3
"""
Docking Pose Consistency Tool — Streamlit GUI
==============================================
Interactive web interface for comparing docked ligand poses against a
reference pose and evaluating binding mode conservation.

Launch:
    streamlit run streamlit_app.py
"""

import io
import shutil
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Docking Pose Consistency Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 { margin: 0 0 0.3rem 0; font-size: 2rem; }
    .main-header p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid;
    }
    .metric-card h4 { margin: 0; font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; margin: 0.3rem 0; }
    .mc-green  { border-color: #2ecc71; } .mc-green  .value { color: #2ecc71; }
    .mc-orange { border-color: #f39c12; } .mc-orange .value { color: #f39c12; }
    .mc-red    { border-color: #e74c3c; } .mc-red    .value { color: #e74c3c; }
    .mc-blue   { border-color: #3498db; } .mc-blue   .value { color: #3498db; }
    .mc-purple { border-color: #9b59b6; } .mc-purple .value { color: #9b59b6; }
    .mc-gray   { border-color: #95a5a6; } .mc-gray   .value { color: #95a5a6; }

    /* Classification badges */
    .badge { padding: 3px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; color: white; display: inline-block; }
    .badge-conserved { background: #2ecc71; }
    .badge-partial   { background: #f39c12; }
    .badge-different { background: #e74c3c; }
    .badge-flipped   { background: #9b59b6; }
    .badge-outlier   { background: #95a5a6; }

    /* Step indicator */
    .step-box {
        background: linear-gradient(90deg, #3498db22, transparent);
        border-left: 4px solid #3498db;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* Table tweaks */
    .dataframe th { background: #34495e !important; color: white !important; }

    /* Plotly chart container */
    .plot-frame { border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem; background: white; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _save_uploaded(uploaded_file, target_dir: Path) -> Path:
    """Persist an uploaded file to *target_dir* and return the path."""
    dest = target_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    return dest


def _classification_badge(cls: str) -> str:
    key = cls.split()[0].lower()
    return f'<span class="badge badge-{key}">{cls}</span>'


def _metric_card(title: str, value, css_class: str = "mc-blue") -> str:
    return f"""
    <div class="metric-card {css_class}">
        <h4>{title}</h4>
        <div class="value">{value}</div>
    </div>"""


def _build_zip(output_dir: Path) -> bytes:
    """Create an in-memory ZIP of all output files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(output_dir))
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Inputs & Settings
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/protein.png", width=60)
    st.markdown("## Docking Consistency")
    st.markdown("---")

    st.markdown("### 📂 Input Files")
    protein_file = st.file_uploader(
        "Protein structure (.pdb)", type=["pdb"], help="Receptor PDB file"
    )
    reference_file = st.file_uploader(
        "Reference pose", type=["sdf", "mol2", "pdbqt", "pdb", "mol"],
        help="Trusted docked pose",
    )
    pose_files = st.file_uploader(
        "Docked poses", type=["sdf", "mol2", "pdbqt", "pdb", "mol", "sd"],
        accept_multiple_files=True,
        help="One or more ligand files / multi-mol SDFs",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    rmsd_cutoff = st.slider(
        "RMSD conserved cutoff (Å)", 0.5, 5.0, 2.0, 0.5,
        help="RMSD below this is classified as CONSERVED",
    )
    n_clusters = st.number_input(
        "Clusters (0 = auto)", 0, 20, 0,
        help="Force a specific number of clusters",
    )
    mcs_timeout = st.number_input("MCS timeout (s)", 5, 120, 30)

    top_n = st.number_input(
        "Top N poses per compound", 1, 20, 5,
        help="Keep the best N poses for each compound",
    )

    st.markdown("#### Methods to run")
    run_ifp = st.checkbox("Interaction Fingerprints (IFP)", value=True)
    run_pharma = st.checkbox("Pharmacophore Analysis", value=True)
    run_shape = st.checkbox("Shape Similarity", value=True)
    run_orient = st.checkbox("Orientation Analysis", value=True)

    st.markdown("---")
    run_button = st.button("🚀  Run Analysis", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🧬 Docking Pose Consistency Tool</h1>
    <p>Compare docked ligand poses against a reference — RMSD · MCS · IFP · Pharmacophore · Shape · Orientation</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

if run_button:
    # ── Validation ──
    if not protein_file:
        st.error("Please upload a **protein PDB** file.")
        st.stop()
    if not reference_file:
        st.error("Please upload a **reference pose** file.")
        st.stop()
    if not pose_files:
        st.error("Please upload at least one **docked pose** file.")
        st.stop()

    # ── Persist uploads to a temp directory ──
    work_dir = Path(tempfile.mkdtemp(prefix="dock_consist_"))
    output_dir = work_dir / "output"
    output_dir.mkdir()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    poses_dir = work_dir / "poses"
    poses_dir.mkdir()

    protein_path = _save_uploaded(protein_file, work_dir)
    ref_path = _save_uploaded(reference_file, work_dir)
    for pf in pose_files:
        _save_uploaded(pf, poses_dir)

    # ── Set up logging ──
    from src.utils import setup_logging
    logger = setup_logging(output_dir, verbose=False)

    progress = st.progress(0, text="Initializing…")
    status_area = st.empty()

    def _step(pct: int, msg: str):
        progress.progress(pct, text=msg)
        status_area.markdown(f'<div class="step-box">{msg}</div>', unsafe_allow_html=True)

    t0 = time.time()

    # ── STEP 1: Load molecules ──────────────────────────────────────────
    _step(5, "📥  Step 1/8 — Loading molecules…")
    from src.io_handlers import load_protein, load_reference, load_molecules, write_sdf, select_top_poses

    protein = load_protein(str(protein_path))
    ref = load_reference(str(ref_path))
    if ref is None:
        st.error("Failed to parse the reference pose. Check file format.")
        st.stop()

    queries = load_molecules(str(poses_dir))
    if not queries:
        st.error("No valid docked poses found. Check file formats.")
        st.stop()

    st.sidebar.success(f"✅ Loaded **{len(queries)}** poses")

    # ── STEP 2: RMSD & MCS ─────────────────────────────────────────────
    _step(15, f"📐  Step 2/8 — RMSD & MCS analysis ({len(queries)} compounds, poses unchanged)…")
    from src.alignment import batch_rmsd_analysis

    rmsd_results = batch_rmsd_analysis(ref, queries)

    from rdkit import Chem
    ref_sdf = str(output_dir / "reference_pose.sdf")
    w = Chem.SDWriter(ref_sdf)
    ref.mol.SetProp("_Name", ref.name)
    w.write(ref.mol)
    w.close()
    write_sdf(queries, str(output_dir / "docked_poses.sdf"))

    # ── STEP 3: Interaction fingerprints ────────────────────────────────
    ifp_results: List[Dict] = []
    fp_matrix = None
    if run_ifp:
        _step(30, "🔗  Step 3/8 — Interaction fingerprints…")
        from src.interactions import batch_ifp_analysis
        try:
            ifp_results, fp_matrix = batch_ifp_analysis(str(protein_path), ref, queries)
        except Exception as exc:
            st.warning(f"IFP analysis failed: {exc}")
    else:
        _step(30, "⏭️  Step 3/8 — IFP skipped")

    # ── STEP 4: Pharmacophore ──────────────────────────────────────────
    pharma_results: List[Dict] = []
    if run_pharma:
        _step(45, "💊  Step 4/8 — Pharmacophore consistency…")
        from src.pharmacophore import batch_pharmacophore_analysis
        try:
            pharma_results = batch_pharmacophore_analysis(ref, queries)
        except Exception as exc:
            st.warning(f"Pharmacophore analysis failed: {exc}")
    else:
        _step(45, "⏭️  Step 4/8 — Pharmacophore skipped")

    # ── STEP 5: Orientation ────────────────────────────────────────────
    orient_results: List[Dict] = []
    if run_orient:
        _step(55, "🧭  Step 5/8 — Orientation analysis…")
        from src.orientation import batch_orientation_analysis
        orient_results = batch_orientation_analysis(ref, queries)
    else:
        _step(55, "⏭️  Step 5/8 — Orientation skipped")

    # ── STEP 6: Shape ──────────────────────────────────────────────────
    shape_results: List[Dict] = []
    if run_shape:
        _step(65, "🔷  Step 6/8 — Shape similarity…")
        from src.shape import batch_shape_analysis
        try:
            shape_results = batch_shape_analysis(ref, queries)
        except Exception as exc:
            st.warning(f"Shape analysis failed: {exc}")
    else:
        _step(65, "⏭️  Step 6/8 — Shape skipped")

    # ── STEP 7: Merge & classify ───────────────────────────────────────
    _step(75, "🏷️  Step 7/8 — Merging metrics & classifying…")

    name_to_cid = {q.name: q.compound_id for q in queries}

    merged: Dict[str, Dict] = {}
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

    from src.classification import PoseClassifier
    thresholds = PoseClassifier._default_thresholds()
    thresholds["rmsd_conserved"] = rmsd_cutoff
    classifier = PoseClassifier(thresholds=thresholds)
    classifications = classifier.classify_batch(all_metrics)

    for m, c in zip(all_metrics, classifications):
        m["classification"] = c["classification"]
        m["consensus_score"] = c["consensus_score"]

    # ── Top-N pose selection per compound ──────────────────────────────
    _step(80, f"🏆  Selecting top {top_n} poses per compound…")
    top_records, top_metrics, per_compound = select_top_poses(
        queries, all_metrics, top_n=top_n
    )

    # Write top-N SDF with metadata
    from rdkit import Chem as _Chem_top
    top_sdf_path = output_dir / f"top{top_n}_poses.sdf"
    _tw = _Chem_top.SDWriter(str(top_sdf_path))
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

    # ── STEP 8: Clustering ─────────────────────────────────────────────
    _step(88, "📊  Step 8/8 — Clustering & visualizations…")
    from src.clustering import (
        build_feature_matrix, hierarchical_clustering, compute_pca,
        compute_umap, compute_similarity_matrix, ifp_clustering,
    )

    feature_matrix, feat_names, feat_cols = build_feature_matrix(all_metrics)
    cluster_result = pca_result = umap_result = None
    sim_matrix = sim_names = None

    if len(all_metrics) >= 3:
        cluster_result = hierarchical_clustering(feature_matrix, feat_names, n_clusters=n_clusters)
        for m, label in zip(all_metrics, cluster_result["labels"]):
            m["cluster"] = label
        pca_result = compute_pca(feature_matrix, feat_names)
        umap_result = compute_umap(feature_matrix, feat_names)
        sim_matrix, sim_names = compute_similarity_matrix(all_metrics)

    # ── Save CSVs ─────────────────────────────────────────────────────
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
        "consensus_score", "classification", "cluster",
    ]
    csv_cols = [c for c in csv_cols if c in results_df.columns]
    csv_df = results_df[csv_cols].copy()
    csv_path = output_dir / "pose_consistency_results.csv"
    csv_df.to_csv(csv_path, index=False, float_format="%.3f")

    top_df = pd.DataFrame(top_metrics)
    top_csv_cols = [c for c in csv_cols if c in top_df.columns]
    top_csv_path = output_dir / f"top{top_n}_poses.csv"
    top_df[top_csv_cols].to_csv(top_csv_path, index=False, float_format="%.3f")

    progress.progress(100, text="✅  Analysis complete!")
    elapsed = time.time() - t0
    status_area.empty()

    # ══════════════════════════════════════════════════════════════════
    #  STORE RESULTS IN SESSION STATE for display below the button
    # ══════════════════════════════════════════════════════════════════
    st.session_state["results"] = {
        "all_metrics": all_metrics,
        "top_metrics": top_metrics,
        "per_compound": per_compound,
        "top_n": top_n,
        "top_records": top_records,
        "results_df": results_df,
        "csv_df": csv_df,
        "classifications": classifications,
        "queries": queries,
        "ref": ref,
        "cluster_result": cluster_result,
        "pca_result": pca_result,
        "umap_result": umap_result,
        "sim_matrix": sim_matrix,
        "sim_names": sim_names,
        "ifp_results": ifp_results,
        "fp_matrix": fp_matrix,
        "output_dir": output_dir,
        "plots_dir": plots_dir,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY RESULTS  (persisted in session_state so they survive reruns)
# ═══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    # Landing page
    st.info(
        "**Upload your files** in the sidebar and click **Run Analysis** to begin.\n\n"
        "The tool accepts:\n"
        "- **Protein**: `.pdb`\n"
        "- **Reference & Poses**: `.sdf`, `.mol2`, `.pdbqt`, `.pdb`\n\n"
        "You can upload a single multi-mol SDF or many individual ligand files."
    )
    st.stop()

# Unpack results
R = st.session_state["results"]
all_metrics      = R["all_metrics"]
top_metrics      = R["top_metrics"]
per_compound     = R["per_compound"]
top_n_val        = R["top_n"]
top_records      = R["top_records"]
results_df       = R["results_df"]
csv_df           = R["csv_df"]
classifications  = R["classifications"]
queries          = R["queries"]
ref              = R["ref"]
cluster_result   = R["cluster_result"]
pca_result       = R["pca_result"]
umap_result      = R["umap_result"]
sim_matrix       = R["sim_matrix"]
sim_names        = R["sim_names"]
ifp_results      = R["ifp_results"]
fp_matrix        = R["fp_matrix"]
output_dir       = R["output_dir"]
plots_dir        = R["plots_dir"]
elapsed          = R["elapsed"]


# ── Summary Cards ─────────────────────────────────────────────────────────
st.markdown("## Overview")

n_compounds = len(per_compound)
n_total_poses = len(all_metrics)
n_selected = len(top_metrics)

class_counts: Dict[str, int] = {}
for c in classifications:
    cls = c["classification"]
    class_counts[cls] = class_counts.get(cls, 0) + 1

cols = st.columns(6)
card_data = [
    ("Compounds", n_compounds, "mc-blue"),
    ("Total Poses", n_total_poses, "mc-blue"),
    (f"Top-{top_n_val} Selected", n_selected, "mc-green"),
    ("Conserved", class_counts.get("CONSERVED BINDING MODE", 0), "mc-green"),
    ("Partial", class_counts.get("PARTIAL CONSERVATION", 0), "mc-orange"),
    ("Different / Other", class_counts.get("DIFFERENT POSE", 0)
     + class_counts.get("FLIPPED ORIENTATION", 0)
     + class_counts.get("OUTLIER", 0), "mc-red"),
]
for col, (title, val, css) in zip(cols, card_data):
    col.markdown(_metric_card(title, val, css), unsafe_allow_html=True)

st.caption(
    f"Reference: **{ref.name}** ({ref.num_heavy_atoms} heavy atoms)  ·  "
    f"Analysis time: {elapsed:.1f}s  ·  "
    f"Poses kept as-docked (no alignment)  ·  "
    f"Top {top_n_val} poses per compound selected"
)

# ── Top Poses Per Compound ────────────────────────────────────────────────
st.markdown(f"### 🏆 Top {top_n_val} Poses Per Compound")

# Build a summary table
top_summary_rows = []
for cid, poses in sorted(per_compound.items()):
    best = poses[0]
    top_summary_rows.append({
        "Compound": cid,
        "Poses Found": len([m for m in all_metrics
                            if m.get("compound_id", m.get("compound","")) == cid
                            or m.get("compound","").startswith(cid)]),
        "Selected": len(poses),
        "Best Pose": best["compound"],
        "Best Score": best.get("consensus_score", 0),
        "Best RMSD": best.get("heavy_atom_rmsd"),
        "Best Class": best.get("classification", "N/A"),
    })
top_summary_df = pd.DataFrame(top_summary_rows)
st.dataframe(
    top_summary_df,
    column_config={
        "Best Score": st.column_config.ProgressColumn("Best Score", min_value=0, max_value=1, format="%.3f"),
        "Best RMSD": st.column_config.NumberColumn("Best RMSD (Å)", format="%.2f"),
    },
    use_container_width=True,
    hide_index=True,
)

# Expandable per-compound details
for cid, poses in sorted(per_compound.items()):
    with st.expander(f"**{cid}** — {len(poses)} pose(s) selected"):
        pose_detail = []
        for p in poses:
            pose_detail.append({
                "Rank": p.get("rank", ""),
                "Pose": p["compound"],
                "Consensus": p.get("consensus_score", 0),
                "RMSD (Å)": p.get("heavy_atom_rmsd"),
                "IFP Tanimoto": p.get("ifp_tanimoto"),
                "Shape": p.get("shape_tanimoto"),
                "Classification": p.get("classification", ""),
            })
        st.dataframe(pd.DataFrame(pose_detail), use_container_width=True, hide_index=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_table, tab_class, tab_plots, tab_cluster, tab_3d, tab_detail, tab_download = st.tabs([
    "📋 All Poses",
    "🏷️ Classifications",
    "📊 Plots",
    "🔬 Clustering",
    "🧊 3D Viewer",
    "🔍 Compound Detail",
    "💾 Downloads",
])


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Results Table
# ═══════════════════════════════════════════════════════════════════════════
with tab_table:
    st.markdown("### Full Results Table")
    st.caption("Click column headers to sort.")

    display_cols = [
        "compound", "heavy_atom_rmsd", "mcs_rmsd", "ifp_tanimoto",
        "pharmacophore_score", "shape_tanimoto", "com_distance", "gap_score",
        "principal_axis_angle", "consensus_score", "classification",
    ]
    display_cols = [c for c in display_cols if c in csv_df.columns]
    display_df = csv_df[display_cols].copy()

    col_config = {
        "heavy_atom_rmsd": st.column_config.NumberColumn("RMSD (Å)", format="%.2f"),
        "mcs_rmsd": st.column_config.NumberColumn("MCS RMSD (Å)", format="%.2f"),
        "ifp_tanimoto": st.column_config.ProgressColumn("IFP Tanimoto", min_value=0, max_value=1, format="%.2f"),
        "pharmacophore_score": st.column_config.ProgressColumn("Pharma Score", min_value=0, max_value=1, format="%.2f"),
        "shape_tanimoto": st.column_config.ProgressColumn("Shape Score", min_value=0, max_value=1, format="%.2f"),
        "consensus_score": st.column_config.ProgressColumn("Consensus", min_value=0, max_value=1, format="%.3f"),
        "gap_score": st.column_config.ProgressColumn("Gap Score", min_value=0, max_value=1, format="%.2f", help="Fraction of reference interactions conserved"),
        "com_distance": st.column_config.NumberColumn("COM Dist (Å)", format="%.2f"),
        "principal_axis_angle": st.column_config.NumberColumn("Axis Angle (°)", format="%.1f"),
    }

    st.dataframe(
        display_df,
        column_config=col_config,
        use_container_width=True,
        height=min(35 * len(display_df) + 50, 700),
    )

    # ── Quick Statistics ──
    with st.expander("📈 Quick statistics"):
        num_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.dataframe(display_df[num_cols].describe().T.round(3), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Classifications
# ═══════════════════════════════════════════════════════════════════════════
with tab_class:
    st.markdown("### Pose Classifications")

    cl, cr = st.columns([1, 2])
    with cl:
        import plotly.graph_objects as go

        labels_pie = list(class_counts.keys())
        values_pie = list(class_counts.values())
        color_map = {
            "CONSERVED BINDING MODE": "#2ecc71",
            "PARTIAL CONSERVATION": "#f39c12",
            "DIFFERENT POSE": "#e74c3c",
            "FLIPPED ORIENTATION": "#9b59b6",
            "OUTLIER": "#95a5a6",
        }
        colors_pie = [color_map.get(l, "#333") for l in labels_pie]
        fig_pie = go.Figure(go.Pie(
            labels=labels_pie, values=values_pie,
            marker=dict(colors=colors_pie),
            hole=0.45, textinfo="label+value",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            margin=dict(t=30, b=10, l=10, r=10), height=350,
            showlegend=False, title_text="Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with cr:
        cls_df = pd.DataFrame(classifications)
        cls_df = cls_df[["compound", "classification", "consensus_score"]].copy()
        cls_df = cls_df.sort_values("consensus_score", ascending=False)
        st.dataframe(cls_df, use_container_width=True, height=350)

    # Per-class expandable lists
    st.markdown("---")
    for cls_name in color_map:
        compounds = [c["compound"] for c in classifications if c["classification"] == cls_name]
        if compounds:
            with st.expander(f"{_classification_badge(cls_name)}  ({len(compounds)} compounds)", expanded=False):
                st.write(", ".join(compounds))


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Plots
# ═══════════════════════════════════════════════════════════════════════════
with tab_plots:
    st.markdown("### Visualizations")
    import plotly.express as px
    import plotly.graph_objects as go

    # ── RMSD bar chart ──
    st.markdown("#### RMSD vs Reference")
    rmsd_data = [
        {"compound": m["compound"][:20], "RMSD": m.get("heavy_atom_rmsd")}
        for m in all_metrics if m.get("heavy_atom_rmsd") is not None
    ]
    if rmsd_data:
        rdf = pd.DataFrame(rmsd_data)
        colors_bar = [
            "#2ecc71" if r < rmsd_cutoff else "#f39c12" if r < 4.0 else "#e74c3c"
            for r in rdf["RMSD"]
        ]
        fig_rmsd = go.Figure(go.Bar(
            x=rdf["compound"], y=rdf["RMSD"],
            marker_color=colors_bar,
            text=rdf["RMSD"].round(2), textposition="outside",
        ))
        fig_rmsd.add_hline(y=rmsd_cutoff, line_dash="dash", line_color="#2ecc71",
                           annotation_text=f"Conserved < {rmsd_cutoff} Å")
        fig_rmsd.add_hline(y=4.0, line_dash="dash", line_color="#e74c3c",
                           annotation_text="Inconsistent > 4 Å")
        fig_rmsd.update_layout(
            yaxis_title="RMSD (Å)", xaxis_title="Compound",
            height=450, margin=dict(t=40),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_rmsd, use_container_width=True)

    # ── Multi-method comparison ──
    st.markdown("#### Multi-Method Scores Comparison")
    score_cols = ["ifp_tanimoto", "pharmacophore_score", "shape_tanimoto", "orientation_score"]
    score_labels = ["IFP Tanimoto", "Pharmacophore", "Shape", "Orientation"]
    available_scores = [(c, l) for c, l in zip(score_cols, score_labels)
                        if c in results_df.columns and results_df[c].notna().any()]
    if available_scores:
        melt_data = []
        for m in all_metrics:
            for col, label in available_scores:
                val = m.get(col)
                if val is not None:
                    melt_data.append({"Compound": m["compound"][:18], "Method": label, "Score": val})
        melt_df = pd.DataFrame(melt_data)
        fig_multi = px.bar(
            melt_df, x="Compound", y="Score", color="Method",
            barmode="group", height=450,
            color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
        )
        fig_multi.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 1.05])
        st.plotly_chart(fig_multi, use_container_width=True)

    # ── Similarity Heatmap ──
    if sim_matrix is not None:
        st.markdown("#### Pose Similarity Heatmap")
        short_names = [n[:18] for n in sim_names]
        fig_heat = go.Figure(go.Heatmap(
            z=sim_matrix, x=short_names, y=short_names,
            colorscale="RdYlGn", zmin=0, zmax=1,
            text=np.round(sim_matrix, 2),
            texttemplate="%{text:.2f}" if len(sim_names) <= 20 else "",
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(height=max(500, len(sim_names) * 28), margin=dict(t=40))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Interaction Conservation ──
    if ifp_results:
        st.markdown("#### Interaction Type Conservation")
        int_data = []
        for r in ifp_results:
            name = r.get("compound", "?")[:18]
            int_data.append({"Compound": name, "Type": "H-Bonds", "Count": r.get("n_hbonds", 0)})
            int_data.append({"Compound": name, "Type": "Hydrophobic", "Count": r.get("n_hydrophobic", 0)})
            int_data.append({"Compound": name, "Type": "π-Stacking", "Count": r.get("n_pistacking", 0)})
            int_data.append({"Compound": name, "Type": "Salt Bridges", "Count": r.get("n_saltbridges", 0)})
        int_df = pd.DataFrame(int_data)
        fig_int = px.bar(
            int_df, x="Compound", y="Count", color="Type",
            barmode="group", height=400,
            color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
        )
        fig_int.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_int, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — Clustering
# ═══════════════════════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown("### Clustering Analysis")

    if pca_result is None:
        st.info("Need at least 3 compounds for clustering.")
    else:
        import plotly.express as px

        # PCA scatter
        st.markdown("#### PCA Projection")
        coords = pca_result["coordinates"]
        var = pca_result.get("explained_variance", [0, 0])
        pca_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Compound": pca_result["compound_names"],
            "Classification": [m.get("classification", "?") for m in all_metrics],
            "Consensus": [m.get("consensus_score", 0) for m in all_metrics],
            "Cluster": [str(m.get("cluster", "?")) for m in all_metrics],
        })

        color_by = st.radio("Color by:", ["Classification", "Cluster"], horizontal=True, key="pca_color")

        color_map_cls = {
            "CONSERVED BINDING MODE": "#2ecc71",
            "PARTIAL CONSERVATION": "#f39c12",
            "DIFFERENT POSE": "#e74c3c",
            "FLIPPED ORIENTATION": "#9b59b6",
            "OUTLIER": "#95a5a6",
        }

        if color_by == "Classification":
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Classification",
                hover_name="Compound", hover_data=["Consensus"],
                size="Consensus", size_max=18,
                color_discrete_map=color_map_cls,
                labels={"PC1": f"PC1 ({var[0]*100:.1f}%)", "PC2": f"PC2 ({var[1]*100:.1f}%)"},
            )
        else:
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Cluster",
                hover_name="Compound", hover_data=["Classification", "Consensus"],
                size="Consensus", size_max=18,
                labels={"PC1": f"PC1 ({var[0]*100:.1f}%)", "PC2": f"PC2 ({var[1]*100:.1f}%)"},
            )

        fig_pca.update_layout(height=550, margin=dict(t=40))
        fig_pca.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig_pca, use_container_width=True)

        # Dendrogram
        if cluster_result:
            st.markdown("#### Hierarchical Clustering Dendrogram")
            from scipy.cluster.hierarchy import dendrogram as scipy_dendro
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_d, ax_d = plt.subplots(figsize=(max(10, len(all_metrics) * 0.5), 5))
            scipy_dendro(
                cluster_result["linkage_matrix"],
                labels=[n[:15] for n in cluster_result["compound_names"]],
                ax=ax_d, leaf_rotation=45, leaf_font_size=8,
            )
            ax_d.set_ylabel("Distance")
            ax_d.set_title("Pose Clustering Dendrogram")
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)

            # Cluster membership table
            st.markdown("#### Cluster Membership")
            cluster_table = []
            for cid, members in sorted(cluster_result["clusters"].items()):
                cluster_table.append({"Cluster": cid, "Size": len(members), "Members": ", ".join(members)})
            st.dataframe(pd.DataFrame(cluster_table), use_container_width=True)

        # UMAP
        if umap_result is not None:
            st.markdown("#### UMAP Embedding")
            ucoords = umap_result["coordinates"]
            umap_df = pd.DataFrame({
                "UMAP1": ucoords[:, 0],
                "UMAP2": ucoords[:, 1],
                "Compound": umap_result["compound_names"],
                "Classification": [m.get("classification", "?") for m in all_metrics],
            })
            fig_umap = px.scatter(
                umap_df, x="UMAP1", y="UMAP2", color="Classification",
                hover_name="Compound", color_discrete_map=color_map_cls,
            )
            fig_umap.update_layout(height=500)
            fig_umap.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig_umap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 — 3D Viewer
# ═══════════════════════════════════════════════════════════════════════════
with tab_3d:
    st.markdown("### 3D Pose Overlay")

    try:
        import py3Dmol
        from rdkit import Chem
        from stmol import showmol

        view = py3Dmol.view(width=800, height=550)

        ref_block = Chem.MolToMolBlock(ref.mol)
        view.addModel(ref_block, "sdf")
        view.setStyle({"model": 0}, {"stick": {"color": "#2ecc71", "radius": 0.18}})

        max_display = st.slider("Max poses to display", 1, min(len(queries), 50), min(len(queries), 20), key="n3d")
        for i, q in enumerate(queries[:max_display]):
            if q.mol:
                q_block = Chem.MolToMolBlock(q.mol)
                view.addModel(q_block, "sdf")

                cls = all_metrics[i].get("classification", "") if i < len(all_metrics) else ""
                mol_color = {
                    "CONSERVED BINDING MODE": "#2ecc71",
                    "PARTIAL CONSERVATION": "#f39c12",
                    "DIFFERENT POSE": "#e74c3c",
                    "FLIPPED ORIENTATION": "#9b59b6",
                }.get(cls, "#3498db")

                view.setStyle({"model": i + 1}, {"stick": {"color": mol_color, "radius": 0.12, "opacity": 0.7}})

        view.zoomTo()
        showmol(view, height=550, width=800)
        st.caption("**Green** = reference · Colored by classification · Original docked poses (no alignment applied)")

    except ImportError:
        st.warning(
            "Interactive 3D viewer requires `py3Dmol` and `stmol`.\n\n"
            "Install with: `pip install py3Dmol stmol`\n\n"
            "Meanwhile, download the docked poses SDF and view in PyMOL or Discovery Studio."
        )
    except Exception as e:
        st.warning(f"3D viewer error: {e}")
        st.info("Download the docked poses SDF from the Downloads tab for PyMOL/Discovery Studio viewing.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 6 — Compound Detail
# ═══════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown("### Compound Detail View")

    compound_names = [m["compound"] for m in all_metrics]
    selected = st.selectbox("Select compound:", compound_names)

    if selected:
        m = next(x for x in all_metrics if x["compound"] == selected)
        c = next(x for x in classifications if x["compound"] == selected)

        st.markdown(f"**Classification:** {_classification_badge(c['classification'])}", unsafe_allow_html=True)

        # Radar chart
        import plotly.graph_objects as go

        radar_cats = ["RMSD", "IFP", "Pharmacophore", "Shape", "Orientation"]
        radar_vals = [
            max(0, 1.0 - min(m.get("heavy_atom_rmsd", 10) / 10, 1.0)),
            m.get("ifp_tanimoto", 0) or 0,
            m.get("pharmacophore_score", 0) or 0,
            m.get("shape_tanimoto", 0) or 0,
            m.get("orientation_score", 0) or 0,
        ]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            fillcolor="rgba(52,152,219,0.2)",
            line=dict(color="#3498db", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400, margin=dict(t=50, b=50),
            title=f"Multi-Method Profile: {selected[:25]}",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Gap Analysis Section
        if "conserved_list" in m:
            st.markdown("#### 🔬 Interaction Gap Analysis")
            g1, g2, g3 = st.columns(3)
            with g1:
                st.success(f"**Conserved** ({m.get('conserved_count', 0)})")
                for item in m.get("conserved_list", []):
                    st.caption(f"✅ {item}")
            with g2:
                st.error(f"**Missing (Gaps)** ({m.get('missing_count', 0)})")
                for item in m.get("missing_list", []):
                    st.caption(f"❌ {item}")
            with g3:
                st.info(f"**New Interactions** ({m.get('new_count', 0)})")
                for item in m.get("new_list", []):
                    st.caption(f"✨ {item}")

        # Detailed metrics table
        st.markdown("#### All Metrics")
        detail_items = []
        nice_names = {
            "heavy_atom_rmsd": "Heavy Atom RMSD (Å)",
            "symmetry_corrected_rmsd": "Symmetry-Corrected RMSD (Å)",
            "mcs_rmsd": "MCS RMSD (Å)",
            "mcs_num_atoms": "MCS Atom Count",
            "mcs_fraction_ref": "MCS Fraction (ref)",
            "ifp_tanimoto": "IFP Tanimoto",
            "ifp_dice": "IFP Dice",
            "ifp_overlap_pct": "IFP Overlap %",
            "n_hbonds": "H-Bonds",
            "n_hydrophobic": "Hydrophobic Contacts",
            "n_pistacking": "π-Stacking",
            "n_saltbridges": "Salt Bridges",
            "pharmacophore_score": "Pharmacophore Score",
            "feature_overlap_pct": "Feature Overlap %",
            "matched_features": "Matched Features",
            "usr_similarity": "USR Similarity",
            "usrcat_similarity": "USRCAT Similarity",
            "gaussian_overlap": "Gaussian Overlap",
            "volumetric_overlap_pct": "Volumetric Overlap %",
            "shape_tanimoto": "Shape Tanimoto",
            "com_distance": "COM Distance (Å)",
            "principal_axis_angle": "Principal Axis Angle (°)",
            "orientation_score": "Orientation Score",
            "is_flipped": "Flipped?",
            "gap_score": "Interaction Gap Score",
            "consensus_score": "Consensus Score",
        }
        for key, label in nice_names.items():
            val = m.get(key)
            if val is not None:
                if isinstance(val, float):
                    val = f"{val:.3f}"
                detail_items.append({"Metric": label, "Value": val})

        if detail_items:
            st.dataframe(pd.DataFrame(detail_items), use_container_width=True, hide_index=True)

        # Component scores from classifier
        if "component_scores" in c:
            st.markdown("#### Classification Component Scores")
            comp_df = pd.DataFrame([
                {"Component": k.replace("_", " ").title(), "Score": v}
                for k, v in c["component_scores"].items()
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 7 — Downloads
# ═══════════════════════════════════════════════════════════════════════════
with tab_download:
    st.markdown("### Download Results")

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown(f"#### 🏆 Top-{top_n_val} Poses SDF")
        top_sdf_path = output_dir / f"top{top_n_val}_poses.sdf"
        if top_sdf_path.exists():
            st.download_button(
                f"Download Top-{top_n_val} Poses ({n_selected} poses)",
                data=top_sdf_path.read_bytes(),
                file_name=f"top{top_n_val}_poses.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True,
            )
        st.caption(f"Contains the best {top_n_val} poses for each of {n_compounds} compounds")

    with dl2:
        st.markdown(f"#### 📄 Top-{top_n_val} CSV")
        top_csv_path = output_dir / f"top{top_n_val}_poses.csv"
        if top_csv_path.exists():
            st.download_button(
                f"Download Top-{top_n_val} CSV",
                data=top_csv_path.read_bytes(),
                file_name=f"top{top_n_val}_poses.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption("Metrics for selected poses only")

    with dl3:
        st.markdown("#### 📦 Full ZIP")
        st.download_button(
            "Download All Results",
            data=_build_zip(output_dir),
            file_name="docking_consistency_results.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("All CSVs, SDFs, plots, and scripts")

    st.markdown("---")
    st.markdown("#### All Files")

    dl4, dl5 = st.columns(2)
    with dl4:
        st.download_button(
            "All Poses CSV (complete)",
            data=csv_df.to_csv(index=False, float_format="%.3f"),
            file_name="pose_consistency_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl5:
        docked_path = output_dir / "docked_poses.sdf"
        if docked_path.exists():
            st.download_button(
                "All Docked Poses SDF",
                data=docked_path.read_bytes(),
                file_name="docked_poses.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True,
            )

    st.markdown("#### Individual Files")
    ref_sdf_path = output_dir / "reference_pose.sdf"
    if ref_sdf_path.exists():
        st.download_button(
            "Reference Pose (SDF)", data=ref_sdf_path.read_bytes(),
            file_name="reference_pose.sdf", mime="chemical/x-mdl-sdfile",
        )

    log_path = output_dir / "pose_consistency.log"
    if log_path.exists():
        st.download_button(
            "Analysis Log", data=log_path.read_bytes(),
            file_name="pose_consistency.log", mime="text/plain",
        )

    # PyMOL script
    pml_content = f"""# PyMOL session for docking consistency visualization
bg_color white
set ray_shadow, 0
set stick_radius, 0.15
load reference_pose.sdf, reference
color green, reference
show sticks, reference
load docked_poses.sdf, all_poses
split_states all_poses
delete all_poses
set all_states, on
show sticks
hide everything, hydrogens
zoom visible
"""
    st.download_button(
        "PyMOL Script (.pml)", data=pml_content,
        file_name="pose_overlay.pml", mime="text/plain",
    )
