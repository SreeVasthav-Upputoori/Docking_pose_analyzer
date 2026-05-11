"""Visualization: plots, heatmaps, overlay images, PyMOL sessions."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger, ensure_dir

warnings.filterwarnings("ignore")


def plot_rmsd_histogram(metrics: List[Dict], output_dir: Path) -> str:
    """Generate RMSD distribution histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmsds = [m["heavy_atom_rmsd"] for m in metrics if m.get("heavy_atom_rmsd") is not None]
    if not rmsds:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for r in rmsds:
        if r < 2.0:
            colors.append("#2ecc71")
        elif r < 4.0:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    ax.bar(range(len(rmsds)), rmsds, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=2.0, color="#2ecc71", linestyle="--", alpha=0.7, label="Conserved (<2 Å)")
    ax.axhline(y=4.0, color="#e74c3c", linestyle="--", alpha=0.7, label="Inconsistent (>4 Å)")

    names = [m.get("compound", f"mol_{i}")[:15] for i, m in enumerate(metrics)
             if m.get("heavy_atom_rmsd") is not None]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.set_title("Heavy Atom RMSD vs Reference Pose", fontsize=14)
    ax.legend()
    plt.tight_layout()

    path = str(output_dir / "rmsd_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_similarity_heatmap(sim_matrix: np.ndarray, names: List[str],
                             output_dir: Path, title: str = "Pose Similarity") -> str:
    """Generate similarity heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), max(6, len(names) * 0.5)))
    short_names = [n[:15] for n in names]
    sns.heatmap(sim_matrix, xticklabels=short_names, yticklabels=short_names,
                cmap="RdYlGn", vmin=0, vmax=1, annot=len(names) <= 20,
                fmt=".2f" if len(names) <= 20 else "",
                linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    path = str(output_dir / "similarity_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dendrogram(linkage_matrix: np.ndarray, names: List[str],
                     output_dir: Path, title: str = "Pose Clustering Dendrogram") -> str:
    """Generate clustering dendrogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 6))
    short_names = [n[:15] for n in names]
    dendrogram(linkage_matrix, labels=short_names, ax=ax,
               leaf_rotation=45, leaf_font_size=8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Distance", fontsize=12)
    plt.tight_layout()

    path = str(output_dir / "dendrogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_embedding(pca_result: Dict, labels: Optional[List[int]],
                        classifications: Optional[List[str]],
                        output_dir: Path) -> str:
    """Generate PCA scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = pca_result["coordinates"]
    names = pca_result["compound_names"]
    var = pca_result.get("explained_variance", [0, 0])

    fig, ax = plt.subplots(figsize=(10, 8))

    if classifications:
        class_colors = {
            "CONSERVED BINDING MODE": "#2ecc71",
            "PARTIAL CONSERVATION": "#f39c12",
            "DIFFERENT POSE": "#e74c3c",
            "FLIPPED ORIENTATION": "#9b59b6",
            "OUTLIER": "#95a5a6",
        }
        colors = [class_colors.get(c, "#333333") for c in classifications]
    elif labels:
        from matplotlib.cm import Set1
        unique_labels = sorted(set(labels))
        cmap = {l: Set1(i / max(len(unique_labels), 1)) for i, l in enumerate(unique_labels)}
        colors = [cmap[l] for l in labels]
    else:
        colors = "#3498db"

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100,
                          edgecolors="black", linewidth=0.5, alpha=0.8)

    for i, name in enumerate(names):
        ax.annotate(name[:12], (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    xlabel = f"PC1 ({var[0]*100:.1f}%)" if len(var) > 0 else "PC1"
    ylabel = f"PC2 ({var[1]*100:.1f}%)" if len(var) > 1 else "PC2"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Pose Similarity - PCA Projection", fontsize=14)

    if classifications:
        from matplotlib.patches import Patch
        legend_handles = []
        for cls_name, color in class_colors.items():
            if cls_name in classifications:
                legend_handles.append(Patch(facecolor=color, label=cls_name))
        if legend_handles:
            ax.legend(handles=legend_handles, loc="best", fontsize=8)

    plt.tight_layout()
    path = str(output_dir / "pca_embedding.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_classification_summary(classifications: List[Dict], output_dir: Path) -> str:
    """Generate classification summary pie chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {}
    for c in classifications:
        cls = c["classification"]
        counts[cls] = counts.get(cls, 0) + 1

    class_colors = {
        "CONSERVED BINDING MODE": "#2ecc71",
        "PARTIAL CONSERVATION": "#f39c12",
        "DIFFERENT POSE": "#e74c3c",
        "FLIPPED ORIENTATION": "#9b59b6",
        "OUTLIER": "#95a5a6",
    }

    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [class_colors.get(l, "#333333") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("Pose Classification Distribution", fontsize=14)
    plt.tight_layout()

    path = str(output_dir / "classification_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_interaction_conservation(ifp_results: List[Dict], output_dir: Path) -> str:
    """Interaction type conservation bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interaction_types = ["n_hbonds", "n_hydrophobic", "n_pistacking", "n_saltbridges"]
    display_names = ["H-Bonds", "Hydrophobic", "π-Stacking", "Salt Bridges"]

    data = {d: [] for d in display_names}
    names = []

    for r in ifp_results:
        names.append(r.get("compound", "?")[:12])
        for itype, dname in zip(interaction_types, display_names):
            data[dname].append(r.get(itype, 0))

    if not names:
        return ""

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 6))
    x = np.arange(len(names))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for i, (dname, color) in enumerate(zip(display_names, colors)):
        ax.bar(x + i * width, data[dname], width, label=dname, color=color, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Interaction Type Conservation", fontsize=14)
    ax.legend()
    plt.tight_layout()

    path = str(output_dir / "interaction_conservation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_radar_chart(metrics: Dict, output_dir: Path) -> str:
    """Radar chart for a single compound's multi-method scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = [
        ("RMSD Score", 1.0 - min(metrics.get("heavy_atom_rmsd", 10) / 10, 1.0)),
        ("IFP Similarity", metrics.get("ifp_tanimoto", 0)),
        ("Pharmacophore", metrics.get("pharmacophore_score", 0)),
        ("Shape", metrics.get("shape_tanimoto", 0)),
        ("Orientation", metrics.get("orientation_score", 0)),
    ]

    labels = [c[0] for c in categories]
    values = [c[1] for c in categories]
    values.append(values[0])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#3498db", alpha=0.25)
    ax.plot(angles, values, color="#3498db", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Multi-Method Profile: {metrics.get('compound', 'Unknown')[:20]}", fontsize=12)
    plt.tight_layout()

    name = metrics.get("compound", "unknown").replace(" ", "_")[:20]
    path = str(output_dir / f"radar_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_plotly_heatmap(sim_matrix: np.ndarray, names: List[str],
                             output_dir: Path) -> str:
    """Interactive Plotly heatmap for HTML report."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        short_names = [n[:20] for n in names]
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=short_names, y=short_names,
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            text=np.round(sim_matrix, 2),
            texttemplate="%{text}" if len(names) <= 20 else "",
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(title="Pose Similarity Heatmap", width=800, height=700)

        path = str(output_dir / "interactive_heatmap.html")
        pio.write_html(fig, path)
        return path
    except Exception:
        return ""


def generate_py3dmol_view(ref_record, query_records: list, output_dir: Path) -> str:
    """Generate interactive 3D molecule viewer HTML using py3Dmol."""
    try:
        import py3Dmol
        from rdkit import Chem

        view = py3Dmol.view(width=800, height=600)

        ref_block = Chem.MolToMolBlock(ref_record.mol)
        view.addModel(ref_block, "sdf")
        view.setStyle({"model": 0}, {"stick": {"color": "#2ecc71", "radius": 0.15}})

        for i, qr in enumerate(query_records[:20]):
            if qr.mol:
                q_block = Chem.MolToMolBlock(qr.mol)
                view.addModel(q_block, "sdf")
                view.setStyle({"model": i + 1},
                              {"stick": {"color": "#3498db", "radius": 0.1, "opacity": 0.6}})

        view.zoomTo()

        html_str = view._make_html()
        path = str(output_dir / "3d_overlay.html")
        Path(path).write_text(html_str)
        return path
    except Exception as e:
        get_logger().debug(f"py3Dmol visualization failed: {e}")
        return ""


def generate_pymol_script(ref_record, query_records: list, output_dir: Path) -> str:
    """Generate PyMOL script for session recreation."""
    script_lines = [
        "# PyMOL script for docking consistency visualization",
        "# Load this script in PyMOL: run pose_overlay.pml",
        "",
        "bg_color white",
        "set ray_shadow, 0",
        "set stick_radius, 0.15",
        "",
    ]

    docked_sdf = output_dir / "docked_poses.sdf"
    if docked_sdf.exists():
        script_lines.append(f'load {docked_sdf.name}, all_poses')
        script_lines.append("split_states all_poses")
        script_lines.append("delete all_poses")
        script_lines.append("")

    ref_sdf = output_dir / "reference_pose.sdf"
    if ref_sdf.exists():
        script_lines.append(f'load {ref_sdf.name}, reference')
        script_lines.append('color green, reference')
        script_lines.append('show sticks, reference')
        script_lines.append("")

    script_lines.extend([
        "set all_states, on",
        "show sticks",
        "hide everything, hydrogens",
        "zoom visible",
        "set transparency, 0.3",
        "",
        "# Color by classification (manual)",
        "# select conserved, /state_name",
        "# color green, conserved",
        "# select partial, /state_name",
        "# color orange, partial",
        "# select different, /state_name",
        "# color red, different",
    ])

    path = str(output_dir / "pose_overlay.pml")
    Path(path).write_text("\n".join(script_lines))
    return path
