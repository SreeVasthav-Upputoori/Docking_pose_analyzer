"""HTML Report Generator: interactive report with tables, plots, and 3D viewers."""

from pathlib import Path
from typing import Dict, List, Optional
import base64
import json

import pandas as pd

from .utils import get_logger, ensure_dir


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Docking Pose Consistency Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        header h1 { font-size: 28px; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); text-align: center; }
        .card h3 { font-size: 14px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px; }
        .card .value { font-size: 36px; font-weight: bold; }
        .card .value.green { color: #2ecc71; }
        .card .value.orange { color: #f39c12; }
        .card .value.red { color: #e74c3c; }
        .card .value.blue { color: #3498db; }
        .section { background: white; border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        .section h2 { font-size: 20px; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { background: #34495e; color: white; padding: 10px 8px; text-align: left; cursor: pointer; user-select: none; }
        th:hover { background: #2c3e50; }
        td { padding: 8px; border-bottom: 1px solid #ecf0f1; }
        tr:hover { background: #f8f9fa; }
        .class-badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; color: white; display: inline-block; }
        .class-CONSERVED { background: #2ecc71; }
        .class-PARTIAL { background: #f39c12; }
        .class-DIFFERENT { background: #e74c3c; }
        .class-FLIPPED { background: #9b59b6; }
        .class-OUTLIER { background: #95a5a6; }
        .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
        .plot-container { text-align: center; }
        .plot-container img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .plot-container h4 { margin-bottom: 10px; color: #7f8c8d; }
        .viewer-container { width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
        .nav { position: sticky; top: 0; background: white; z-index: 100; padding: 10px 20px; border-bottom: 1px solid #ddd; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .nav a { color: #3498db; text-decoration: none; margin: 0 15px; font-size: 14px; }
        .nav a:hover { text-decoration: underline; }
        .footer { text-align: center; color: #95a5a6; padding: 20px; font-size: 12px; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Docking Pose Consistency Report</h1>
        <p>Reference: {{REF_NAME}} | Compounds analyzed: {{N_COMPOUNDS}} | Generated: {{TIMESTAMP}}</p>
    </header>

    <nav class="nav">
        <a href="#summary">Summary</a>
        <a href="#table">Results Table</a>
        <a href="#classifications">Classifications</a>
        <a href="#plots">Visualizations</a>
        <a href="#gaps">Gap Analysis</a>
        <a href="#clustering">Clustering</a>
        <a href="#viewer">3D Viewer</a>
    </nav>

    <div id="summary" class="summary-cards">
        {{SUMMARY_CARDS}}
    </div>

    <div id="table" class="section">
        <h2>Detailed Results</h2>
        <div style="overflow-x: auto;">
            {{RESULTS_TABLE}}
        </div>
    </div>

    <div id="classifications" class="section">
        <h2>Classification Summary</h2>
        <div class="plot-grid">
            {{CLASSIFICATION_PLOT}}
            {{CLASSIFICATION_TABLE}}
        </div>
    </div>

    <div id="plots" class="section">
        <h2>Visualizations</h2>
        <div class="plot-grid">
            {{PLOTS}}
        </div>
    </div>

    <div id="gaps" class="section">
        <h2>Interaction Gap Analysis</h2>
        <p>Comparison of protein-ligand interactions between reference and query poses.</p>
        <div style="overflow-x: auto;">
            {{GAP_ANALYSIS_TABLE}}
        </div>
    </div>

    <div id="clustering" class="section">
        <h2>Clustering Analysis</h2>
        <div class="plot-grid">
            {{CLUSTERING_PLOTS}}
        </div>
    </div>

    <div id="viewer" class="section">
        <h2>3D Pose Overlay</h2>
        {{VIEWER_3D}}
    </div>

    <div class="footer">
        <p>Docking Consistency Tool v1.0 | Methods: RMSD, MCS, IFP, Pharmacophore, Shape, Orientation</p>
    </div>
</div>

<script>
document.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const asc = th.dataset.sort !== 'asc';
        th.dataset.sort = asc ? 'asc' : 'desc';
        rows.sort((a, b) => {
            let va = a.children[idx].textContent.trim();
            let vb = b.children[idx].textContent.trim();
            let na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
            return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        });
        rows.forEach(r => tbody.appendChild(r));
    });
});
</script>
</body>
</html>"""


def _img_to_base64(img_path: str) -> str:
    """Convert image to base64 data URI."""
    p = Path(img_path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = p.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def _classification_badge(cls_name: str) -> str:
    """Generate HTML badge for classification."""
    short = cls_name.split()[0].upper()
    return f'<span class="class-badge class-{short}">{cls_name}</span>'


def _build_summary_cards(n_compounds: int, classifications: List[Dict]) -> str:
    """Build summary statistics cards."""
    counts = {}
    for c in classifications:
        cls = c["classification"]
        counts[cls] = counts.get(cls, 0) + 1

    conserved = counts.get("CONSERVED BINDING MODE", 0)
    partial = counts.get("PARTIAL CONSERVATION", 0)
    different = counts.get("DIFFERENT POSE", 0)
    flipped = counts.get("FLIPPED ORIENTATION", 0)
    outlier = counts.get("OUTLIER", 0)

    cards = f"""
    <div class="card"><h3>Total Compounds</h3><div class="value blue">{n_compounds}</div></div>
    <div class="card"><h3>Conserved</h3><div class="value green">{conserved}</div></div>
    <div class="card"><h3>Partial</h3><div class="value orange">{partial}</div></div>
    <div class="card"><h3>Different</h3><div class="value red">{different}</div></div>
    <div class="card"><h3>Flipped</h3><div class="value" style="color:#9b59b6">{flipped}</div></div>
    <div class="card"><h3>Outlier</h3><div class="value" style="color:#95a5a6">{outlier}</div></div>
    """
    return cards


def _build_results_table(df: pd.DataFrame) -> str:
    """Build sortable HTML table from results DataFrame."""
    display_cols = [
        "compound", "heavy_atom_rmsd", "mcs_rmsd", "shape_tanimoto",
        "ifp_tanimoto",        "pharmacophore_score", "com_distance", "gap_score",
        "principal_axis_angle", "consensus_score", "classification",
    ]
    cols = [c for c in display_cols if c in df.columns]
    if not cols:
        return "<p>No data available.</p>"

    header_names = {
        "compound": "Compound",
        "heavy_atom_rmsd": "RMSD (Å)",
        "mcs_rmsd": "MCS RMSD (Å)",
        "shape_tanimoto": "Shape Score",
        "ifp_tanimoto": "IFP Tanimoto",
        "pharmacophore_score": "Pharma Score",
        "com_distance": "COM Dist (Å)",
        "principal_axis_angle": "Axis Angle (°)",
        "gap_score": "Gap Score",
        "consensus_score": "Consensus",
        "classification": "Classification",
    }

    html = "<table>\n<thead><tr>"
    for c in cols:
        html += f"<th>{header_names.get(c, c)}</th>"
    html += "</tr></thead>\n<tbody>\n"

    for _, row in df.iterrows():
        html += "<tr>"
        for c in cols:
            val = row.get(c, "")
            if c == "classification" and isinstance(val, str):
                html += f"<td>{_classification_badge(val)}</td>"
            elif isinstance(val, float):
                html += f"<td>{val:.3f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>\n"

    html += "</tbody>\n</table>"
    return html


def generate_html_report(
    ref_name: str,
    results_df: pd.DataFrame,
    classifications: List[Dict],
    plot_paths: Dict[str, str],
    viewer_html_path: Optional[str],
    output_dir: Path,
) -> str:
    """Generate the complete HTML report."""
    logger = get_logger()
    from datetime import datetime

    html = HTML_TEMPLATE
    html = html.replace("{{REF_NAME}}", ref_name)
    html = html.replace("{{N_COMPOUNDS}}", str(len(results_df)))
    html = html.replace("{{TIMESTAMP}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    html = html.replace("{{SUMMARY_CARDS}}", _build_summary_cards(len(results_df), classifications))
    html = html.replace("{{RESULTS_TABLE}}", _build_results_table(results_df))

    # Gap Analysis Table
    gap_html = "<table><thead><tr><th>Compound</th><th>Gap Score</th><th>Conserved</th><th>Missing (Gaps)</th><th>New</th></tr></thead><tbody>"
    for _, row in results_df.iterrows():
        cons_list = row.get("conserved_list")
        if isinstance(cons_list, list):
            cons = "<br>".join(cons_list)
            miss = "<br>".join(row.get("missing_list", []))
            new = "<br>".join(row.get("new_list", []))
            gap_score = row.get("gap_score", 0)
            gap_html += f"<tr><td>{row['compound']}</td><td>{gap_score:.2f}</td><td style='color:#27ae60'>{cons}</td><td style='color:#c0392b'>{miss}</td><td style='color:#2980b9'>{new}</td></tr>"
    gap_html += "</tbody></table>"
    html = html.replace("{{GAP_ANALYSIS_TABLE}}", gap_html)

    cls_plot = ""
    if "classification_summary" in plot_paths and plot_paths["classification_summary"]:
        b64 = _img_to_base64(plot_paths["classification_summary"])
        if b64:
            cls_plot = f'<div class="plot-container"><img src="{b64}" alt="Classification Summary"></div>'
    html = html.replace("{{CLASSIFICATION_PLOT}}", cls_plot)

    cls_table = "<div><table><thead><tr><th>Compound</th><th>Classification</th><th>Score</th></tr></thead><tbody>"
    for c in classifications:
        cls_table += f"<tr><td>{c['compound']}</td><td>{_classification_badge(c['classification'])}</td><td>{c['consensus_score']:.3f}</td></tr>"
    cls_table += "</tbody></table></div>"
    html = html.replace("{{CLASSIFICATION_TABLE}}", cls_table)

    plots_html = ""
    plot_order = ["rmsd_histogram", "similarity_heatmap", "interaction_conservation"]
    for key in plot_order:
        path = plot_paths.get(key, "")
        if path:
            b64 = _img_to_base64(path)
            if b64:
                title = key.replace("_", " ").title()
                plots_html += f'<div class="plot-container"><h4>{title}</h4><img src="{b64}" alt="{title}"></div>\n'

    for key, path in plot_paths.items():
        if key.startswith("radar_") and path:
            b64 = _img_to_base64(path)
            if b64:
                plots_html += f'<div class="plot-container"><h4>Radar: {key[6:]}</h4><img src="{b64}" alt="Radar"></div>\n'

    html = html.replace("{{PLOTS}}", plots_html)

    cluster_html = ""
    for key in ["dendrogram", "pca_embedding"]:
        path = plot_paths.get(key, "")
        if path:
            b64 = _img_to_base64(path)
            if b64:
                title = key.replace("_", " ").title()
                cluster_html += f'<div class="plot-container"><h4>{title}</h4><img src="{b64}" alt="{title}"></div>\n'
    html = html.replace("{{CLUSTERING_PLOTS}}", cluster_html)

    viewer_content = ""
    if viewer_html_path and Path(viewer_html_path).exists():
        viewer_content = f'<iframe src="{Path(viewer_html_path).name}" style="width:100%;height:500px;border:none;border-radius:8px;"></iframe>'
    else:
        viewer_content = '<p style="color:#95a5a6;">3D viewer requires py3Dmol. View docked_poses.sdf in PyMOL or Discovery Studio.</p>'
    html = html.replace("{{VIEWER_3D}}", viewer_content)

    report_path = str(output_dir / "report.html")
    Path(report_path).write_text(html, encoding="utf-8")
    logger.info(f"HTML report generated: {report_path}")

    return report_path
