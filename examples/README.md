# Example Dataset Structure

Place your files in this directory following this structure:

```
examples/
├── protein.pdb              # Protein structure (receptor)
├── reference_pose.sdf       # Trusted reference docked pose
└── docked_poses/            # Directory of docked compounds
    ├── compound_01.sdf
    ├── compound_02.sdf
    ├── compound_03.mol2
    ├── compound_04.pdbqt
    └── ...
```

## Supported Input Formats

| Format | Extension | Source |
|--------|-----------|--------|
| SDF    | .sdf, .sd | AutoDock Vina, Glide, MOE, GOLD, Discovery Studio |
| MOL2   | .mol2     | GOLD, MOE, Discovery Studio |
| PDBQT  | .pdbqt    | AutoDock Vina, AutoDock4 |
| PDB    | .pdb      | Various docking tools |

## Running the Example

```bash
python pose_consistency.py \
  --protein examples/protein.pdb \
  --reference examples/reference_pose.sdf \
  --poses examples/docked_poses/ \
  -o example_output/
```

## Expected Outputs

```
example_output/
├── pose_consistency_results.csv    # Full metrics table
├── report.html                     # Interactive HTML report
├── reference_pose.sdf              # Reference structure
├── docked_poses.sdf                # All docked structures (original poses, unmodified)
├── best_pose.sdf                   # Best-scoring pose selected by consensus
├── pose_overlay.pml                # PyMOL visualization script
├── 3d_overlay.html                 # Interactive 3D viewer
├── pose_consistency.log            # Analysis log
└── plots/
    ├── rmsd_histogram.png
    ├── similarity_heatmap.png
    ├── dendrogram.png
    ├── pca_embedding.png
    ├── classification_summary.png
    ├── interaction_conservation.png
    ├── interactive_heatmap.html
    └── radar_*.png
```
