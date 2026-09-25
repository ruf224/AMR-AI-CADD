# AMR-AI-CADD NDM-1 — FINAL VERSION

## Purpose
A realistic browser-deployable AI/CADD demonstrator for NDM-1 inhibitor prioritization.

## What makes this version different
- Uses a published 686-compound NDM-1 IC50 modeling dataset rather than synthetic activity values.
- Uses RDKit molecular descriptors + Morgan fingerprints for the ligand-based model.
- Reports random-forest prediction uncertainty as tree dispersion.
- Adds nearest-neighbor Tanimoto similarity as an applicability-domain indicator.
- Provides scaffold-split validation.
- Embeds experimental NDM-1 structures from RCSB PDB using the Mol* viewer.
- Adds an explicitly labeled structure-context heuristic rather than fake docking scores.
- Generates conservative, RDKit-valid analogues.
- Separates ADMET endpoints rather than fabricating a single generic ADMET score.
- Builds a multi-objective candidate portfolio and exports CSV files.
- Includes a research-methods summary suitable for adaptation into a dissertation/manuscript.

## Data provenance
NDM-1 dataset:
https://github.com/georgeyuricadd/IC50-dataset/blob/main/NDM686concise.csv

The underlying study states that 703 NDM-1 inhibitor IC50 activities were compiled from ChEMBL and literature and curated to 686 non-redundant molecules.

RCSB PDB examples:
4U4L, 6IBS, 6EFJ, 6MDU.

TDC ADMET benchmark families:
AqSolDB, BBB_Martins, Pgp_Broccatelli, CYP2C9/2D6/3A4 inhibition, hERG, AMES, DILI.

## Scientific boundaries
This is an AI-assisted prioritization system, not a validated clinical decision system.
The structure-aware score is not molecular docking, a binding pose, or a free-energy estimate.
The RF uncertainty is an uncertainty proxy, not a calibrated confidence interval.
Generated analogues are computational hypotheses and require synthesis, assay and downstream validation.

## Deployment
Upload the folder to GitHub and deploy `app.py` using Streamlit Community Cloud.

requirements.txt is intentionally lightweight.
