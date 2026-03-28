# mt-trna-scripts
Mitochondrial tRNA Structural Evolution Across Tetrapod Species

Dataset: 19,350 tRNAs · 971 species · 70 orders · 4 classes

---

## Script Overview

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `01_data_acquisition.py` | NCBI data retrieval, tRNA extraction, QC, deduplication | `trna_final.csv`, `species_metadata.csv` |
| `02_structure_analysis.py` | ViennaRNA folding, domain annotation, GC/Tm/arm-loss | `trna_structure.csv`, `species_means.csv` |
| `03_statistical_analysis.py` | Mann-Whitney, Spearman, chi-square, Ti/Tv, MI, thermal reserve | `stats_*.csv`, `titv_results.json`, `mi_results.json`, `thermal_reserve.csv` |
| `04_figures.py` | All 6 main figures | `fig1_*.svg/png` … `fig6_*.svg/png` |
| `05_phylogeny.py` | OTL tree construction (Python) | `tetrapod_otl_dendropy.nwk`, `tree_annotations.csv` |
| `05_phylogeny.R` | Phylogenetic tree visualization (R/ggtree) | `FigS_phylogeny.svg/png` |
| `06_supplementary_datasets.py` | Assemble datasets S1–S10 | `dataset_S1_*.csv` … `dataset_S10_*.csv` |

---

## Execution Order

```bash
# 1. Data acquisition (~2–4 hours, NCBI rate-limited)
python 01_data_acquisition.py

# 2. Structure analysis (~30–60 min, ViennaRNA)
python 02_structure_analysis.py

# 3. Statistical analyses (~20–40 min, MAFFT alignments)
python 03_statistical_analysis.py

# 4. Figures
python 04_figures.py

# 5a. Phylogeny construction (Python, ~30 min)
python 05_phylogeny.py

# 5b. Phylogeny visualization (R)
Rscript 05_phylogeny.R

# 6. Supplementary datasets
python 06_supplementary_datasets.py
```

---

## Dependencies

**Python packages:**
```
biopython >= 1.80
pandas >= 1.5
numpy >= 1.23
scipy >= 1.9
matplotlib >= 3.6
seaborn >= 0.12
ViennaRNA (RNA) >= 2.5
dendropy >= 4.5
```

**External tools:**
- `mafft` (v7+) — multiple sequence alignment (used in script 03)

**R packages:**
- `ape`, `ggtree`, `ggplot2`, `dplyr`, `ggnewscale`

---

## Data Flow

```
NCBI GenBank
    │
    ▼
01_data_acquisition.py
    │  trna_final.csv (19,350 tRNAs)
    │  species_metadata.csv (971 species)
    ▼
02_structure_analysis.py
    │  trna_structure.csv
    │  species_means.csv
    ▼
03_statistical_analysis.py ──────────────────────────────────┐
    │  stats_ecto_vs_endo.csv                                 │
    │  stats_body_temp_corr.csv                               │
    │  titv_results.json                                      │
    │  mi_results.json                                        │
    │  thermal_reserve.csv                                    │
    ▼                                                         │
04_figures.py ◄───────────────────────────────────────────────┘
    │  fig1–fig6 (.svg + .png)
    ▼
05_phylogeny.py → 05_phylogeny.R
    │  FigS_phylogeny.svg/png
    ▼
06_supplementary_datasets.py
    │  dataset_S1–S10 (.csv)
```

---

## Notes on tRNA Coverage

- **20 of 22 canonical mt-tRNA types** are retained per species after QC.
- The two tRNA-Ser isoacceptors (including the D-arm-lacking AGY type, typically 45–54 nt) and two tRNA-Leu isoacceptors are each represented by a single sequence per species after deduplication.
- The 55–95 nt length filter excludes the structurally degenerate AGY-type tRNA-Ser, which may slightly underestimate D-arm loss rates.
- Species with fewer than 10 tRNA types recovered are excluded (incomplete mitogenomes).
