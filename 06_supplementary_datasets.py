#!/usr/bin/env python3
"""
06_supplementary_datasets.py
============================
Assemble all supplementary datasets (S1–S10) for the manuscript.

Dataset descriptions:
  S1  — GenBank accession list (per-accession tRNA counts)
  S2  — Complete tRNA sequence table (all 19,350 sequences)
  S3  — tRNA structural parameters (per-tRNA)
  S4  — Species-level structural means
  S5  — Species metadata (taxonomy, thermoregulation, body temperature)
  S6  — Ectotherm vs endotherm statistical comparisons
  S7  — Body temperature Spearman correlations (order-level)
  S8  — Ti/Tv ratios by thermoregulation group and class
  S9  — Mutual information (co-evolutionary signal) by tRNA region
  S10 — Thermal reserve estimates for ectotherm species

All datasets are saved as CSV to /mnt/results/intermediate_data/
with standardized column names and documentation headers.

Dependencies: pandas, numpy, json
"""

import os
import json
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
INDIR   = "/workspace/data_v2"
OUTDIR  = "/mnt/results/intermediate_data"

os.makedirs(OUTDIR, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def save_dataset(df, name, description):
    path = os.path.join(OUTDIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  {name}: {len(df):,} rows × {len(df.columns)} cols  — {description}")
    return path

# ── Dataset S1: GenBank accession list ───────────────────────────────────────
def build_s1(trna_df):
    """
    Per-accession summary: how many tRNA sequences were extracted from
    each GenBank record. Note: many records are partial mitogenomes,
    so n_tRNAs < 22 is expected. Species-level complete sets are in S2/S3.
    """
    s1 = (trna_df.groupby(["accession", "order", "class", "thermo", "species_clean"])
          .size()
          .reset_index(name="n_tRNAs"))
    s1 = s1.sort_values(["class", "order", "species_clean"]).reset_index(drop=True)
    s1.columns = ["GenBank_Accession", "Order", "Class",
                  "Thermoregulation", "Species", "n_tRNAs"]
    return s1

# ── Dataset S2: tRNA sequence table ──────────────────────────────────────────
def build_s2(trna_df, sp_meta):
    """
    Complete deduplicated tRNA sequence table (1 sequence per species per type).
    Includes taxonomy and thermoregulation metadata.
    """
    s2 = trna_df[["accession", "species_clean", "class", "order",
                  "thermo", "body_temp", "trna_type", "sequence", "length"]].copy()
    s2.columns = ["GenBank_Accession", "Species", "Class", "Order",
                  "Thermoregulation", "Body_Temp_C", "tRNA_Type",
                  "Sequence", "Length_nt"]
    s2 = s2.sort_values(["Class", "Order", "Species", "tRNA_Type"]).reset_index(drop=True)
    return s2

# ── Dataset S3: tRNA structural parameters ───────────────────────────────────
def build_s3(struct_df):
    """
    Per-tRNA structural parameters from ViennaRNA folding and domain analysis.
    """
    cols = [
        "species_clean", "class", "order", "thermo", "trna_type",
        "sequence", "structure", "mfe", "tm",
        "gc_total", "gc_stem", "gc_acceptor", "gc_d_stem",
        "gc_anticodon", "gc_tpsic", "gu_wobble",
        "d_arm_present", "t_arm_present",
    ]
    available = [c for c in cols if c in struct_df.columns]
    s3 = struct_df[available].copy()
    s3 = s3.rename(columns={
        "species_clean": "Species",
        "class":         "Class",
        "order":         "Order",
        "thermo":        "Thermoregulation",
        "trna_type":     "tRNA_Type",
        "sequence":      "Sequence",
        "structure":     "DotBracket_Structure",
        "mfe":           "MFE_kcal_mol",
        "tm":            "Predicted_Tm_C",
        "gc_total":      "GC_total",
        "gc_stem":       "GC_stem",
        "gc_acceptor":   "GC_acceptor_stem",
        "gc_d_stem":     "GC_D_stem",
        "gc_anticodon":  "GC_anticodon_stem",
        "gc_tpsic":      "GC_TpsiC_stem",
        "gu_wobble":     "GU_wobble_fraction",
        "d_arm_present": "D_arm_present",
        "t_arm_present": "T_arm_present",
    })
    s3 = s3.sort_values(["Class", "Order", "Species", "tRNA_Type"]).reset_index(drop=True)
    return s3

# ── Dataset S4: Species-level structural means ────────────────────────────────
def build_s4(sp_means):
    """
    Per-species mean structural parameters (averaged across all tRNA types).
    """
    cols = [
        "species_clean", "class", "order", "thermo", "body_temp",
        "n_trnas", "mfe", "tm", "gc_total", "gc_stem", "gc_acceptor",
        "gc_d_stem", "gc_anticodon", "gc_tpsic", "gu_wobble",
        "d_arm_loss", "t_arm_loss",
    ]
    available = [c for c in cols if c in sp_means.columns]
    s4 = sp_means[available].copy()
    s4 = s4.rename(columns={
        "species_clean": "Species",
        "class":         "Class",
        "order":         "Order",
        "thermo":        "Thermoregulation",
        "body_temp":     "Body_Temp_C",
        "n_trnas":       "n_tRNA_types",
        "mfe":           "Mean_MFE_kcal_mol",
        "tm":            "Mean_Tm_C",
        "gc_total":      "Mean_GC_total",
        "gc_stem":       "Mean_GC_stem",
        "gc_acceptor":   "Mean_GC_acceptor",
        "gc_d_stem":     "Mean_GC_D_stem",
        "gc_anticodon":  "Mean_GC_anticodon",
        "gc_tpsic":      "Mean_GC_TpsiC",
        "gu_wobble":     "Mean_GU_wobble",
        "d_arm_loss":    "D_arm_loss_rate",
        "t_arm_loss":    "T_arm_loss_rate",
    })
    s4 = s4.sort_values(["Class", "Order", "Species"]).reset_index(drop=True)
    return s4

# ── Dataset S5: Species metadata ──────────────────────────────────────────────
def build_s5(sp_meta):
    """
    Species-level taxonomy and thermoregulation metadata.
    """
    s5 = sp_meta.copy()
    # Standardize column names
    rename_map = {
        "species":       "species",
        "species_clean": "species",
        "n_trna_types":  "n_trna_types",
        "n_trnas":       "n_trnas",
        "order":         "order",
        "class":         "class",
        "thermo":        "thermoregulation",
        "body_temp":     "body_temp_C",
        "accession":     "representative_accession",
    }
    s5 = s5.rename(columns={k: v for k, v in rename_map.items() if k in s5.columns})
    s5 = s5.sort_values(["class", "order", "species"]).reset_index(drop=True)
    return s5

# ── Dataset S6: Ecto vs endo statistics ──────────────────────────────────────
def build_s6(stats_path):
    """Mann-Whitney U test results: ectotherm vs endotherm."""
    if os.path.exists(stats_path):
        return pd.read_csv(stats_path)
    return pd.DataFrame()

# ── Dataset S7: Body temperature correlations ─────────────────────────────────
def build_s7(corr_path):
    """Spearman correlations between body temperature and structural variables."""
    if os.path.exists(corr_path):
        return pd.read_csv(corr_path)
    return pd.DataFrame()

# ── Dataset S8: Ti/Tv results ─────────────────────────────────────────────────
def build_s8(titv_path):
    """Ti/Tv ratios in stem vs loop regions by thermoregulation group and class."""
    if not os.path.exists(titv_path):
        return pd.DataFrame()
    with open(titv_path) as f:
        titv = json.load(f)
    rows = []
    for group, vals in titv.items():
        rows.append({
            "Group":        group,
            "TiTv_stem":    vals.get("titv_stem"),
            "TiTv_loop":    vals.get("titv_loop"),
            "Stem_loop_ratio": (vals["titv_stem"] / vals["titv_loop"]
                                if vals.get("titv_stem") and vals.get("titv_loop")
                                else None),
            "n_stem_cols":  vals.get("n_stem_cols"),
            "n_loop_cols":  vals.get("n_loop_cols"),
            "n_sequences":  vals.get("n_seqs"),
        })
    return pd.DataFrame(rows)

# ── Dataset S9: Mutual information ───────────────────────────────────────────
def build_s9(mi_path):
    """Mutual information (co-evolutionary signal) by tRNA region."""
    if not os.path.exists(mi_path):
        return pd.DataFrame()
    with open(mi_path) as f:
        mi = json.load(f)
    rows = []
    for group, regions in mi.items():
        for region, vals in regions.items():
            rows.append({
                "Group":    group,
                "Region":   region,
                "Mean_MI":  vals.get("mean_mi"),
                "n_pairs":  vals.get("n_pairs"),
            })
    return pd.DataFrame(rows)

# ── Dataset S10: Thermal reserve ─────────────────────────────────────────────
def build_s10(thermal_path):
    """Thermal reserve (predicted Tm − habitat Tmax) for ectotherm species."""
    if os.path.exists(thermal_path):
        df = pd.read_csv(thermal_path)
        df = df.rename(columns={
            "species_clean": "Species",
            "class":         "Class",
            "order":         "Order",
            "thermo":        "Thermoregulation",
            "tm":            "Predicted_Tm_C",
            "habitat_tmax":  "Habitat_Tmax_C",
            "thermal_reserve": "Thermal_Reserve_C",
        })
        return df.sort_values(["Class", "Order", "Species"]).reset_index(drop=True)
    return pd.DataFrame()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading input data...")
    trna_df   = pd.read_csv(os.path.join(INDIR, "trna_final.csv"))
    struct_df = pd.read_csv(os.path.join(INDIR, "trna_structure.csv"))
    sp_means  = pd.read_csv(os.path.join(INDIR, "species_means.csv"))
    sp_meta   = pd.read_csv(os.path.join(INDIR, "species_metadata.csv"))

    print(f"\nBuilding supplementary datasets...")
    print(f"{'='*60}")

    datasets = {
        "dataset_S1_genbank_accessions":    build_s1(trna_df),
        "dataset_S2_trna_sequences":        build_s2(trna_df, sp_meta),
        "dataset_S3_trna_structure":        build_s3(struct_df),
        "dataset_S4_species_struct_means":  build_s4(sp_means),
        "dataset_S5_species_metadata":      build_s5(sp_meta),
        "dataset_S6_ecto_vs_endo_stats":    build_s6(
            os.path.join(INDIR, "stats_ecto_vs_endo.csv")),
        "dataset_S7_body_temp_corr":        build_s7(
            os.path.join(INDIR, "stats_body_temp_corr.csv")),
        "dataset_S8_titv_results":          build_s8(
            os.path.join(INDIR, "titv_results.json")),
        "dataset_S9_mi_results":            build_s9(
            os.path.join(INDIR, "mi_results.json")),
        "dataset_S10_thermal_reserve":      build_s10(
            os.path.join(INDIR, "thermal_reserve.csv")),
    }

    descriptions = {
        "dataset_S1_genbank_accessions":   "GenBank accession list with per-record tRNA counts",
        "dataset_S2_trna_sequences":       "Complete deduplicated tRNA sequence table",
        "dataset_S3_trna_structure":       "Per-tRNA structural parameters (ViennaRNA)",
        "dataset_S4_species_struct_means": "Species-level mean structural parameters",
        "dataset_S5_species_metadata":     "Species taxonomy and thermoregulation metadata",
        "dataset_S6_ecto_vs_endo_stats":   "Mann-Whitney U: ectotherm vs endotherm",
        "dataset_S7_body_temp_corr":       "Spearman correlations with body temperature",
        "dataset_S8_titv_results":         "Ti/Tv ratios in stem vs loop regions",
        "dataset_S9_mi_results":           "Mutual information by tRNA region",
        "dataset_S10_thermal_reserve":     "Thermal reserve for ectotherm species",
    }

    for name, df in datasets.items():
        if len(df) > 0:
            save_dataset(df, name, descriptions[name])
        else:
            print(f"  {name}: SKIPPED (source file not found)")

    print(f"\n{'='*60}")
    print(f"All datasets saved to: {OUTDIR}/")
    print(f"\nDataset summary:")
    print(f"  S1: {len(datasets['dataset_S1_genbank_accessions']):,} GenBank accessions")
    print(f"  S2: {len(datasets['dataset_S2_trna_sequences']):,} tRNA sequences")
    print(f"  S3: {len(datasets['dataset_S3_trna_structure']):,} structural annotations")
    print(f"  S4: {len(datasets['dataset_S4_species_struct_means']):,} species")
    print(f"  S5: {len(datasets['dataset_S5_species_metadata']):,} species")
