#!/usr/bin/env python3
"""
02_structure_analysis.py
========================
tRNA secondary structure analysis.

Steps:
  1. Load trna_final.csv (output of 01_data_acquisition.py)
  2. Fold each sequence with ViennaRNA → dot-bracket + MFE
  3. Parse dot-bracket topology → assign stem/loop domains
     (acceptor stem, D-stem, anticodon stem, TψC stem, loops)
  4. Compute per-domain GC content (paired positions only)
  5. Detect T-arm and D-arm loss (< 2 detected stem pairs)
  6. Predict melting temperature (Tm) via Marmur-Doty/Wallace formula
  7. Aggregate species-level means
  8. Save: trna_structure.csv, species_means.csv

Output files (written to OUTDIR):
  trna_structure.csv   — per-tRNA structural parameters
  species_means.csv    — per-species mean structural parameters

Dependencies: ViennaRNA (RNA), pandas, numpy
"""

import os
import re
import numpy as np
import pandas as pd
import RNA   # ViennaRNA Python bindings

# ── Configuration ─────────────────────────────────────────────────────────────
INDIR  = "/workspace/data_v2"
OUTDIR = "/workspace/data_v2"

# ── Step 1: Load data ─────────────────────────────────────────────────────────
def load_trna_data(path):
    df = pd.read_csv(path)
    # Ensure uppercase RNA sequences
    df["sequence"] = df["sequence"].str.upper().str.replace("T", "U", regex=False)
    print(f"Loaded {len(df):,} tRNAs from {df['species_clean'].nunique()} species")
    return df

# ── Step 2: ViennaRNA folding ─────────────────────────────────────────────────
def fold_sequence(seq):
    """Fold RNA sequence; return (dot-bracket structure, MFE kcal/mol)."""
    try:
        structure, mfe = RNA.fold(seq)
        return structure, round(float(mfe), 3)
    except Exception:
        return None, None

# ── Step 3: Dot-bracket topology parsing ─────────────────────────────────────
def build_pair_map(structure):
    """Build {i: j} pairing map from dot-bracket string."""
    pairs = {}
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j
    return pairs

def detect_stems(pairs, n):
    """
    Detect contiguous stem blocks from pairing map.
    Returns list of (5'_start, 5'_end, 3'_start, 3'_end) sorted by 5' position.
    A stem block = maximal run of consecutive nested pairs.
    """
    opening = sorted([(i, pairs[i]) for i in pairs if pairs[i] > i])
    if not opening:
        return []
    stems = []
    i = 0
    while i < len(opening):
        s5, s3 = [opening[i][0]], [opening[i][1]]
        j = i + 1
        while j < len(opening):
            pi, pj = opening[j]
            if pi == s5[-1] + 1 and pj == s3[-1] - 1:
                s5.append(pi)
                s3.append(pj)
                j += 1
            else:
                break
        stems.append((s5[0], s5[-1], s3[-1], s3[0]))
        i = j
    return stems

def assign_trna_domains(seq, structure):
    """
    Assign each nucleotide to a tRNA domain using stem topology.

    Canonical cloverleaf order (5'→3'):
      acceptor_stem(5') → D_stem → anticodon_stem → [var_loop] → TψC_stem → acceptor_stem(3')

    Returns:
      domain_map : {position: domain_label}
      arm_flags  : {'d_arm_present': bool, 't_arm_present': bool}
    """
    n = len(seq)
    pairs = build_pair_map(structure)
    stems = detect_stems(pairs, n)

    domain_map = {}
    arm_flags  = {"d_arm_present": False, "t_arm_present": False}

    if len(stems) < 2:
        return domain_map, arm_flags

    # Assign stems by position order
    # Stem 0 = acceptor stem (spans both ends of sequence)
    # Remaining stems ordered 5'→3': D_stem, anticodon_stem, [var], TψC_stem
    acc_stem = stems[0]
    inner_stems = stems[1:]

    # Acceptor stem
    for pos in list(range(acc_stem[0], acc_stem[1] + 1)) + \
               list(range(acc_stem[2], acc_stem[3] + 1)):
        domain_map[pos] = "acceptor_stem"

    # Inner stems: assign by order
    stem_labels = ["D_stem", "anticodon_stem", "TpsiC_stem"]
    for k, stem in enumerate(inner_stems[:3]):
        label = stem_labels[k] if k < len(stem_labels) else f"stem_{k}"
        n_pairs = stem[1] - stem[0] + 1
        if label == "D_stem" and n_pairs >= 2:
            arm_flags["d_arm_present"] = True
        if label == "TpsiC_stem" and n_pairs >= 2:
            arm_flags["t_arm_present"] = True
        for pos in list(range(stem[0], stem[1] + 1)) + \
                   list(range(stem[2], stem[3] + 1)):
            domain_map[pos] = label

    # Unpaired positions → loops
    for pos in range(n):
        if pos not in domain_map:
            domain_map[pos] = "loop"

    return domain_map, arm_flags

# ── Step 4: GC content (paired positions only) ───────────────────────────────
GC_BASES = set("GC")

def gc_fraction(seq, positions):
    """GC fraction at specified positions."""
    bases = [seq[p] for p in positions if p < len(seq)]
    if not bases:
        return np.nan
    return sum(b in GC_BASES for b in bases) / len(bases)

def gu_wobble_fraction(seq, pairs):
    """Fraction of G:U wobble pairs among all base pairs."""
    if not pairs:
        return np.nan
    gu = sum(1 for i, j in pairs.items()
             if j > i and {seq[i], seq[j]} == {"G", "U"})
    total = sum(1 for i, j in pairs.items() if j > i)
    return gu / total if total > 0 else np.nan

# ── Step 5: Tm prediction ─────────────────────────────────────────────────────
def predict_tm(seq, na_conc_molar=0.1):
    """
    Predict melting temperature using Marmur-Doty/Wallace empirical formula:
      Tm = 0.41 × (%GC) + 16.6 × log10([Na+]) + 81.5 − 675/n
    where n = sequence length, [Na+] in molar.

    Reference: Marmur & Doty (1962); Wallace et al. (1979).
    """
    n = len(seq)
    if n < 10:
        return np.nan
    gc_pct = sum(b in GC_BASES for b in seq) / n * 100
    tm = (0.41 * gc_pct
          + 16.6 * np.log10(na_conc_molar)
          + 81.5
          - 675.0 / n)
    return round(tm, 2)

# ── Core per-tRNA annotation ──────────────────────────────────────────────────
def annotate_trna(row):
    """Annotate a single tRNA row; return dict of structural parameters."""
    seq       = row["sequence"]
    structure, mfe = fold_sequence(seq)

    result = {
        "structure":      structure,
        "mfe":            mfe,
        "tm":             predict_tm(seq),
        "gc_total":       gc_fraction(seq, range(len(seq))),
        "gc_stem":        np.nan,
        "gc_acceptor":    np.nan,
        "gc_d_stem":      np.nan,
        "gc_anticodon":   np.nan,
        "gc_tpsic":       np.nan,
        "gu_wobble":      np.nan,
        "d_arm_present":  False,
        "t_arm_present":  False,
    }

    if structure is None:
        return result

    pairs      = build_pair_map(structure)
    domain_map, arm_flags = assign_trna_domains(seq, structure)

    result["d_arm_present"] = arm_flags["d_arm_present"]
    result["t_arm_present"] = arm_flags["t_arm_present"]
    result["gu_wobble"]     = gu_wobble_fraction(seq, pairs)

    # GC content per domain (paired positions only)
    stem_domains = {
        "gc_stem":      ["acceptor_stem", "D_stem", "anticodon_stem", "TpsiC_stem"],
        "gc_acceptor":  ["acceptor_stem"],
        "gc_d_stem":    ["D_stem"],
        "gc_anticodon": ["anticodon_stem"],
        "gc_tpsic":     ["TpsiC_stem"],
    }
    paired_pos = set(pairs.keys())
    for col, domains in stem_domains.items():
        pos = [p for p, d in domain_map.items()
               if d in domains and p in paired_pos]
        result[col] = gc_fraction(seq, pos)

    return result

# ── Step 6: Run annotation ────────────────────────────────────────────────────
def annotate_all(df, log_interval=2000):
    """Annotate all tRNAs; return DataFrame with structural columns appended."""
    print(f"Annotating {len(df):,} tRNAs with ViennaRNA...")
    records = []
    for i, row in df.iterrows():
        records.append(annotate_trna(row))
        if (i + 1) % log_interval == 0:
            print(f"  [{(i+1)/len(df)*100:5.1f}%] {i+1:,}/{len(df):,}")

    df_struct = pd.concat([df.reset_index(drop=True),
                           pd.DataFrame(records)], axis=1)
    print(f"Annotation complete: {df_struct['structure'].notna().sum():,} folded")
    return df_struct

# ── Step 7: Species-level aggregation ────────────────────────────────────────
def build_species_means(df_struct):
    """Compute per-species mean structural parameters."""
    numeric_cols = [
        "mfe", "tm", "gc_total", "gc_stem", "gc_acceptor",
        "gc_d_stem", "gc_anticodon", "gc_tpsic", "gu_wobble",
    ]
    binary_cols = ["d_arm_present", "t_arm_present"]

    agg_dict = {c: "mean" for c in numeric_cols if c in df_struct.columns}
    agg_dict.update({c: "mean" for c in binary_cols if c in df_struct.columns})
    agg_dict["trna_type"] = "count"

    sp_means = (df_struct.groupby("species_clean")
                .agg(agg_dict)
                .rename(columns={"trna_type": "n_trnas"})
                .reset_index())

    # Merge metadata
    meta_cols = ["species_clean", "class", "thermo", "order", "body_temp"]
    meta = df_struct[meta_cols].drop_duplicates("species_clean")
    sp_means = sp_means.merge(meta, on="species_clean", how="left")

    # Rename loss columns to rates
    sp_means["d_arm_loss"] = 1 - sp_means["d_arm_present"]
    sp_means["t_arm_loss"] = 1 - sp_means["t_arm_present"]

    return sp_means

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load
    df = load_trna_data(os.path.join(INDIR, "trna_final.csv"))

    # Annotate
    df_struct = annotate_all(df)

    # Species means
    sp_means = build_species_means(df_struct)

    # Save
    struct_out = os.path.join(OUTDIR, "trna_structure.csv")
    means_out  = os.path.join(OUTDIR, "species_means.csv")

    df_struct.to_csv(struct_out, index=False)
    sp_means.to_csv(means_out, index=False)

    print(f"\n{'='*55}")
    print(f"STRUCTURE SUMMARY")
    print(f"{'='*55}")
    print(f"tRNAs annotated : {len(df_struct):,}")
    print(f"Species         : {sp_means.shape[0]:,}")
    print(f"\nMFE by class (mean ± SD):")
    for cls, grp in df_struct.groupby("class"):
        v = grp["mfe"].dropna()
        print(f"  {cls:<12} {v.mean():.2f} ± {v.std():.2f} kcal/mol")
    print(f"\nT-arm loss rate by class:")
    for cls, grp in df_struct.groupby("class"):
        rate = (~grp["t_arm_present"]).mean() * 100
        print(f"  {cls:<12} {rate:.1f}%")
    print(f"\nSaved: {struct_out}")
    print(f"Saved: {means_out}")
