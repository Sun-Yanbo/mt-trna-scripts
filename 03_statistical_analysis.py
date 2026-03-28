#!/usr/bin/env python3
"""
03_statistical_analysis.py
==========================
Core statistical analyses for the tetrapod mt-tRNA study.

Analyses:
  A. Ectotherm vs endotherm structural comparison (Mann-Whitney U)
  B. Body temperature correlations (Spearman, order-level)
  C. T-arm / D-arm loss rates (chi-square across classes)
  D. Ti/Tv ratio in stem vs loop regions (MAFFT + structure-based)
  E. Mutual information (co-evolutionary signal) in tRNA stems
  F. Thermal reserve analysis for ectotherms (Tm − habitat Tmax)

Outputs (written to OUTDIR):
  stats_ecto_vs_endo.csv       — Mann-Whitney results, all variables
  stats_body_temp_corr.csv     — Spearman correlations with body temperature
  stats_arm_loss.csv           — Chi-square arm loss results
  titv_results.json            — Ti/Tv by thermoregulation group and class
  mi_results.json              — MI by region (ecto vs endo)
  thermal_reserve.csv          — Per-species thermal reserve estimates

Dependencies: pandas, numpy, scipy, subprocess (mafft), json
"""

import os
import json
import subprocess
import tempfile
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
INDIR  = "/workspace/data_v2"
OUTDIR = "/workspace/data_v2"

# Habitat Tmax values for ectotherm orders (°C), from literature
# Sources: IUCN range maps + WorldClim Tmax data
HABITAT_TMAX = {
    "Anura":          28.5,
    "Caudata":        22.0,
    "Gymnophiona":    30.0,
    "Squamata":       35.0,
    "Testudines":     32.0,
    "Crocodilia":     34.0,
    "Rhynchocephalia":25.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def mannwhitney_summary(group_a, group_b, label_a="ecto", label_b="endo"):
    """Run Mann-Whitney U test; return summary dict."""
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 3 or len(b) < 3:
        return None
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Effect size: rank-biserial correlation
    n1, n2 = len(a), len(b)
    r = 1 - (2 * stat) / (n1 * n2)
    return {
        f"n_{label_a}": n1,
        f"n_{label_b}": n2,
        f"mean_{label_a}": round(a.mean(), 4),
        f"mean_{label_b}": round(b.mean(), 4),
        "U_stat": round(stat, 1),
        "p_value": p,
        "effect_r": round(r, 4),
    }

def run_mafft(sequences, labels):
    """Run MAFFT on sequences; return list of aligned sequences."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            f.write(f">{lab}_{i}\n{seq}\n")
        fasta_in = f.name
    try:
        result = subprocess.run(
            ["mafft", "--auto", "--quiet", fasta_in],
            capture_output=True, text=True, timeout=120
        )
        os.unlink(fasta_in)
    except Exception as e:
        os.unlink(fasta_in)
        return []
    aln = {}
    cur = None
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:]
            aln[cur] = ""
        elif cur:
            aln[cur] += line.upper()
    return list(aln.values())

def get_paired_positions(structure):
    """Return set of base-paired positions from dot-bracket string."""
    paired = set()
    stack  = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                paired.add(i)
                paired.add(j)
    return paired

def map_orig_to_aln(orig_seq, aln_seq):
    """Map original sequence positions to alignment column indices."""
    orig_to_aln = {}
    orig_pos = 0
    for aln_pos, c in enumerate(aln_seq):
        if c != "-":
            orig_to_aln[orig_pos] = aln_pos
            orig_pos += 1
    return orig_to_aln

# ── Analysis A: Ectotherm vs endotherm comparison ────────────────────────────
def analysis_ecto_vs_endo(df_struct):
    """Mann-Whitney U tests for all structural variables, ecto vs endo."""
    print("\n=== Analysis A: Ectotherm vs Endotherm ===")
    variables = [
        ("mfe",           "MFE (kcal/mol)"),
        ("tm",            "Predicted Tm (°C)"),
        ("gc_stem",       "Stem GC fraction"),
        ("gc_acceptor",   "Acceptor stem GC"),
        ("gc_d_stem",     "D-stem GC"),
        ("gc_anticodon",  "Anticodon stem GC"),
        ("gc_tpsic",      "TψC stem GC"),
        ("gu_wobble",     "G:U wobble fraction"),
        ("d_arm_present", "D-arm present"),
        ("t_arm_present", "T-arm present"),
    ]
    results = []
    ecto = df_struct[df_struct["thermo"] == "ectotherm"]
    endo = df_struct[df_struct["thermo"] == "endotherm"]

    for col, label in variables:
        if col not in df_struct.columns:
            continue
        res = mannwhitney_summary(ecto[col], endo[col])
        if res:
            res["variable"] = label
            results.append(res)
            print(f"  {label:<25} ecto={res['mean_ecto']:.4f}  "
                  f"endo={res['mean_endo']:.4f}  p={res['p_value']:.2e}")

    df_res = pd.DataFrame(results)
    # Bonferroni correction
    df_res["p_bonferroni"] = (df_res["p_value"] * len(df_res)).clip(upper=1.0)
    return df_res

# ── Analysis B: Body temperature correlations ─────────────────────────────────
def analysis_body_temp_corr(sp_means):
    """Spearman correlations between body temperature and structural variables
    at the order level (to avoid pseudoreplication)."""
    print("\n=== Analysis B: Body Temperature Correlations (order-level) ===")
    order_means = (sp_means.groupby("order")
                   .agg(body_temp=("body_temp", "mean"),
                        mfe=("mfe", "mean"),
                        tm=("tm", "mean"),
                        gc_stem=("gc_stem", "mean"),
                        gc_acceptor=("gc_acceptor", "mean"),
                        gu_wobble=("gu_wobble", "mean"),
                        t_arm_loss=("t_arm_loss", "mean"),
                        n_species=("species_clean", "count"))
                   .reset_index())

    variables = ["mfe", "tm", "gc_stem", "gc_acceptor", "gu_wobble", "t_arm_loss"]
    results = []
    for col in variables:
        sub = order_means[["body_temp", col]].dropna()
        if len(sub) < 5:
            continue
        rho, p = stats.spearmanr(sub["body_temp"], sub[col])
        results.append({
            "variable": col,
            "n_orders": len(sub),
            "spearman_rho": round(rho, 4),
            "p_value": p,
        })
        print(f"  {col:<20} rho={rho:.3f}  p={p:.2e}  n={len(sub)}")

    return pd.DataFrame(results)

# ── Analysis C: Arm loss chi-square ──────────────────────────────────────────
def analysis_arm_loss(df_struct):
    """Chi-square tests for T-arm and D-arm loss across classes."""
    print("\n=== Analysis C: Arm Loss Chi-Square ===")
    results = []
    for arm, col in [("T-arm", "t_arm_present"), ("D-arm", "d_arm_present")]:
        if col not in df_struct.columns:
            continue
        ct = pd.crosstab(df_struct["class"], df_struct[col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        print(f"  {arm}: chi2={chi2:.1f}, df={dof}, p={p:.2e}")
        results.append({"arm": arm, "chi2": chi2, "df": dof, "p_value": p})

        # Ecto vs endo
        ct2 = pd.crosstab(df_struct["thermo"], df_struct[col])
        chi2_2, p2, dof2, _ = stats.chi2_contingency(ct2)
        print(f"  {arm} (ecto vs endo): chi2={chi2_2:.1f}, p={p2:.2e}")
        results.append({"arm": f"{arm}_ecto_vs_endo",
                        "chi2": chi2_2, "df": dof2, "p_value": p2})
    return pd.DataFrame(results)

# ── Analysis D: Ti/Tv in stem vs loop ────────────────────────────────────────
TRANSITIONS   = {frozenset(["A","G"]), frozenset(["C","U"]),
                 frozenset(["C","T"]), frozenset(["T","U"])}
TRANSVERSIONS = {frozenset(["A","C"]), frozenset(["A","U"]), frozenset(["A","T"]),
                 frozenset(["G","C"]), frozenset(["G","U"]), frozenset(["G","T"])}

def classify_sub(b1, b2):
    p = frozenset([b1.upper(), b2.upper()])
    if p in TRANSITIONS:   return "Ti"
    if p in TRANSVERSIONS: return "Tv"
    return None

def compute_titv(df_struct, trna_type="tRNA-Phe", max_seqs=150, seed=42):
    """
    Compute Ti/Tv ratio in stem vs loop positions using MAFFT alignment
    and ViennaRNA structure-based stem/loop classification.
    Returns dict: {group: {titv_stem, titv_loop, n_stem_cols, n_loop_cols}}
    """
    print(f"\n=== Analysis D: Ti/Tv ({trna_type}) ===")
    np.random.seed(seed)
    results = {}

    groups = {
        "ectotherm": df_struct[df_struct["thermo"] == "ectotherm"],
        "endotherm": df_struct[df_struct["thermo"] == "endotherm"],
    }
    # Also by class
    for cls in df_struct["class"].unique():
        groups[cls] = df_struct[df_struct["class"] == cls]

    for group_name, grp in groups.items():
        sub = grp[grp["trna_type"] == trna_type].dropna(
            subset=["sequence", "structure"])
        if len(sub) < 10:
            continue
        if len(sub) > max_seqs:
            sub = sub.sample(max_seqs, random_state=seed)

        seqs    = sub["sequence"].tolist()
        structs = sub["structure"].tolist()
        labels  = sub["class"].tolist()

        aln_seqs = run_mafft(seqs, labels)
        if not aln_seqs or len(aln_seqs) < 4:
            continue

        aln_len = len(aln_seqs[0])

        # Build stem/loop column map by majority vote
        stem_votes  = np.zeros(aln_len, dtype=int)
        total_votes = np.zeros(aln_len, dtype=int)
        for seq, struct, aln_seq in zip(seqs, structs, aln_seqs):
            if not struct or len(struct) != len(seq):
                continue
            paired      = get_paired_positions(struct)
            orig_to_aln = map_orig_to_aln(seq, aln_seq)
            for orig_pos, aln_pos in orig_to_aln.items():
                total_votes[aln_pos] += 1
                if orig_pos in paired:
                    stem_votes[aln_pos] += 1

        is_stem = np.array([
            (stem_votes[c] / total_votes[c] > 0.50) if total_votes[c] > 0 else False
            for c in range(aln_len)
        ])
        stem_cols = set(np.where(is_stem)[0])
        loop_cols = set(np.where(~is_stem)[0])

        # Count Ti/Tv at stem vs loop columns
        ti_stem = tv_stem = ti_loop = tv_loop = 0
        for i in range(len(aln_seqs)):
            for j in range(i + 1, len(aln_seqs)):
                for pos in range(aln_len):
                    b1, b2 = aln_seqs[i][pos], aln_seqs[j][pos]
                    if b1 == "-" or b2 == "-" or b1 == b2:
                        continue
                    sub_type = classify_sub(b1, b2)
                    if sub_type is None:
                        continue
                    if pos in stem_cols:
                        if sub_type == "Ti": ti_stem += 1
                        else:                tv_stem += 1
                    elif pos in loop_cols:
                        if sub_type == "Ti": ti_loop += 1
                        else:                tv_loop += 1

        titv_stem = ti_stem / tv_stem if tv_stem > 0 else np.nan
        titv_loop = ti_loop / tv_loop if tv_loop > 0 else np.nan
        results[group_name] = {
            "titv_stem":   round(titv_stem, 4) if not np.isnan(titv_stem) else None,
            "titv_loop":   round(titv_loop, 4) if not np.isnan(titv_loop) else None,
            "n_stem_cols": len(stem_cols),
            "n_loop_cols": len(loop_cols),
            "n_seqs":      len(sub),
        }
        print(f"  {group_name:<12} stem={titv_stem:.3f}  loop={titv_loop:.3f}  "
              f"(n={len(sub)})")

    return results

# ── Analysis E: Mutual information ───────────────────────────────────────────
def entropy(x):
    vals, cnts = np.unique(x, return_counts=True)
    p = cnts / cnts.sum()
    return -np.sum(p * np.log2(p + 1e-10))

def compute_mi_for_group(sequences, structures, labels, max_seqs=200, seed=42):
    """
    Compute mutual information between paired positions in tRNA stems.
    Returns dict: {region: mean_MI}
    """
    np.random.seed(seed)
    if len(sequences) > max_seqs:
        idx = np.random.choice(len(sequences), max_seqs, replace=False)
        sequences  = [sequences[i]  for i in idx]
        structures = [structures[i] for i in idx]
        labels     = [labels[i]     for i in idx]

    aln_seqs = run_mafft(sequences, labels)
    if not aln_seqs or len(aln_seqs) < 4:
        return {}

    aln_len = len(aln_seqs[0])
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    mat = np.full((len(aln_seqs), aln_len), -1, dtype=np.int8)
    for i, seq in enumerate(aln_seqs):
        for j, b in enumerate(seq):
            mat[i, j] = base_map.get(b, -1)

    # Build stem/loop column map
    stem_votes  = np.zeros(aln_len, dtype=int)
    total_votes = np.zeros(aln_len, dtype=int)
    for seq, struct, aln_seq in zip(sequences, structures, aln_seqs):
        if not struct or len(struct) != len(seq):
            continue
        paired      = get_paired_positions(struct)
        orig_to_aln = map_orig_to_aln(seq, aln_seq)
        for orig_pos, aln_pos in orig_to_aln.items():
            total_votes[aln_pos] += 1
            if orig_pos in paired:
                stem_votes[aln_pos] += 1

    is_stem = np.array([
        (stem_votes[c] / total_votes[c] > 0.50) if total_votes[c] > 0 else False
        for c in range(aln_len)
    ])
    stem_cols = np.where(is_stem)[0]
    loop_cols = np.where(~is_stem)[0]

    # Assign stem columns to regions by position order
    n_stem = len(stem_cols)
    q1, q2 = n_stem // 4, 3 * n_stem // 4
    region_cols = {
        "acceptor":  set(stem_cols[:q1].tolist() + stem_cols[-q1:].tolist()),
        "anticodon": set(stem_cols[q1:q2].tolist()),
        "t_arm":     set(stem_cols[q2:].tolist()),
        "loop":      set(loop_cols.tolist()),
        "all_stems": set(stem_cols.tolist()),
    }

    # Compute MI for each region
    mi_by_region = {}
    for region, cols in region_cols.items():
        col_list = sorted(cols)
        mi_vals  = []
        for a in range(len(col_list)):
            for b in range(a + 1, len(col_list)):
                c1 = mat[:, col_list[a]]
                c2 = mat[:, col_list[b]]
                mask = (c1 >= 0) & (c2 >= 0)
                if mask.sum() < 10:
                    continue
                h1  = entropy(c1[mask])
                h2  = entropy(c2[mask])
                h12 = entropy(c1[mask] * 4 + c2[mask])
                mi_vals.append(h1 + h2 - h12)
        mi_by_region[region] = {
            "mean_mi": round(float(np.mean(mi_vals)), 4) if mi_vals else None,
            "n_pairs": len(mi_vals),
        }
    return mi_by_region

def analysis_mi(df_struct, trna_type="tRNA-Phe"):
    """Run MI analysis for ecto vs endo groups."""
    print(f"\n=== Analysis E: Mutual Information ({trna_type}) ===")
    results = {}
    for group in ["ectotherm", "endotherm"]:
        sub = df_struct[
            (df_struct["trna_type"] == trna_type) &
            (df_struct["thermo"] == group)
        ].dropna(subset=["sequence", "structure"])
        print(f"  {group} ({len(sub)} seqs)...")
        mi = compute_mi_for_group(
            sub["sequence"].tolist(),
            sub["structure"].tolist(),
            sub["class"].tolist(),
        )
        results[group] = mi
        for region, vals in mi.items():
            print(f"    {region:<12} MI={vals['mean_mi']}  n_pairs={vals['n_pairs']}")
    return results

# ── Analysis F: Thermal reserve ───────────────────────────────────────────────
def analysis_thermal_reserve(sp_means):
    """
    Compute thermal reserve = predicted Tm − habitat Tmax for ectotherms.
    Higher thermal reserve → greater buffer against heat stress.
    """
    print("\n=== Analysis F: Thermal Reserve (ectotherms) ===")
    ecto = sp_means[sp_means["thermo"] == "ectotherm"].copy()
    ecto["habitat_tmax"] = ecto["order"].map(HABITAT_TMAX)
    ecto["thermal_reserve"] = ecto["tm"] - ecto["habitat_tmax"]

    print(f"  Ectotherm species: {len(ecto)}")
    print(f"  Thermal reserve (mean ± SD): "
          f"{ecto['thermal_reserve'].mean():.2f} ± "
          f"{ecto['thermal_reserve'].std():.2f} °C")
    print(f"\n  By order:")
    for order, grp in ecto.groupby("order"):
        tr = grp["thermal_reserve"].dropna()
        print(f"    {order:<20} {tr.mean():.2f} ± {tr.std():.2f} °C  (n={len(tr)})")

    return ecto[["species_clean", "class", "order", "thermo",
                 "tm", "habitat_tmax", "thermal_reserve"]]

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    df_struct = pd.read_csv(os.path.join(INDIR, "trna_structure.csv"))
    sp_means  = pd.read_csv(os.path.join(INDIR, "species_means.csv"))

    print(f"Loaded: {len(df_struct):,} tRNAs, {sp_means.shape[0]} species")

    # Run analyses
    df_ecto_endo = analysis_ecto_vs_endo(df_struct)
    df_corr      = analysis_body_temp_corr(sp_means)
    df_arm_loss  = analysis_arm_loss(df_struct)
    titv_results = compute_titv(df_struct)
    mi_results   = analysis_mi(df_struct)
    df_thermal   = analysis_thermal_reserve(sp_means)

    # Save
    df_ecto_endo.to_csv(os.path.join(OUTDIR, "stats_ecto_vs_endo.csv"),   index=False)
    df_corr.to_csv(     os.path.join(OUTDIR, "stats_body_temp_corr.csv"), index=False)
    df_arm_loss.to_csv( os.path.join(OUTDIR, "stats_arm_loss.csv"),       index=False)
    df_thermal.to_csv(  os.path.join(OUTDIR, "thermal_reserve.csv"),      index=False)

    with open(os.path.join(OUTDIR, "titv_results.json"), "w") as f:
        json.dump(titv_results, f, indent=2)
    with open(os.path.join(OUTDIR, "mi_results.json"), "w") as f:
        json.dump(mi_results, f, indent=2)

    print(f"\nAll results saved to {OUTDIR}/")
