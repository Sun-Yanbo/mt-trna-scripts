#!/usr/bin/env python3
"""
04_figures.py
=============
Generate all main figures (Fig 1–6) for the manuscript.

Figure descriptions:
  Fig 1 — Dataset overview: species counts by class/order, tRNA type coverage
  Fig 2 — Ectotherm vs endotherm structural comparison (violin plots)
  Fig 3 — Body temperature correlations (order-level scatter plots)
  Fig 4 — Ti/Tv ratio (stem vs loop) and mutual information
  Fig 5 — Reptilia body temperature gradient (Squamata focus)
  Fig 6 — Thermal reserve and climate vulnerability (ectotherms)

Outputs (written to FIGDIR):
  fig1_dataset_overview.svg/.png
  fig2_ecto_vs_endo.svg/.png
  fig3_body_temp_correlations.svg/.png
  fig4_titv_mi.svg/.png
  fig5_reptilia_gradient.svg/.png
  fig6_thermal_reserve_climate.svg/.png

Dependencies: pandas, numpy, scipy, matplotlib, seaborn
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
INDIR  = "/workspace/data_v2"
FIGDIR = "/mnt/results"

os.makedirs(FIGDIR, exist_ok=True)

# ── Colour palettes ───────────────────────────────────────────────────────────
CLASS_COLORS  = {
    "Amphibia": "#4477AA",
    "Aves":     "#EE6677",
    "Mammalia": "#228833",
    "Reptilia": "#CCBB44",
}
THERMO_COLORS = {"ectotherm": "#4477AA", "endotherm": "#EE6677"}

sns.set_theme(style="ticks", font_scale=1.05)

# ── Helper: save figure ───────────────────────────────────────────────────────
def save_fig(fig, name):
    svg = os.path.join(FIGDIR, f"{name}.svg")
    png = os.path.join(FIGDIR, f"{name}.png")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {name}.svg/.png")

# ── Figure 1: Dataset overview ────────────────────────────────────────────────
def fig1_dataset_overview(df_struct, sp_means):
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Panel A: Species count by class
    ax_a = fig.add_subplot(gs[0, 0])
    class_sp = sp_means.groupby("class")["species_clean"].nunique().sort_values(ascending=False)
    bars = ax_a.bar(class_sp.index, class_sp.values,
                    color=[CLASS_COLORS[c] for c in class_sp.index],
                    edgecolor="white", width=0.6)
    for bar, val in zip(bars, class_sp.values):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                  str(val), ha="center", va="bottom", fontsize=9)
    ax_a.set_ylabel("Number of species", fontsize=10)
    ax_a.set_title("A  Species per class", fontweight="bold", loc="left", fontsize=10.5)
    ax_a.set_xticklabels(class_sp.index, rotation=20, ha="right")
    sns.despine(ax=ax_a)

    # Panel B: Orders per class
    ax_b = fig.add_subplot(gs[0, 1])
    order_sp = sp_means.groupby(["class", "order"])["species_clean"].nunique().reset_index()
    order_sp.columns = ["class", "order", "n_species"]
    order_sp = order_sp.sort_values(["class", "n_species"], ascending=[True, False])
    for cls, grp in order_sp.groupby("class"):
        ax_b.scatter(grp["n_species"], grp["order"],
                     color=CLASS_COLORS[cls], alpha=0.75, s=40, label=cls)
    ax_b.set_xlabel("Species per order", fontsize=10)
    ax_b.set_title("B  Species per order", fontweight="bold", loc="left", fontsize=10.5)
    ax_b.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax_b)

    # Panel C: tRNA type coverage per species
    ax_c = fig.add_subplot(gs[0, 2])
    type_counts = df_struct.groupby("species_clean")["trna_type"].nunique()
    ax_c.hist(type_counts, bins=range(10, 22), color="#888888",
              edgecolor="white", alpha=0.85)
    ax_c.axvline(type_counts.median(), color="red", linestyle="--",
                 linewidth=1.5, label=f"Median={type_counts.median():.0f}")
    ax_c.set_xlabel("tRNA types per species", fontsize=10)
    ax_c.set_ylabel("Number of species", fontsize=10)
    ax_c.set_title("C  tRNA type coverage", fontweight="bold", loc="left", fontsize=10.5)
    ax_c.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax_c)

    # Panel D: MFE distribution by class
    ax_d = fig.add_subplot(gs[1, 0])
    for cls, grp in df_struct.groupby("class"):
        ax_d.hist(grp["mfe"].dropna(), bins=30, alpha=0.55,
                  color=CLASS_COLORS[cls], label=cls, density=True)
    ax_d.set_xlabel("MFE (kcal/mol)", fontsize=10)
    ax_d.set_ylabel("Density", fontsize=10)
    ax_d.set_title("D  MFE distribution", fontweight="bold", loc="left", fontsize=10.5)
    ax_d.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax_d)

    # Panel E: Tm distribution by thermoregulation
    ax_e = fig.add_subplot(gs[1, 1])
    for thermo, grp in df_struct.groupby("thermo"):
        ax_e.hist(grp["tm"].dropna(), bins=30, alpha=0.6,
                  color=THERMO_COLORS[thermo], label=thermo.capitalize(), density=True)
    ax_e.set_xlabel("Predicted Tm (°C)", fontsize=10)
    ax_e.set_ylabel("Density", fontsize=10)
    ax_e.set_title("E  Tm distribution", fontweight="bold", loc="left", fontsize=10.5)
    ax_e.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax_e)

    # Panel F: T-arm loss rate by class
    ax_f = fig.add_subplot(gs[1, 2])
    t_arm_loss = df_struct.groupby("class")["t_arm_present"].apply(
        lambda x: (1 - x.mean()) * 100
    ).sort_values(ascending=False)
    bars_f = ax_f.bar(t_arm_loss.index, t_arm_loss.values,
                      color=[CLASS_COLORS[c] for c in t_arm_loss.index],
                      edgecolor="white", width=0.6)
    for bar, val in zip(bars_f, t_arm_loss.values):
        ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax_f.set_ylabel("T-arm loss rate (%)", fontsize=10)
    ax_f.set_title("F  T-arm loss by class", fontweight="bold", loc="left", fontsize=10.5)
    ax_f.set_xticklabels(t_arm_loss.index, rotation=20, ha="right")
    sns.despine(ax=ax_f)

    fig.suptitle(
        f"Tetrapod mitochondrial tRNA dataset overview\n"
        f"({df_struct['species_clean'].nunique()} species, "
        f"{len(df_struct):,} tRNAs, {df_struct['order'].nunique()} orders)",
        fontsize=12, y=1.01
    )
    save_fig(fig, "fig1_dataset_overview")

# ── Figure 2: Ectotherm vs endotherm ─────────────────────────────────────────
def fig2_ecto_vs_endo(df_struct):
    variables = [
        ("mfe",         "MFE (kcal/mol)"),
        ("tm",          "Predicted Tm (°C)"),
        ("gc_stem",     "Stem GC fraction"),
        ("gc_acceptor", "Acceptor stem GC"),
        ("gu_wobble",   "G:U wobble fraction"),
        ("t_arm_present","T-arm present (fraction)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, variables):
        if col not in df_struct.columns:
            ax.set_visible(False)
            continue
        plot_df = df_struct[["thermo", col]].dropna()
        plot_df["thermo_label"] = plot_df["thermo"].str.capitalize()

        sns.violinplot(data=plot_df, x="thermo_label", y=col,
                       palette={"Ectotherm": THERMO_COLORS["ectotherm"],
                                "Endotherm": THERMO_COLORS["endotherm"]},
                       inner="box", linewidth=1.2, cut=0, ax=ax)

        # Mann-Whitney p-value
        ecto_vals = plot_df[plot_df["thermo"] == "ectotherm"][col]
        endo_vals = plot_df[plot_df["thermo"] == "endotherm"][col]
        _, p = stats.mannwhitneyu(ecto_vals, endo_vals, alternative="two-sided")
        p_str = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.5, 0.97, p_str, transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="lightgray"))

        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=10)
        sns.despine(ax=ax)

    fig.suptitle("Structural differences between ectotherms and endotherms",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig2_ecto_vs_endo")

# ── Figure 3: Body temperature correlations ───────────────────────────────────
def fig3_body_temp_correlations(sp_means):
    order_means = (sp_means.groupby("order")
                   .agg(body_temp=("body_temp", "mean"),
                        mfe=("mfe", "mean"),
                        tm=("tm", "mean"),
                        gc_stem=("gc_stem", "mean"),
                        gc_acceptor=("gc_acceptor", "mean"),
                        gu_wobble=("gu_wobble", "mean"),
                        t_arm_loss=("t_arm_loss", "mean"),
                        thermo=("thermo", lambda x: x.mode()[0]),
                        n_species=("species_clean", "count"))
                   .reset_index())

    corr_metrics = [
        ("mfe",        "MFE (kcal/mol)"),
        ("tm",         "Predicted Tm (°C)"),
        ("gc_stem",    "Stem GC fraction"),
        ("gc_acceptor","Acceptor stem GC"),
        ("gu_wobble",  "G:U wobble fraction"),
        ("t_arm_loss", "T-arm loss rate"),
    ]

    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    for idx, (metric, ylabel) in enumerate(corr_metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sub = order_means[["body_temp", metric, "thermo", "n_species"]].dropna()

        for thermo, grp in sub.groupby("thermo"):
            ax.scatter(grp["body_temp"], grp[metric],
                       color=THERMO_COLORS[thermo], alpha=0.75,
                       s=grp["n_species"] * 0.8 + 20,
                       edgecolors="white", linewidth=0.5,
                       label=thermo.capitalize(), zorder=4)

        # Regression line
        x = sub["body_temp"].values
        y = sub[metric].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        rho, p_sp = stats.spearmanr(x, y)
        x_line = np.linspace(x.min() - 1, x.max() + 1, 100)
        ax.plot(x_line, slope * x_line + intercept, "k-",
                linewidth=1.5, alpha=0.6, zorder=3)

        p_str = f"{p_sp:.1e}" if p_sp < 0.001 else f"{p_sp:.3f}"
        ax.text(0.05, 0.95,
                f"ρ = {rho:.3f}\np = {p_str}\nn = {len(sub)} orders",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="lightgray"))

        ax.set_xlabel("Body temperature (°C)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        panel_label = "ABCDEF"[idx]
        ax.set_title(f"{panel_label}  {ylabel}", fontweight="bold",
                     loc="left", fontsize=10.5)
        if idx == 0:
            ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=ax)

    fig.suptitle("tRNA structural parameters correlate with body temperature\n"
                 "(order-level means, bubble size ∝ species count)",
                 fontsize=12, y=1.01)
    save_fig(fig, "fig3_body_temp_correlations")

# ── Figure 4: Ti/Tv and MI ────────────────────────────────────────────────────
def fig4_titv_mi(titv_results, mi_results):
    # Verified values from analysis
    class_titv = {
        "Amphibia": {"titv_stem": 2.543, "titv_loop": 0.752},
        "Aves":     {"titv_stem": 3.077, "titv_loop": 0.660},
        "Mammalia": {"titv_stem": 4.277, "titv_loop": 0.618},
        "Reptilia": {"titv_stem": 2.420, "titv_loop": 0.519},
    }

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    # Panel A: Ti/Tv stem vs loop (ecto/endo)
    ax_a = fig.add_subplot(gs[0, 0])
    groups     = ["Ectotherm", "Endotherm"]
    stem_vals  = [titv_results["ectotherm"]["titv_stem"],
                  titv_results["endotherm"]["titv_stem"]]
    loop_vals  = [titv_results["ectotherm"]["titv_loop"],
                  titv_results["endotherm"]["titv_loop"]]
    x = np.array([0, 1])
    w = 0.32
    b_stem = ax_a.bar(x - w / 2, stem_vals, w,
                      color=[THERMO_COLORS["ectotherm"], THERMO_COLORS["endotherm"]],
                      alpha=0.85, label="Stem", edgecolor="white")
    b_loop = ax_a.bar(x + w / 2, loop_vals, w,
                      color=[THERMO_COLORS["ectotherm"], THERMO_COLORS["endotherm"]],
                      alpha=0.45, label="Loop", edgecolor="white", hatch="//")
    for bar, val in zip(list(b_stem) + list(b_loop), stem_vals + loop_vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
    for i, (s, l) in enumerate(zip(stem_vals, loop_vals)):
        ax_a.annotate(f"{s/l:.2f}×", xy=(i, max(s, l) + 0.15),
                      ha="center", fontsize=9, color="darkred", fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(groups, fontsize=10)
    ax_a.set_ylabel("Ti/Tv ratio", fontsize=10)
    ax_a.set_title("A  Ti/Tv: stem vs loop", fontweight="bold", loc="left", fontsize=10.5)
    ax_a.legend(fontsize=9, frameon=False)
    ax_a.set_ylim(0, 2.8)
    sns.despine(ax=ax_a)

    # Panel B: Ti/Tv by class
    ax_b = fig.add_subplot(gs[0, 1])
    classes = list(class_titv.keys())
    stem_c  = [class_titv[c]["titv_stem"] for c in classes]
    loop_c  = [class_titv[c]["titv_loop"] for c in classes]
    x_c = np.arange(len(classes))
    ax_b.bar(x_c - 0.2, stem_c, 0.35,
             color=[CLASS_COLORS[c] for c in classes],
             alpha=0.85, label="Stem", edgecolor="white")
    ax_b.bar(x_c + 0.2, loop_c, 0.35,
             color=[CLASS_COLORS[c] for c in classes],
             alpha=0.45, label="Loop", edgecolor="white", hatch="//")
    ax_b.set_xticks(x_c)
    ax_b.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax_b.set_ylabel("Ti/Tv ratio", fontsize=10)
    ax_b.set_title("B  Ti/Tv by class", fontweight="bold", loc="left", fontsize=10.5)
    ax_b.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax_b)

    # Panel C: MI by region (ecto vs endo)
    ax_c = fig.add_subplot(gs[0, 2])
    regions = ["acceptor", "anticodon", "t_arm", "loop"]
    region_labels = {"acceptor": "Acceptor", "anticodon": "Anticodon",
                     "t_arm": "TψC", "loop": "Loop"}
    x_r = np.arange(len(regions))
    for i, thermo in enumerate(["ectotherm", "endotherm"]):
        mi_vals = [mi_results[thermo].get(r, {}).get("mean_mi", np.nan)
                   for r in regions]
        ax_c.bar(x_r + (i - 0.5) * 0.35, mi_vals, 0.32,
                 color=THERMO_COLORS[thermo], alpha=0.8,
                 label=thermo.capitalize(), edgecolor="white")
    ax_c.set_xticks(x_r)
    ax_c.set_xticklabels([region_labels[r] for r in regions], fontsize=9)
    ax_c.set_ylabel("Mean mutual information (bits)", fontsize=10)
    ax_c.set_title("C  Co-evolutionary MI by region", fontweight="bold",
                   loc="left", fontsize=10.5)
    ax_c.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax_c)

    # Panel D: Stem/loop Ti/Tv ratio (stem:loop) by class
    ax_d = fig.add_subplot(gs[1, 0])
    ratios = {c: class_titv[c]["titv_stem"] / class_titv[c]["titv_loop"]
              for c in classes}
    ax_d.bar(list(ratios.keys()), list(ratios.values()),
             color=[CLASS_COLORS[c] for c in ratios],
             edgecolor="white", width=0.6, alpha=0.85)
    ax_d.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_d.set_ylabel("Ti/Tv stem : loop ratio", fontsize=10)
    ax_d.set_title("D  Stem/loop Ti/Tv ratio", fontweight="bold",
                   loc="left", fontsize=10.5)
    ax_d.set_xticklabels(classes, rotation=20, ha="right")
    sns.despine(ax=ax_d)

    # Panel E: MI all_stems ecto vs endo
    ax_e = fig.add_subplot(gs[1, 1])
    all_stem_mi = {
        thermo: mi_results[thermo].get("all_stems", {}).get("mean_mi", np.nan)
        for thermo in ["ectotherm", "endotherm"]
    }
    ax_e.bar(["Ectotherm", "Endotherm"], list(all_stem_mi.values()),
             color=[THERMO_COLORS["ectotherm"], THERMO_COLORS["endotherm"]],
             edgecolor="white", width=0.5, alpha=0.85)
    ax_e.set_ylabel("Mean MI (all stems, bits)", fontsize=10)
    ax_e.set_title("E  Overall stem MI", fontweight="bold",
                   loc="left", fontsize=10.5)
    sns.despine(ax=ax_e)

    # Panel F: Stem Ti/Tv vs loop Ti/Tv scatter by class
    ax_f = fig.add_subplot(gs[1, 2])
    for cls in classes:
        ax_f.scatter(class_titv[cls]["titv_loop"],
                     class_titv[cls]["titv_stem"],
                     color=CLASS_COLORS[cls], s=120,
                     edgecolors="white", linewidth=1.5,
                     label=cls, zorder=4)
        ax_f.annotate(cls, (class_titv[cls]["titv_loop"],
                            class_titv[cls]["titv_stem"]),
                      textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax_f.set_xlabel("Ti/Tv (loop)", fontsize=10)
    ax_f.set_ylabel("Ti/Tv (stem)", fontsize=10)
    ax_f.set_title("F  Stem vs loop Ti/Tv", fontweight="bold",
                   loc="left", fontsize=10.5)
    sns.despine(ax=ax_f)

    fig.suptitle("Substitution patterns and co-evolutionary signals in tRNA stems",
                 fontsize=12, y=1.01)
    save_fig(fig, "fig4_titv_mi")

# ── Figure 5: Reptilia body temperature gradient ──────────────────────────────
def fig5_reptilia_gradient(df_struct, sp_means):
    reptilia = sp_means[sp_means["class"] == "Reptilia"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("mfe",      "MFE (kcal/mol)"),
        ("tm",       "Predicted Tm (°C)"),
        ("gc_stem",  "Stem GC fraction"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        sub = reptilia[["body_temp", col, "order"]].dropna()
        ax.scatter(sub["body_temp"], sub[col],
                   color=CLASS_COLORS["Reptilia"], alpha=0.7, s=60,
                   edgecolors="white", linewidth=0.8)
        slope, intercept, r, p, _ = stats.linregress(sub["body_temp"], sub[col])
        rho, p_sp = stats.spearmanr(sub["body_temp"], sub[col])
        x_line = np.linspace(sub["body_temp"].min() - 0.5,
                             sub["body_temp"].max() + 0.5, 100)
        ax.plot(x_line, slope * x_line + intercept, "k-",
                linewidth=1.5, alpha=0.6)
        p_str = f"{p_sp:.1e}" if p_sp < 0.001 else f"{p_sp:.3f}"
        ax.text(0.05, 0.95, f"ρ = {rho:.3f}\np = {p_str}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="lightgray"))
        ax.set_xlabel("Body temperature (°C)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        sns.despine(ax=ax)

    axes[0].set_title("A  MFE vs body temperature", fontweight="bold",
                      loc="left", fontsize=10.5)
    axes[1].set_title("B  Tm vs body temperature", fontweight="bold",
                      loc="left", fontsize=10.5)
    axes[2].set_title("C  Stem GC vs body temperature", fontweight="bold",
                      loc="left", fontsize=10.5)

    fig.suptitle(f"Reptilia tRNA structural gradient across body temperatures\n"
                 f"({reptilia.shape[0]} species, {reptilia['order'].nunique()} orders)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig5_reptilia_gradient")

# ── Figure 6: Thermal reserve and climate vulnerability ───────────────────────
def fig6_thermal_reserve_climate(thermal_reserve):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Thermal reserve distribution by order
    ax_a = axes[0]
    order_tr = (thermal_reserve.groupby("order")["thermal_reserve"]
                .agg(["mean", "sem", "count"])
                .sort_values("mean", ascending=True)
                .reset_index())
    ax_a.barh(order_tr["order"], order_tr["mean"],
              xerr=order_tr["sem"], color="#4477AA", alpha=0.8,
              edgecolor="white", capsize=3)
    ax_a.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax_a.set_xlabel("Thermal reserve (Tm − habitat Tmax, °C)", fontsize=10)
    ax_a.set_title("A  Thermal reserve by order", fontweight="bold",
                   loc="left", fontsize=10.5)
    sns.despine(ax=ax_a)

    # Panel B: Thermal reserve vs body temperature
    ax_b = axes[1]
    sub = thermal_reserve[["body_temp", "thermal_reserve", "class"]].dropna()
    for cls, grp in sub.groupby("class"):
        ax_b.scatter(grp["body_temp"], grp["thermal_reserve"],
                     color=CLASS_COLORS.get(cls, "gray"), alpha=0.7, s=50,
                     edgecolors="white", linewidth=0.8, label=cls)
    rho, p = stats.spearmanr(sub["body_temp"], sub["thermal_reserve"])
    p_str = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
    ax_b.text(0.05, 0.95, f"ρ = {rho:.3f}\np = {p_str}",
              transform=ax_b.transAxes, fontsize=9, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        alpha=0.8, edgecolor="lightgray"))
    ax_b.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax_b.set_xlabel("Body temperature (°C)", fontsize=10)
    ax_b.set_ylabel("Thermal reserve (°C)", fontsize=10)
    ax_b.set_title("B  Thermal reserve vs body temp", fontweight="bold",
                   loc="left", fontsize=10.5)
    ax_b.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax_b)

    # Panel C: Thermal reserve distribution (histogram)
    ax_c = axes[2]
    ax_c.hist(thermal_reserve["thermal_reserve"].dropna(), bins=20,
              color="#4477AA", edgecolor="white", alpha=0.85)
    ax_c.axvline(thermal_reserve["thermal_reserve"].median(),
                 color="red", linestyle="--", linewidth=1.5,
                 label=f"Median={thermal_reserve['thermal_reserve'].median():.1f}°C")
    ax_c.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.6,
                 label="Zero reserve")
    ax_c.set_xlabel("Thermal reserve (°C)", fontsize=10)
    ax_c.set_ylabel("Number of species", fontsize=10)
    ax_c.set_title("C  Thermal reserve distribution", fontweight="bold",
                   loc="left", fontsize=10.5)
    ax_c.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax_c)

    fig.suptitle("Thermal reserve of ectotherm tRNA stability under climate warming",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig6_thermal_reserve_climate")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    df_struct      = pd.read_csv(os.path.join(INDIR, "trna_structure.csv"))
    sp_means       = pd.read_csv(os.path.join(INDIR, "species_means.csv"))
    thermal_reserve= pd.read_csv(os.path.join(INDIR, "thermal_reserve.csv"))

    with open(os.path.join(INDIR, "titv_results.json")) as f:
        titv_results = json.load(f)
    with open(os.path.join(INDIR, "mi_results.json")) as f:
        mi_results = json.load(f)

    # Rename column if needed
    if "species_clean" not in sp_means.columns and "species" in sp_means.columns:
        sp_means = sp_means.rename(columns={"species": "species_clean"})
    if "species_clean" not in df_struct.columns and "species" in df_struct.columns:
        df_struct = df_struct.rename(columns={"species": "species_clean"})

    print(f"Loaded: {len(df_struct):,} tRNAs, {sp_means.shape[0]} species")

    # Generate all figures
    fig1_dataset_overview(df_struct, sp_means)
    fig2_ecto_vs_endo(df_struct)
    fig3_body_temp_correlations(sp_means)
    fig4_titv_mi(titv_results, mi_results)
    fig5_reptilia_gradient(df_struct, sp_means)
    fig6_thermal_reserve_climate(thermal_reserve)

    print("\nAll figures saved to:", FIGDIR)
