#!/usr/bin/env Rscript
# 05_phylogeny.R
# ==============
# Visualize the tetrapod mitochondrial tRNA phylogenetic tree using ggtree.
#
# Input files (from 05_phylogeny.py):
#   /workspace/tetrapod_otl_dendropy.nwk   — labelled Newick cladogram
#   /workspace/tree_annotations.csv        — tip metadata
#
# Output:
#   /mnt/results/FigS_phylogeny.svg
#   /mnt/results/FigS_phylogeny.png
#
# Dependencies: ape, ggtree, ggplot2, dplyr, ggnewscale

suppressPackageStartupMessages({
  library(ape)
  library(ggtree)
  library(ggplot2)
  library(dplyr)
  library(ggnewscale)
})

# ── Configuration ──────────────────────────────────────────────────────────────
TREE_FILE <- "/workspace/tetrapod_otl_dendropy.nwk"
ANN_FILE  <- "/workspace/tree_annotations.csv"
OUT_SVG   <- "/mnt/results/FigS_phylogeny.svg"
OUT_PNG   <- "/mnt/results/FigS_phylogeny.png"

CLASS_COLORS  <- c(
  "Amphibia" = "#4DAF4A",   # green
  "Aves"     = "#377EB8",   # blue
  "Mammalia" = "#FF7F00",   # orange
  "Reptilia" = "#E41A1C"    # red
)
CLASS_BG <- c(
  "Amphibia" = "#E8F5E9",
  "Aves"     = "#E3F2FD",
  "Mammalia" = "#FFF3E0",
  "Reptilia" = "#FFEBEE"
)

# ── Load tree and annotations ──────────────────────────────────────────────────
tree_raw <- read.tree(TREE_FILE)
ann      <- read.csv(ANN_FILE, stringsAsFactors = FALSE)

cat("Tips in tree:", length(tree_raw$tip.label), "\n")

# ── Root on Amphibia (outgroup) ────────────────────────────────────────────────
amphibia_tips <- ann$tip_label_underscore[
  !is.na(ann$class) & ann$class == "Amphibia"
]
amphibia_tips <- intersect(amphibia_tips, tree_raw$tip.label)
cat("Amphibia tips for outgroup:", length(amphibia_tips), "\n")

if (length(amphibia_tips) >= 2) {
  amphibia_mrca <- getMRCA(tree_raw, amphibia_tips)
  tree_rooted   <- root(tree_raw, node = amphibia_mrca, resolve.root = TRUE)
} else if (length(amphibia_tips) == 1) {
  tree_rooted <- root(tree_raw, outgroup = amphibia_tips[1], resolve.root = TRUE)
} else {
  tree_rooted <- tree_raw
  warning("No Amphibia tips found for rooting")
}
cat("Rooted:", is.rooted(tree_rooted), "| Tips:", length(tree_rooted$tip.label), "\n")

# ── Prepare annotation data frame ─────────────────────────────────────────────
ann_df <- ann %>%
  select(tip_label_underscore, class, order, thermo) %>%
  rename(label = tip_label_underscore) %>%
  filter(!is.na(class))

# ── Identify orders with ≥10 species for labelling ────────────────────────────
order_counts <- ann_df %>%
  group_by(order) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n >= 10)
orders_to_label <- order_counts$order
cat("Orders to label:", length(orders_to_label), "\n")

# ── Build ggtree plot ──────────────────────────────────────────────────────────
p <- ggtree(tree_rooted, layout = "fan", open.angle = 15,
            size = 0.15, color = "grey40") %<+% ann_df

# Tip points coloured by class
p <- p +
  geom_tippoint(
    aes(color = class, shape = thermo),
    size  = 0.9,
    alpha = 0.85,
    na.rm = TRUE
  ) +
  scale_color_manual(
    values = CLASS_COLORS,
    name   = "Class",
    na.value = "grey70",
    guide  = guide_legend(override.aes = list(size = 3))
  ) +
  scale_shape_manual(
    values = c("ectotherm" = 17, "endotherm" = 16),   # triangle / circle
    name   = "Thermoregulation",
    labels = c("ectotherm" = "Ectotherm", "endotherm" = "Endotherm"),
    na.value = 1,
    guide  = guide_legend(override.aes = list(size = 3))
  )

# Clade highlight rectangles for each class
for (cls in names(CLASS_COLORS)) {
  cls_tips <- ann_df$label[!is.na(ann_df$class) & ann_df$class == cls]
  cls_tips <- intersect(cls_tips, tree_rooted$tip.label)
  if (length(cls_tips) >= 2) {
    mrca_node <- getMRCA(tree_rooted, cls_tips)
    p <- p + geom_hilight(
      node  = mrca_node,
      fill  = CLASS_BG[cls],
      alpha = 0.35,
      extend = 0.5
    )
  }
}

# Order-level arc labels (geom_cladelab) for orders with ≥10 species
for (ord in orders_to_label) {
  ord_tips <- ann_df$label[!is.na(ann_df$order) & ann_df$order == ord]
  ord_tips <- intersect(ord_tips, tree_rooted$tip.label)
  if (length(ord_tips) >= 2) {
    mrca_node <- getMRCA(tree_rooted, ord_tips)
    cls_of_ord <- ann_df$class[ann_df$order == ord & !is.na(ann_df$class)][1]
    p <- p + geom_cladelab(
      node     = mrca_node,
      label    = ord,
      angle    = "auto",
      fontsize = 1.8,
      offset   = 0.5,
      offset.text = 0.3,
      color    = CLASS_COLORS[cls_of_ord],
      barsize  = 0.4,
      hjust    = 0.5
    )
  }
}

# Theme and legend
p <- p +
  theme_tree2() +
  theme(
    legend.position  = c(0.08, 0.15),
    legend.text      = element_text(size = 8),
    legend.title     = element_text(size = 9, face = "bold"),
    legend.background= element_rect(fill = "white", color = NA),
    plot.title       = element_text(size = 11, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 8,  hjust = 0.5, color = "grey40"),
  ) +
  labs(
    title    = "Tetrapod mitochondrial tRNA phylogeny",
    subtitle = paste0(
      length(tree_rooted$tip.label), " species | ",
      "cladogram, no branch lengths | ",
      "Open Tree of Life topology"
    )
  )

# ── Save ───────────────────────────────────────────────────────────────────────
# PNG (primary output)
png(OUT_PNG, width = 4800, height = 4800, res = 300)
print(p)
dev.off()
cat("Saved:", OUT_PNG, "\n")

# SVG
svg(OUT_SVG, width = 16, height = 16)
print(p)
dev.off()
cat("Saved:", OUT_SVG, "\n")

cat("\nDone.\n")
