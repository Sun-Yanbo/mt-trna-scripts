#!/usr/bin/env python3
"""
05_phylogeny.py
===============
Construct a species-level phylogenetic tree for all 971 tetrapod species
using the Open Tree of Life (OTL) API.

Steps:
  1. Load species list from species_metadata.csv
  2. Query NCBI Taxonomy API to obtain family-level classification
  3. Query OTL TNRS (Taxonomic Name Resolution Service) to get OTT IDs
  4. Request OTL induced_subtree to extract the phylogenetic topology
  5. Parse and label tree tips with species names
  6. Save: tetrapod_otl.nwk (Newick), tree_annotations.csv

The resulting tree is a cladogram (topology only, no branch lengths).
Visualization is handled by 05_phylogeny.R (ggtree).

Output files (written to WORKSPACE):
  /workspace/tetrapod_otl_dendropy.nwk   — labelled Newick tree
  /workspace/tree_annotations.csv        — tip metadata (class, order, thermo)
  /workspace/species_ott_ids.csv         — OTT ID mapping for all species
  /workspace/species_taxonomy_full.csv   — NCBI family-level taxonomy

Dependencies: biopython, pandas, dendropy, requests, urllib
"""

import os
import re
import time
import json
import urllib.request
import urllib.error
import urllib.parse
import pandas as pd
import dendropy
from Bio import Entrez

# ── Configuration ─────────────────────────────────────────────────────────────
Entrez.email = "your.email@institution.edu"   # <-- replace
INDIR     = "/workspace/data_v2"
WORKSPACE = "/workspace"

# ── Step 1: Load species list ─────────────────────────────────────────────────
def load_species_list(path):
    meta = pd.read_csv(path)
    print(f"Species to process: {len(meta)}")
    return meta

# ── Step 2: NCBI Taxonomy — get family for each species ──────────────────────
def get_ncbi_family(species_name, retries=3):
    """Query NCBI Taxonomy for the family of a species."""
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="taxonomy", term=f'"{species_name}"[Scientific Name]')
            record = Entrez.read(handle)
            handle.close()
            if not record["IdList"]:
                return None
            taxid = record["IdList"][0]
            handle2 = Entrez.efetch(db="taxonomy", id=taxid, rettype="xml")
            tax_record = Entrez.read(handle2)
            handle2.close()
            if not tax_record:
                return None
            lineage = tax_record[0].get("LineageEx", [])
            for node in lineage:
                if node.get("Rank") == "family":
                    return node.get("ScientificName")
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None

def fetch_all_families(species_list, checkpoint_path=None):
    """Fetch family for all species; supports checkpointing."""
    results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        df_ckpt = pd.read_csv(checkpoint_path)
        results = dict(zip(df_ckpt["species"], df_ckpt["family"]))
        print(f"Loaded {len(results)} cached family assignments")

    remaining = [s for s in species_list if s not in results]
    print(f"Querying NCBI Taxonomy for {len(remaining)} species...")

    for i, sp in enumerate(remaining):
        family = get_ncbi_family(sp)
        results[sp] = family
        time.sleep(0.35)   # NCBI rate limit: ≤3 requests/sec
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(remaining)} done")
            if checkpoint_path:
                pd.DataFrame(list(results.items()),
                             columns=["species", "family"]).to_csv(
                    checkpoint_path, index=False)

    return results

# ── Step 3: OTL TNRS — get OTT IDs ──────────────────────────────────────────
OTL_TNRS_URL = "https://api.opentreeoflife.org/v3/tnrs/match_names"

def query_otl_tnrs(names, batch_size=250):
    """
    Query OTL TNRS to resolve species names to OTT IDs.
    Returns dict: {species_name: ott_id}
    """
    ott_map = {}
    for start in range(0, len(names), batch_size):
        batch = names[start: start + batch_size]
        payload = json.dumps({
            "names": batch,
            "do_approximate_matching": False,
            "include_suppressed": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            OTL_TNRS_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            for result in data.get("results", []):
                name = result.get("name", "")
                matches = result.get("matches", [])
                if matches:
                    best = matches[0]
                    ott_id = best.get("taxon", {}).get("ott_id")
                    ott_map[name] = ott_id
        except Exception as e:
            print(f"  TNRS batch {start} failed: {e}")
        time.sleep(0.5)
        print(f"  TNRS: {min(start+batch_size, len(names))}/{len(names)} processed")

    return ott_map

# ── Step 4: OTL induced_subtree ───────────────────────────────────────────────
OTL_SUBTREE_URL = "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree"

def get_otl_subtree(ott_ids, max_retries=5):
    """
    Request OTL induced_subtree for a list of OTT IDs.
    Iteratively removes pruned/invalid IDs until the request succeeds.
    Returns raw Newick string.
    """
    ids_to_try = list(set(ott_ids))
    pruned_ids = set()

    for attempt in range(max_retries):
        payload = json.dumps({
            "ott_ids": ids_to_try,
            "label_format": "id",
        }).encode("utf-8")
        req = urllib.request.Request(
            OTL_SUBTREE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            newick = data.get("newick", "")
            print(f"  Subtree obtained: {len(ids_to_try)} IDs, "
                  f"newick length={len(newick)}")
            return newick, pruned_ids
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            # Parse pruned IDs from error message
            pruned = re.findall(r'\b(\d{5,})\b', body)
            if pruned:
                new_pruned = set(int(p) for p in pruned)
                pruned_ids |= new_pruned
                ids_to_try = [i for i in ids_to_try if i not in pruned_ids]
                print(f"  Attempt {attempt+1}: removed {len(new_pruned)} pruned IDs, "
                      f"{len(ids_to_try)} remaining")
            else:
                print(f"  HTTP error: {e.code} — {body[:200]}")
                break
        time.sleep(2)

    return None, pruned_ids

# ── Step 5: Parse and label tree ─────────────────────────────────────────────
def label_tree_tips(newick_raw, ott_to_species):
    """
    Replace OTT ID labels (e.g. 'ott12345') with species names.
    Returns labelled Newick string.
    """
    def replace_ott(match):
        ott_id = int(match.group(1))
        sp = ott_to_species.get(ott_id, f"ott{ott_id}")
        # Sanitize: replace spaces with underscores
        return sp.replace(" ", "_")

    labelled = re.sub(r"ott(\d+)", replace_ott, newick_raw)
    return labelled

def parse_and_save_tree(newick_labelled, out_path):
    """Parse with dendropy and save as clean Newick."""
    tree = dendropy.Tree.get(data=newick_labelled, schema="newick")
    # Remove branch lengths (cladogram)
    for edge in tree.preorder_edge_iter():
        edge.length = None
    tree.write(path=out_path, schema="newick")
    n_tips = len(tree.leaf_node_iter())
    print(f"Tree saved: {n_tips} tips → {out_path}")
    return tree

# ── Step 6: Build annotation table ───────────────────────────────────────────
def build_tree_annotations(tree, sp_meta):
    """Build tip annotation table for ggtree visualization."""
    tip_labels = [leaf.taxon.label.replace("_", " ")
                  for leaf in tree.leaf_node_iter()]
    ann = pd.DataFrame({"tip_label": tip_labels})
    ann["tip_label_underscore"] = ann["tip_label"].str.replace(" ", "_")

    # Merge with species metadata
    sp_meta_clean = sp_meta.copy()
    sp_meta_clean["species_underscore"] = sp_meta_clean["species"].str.replace(" ", "_")
    ann = ann.merge(
        sp_meta_clean[["species_underscore", "class", "order", "thermo", "body_temp"]],
        left_on="tip_label_underscore",
        right_on="species_underscore",
        how="left",
    )
    return ann

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load species list
    sp_meta = load_species_list(os.path.join(INDIR, "species_metadata.csv"))
    species_list = sp_meta["species"].tolist()

    # Step 2: NCBI family lookup
    tax_path = os.path.join(WORKSPACE, "species_taxonomy_full.csv")
    family_map = fetch_all_families(species_list, checkpoint_path=tax_path)
    df_tax = pd.DataFrame(list(family_map.items()), columns=["species", "family"])
    df_tax.to_csv(tax_path, index=False)
    print(f"Taxonomy saved: {tax_path}")

    # Step 3: OTL TNRS
    ott_path = os.path.join(WORKSPACE, "species_ott_ids.csv")
    if os.path.exists(ott_path):
        df_ott = pd.read_csv(ott_path)
        print(f"Loaded existing OTT IDs: {df_ott['ott_id'].notna().sum()}/{len(df_ott)}")
    else:
        ott_map = query_otl_tnrs(species_list)
        df_ott = pd.DataFrame([
            {"species": sp, "ott_id": ott_map.get(sp)}
            for sp in species_list
        ])
        df_ott = df_ott.merge(sp_meta[["species", "class", "order", "thermo"]],
                              on="species", how="left")
        df_ott.to_csv(ott_path, index=False)
        print(f"OTT IDs saved: {ott_path}")

    # Step 4: OTL induced_subtree
    valid_ott = df_ott.dropna(subset=["ott_id"])
    ott_ids   = valid_ott["ott_id"].astype(int).unique().tolist()
    print(f"\nRequesting OTL subtree for {len(ott_ids)} OTT IDs...")
    newick_raw, pruned = get_otl_subtree(ott_ids)

    if newick_raw is None:
        raise RuntimeError("Failed to obtain OTL subtree")

    # Save raw newick
    raw_nwk = os.path.join(WORKSPACE, "tetrapod_otl_raw.nwk")
    with open(raw_nwk, "w") as f:
        f.write(newick_raw)

    # Step 5: Label tips
    ott_to_species = dict(zip(
        valid_ott["ott_id"].astype(int),
        valid_ott["species"]
    ))
    newick_labelled = label_tree_tips(newick_raw, ott_to_species)

    # Parse and save
    nwk_out = os.path.join(WORKSPACE, "tetrapod_otl_dendropy.nwk")
    tree    = parse_and_save_tree(newick_labelled, nwk_out)

    # Step 6: Annotations
    ann = build_tree_annotations(tree, sp_meta)
    ann_out = os.path.join(WORKSPACE, "tree_annotations.csv")
    ann.to_csv(ann_out, index=False)
    print(f"Annotations saved: {ann_out}")

    # Coverage summary
    n_in_tree = ann["class"].notna().sum()
    print(f"\n{'='*50}")
    print(f"COVERAGE SUMMARY")
    print(f"{'='*50}")
    print(f"Total species in dataset : {len(species_list)}")
    print(f"Species placed in tree   : {n_in_tree} ({n_in_tree/len(species_list)*100:.1f}%)")
    print(f"Missing from tree        : {len(species_list) - n_in_tree}")
    print(f"\nClass distribution in tree:")
    print(ann.groupby("class")["tip_label"].count().to_string())
