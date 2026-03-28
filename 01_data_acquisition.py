#!/usr/bin/env python3
"""
01_data_acquisition.py
======================
Tetrapod mitochondrial tRNA dataset construction.

Steps:
  1. Define taxonomy metadata for 70 orders across 4 classes
  2. Query NCBI for mitochondrial genome accessions (≤50 per order)
  3. Batch-download GenBank records
  4. Extract tRNA features (length filter: 55–95 nt)
  5. Species-level deduplication (1 sequence per species per tRNA type)
  6. Remove species with <10 tRNA types (incomplete mitogenomes)
  7. Save: trna_final.csv, species_metadata.csv, order_accessions.json

Output files (written to OUTDIR):
  trna_final.csv          — 19,350 tRNAs × 9 columns
  species_metadata.csv    — 971 species × 8 columns
  order_accessions.json   — raw accession lists per order

Dependencies: biopython, pandas
"""

import os
import re
import json
import time
import pickle
from collections import defaultdict

import pandas as pd
from Bio import Entrez, SeqIO

# ── Configuration ─────────────────────────────────────────────────────────────
Entrez.email = "your.email@institution.edu"   # <-- replace
Entrez.tool  = "TetrapodMtTRNA"
OUTDIR       = "/workspace/data_v2"
MAX_PER_ORDER = 50
BATCH_SIZE    = 50

os.makedirs(OUTDIR, exist_ok=True)

# ── Step 1: Order taxonomy metadata ──────────────────────────────────────────
# Format: order_name -> {class, thermo, body_temp (°C), taxid}
TETRAPOD_ORDERS = {
    # Amphibia — ectotherm
    "Anura":          {"class": "Amphibia", "thermo": "ectotherm", "body_temp": 20, "taxid": "8342"},
    "Caudata":        {"class": "Amphibia", "thermo": "ectotherm", "body_temp": 18, "taxid": "8293"},
    "Gymnophiona":    {"class": "Amphibia", "thermo": "ectotherm", "body_temp": 22, "taxid": "8445"},
    # Reptilia — ectotherm
    "Squamata":       {"class": "Reptilia", "thermo": "ectotherm", "body_temp": 30, "taxid": "8509"},
    "Testudines":     {"class": "Reptilia", "thermo": "ectotherm", "body_temp": 28, "taxid": "8459"},
    "Crocodilia":     {"class": "Reptilia", "thermo": "ectotherm", "body_temp": 30, "taxid": "8493"},
    "Rhynchocephalia":{"class": "Reptilia", "thermo": "ectotherm", "body_temp": 22, "taxid": "8508"},
    # Aves — endotherm (body_temp 38–41 °C; order-specific values below)
    "Passeriformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "9126"},
    "Psittaciformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "9223"},
    "Accipitriformes":{"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "2558684"},
    "Anseriformes":   {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8826"},
    "Galliformes":    {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8976"},
    "Columbiformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8929"},
    "Strigiformes":   {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "30458"},
    "Piciformes":     {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "9219"},
    "Coraciiformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "30459"},
    "Cuculiformes":   {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8940"},
    "Falconiformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8948"},
    "Charadriiformes":{"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8906"},
    "Gruiformes":     {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8916"},
    "Ciconiiformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8920"},
    "Pelecaniformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8908"},
    "Suliformes":     {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "693979"},
    "Sphenisciformes":{"class": "Aves", "thermo": "endotherm", "body_temp": 38, "taxid": "9230"},
    "Procellariiformes":{"class":"Aves","thermo": "endotherm", "body_temp": 38, "taxid": "8955"},
    "Gaviiformes":    {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "30459"},
    "Podicipediformes":{"class":"Aves","thermo": "endotherm", "body_temp": 41, "taxid": "30461"},
    "Phoenicopteriformes":{"class":"Aves","thermo":"endotherm","body_temp": 41, "taxid": "9214"},
    "Opisthocomiformes":{"class":"Aves","thermo":"endotherm", "body_temp": 41, "taxid": "30464"},
    "Musophagiformes":{"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "56313"},
    "Bucerotiformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "57379"},
    "Trogoniformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "57381"},
    "Leptosomiformes":{"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "1549675"},
    "Caprimulgiformes":{"class":"Aves","thermo": "endotherm", "body_temp": 41, "taxid": "8936"},
    "Apodiformes":    {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8930"},
    "Otidiformes":    {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8931"},
    "Mesitornithiformes":{"class":"Aves","thermo":"endotherm","body_temp": 41, "taxid": "8927"},
    "Pterocliformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "8932"},
    "Cariamiformes":  {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "57386"},
    "Cathartiformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "9376"},
    "Eurypygiformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 41, "taxid": "1549676"},
    "Rheiformes":     {"class": "Aves", "thermo": "endotherm", "body_temp": 38, "taxid": "8798"},
    "Struthioniformes":{"class":"Aves","thermo": "endotherm", "body_temp": 38, "taxid": "8798"},
    "Casuariiformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 38, "taxid": "8798"},
    "Apterygiformes": {"class": "Aves", "thermo": "endotherm", "body_temp": 38, "taxid": "8798"},
    "Tinamiformes":   {"class": "Aves", "thermo": "endotherm", "body_temp": 38, "taxid": "8802"},
    # Mammalia — endotherm
    "Primates":       {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9443"},
    "Rodentia":       {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9989"},
    "Carnivora":      {"class": "Mammalia", "thermo": "endotherm", "body_temp": 38, "taxid": "33554"},
    "Chiroptera":     {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9397"},
    "Artiodactyla":   {"class": "Mammalia", "thermo": "endotherm", "body_temp": 38, "taxid": "91561"},
    "Perissodactyla": {"class": "Mammalia", "thermo": "endotherm", "body_temp": 38, "taxid": "9787"},
    "Cetacea":        {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9721"},
    "Lagomorpha":     {"class": "Mammalia", "thermo": "endotherm", "body_temp": 38, "taxid": "9975"},
    "Insectivora":    {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "9362"},
    "Erinaceomorpha": {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "9362"},
    "Soricomorpha":   {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "9362"},
    "Afrosoricida":   {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "311790"},
    "Macroscelidea":  {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9362"},
    "Proboscidea":    {"class": "Mammalia", "thermo": "endotherm", "body_temp": 36, "taxid": "9780"},
    "Sirenia":        {"class": "Mammalia", "thermo": "endotherm", "body_temp": 36, "taxid": "9778"},
    "Hyracoidea":     {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9779"},
    "Tubulidentata":  {"class": "Mammalia", "thermo": "endotherm", "body_temp": 36, "taxid": "9818"},
    "Pholidota":      {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "9973"},
    "Pilosa":         {"class": "Mammalia", "thermo": "endotherm", "body_temp": 32, "taxid": "9357"},
    "Cingulata":      {"class": "Mammalia", "thermo": "endotherm", "body_temp": 32, "taxid": "9361"},
    "Didelphimorphia":{"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "9265"},
    "Diprotodontia":  {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "38558"},
    "Monotremata":    {"class": "Mammalia", "thermo": "endotherm", "body_temp": 32, "taxid": "9255"},
    "Scandentia":     {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9372"},
    "Dermoptera":     {"class": "Mammalia", "thermo": "endotherm", "body_temp": 37, "taxid": "9376"},
    "Eulipotyphla":   {"class": "Mammalia", "thermo": "endotherm", "body_temp": 35, "taxid": "9362"},
}

# ── Step 2: NCBI accession search ─────────────────────────────────────────────
TRNA_RE = re.compile(r'tRNA-([A-Z][a-z]{2})', re.IGNORECASE)

def search_ncbi_mitogenomes(taxid, max_records=MAX_PER_ORDER, retries=3):
    """Search NCBI for complete mitochondrial genomes for a given taxid."""
    query = (f"txid{taxid}[Organism:exp] AND mitochondrion[filter] "
             f"AND complete genome[title] AND 10000:25000[SLEN]")
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="nucleotide", term=query,
                                    retmax=max_records, usehistory="y")
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  Search failed for taxid {taxid}: {e}")
                return []

def fetch_accessions_for_all_orders(orders_dict, max_per_order=MAX_PER_ORDER):
    """Query NCBI for all orders; return {order: [gi_id, ...]}."""
    accession_map = {}
    print(f"Searching NCBI for {len(orders_dict)} orders...")
    for order, meta in orders_dict.items():
        ids = search_ncbi_mitogenomes(meta["taxid"], max_per_order)
        accession_map[order] = ids
        print(f"  {order}: {len(ids)} records")
        time.sleep(0.4)
    return accession_map

# ── Step 3: Batch GenBank download ────────────────────────────────────────────
def extract_trnas_from_record(record, order):
    """Extract tRNA features from a SeqRecord; return list of dicts."""
    trnas = []
    species = record.annotations.get("organism", "Unknown")
    accession = record.id
    meta = TETRAPOD_ORDERS[order]

    for feat in record.features:
        if feat.type != "tRNA":
            continue
        # Parse product name
        product = ""
        for key in ("product", "note"):
            for val in feat.qualifiers.get(key, []):
                m = TRNA_RE.search(val)
                if m:
                    product = f"tRNA-{m.group(1).capitalize()}"
                    break
            if product:
                break
        if not product:
            continue
        # Extract sequence
        try:
            seq = str(feat.extract(record.seq)).upper()
        except Exception:
            continue
        # Length filter: canonical mt-tRNA range
        if not (55 <= len(seq) <= 95):
            continue
        # Ambiguous bases filter
        if re.search(r'[^ACGT]', seq):
            continue
        trnas.append({
            "accession": accession,
            "order":     order,
            "class":     meta["class"],
            "thermo":    meta["thermo"],
            "body_temp": meta["body_temp"],
            "species":   species,
            "trna_type": product,
            "sequence":  seq,
            "length":    len(seq),
        })
    return trnas

def batch_download_and_extract(order_accessions, batch_size=BATCH_SIZE,
                                checkpoint_path=None):
    """
    Download GenBank records in batches and extract tRNA features.
    Supports checkpointing for resumable downloads.
    """
    # Build flat list: (order, gi_id)
    flat_list = [(order, uid)
                 for order, uids in order_accessions.items()
                 for uid in uids]
    print(f"Total records to fetch: {len(flat_list):,}")

    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        all_trnas    = ckpt["all_trnas"]
        done_indices = ckpt["done_indices"]
        print(f"Resuming from checkpoint: {len(done_indices):,} done, "
              f"{len(all_trnas):,} tRNAs collected")
    else:
        all_trnas    = []
        done_indices = set()

    remaining = [(i, order, uid) for i, (order, uid) in enumerate(flat_list)
                 if i not in done_indices]
    print(f"Remaining: {len(remaining):,}")

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start: batch_start + batch_size]
        uids  = [uid for _, _, uid in batch]

        records = None
        for attempt in range(3):
            try:
                handle  = Entrez.efetch(db="nucleotide", id=",".join(uids),
                                        rettype="gb", retmode="text")
                records = list(SeqIO.parse(handle, "genbank"))
                handle.close()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Batch {batch_start} failed: {e}")

        if records is None:
            continue

        # Match records to orders by position
        for (idx, order, uid), rec in zip(batch, records):
            trnas = extract_trnas_from_record(rec, order)
            all_trnas.extend(trnas)
            done_indices.add(idx)

        time.sleep(0.4)

        pct = len(done_indices) / len(flat_list) * 100
        if (batch_start // batch_size) % 5 == 0:
            print(f"  [{pct:5.1f}%] done={len(done_indices):,}  "
                  f"tRNAs={len(all_trnas):,}")

        # Save checkpoint every 10 batches
        if checkpoint_path and (batch_start // batch_size) % 10 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"all_trnas": all_trnas,
                             "done_indices": done_indices}, f)

    print(f"\nDownload complete: {len(done_indices):,} accessions, "
          f"{len(all_trnas):,} raw tRNAs")
    return all_trnas

# ── Step 4–6: QC and deduplication ───────────────────────────────────────────
VALID_TRNA_TYPES = {
    "tRNA-Ala", "tRNA-Arg", "tRNA-Asn", "tRNA-Asp", "tRNA-Cys",
    "tRNA-Gln", "tRNA-Glu", "tRNA-Gly", "tRNA-His", "tRNA-Ile",
    "tRNA-Leu", "tRNA-Lys", "tRNA-Met", "tRNA-Phe", "tRNA-Pro",
    "tRNA-Ser", "tRNA-Thr", "tRNA-Trp", "tRNA-Tyr", "tRNA-Val",
}

def clean_species_name(s):
    """Retain only genus + species epithet (first two words)."""
    parts = s.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else s

def qc_and_deduplicate(raw_trnas):
    """
    QC pipeline:
      1. Remove non-standard tRNA types
      2. Standardize species names
      3. Deduplicate: keep longest sequence per species per tRNA type
      4. Remove species with <10 tRNA types (incomplete mitogenomes)
    """
    df = pd.DataFrame(raw_trnas)
    print(f"Starting QC: {len(df):,} raw records")

    # 1. Valid tRNA types only
    df = df[df["trna_type"].isin(VALID_TRNA_TYPES)].copy()
    print(f"After type filter: {len(df):,}")

    # 2. Clean species names
    df["species_clean"] = df["species"].apply(clean_species_name)

    # 3. Deduplicate: one sequence per species per tRNA type (keep longest)
    df = (df.sort_values("length", ascending=False)
            .drop_duplicates(subset=["species_clean", "trna_type"])
            .reset_index(drop=True))
    print(f"After deduplication: {len(df):,} (1 per species per type)")

    # 4. Remove species with <10 tRNA types
    type_counts = df.groupby("species_clean")["trna_type"].nunique()
    good_species = type_counts[type_counts >= 10].index
    df = df[df["species_clean"].isin(good_species)].copy()
    print(f"After removing incomplete species: {len(df):,}")

    return df

# ── Step 7: Save outputs ──────────────────────────────────────────────────────
def build_species_metadata(df):
    """Aggregate per-species metadata from the deduplicated tRNA table."""
    meta = (df.groupby("species_clean")
              .agg(
                  n_trna_types=("trna_type", "nunique"),
                  n_trnas=("trna_type", "count"),
                  order=("order", "first"),
                  cls=("class", "first"),
                  thermo=("thermo", "first"),
                  body_temp=("body_temp", "first"),
                  accession=("accession", "first"),
              )
              .reset_index()
              .rename(columns={"species_clean": "species", "cls": "class"}))
    return meta

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    acc_json = os.path.join(OUTDIR, "order_accessions.json")
    ckpt_pkl = os.path.join(OUTDIR, "trna_records_checkpoint.pkl")

    # Step 2: Search NCBI (skip if already done)
    if not os.path.exists(acc_json):
        order_accessions = fetch_accessions_for_all_orders(TETRAPOD_ORDERS)
        with open(acc_json, "w") as f:
            json.dump(order_accessions, f, indent=2)
        print(f"Saved accession map: {acc_json}")
    else:
        with open(acc_json) as f:
            order_accessions = json.load(f)
        print(f"Loaded existing accession map: {sum(len(v) for v in order_accessions.values())} records")

    # Step 3: Download and extract
    raw_trnas = batch_download_and_extract(
        order_accessions, checkpoint_path=ckpt_pkl
    )

    # Steps 4–6: QC and deduplication
    df_final = qc_and_deduplicate(raw_trnas)

    # Step 7: Save
    trna_out = os.path.join(OUTDIR, "trna_final.csv")
    meta_out = os.path.join(OUTDIR, "species_metadata.csv")

    df_final.to_csv(trna_out, index=False)
    build_species_metadata(df_final).to_csv(meta_out, index=False)

    print(f"\n{'='*55}")
    print(f"FINAL DATASET")
    print(f"{'='*55}")
    print(f"tRNAs    : {len(df_final):,}")
    print(f"Species  : {df_final['species_clean'].nunique():,}")
    print(f"Orders   : {df_final['order'].nunique()}")
    print(f"tRNA types: {df_final['trna_type'].nunique()}")
    print(f"\nSaved: {trna_out}")
    print(f"Saved: {meta_out}")
