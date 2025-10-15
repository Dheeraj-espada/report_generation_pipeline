#!/usr/bin/env python3
"""
create_health_weights.py

Derive evidence-weighted genus scores from a downloaded SalivaDB CSV.
This script is robust and handles two common SalivaDB formats:
1. Long-form with columns: 'Biomarker Name', 'Regulation'
2. Wide-form with columns: 'Genus', 'Up', 'Down'

Output: salivadb_genus_weights.csv with a 'HealthWeight' column in [-1, 1].
"""
import pandas as pd
import numpy as np
import os
import re
import argparse
import json
from datetime import datetime

# Define higher-level taxa and invalid names to exclude from final weights
EXCLUDE_TAXA = {
    "Bacteria", "Eubacteria", "Proteobacteria", "Firmicutes", "Actinobacteria",
    "Bacteroidetes", "Fusobacteria", "Spirochaetes", "Cyanobacteria", "Tenericutes",
    "Bacteroidia", "Flavobacteria", "Betaproteobacteria", "Gammaproteobacteria",
    "Deltaproteobacteria", "Flavobacteriales", "Spirochaetales", "Bacteroidales",
    "Clostridiales", "Unknown", "Unclassified", ""
}

def clean_genus(name: str) -> str:
    """Extract a clean genus name from a biomarker string."""
    s = str(name).replace("Candidatus ", "").strip()
    s = re.sub(r"\[|\]|[a-z]__", "", s)
    tokens = s.split()
    
    if not tokens:
        return "Unknown"
    
    # Exclude ambiguous single-letter abbreviations like "T." for Treponema/Tannerella
    if re.match(r"^[A-Z]\.$", tokens[0]):
        return "Unknown"
    
    # Find the first capitalized token, which is usually the genus
    for t in tokens:
        if re.match(r"^[A-Z][a-z]+$", t):
            return t
    
    return "Unknown"

def build_from_long_form(df: pd.DataFrame, min_evidence: int) -> pd.DataFrame:
    """Build weights from a CSV with 'Biomarker Name' and 'Regulation' columns."""
    required = {"Biomarker Name", "Regulation"}
    if not required.issubset(df.columns):
        return pd.DataFrame()  # Return empty if schema doesn't match

    x = df[list(required)].dropna()
    x = x[x["Regulation"].isin(["Upregulated", "Downregulated"])].copy()
    x["Genus"] = x["Biomarker Name"].apply(clean_genus)
    
    counts = x.groupby(["Genus", "Regulation"]).size().unstack(fill_value=0)
    if "Upregulated" not in counts:
        counts["Upregulated"] = 0
    if "Downregulated" not in counts:
        counts["Downregulated"] = 0
    
    counts.rename(
        columns={"Upregulated": "N_disease", "Downregulated": "N_healthy"},
        inplace=True
    )
    return calculate_weights(counts, min_evidence)

def build_from_wide_form(df: pd.DataFrame, min_evidence: int) -> pd.DataFrame:
    """Build weights from a CSV with 'Genus', 'Up', and 'Down' columns."""
    cols = {c.lower(): c for c in df.columns}
    g_col = cols.get('genus')
    up_col = cols.get('up')
    down_col = cols.get('down')
    
    if not all([g_col, up_col, down_col]):
        return pd.DataFrame()  # Return empty if schema doesn't match
        
    df = df.rename(columns={
        g_col: 'Genus',
        up_col: 'N_disease',
        down_col: 'N_healthy'
    })
    df['Genus'] = df['Genus'].apply(clean_genus)
    df = df.groupby('Genus')[['N_healthy', 'N_disease']].sum()
    return calculate_weights(df, min_evidence)

def calculate_weights(counts: pd.DataFrame, min_evidence: int) -> pd.DataFrame:
    """Calculate HealthWeight from N_healthy and N_disease counts."""
    counts = counts[~counts.index.isin(EXCLUDE_TAXA)]
    counts = counts[~counts.index.str.match(r"^[A-Z]$")]  # Exclude single-letter genus names
    
    total_evidence = counts["N_healthy"] + counts["N_disease"]
    confident_genera = counts[total_evidence >= min_evidence].copy()
    
    if confident_genera.empty:
        return pd.DataFrame()
    
    # HealthWeight = (Healthy Mentions - Disease Mentions) / Total Mentions
    # Ranges from -1 (always disease) to +1 (always healthy)
    confident_genera["HealthWeight"] = (
        (confident_genera["N_healthy"] - confident_genera["N_disease"]) /
        (confident_genera["N_healthy"] + confident_genera["N_disease"])
    ).fillna(0.0)
    
    return confident_genera[["HealthWeight"]]

def print_summary(weights: pd.DataFrame):
    """Print a summary of the generated weights."""
    beneficial = weights[weights['HealthWeight'] > 0].sort_values(
        'HealthWeight', ascending=False
    )
    pathogenic = weights[weights['HealthWeight'] < 0].sort_values(
        'HealthWeight', ascending=True
    )
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(weights)} genera with evidence-based health weights")
    print(f"{'='*70}")
    
    print(f"\n🟢 Top 10 Most Beneficial Genera:")
    print(beneficial.head(10).to_string())
    
    print(f"\n🔴 Top 10 Most Pathogenic Genera:")
    print(pathogenic.head(10).to_string())
    
    print(f"\n📊 Statistics:")
    print(f"  Mean HealthWeight: {weights['HealthWeight'].mean():.3f}")
    print(f"  Std Dev: {weights['HealthWeight'].std():.3f}")
    print(f"  Beneficial genera: {len(beneficial)}")
    print(f"  Pathogenic genera: {len(pathogenic)}")
    print(f"  Neutral genera: {len(weights[weights['HealthWeight'] == 0])}")

def main():
    parser = argparse.ArgumentParser(
        description="Create evidence-based genus health weights from a SalivaDB CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to SalivaDB CSV file"
    )
    parser.add_argument(
        "--out",
        default="salivadb_genus_weights.csv",
        help="Output weights CSV file"
    )
    parser.add_argument(
        "--min_evidence",
        type=int,
        default=3,
        help="Minimum total mentions (Up+Down) required for inclusion"
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input file not found: {args.csv}")

    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Attempt to build weights using both schema strategies
    print("Attempting long-form schema (Biomarker Name, Regulation)...")
    weights = build_from_long_form(df, args.min_evidence)
    
    if weights.empty:
        print("Long-form failed. Attempting wide-form schema (Genus, Up, Down)...")
        weights = build_from_wide_form(df, args.min_evidence)
    
    if weights.empty:
        raise SystemExit(
            "\n❌ ERROR: No valid schema detected or no genera met the evidence threshold.\n"
            "Expected columns: ('Biomarker Name', 'Regulation') OR ('Genus', 'Up', 'Down').\n"
            f"Minimum evidence threshold: {args.min_evidence}"
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save weights
    weights.sort_index().to_csv(args.out)

    # Create metadata file for provenance
    meta = {
        "source_csv": os.path.basename(args.csv),
        "generated_at": datetime.now().isoformat(),
        "min_evidence": args.min_evidence,
        "n_genera": int(len(weights)),
        "schema_version": "2.0"
    }
    meta_path = args.out.replace(".csv", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Wrote {args.out} (+ .meta.json) with {len(weights)} genera.")
    
    # Print summary
    print_summary(weights)

if __name__ == "__main__":
    main()
