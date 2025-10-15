#!/usr/bin/env python3
"""
prepare_pipeline_data.py

Builds cohort-derived references used by the scorer:

1) reference_db/reference_rel.tsv — genus-level relative abundances
2) reference_db/genus_percentiles.json (p50/p90/p97.5 per genus)
3) reference_db/reference_features.tsv (derived metrics per sample)
4) reference_db/reference_stats.json (median/MAD for features)

Inputs (CLI overrides):
- feature-table.tsv (BIOM TSV, skip first comment row)
- asv_to_genus_map.json (ASV → genus)
- salivadb_genus_weights.csv (genus weights)
- healthy_ids.txt (optional; newline sample IDs)

Usage:
  python prepare_pipeline_data.py \
    --feature_table feature-table.tsv \
    --asv_map asv_to_genus_map.json \
    --weights_csv salivadb_genus_weights.csv \
    --scfa_genera Veillonella Lachnospira \
    --inflam_genera Porphyromonas Treponema \
    --min_depth 5000 \
    --min_presence 2 \
    --out_rel reference_db/reference_rel.tsv \
    --out_percentiles reference_db/genus_percentiles.json \
    --out_features reference_db/reference_features.tsv \
    --out_stats reference_db/reference_stats.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, entropy
from datetime import datetime

def read_feature_table(path: str) -> pd.DataFrame:
    """Read BIOM-format TSV file, skipping the first comment row."""
    print(f"Reading feature table from {path}...")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    print(f"  Loaded {df.shape[0]} ASVs across {df.shape[1]} samples")
    return df

def map_to_genus(rel: pd.DataFrame, asv_map_path: str) -> pd.DataFrame:
    """Map ASV-level abundances to genus level."""
    print(f"Mapping ASVs to genus using {asv_map_path}...")
    mapping = json.load(open(asv_map_path))
    rel = rel.copy()
    rel['genus'] = rel.index.map(lambda x: mapping.get(x, 'unclassified_genus'))
    
    # Aggregate by genus
    genus_rel = rel.groupby('genus').sum()
    print(f"  Aggregated to {genus_rel.shape[0]} genera")
    return genus_rel

def filter_samples(
    counts: pd.DataFrame,
    min_depth: int = 5000,
    min_presence: int = 2
) -> pd.DataFrame:
    """Filter samples by sequencing depth and feature presence."""
    print(f"\nFiltering samples...")
    initial_n = counts.shape[1]
    
    # Filter by depth
    total_counts = counts.sum(axis=0)
    counts = counts.loc[:, total_counts >= min_depth]
    print(f"  After depth filter (>={min_depth}): {counts.shape[1]} samples")
    
    # Filter by presence (ASV must be in at least N samples)
    presence = (counts > 0).sum(axis=1)
    counts = counts.loc[presence >= min_presence, :]
    print(f"  After presence filter (>={min_presence} samples): {counts.shape[0]} ASVs")
    print(f"  Removed {initial_n - counts.shape[1]} samples total")
    
    return counts

def compute_percentiles(rel: pd.DataFrame) -> dict:
    """Compute percentiles (p50, p90, p97.5) for each genus across samples."""
    print("\nComputing genus percentiles...")
    out = {}
    for genus, row in rel.iterrows():
        vals = row.values
        p50, p90, p975 = np.nanpercentile(vals, [50, 90, 97.5])
        out[genus] = {
            'p50': float(p50),
            'p90': float(p90),
            'p97_5': float(p975)
        }
    print(f"  Computed percentiles for {len(out)} genera")
    return out

def compute_reference_stats(feats: pd.DataFrame) -> dict:
    """Compute robust reference statistics (median, MAD, mean, std) for features."""
    print("\nComputing reference statistics...")
    stats = {}
    for feat in feats.columns:
        vals = feats[feat].dropna().values
        med = float(np.median(vals))
        mad = float(median_abs_deviation(vals, scale='normal')) if len(vals) > 1 else 0.0
        mean = float(np.mean(vals)) if len(vals) > 0 else 0.0
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        stats[feat] = {
            'median': med,
            'mad': mad,
            'mean': mean,
            'std': std
        }
    print(f"  Computed stats for {len(stats)} features")
    return stats

def compute_features(
    rel: pd.DataFrame,
    weights: pd.Series,
    scfa_g: set,
    inflam_g: set
) -> pd.DataFrame:
    """Extract key microbiome features for each sample."""
    print("\nExtracting microbiome features...")
    rows = []
    
    beneficial_genera = set(weights[weights > 0].index)
    pathogenic_genera = set(weights[weights < 0].index)
    
    for sample in rel.columns:
        p = rel[sample]
        
        # Shannon diversity
        shannon = float(entropy(p[p > 0]))
        
        # Beneficial and pathogenic sums
        beneficial = float(p[p.index.isin(beneficial_genera)].sum())
        pathogen = float(p[p.index.isin(pathogenic_genera)].sum())
        
        # SCFA producers and inflammatory genera
        scfa = float(p[p.index.isin(scfa_g)].sum())
        inflammation = float(p[p.index.isin(inflam_g)].sum())
        
        # Log ratio (compositionally appropriate)
        bp_log_ratio = float(np.log((beneficial + 1e-6) / (pathogen + 1e-6)))
        
        rows.append([
            sample, shannon, beneficial, pathogen,
            scfa, inflammation, bp_log_ratio
        ])
    
    df = pd.DataFrame(rows, columns=[
        'sample', 'shannon', 'beneficial', 'pathogen_load',
        'scfa', 'inflammation', 'bp_log_ratio'
    ]).set_index('sample')
    
    print(f"  Extracted features for {len(df)} samples")
    return df

def save_metadata(args, weights_len: int, n_samples: int):
    """Save processing metadata for reproducibility."""
    meta = {
        "generated_at": datetime.now().isoformat(),
        "feature_table": os.path.basename(args.feature_table),
        "asv_map": os.path.basename(args.asv_map),
        "weights_csv": os.path.basename(args.weights_csv),
        "n_genera_weights": weights_len,
        "n_samples": n_samples,
        "min_depth": args.min_depth,
        "min_presence": args.min_presence,
        "scfa_genera": args.scfa_genera,
        "inflam_genera": args.inflam_genera,
        "version": "2.0"
    }
    
    meta_path = os.path.join(
        os.path.dirname(args.out_stats),
        "pipeline_metadata.json"
    )
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ Saved metadata to {meta_path}")

def main(args):
    # Create output directories
    for out in [args.out_rel, args.out_percentiles, args.out_features, args.out_stats]:
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print("="*70)
    print("MICROBIOME REFERENCE DATABASE BUILDER")
    print("="*70)

    # 1) Load and filter feature table
    counts = read_feature_table(args.feature_table)
    counts = filter_samples(counts, args.min_depth, args.min_presence)
    
    # 2) Normalize to relative abundance
    print("\nNormalizing to relative abundances...")
    rel = counts.div(counts.sum(axis=0), axis=1)
    rel.to_csv(args.out_rel, sep="\t")
    print(f"  Saved ASV-level relative abundances to {args.out_rel}")

    # 3) Map to genus level
    genus_rel = map_to_genus(rel, args.asv_map)

    # 4) Compute genus percentiles
    pct = compute_percentiles(genus_rel)
    with open(args.out_percentiles, 'w') as f:
        json.dump(pct, f, indent=2)
    print(f"  Saved genus percentiles to {args.out_percentiles}")

    # 5) Load health weights
    print(f"\nLoading health weights from {args.weights_csv}...")
    weights = pd.read_csv(args.weights_csv, index_col=0)['HealthWeight']
    print(f"  Loaded weights for {len(weights)} genera")

    # 6) Extract features
    scfa_set = set(args.scfa_genera)
    inflam_set = set(args.inflam_genera)
    feats = compute_features(genus_rel, weights, scfa_set, inflam_set)
    feats.to_csv(args.out_features, sep="\t")
    print(f"  Saved features to {args.out_features}")

    # 7) Compute reference statistics
    stats = compute_reference_stats(feats)
    with open(args.out_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved reference stats to {args.out_stats}")

    # 8) Save metadata
    save_metadata(args, len(weights), genus_rel.shape[1])

    print("\n" + "="*70)
    print("✅ REFERENCE DATABASE GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {args.out_rel}")
    print(f"  2. {args.out_percentiles}")
    print(f"  3. {args.out_features}")
    print(f"  4. {args.out_stats}")
    print(f"\nReady for batch scoring with batch_health_scorer.py")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Build reference database for microbiome health scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        '--feature_table',
        default='feature-table.tsv',
        help='Input feature table (BIOM TSV format)'
    )
    p.add_argument(
        '--asv_map',
        default='asv_to_genus_map.json',
        help='ASV to genus mapping JSON'
    )
    p.add_argument(
        '--weights_csv',
        default='salivadb_genus_weights.csv',
        help='Genus health weights CSV'
    )
    p.add_argument(
        '--scfa_genera',
        nargs='+',
        default=['Veillonella', 'Lachnospira', 'Eubacterium', 'Propionibacterium', 'Megasphaera'],
        help='SCFA-producing genera'
    )
    p.add_argument(
        '--inflam_genera',
        nargs='+',
        default=['Porphyromonas', 'Treponema', 'Prevotella', 'Campylobacter'],
        help='Inflammatory genera'
    )
    p.add_argument(
        '--min_depth',
        type=int,
        default=5000,
        help='Minimum sequencing depth per sample'
    )
    p.add_argument(
        '--min_presence',
        type=int,
        default=2,
        help='Minimum number of samples an ASV must appear in'
    )
    p.add_argument(
        '--out_rel',
        default='reference_db/reference_rel.tsv',
        help='Output: ASV-level relative abundances'
    )
    p.add_argument(
        '--out_percentiles',
        default='reference_db/genus_percentiles.json',
        help='Output: genus percentiles'
    )
    p.add_argument(
        '--out_features',
        default='reference_db/reference_features.tsv',
        help='Output: derived features per sample'
    )
    p.add_argument(
        '--out_stats',
        default='reference_db/reference_stats.json',
        help='Output: reference statistics'
    )
    
    args = p.parse_args()
    main(args)
