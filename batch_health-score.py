#!/usr/bin/env python3
"""
batch_health_scorer.py
Scores every sample in feature-table.tsv into JSON files under patient_reports/.
Uses ERR (genus_percentiles.json) + robust global stats (reference_stats.json).
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd
from math import erf, sqrt

# ---------- Statistical Helpers ----------
def robust_percentile(x: float, stats: dict) -> float:
    """
    Compute robust percentile using MAD (Median Absolute Deviation).
    Falls back to mean/std if MAD is unavailable.
    """
    if "median" in stats and "mad" in stats and stats["mad"] not in (None, 0):
        z = 0.6745 * (x - stats["median"]) / stats["mad"]
    else:
        sd = stats.get("std", 1.0) or 1.0
        z = (x - stats.get("mean", 0.0)) / sd
    p = 50 * (1 + erf(z / sqrt(2)))
    return float(np.clip(p, 0, 100))

def safe_log_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    """Compute log ratio with epsilon to handle zeros."""
    return float(np.log((a + eps) / (b + eps)))

def exceeds(g_abund: pd.Series, genus: str, label: str, gp: dict) -> bool:
    """Check if genus abundance exceeds the specified percentile threshold."""
    tab = gp.get(genus)
    if not tab or label not in tab:
        return False
    return float(g_abund.get(genus, 0.0)) > float(tab[label])

# ---------- Disease Risk Panels with ERR Gates ----------
PANELS = [
    {
        "name": "Oral Cancer (OSCC)",
        "threshold": 0.05,
        "genera": [
            {"name": "Fusobacterium", "w": 1.0},
            {"name": "Peptostreptococcus", "w": 0.3}
        ],
        "all_of_percentile": {"Fusobacterium": "p97_5"},
        "any_of_percentile": {"Peptostreptococcus": "p90"}
    },
    {
        "name": "Periodontitis (Red Complex)",
        "threshold": 0.02,
        "genera": [
            {"name": "Porphyromonas", "w": 0.5},
            {"name": "Tannerella", "w": 0.5},
            {"name": "Treponema", "w": 0.5}
        ],
        "at_least_k": {
            "label": "p90",
            "genera": ["Porphyromonas", "Tannerella", "Treponema"],
            "k": 2
        }
    },
    {
        "name": "Halitosis (VSC)",
        "threshold": 0.01,
        "genera": [
            {"name": "Solobacterium", "w": 1.0},
            {"name": "Fusobacterium", "w": 0.5}
        ],
        "any_of_percentile": {"Solobacterium": "p90"}
    },
    {
        "name": "Dental Caries",
        "threshold": 0.03,
        "genera": [
            {"name": "Lactobacillus", "w": 0.6},
            {"name": "Scardovia", "w": 0.4},
            {"name": "Streptococcus", "w": 0.5}
        ],
        "any_of_percentile": {
            "Lactobacillus": "p90",
            "Scardovia": "p90"
        }
    },
    {
        "name": "Oral Candidiasis (Thrush)",
        "threshold": 0.001,
        "genera": [{"name": "Candida", "w": 1.0}],
        "any_of_percentile": {"Candida": "p97_5"}
    },
    {
        "name": "Systemic Inflammation Link",
        "threshold": 0.05,
        "genera": [
            {"name": "Prevotella", "w": 0.5},
            {"name": "Campylobacter", "w": 0.5}
        ],
        "any_of_percentile": {
            "Prevotella": "p97_5",
            "Campylobacter": "p90"
        }
    }
]

def pass_gates(abund: pd.Series, panel: dict, gp: dict) -> bool:
    """Check if sample passes all ERR gates for this panel."""
    # all_of: ALL specified genera must exceed their percentile
    allmap = panel.get("all_of_percentile", {})
    if allmap and not all(exceeds(abund, g, lbl, gp) for g, lbl in allmap.items()):
        return False
    
    # any_of: AT LEAST ONE must exceed
    anymap = panel.get("any_of_percentile", {})
    if anymap and not any(exceeds(abund, g, lbl, gp) for g, lbl in anymap.items()):
        return False
    
    # at_least_k: At least K out of N genera must exceed
    ak = panel.get("at_least_k")
    if ak:
        label = ak["label"]
        genera = ak["genera"]
        k = int(ak.get("k", len(genera)))
        hits = sum(exceeds(abund, g, label, gp) for g in genera)
        if hits < k:
            return False
    
    return True

def panel_score(abund: pd.Series, panel: dict) -> tuple[float, bool]:
    """Calculate weighted panel score and check if threshold exceeded."""
    s = 0.0
    for item in panel.get("genera", []):
        s += float(abund.get(item["name"], 0.0)) * float(item.get("w", 1.0))
    return s, (s >= float(panel.get("threshold", 0.0)))

# ---------- Core Analysis Function ----------
def load_feature_table_to_genus_abund(feature_tsv: str, asv_map: dict) -> pd.DataFrame:
    """Load feature table and aggregate to genus level."""
    raw = pd.read_csv(feature_tsv, sep="\t", skiprows=1, index_col=0)
    rel = raw.div(raw.sum(axis=0), axis=1)
    rel["genus"] = rel.index.map(asv_map).fillna("Unknown")
    return rel.groupby("genus").sum()

def analyze_sample(
    abund: pd.Series,
    ref_stats: dict,
    weights_df: pd.DataFrame,
    gp: dict
) -> dict:
    """Analyze a single sample and return clinical report."""
    # Normalize
    abund = abund.clip(lower=0) / max(1e-12, abund.sum())
    
    # Define genus sets
    beneficial = set(weights_df[weights_df["HealthWeight"] > 0].index)
    pathogenic = set(weights_df[weights_df["HealthWeight"] < 0].index)
    scfa = {"Veillonella", "Prevotella", "Eubacterium", "Propionibacterium", "Megasphaera"}

    # Calculate metrics
    p = abund[abund > 0].values
    shannon = float(-np.sum(p * np.log(p))) if p.size else 0.0
    benef = float(abund[abund.index.isin(beneficial)].sum())
    patho = float(abund[abund.index.isin(pathogenic)].sum())
    scfa_sum = float(abund[abund.index.isin(scfa)].sum())
    bp_lr = safe_log_ratio(benef, patho)

    # Component scores (0–100 via robust percentiles)
    sh = robust_percentile(shannon, ref_stats["shannon"])
    ra = robust_percentile(bp_lr, ref_stats["bp_log_ratio"])
    sc = robust_percentile(scfa_sum, ref_stats["scfa"])
    pa = 100 - robust_percentile(patho, ref_stats["pathogen_load"])

    # Composite → calibrated CHI via piecewise interpolation
    w = {"shannon": 0.25, "ratio": 0.30, "scfa": 0.20, "pathogen": 0.25}
    composite = (
        w["shannon"] * sh +
        w["ratio"] * ra +
        w["scfa"] * sc +
        w["pathogen"] * pa
    )
    
    # Map [p5, p50, p95] → [-5, 0, +5]
    x, y = [35.0, 50.0, 70.0], [-5.0, 0.0, +5.0]
    chi = float(np.clip(np.interp(composite, x, y), -10, 10))
    
    category = (
        "Excellent" if chi >= 3 else
        "Good" if chi >= 1 else
        "Average" if chi >= -1 else
        "Below Average" if chi >= -3 else
        "Non-Ideal"
    )

    # Disease risks (ERR-gated)
    risks = {}
    for panel in PANELS:
        if not pass_gates(abund, panel, gp):
            continue
        s, fired = panel_score(abund, panel)
        if fired:
            risks[panel["name"]] = {
                "panel_score": round(s, 4),
                "threshold": panel["threshold"],
                "level": "High" if s >= 2 * panel["threshold"] else "Moderate",
                "marker": ", ".join([g["name"] for g in panel["genera"]]),
            }

    return {
        "clinical_health_index": round(chi, 2),
        "health_category": {
            "label": category,
            "color": (
                "green" if category in {"Excellent", "Good"} else
                "yellow" if category == "Average" else
                "red"
            )
        },
        "key_metrics": {
            "shannon_diversity": round(shannon, 3),
            "beneficial_pathogen_log_ratio": round(bp_lr, 3),
            "scfa_producer_abundance": round(scfa_sum, 4),
            "pathogen_load": round(patho, 4)
        },
        "composite_score": round(composite, 1),
        "disease_risks": risks,
        "raw_abundances": abund.to_dict()
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Batch score patient samples with robust ERR-gated risk detection"
    )
    ap.add_argument("--feature_tsv", default="feature-table.tsv")
    ap.add_argument("--asv_map", default="asv_to_genus_map.json")
    ap.add_argument("--weights_csv", default="salivadb_genus_weights.csv")
    ap.add_argument("--ref_stats", default="reference_stats.json")
    ap.add_argument("--genus_percentiles", default="genus_percentiles.json")
    ap.add_argument("--outdir", default="patient_reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # Load reference data
    asv_map = json.load(open(args.asv_map))
    weights = pd.read_csv(args.weights_csv, index_col=0)
    ref_stats = json.load(open(args.ref_stats))
    gp = json.load(open(args.genus_percentiles))

    # Process samples
    genus_abund = load_feature_table_to_genus_abund(args.feature_tsv, asv_map)

    for sample_id in genus_abund.columns:
        res = analyze_sample(genus_abund[sample_id], ref_stats, weights, gp)
        res["sample_id"] = sample_id
        
        out = os.path.join(args.outdir, f"{sample_id}_report_data.json")
        json.dump(res, open(out, "w"), indent=2)
        print(f"✓ {sample_id} → {out}")

    print("\n✅ Batch scoring complete.")

if __name__ == "__main__":
    main()
