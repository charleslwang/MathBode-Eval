#!/usr/bin/env python3
"""
score_mathbode.py
Compute MathBode scores (MB-Core and MB-Plus) from summary CSV/Parquet files.

Usage:
  python score_mathbode.py --summaries_dir summaries --outdir scores \
    --g_scale 0.25 --phi_scale 45 --wG 0.5 --wPhi 0.5 --alpha 0.5 --beta 0.5 --gamma 0.5
"""

import os, re, glob, math, argparse
import numpy as np
import pandas as pd

# ----------------------------- Config -----------------------------

FAMILY_ORDER = [
    "linear_solve",
    "ratio_saturation",
    "exponential_interest",
    "linear_system",
    "similar_triangles",
]

FAMILY_PRETTY = {
    "linear_solve": "Linear Solve",
    "ratio_saturation": "Ratio Saturation",
    "exponential_interest": "Exponential Interest",
    "linear_system": "Linear System",
    "similar_triangles": "Similar Triangles",
}

# Pretty names (regex -> pretty)
MODEL_PRETTY_OVERRIDES = {
    r"openai.*gpt[-_]?4o": "GPT-4o",
    r"meta.*llama.*guard|meta.*llama.*instruct|^llama.*": "Llama 4 Instruct",
    r"deepseek.*v3\.?1": "DeepSeek V3.1",
    r"mistral.*mixtral.*8x7b": "Mixtral 8×7B",
    r"qwen.*235b.*a22b.*instruct": "Qwen3 235B Instruct",
    r"qwen.*": "Qwen3",
}

MID_BAND_FREQS = {4, 8}

# ----------------------------- IO utils -----------------------------

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv",):
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported summary format: {path}")

def _guess_model_name(path_or_stem: str) -> str:
    stem = os.path.basename(path_or_stem)
    stem = re.sub(r"^summary[_-]?", "", stem, flags=re.I)
    stem = re.sub(r"\.(csv|parquet|pq)$", "", stem, flags=re.I)
    raw = stem
    for pat, pretty in MODEL_PRETTY_OVERRIDES.items():
        if re.search(pat, raw, flags=re.I):
            return pretty
    raw = raw.replace("_", " ").replace("-", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Harmonize column names/alternates
    ren = {}
    if "h2_over_h1" not in df.columns and "h2_over_h1_model" in df.columns:
        ren["h2_over_h1_model"] = "h2_over_h1"
    if "phase_deg" not in df.columns:
        for c in ("phase_err_deg", "phase_deg_model_minus_truth"):
            if c in df.columns:
                ren[c] = "phase_deg"; break
    if ren:
        df = df.rename(columns=ren)

    # Types
    if "frequency_cycles" in df.columns:
        df["frequency_cycles"] = pd.to_numeric(df["frequency_cycles"], errors="coerce")
    for c in ("gain","phase_deg","r2_model","compliance_rate",
              "h2_over_h1","res_rms_norm","res_acf1"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "family" in df.columns:
        df["family"] = df["family"].astype(str)
    return df

def load_summaries(summaries_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(summaries_dir, "summary*.csv"))) + \
            sorted(glob.glob(os.path.join(summaries_dir, "summary*.parquet"))) + \
            sorted(glob.glob(os.path.join(summaries_dir, "*.csv"))) + \
            sorted(glob.glob(os.path.join(summaries_dir, "*.parquet")))
    models = {}
    for p in paths:
        try:
            df = _read_any(p)
        except Exception:
            continue
        if not {"family","frequency_cycles"}.issubset(df.columns):
            continue
        df = _normalize_columns(df)
        pretty = _guess_model_name(p)
        models[pretty] = df
    return models

# ----------------------------- Score math -----------------------------

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def compute_family_scores(df: pd.DataFrame,
                          g_scale: float,
                          phi_scale_deg: float,
                          wG: float,
                          wPhi: float,
                          alpha: float,
                          beta: float,
                          gamma: float) -> dict:
    """
    Returns dict: family -> metrics + MB-Core + MB-Plus
    """
    out = {}
    for fam in FAMILY_ORDER:
        sub = df[(df["family"] == fam) & (df["frequency_cycles"].isin(MID_BAND_FREQS))].copy()
        if sub.empty or "gain" not in sub.columns or "phase_deg" not in sub.columns:
            continue

        # Core mid-band stats
        mean_abs_gerr = float((sub["gain"] - 1.0).abs().mean())
        mean_abs_phi  = float(sub["phase_deg"].abs().mean())

        EG   = _clip01(mean_abs_gerr / g_scale)
        Ephi = _clip01(mean_abs_phi  / phi_scale_deg)
        mb_core = _clip01(1.0 - (wG * EG + wPhi * Ephi))

        # Modifiers (optional)
        # R^2
        if "r2_model" in sub.columns and sub["r2_model"].notna().any():
            Q_r2 = _clip01(sub["r2_model"].mean())
        else:
            Q_r2, alpha_use = 1.0, 0.0  # ignore if missing
            alpha = alpha_use

        # Residual ACF(1): want near zero
        if "res_acf1" in sub.columns and sub["res_acf1"].notna().any():
            Q_acf = _clip01(1.0 - float(sub["res_acf1"].abs().mean()))
        else:
            Q_acf, beta_use = 1.0, 0.0
            beta = beta_use

        # Compliance
        comp_src = df[df["family"] == fam]["compliance_rate"] if "compliance_rate" in df.columns else pd.Series([], dtype=float)
        if comp_src.notna().any():
            Q_comp = _clip01(float(comp_src.mean()))
        else:
            Q_comp, gamma_use = 1.0, 0.0
            gamma = gamma_use

        mb_plus = mb_core * (Q_r2 ** alpha) * (Q_acf ** beta) * (Q_comp ** gamma)
        mb_plus = _clip01(mb_plus)

        out[fam] = {
            "Family": FAMILY_PRETTY.get(fam, fam),
            "GainAbsErrMid": mean_abs_gerr,
            "PhaseAbsDegMid": mean_abs_phi,
            "R2_mean": float(sub["r2_model"].mean()) if "r2_model" in sub.columns and sub["r2_model"].notna().any() else np.nan,
            "ACF1_abs_mean": float(sub["res_acf1"].abs().mean()) if "res_acf1" in sub.columns and sub["res_acf1"].notna().any() else np.nan,
            "Compliance_mean": float(comp_src.mean()) if comp_src.notna().any() else np.nan,
            "MB_Core": mb_core,
            "MB_Plus": mb_plus,
        }
    return out

def aggregate_overall(per_family: dict) -> dict:
    """Average across available families (uniform)"""
    if not per_family:
        return {"Overall_MB_Core": np.nan, "Overall_MB_Plus": np.nan,
                "Num_Families": 0}
    core_vals = [d["MB_Core"] for d in per_family.values() if np.isfinite(d["MB_Core"])]
    plus_vals = [d["MB_Plus"] for d in per_family.values() if np.isfinite(d["MB_Plus"])]
    overall_core = float(np.mean(core_vals)) if core_vals else np.nan
    overall_plus = float(np.mean(plus_vals)) if plus_vals else np.nan
    return {
        "Overall_MB_Core": overall_core,
        "Overall_MB_Plus": overall_plus,
        "Num_Families": len(per_family),
    }

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute MathBode scores from summary files.")
    ap.add_argument("--summaries_dir", required=True, help="Directory containing summary_*.csv/.parquet files.")
    ap.add_argument("--outdir", required=True, help="Output directory for score CSVs.")
    # Scales / weights
    ap.add_argument("--g_scale", type=float, default=0.25, help="Gain error scale for normalization (default 0.25).")
    ap.add_argument("--phi_scale", type=float, default=45.0, help="Phase error scale in degrees (default 45).")
    ap.add_argument("--wG", type=float, default=0.5, help="Weight for gain error in MB-Core (default 0.5).")
    ap.add_argument("--wPhi", type=float, default=0.5, help="Weight for phase error in MB-Core (default 0.5).")
    # Modifiers
    ap.add_argument("--alpha", type=float, default=0.5, help="Exponent for R^2 modifier in MB-Plus (default 0.5).")
    ap.add_argument("--beta", type=float, default=0.5, help="Exponent for ACF(1) modifier in MB-Plus (default 0.5).")
    ap.add_argument("--gamma", type=float, default=0.5, help="Exponent for compliance modifier in MB-Plus (default 0.5).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models = load_summaries(args.summaries_dir)
    if not models:
        raise SystemExit(f"No usable summaries found in: {args.summaries_dir}")

    # Per-family long table
    per_family_rows = []
    overall_rows = []

    for model_name, df in sorted(models.items()):
        fam_scores = compute_family_scores(
            df,
            g_scale=args.g_scale,
            phi_scale_deg=args.phi_scale,
            wG=args.wG,
            wPhi=args.wPhi,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )

        for fam_key in FAMILY_ORDER:
            if fam_key in fam_scores:
                d = fam_scores[fam_key]
                per_family_rows.append({
                    "Model": model_name,
                    "Family": d["Family"],
                    "GainAbsErrMid": d["GainAbsErrMid"],
                    "PhaseAbsDegMid": d["PhaseAbsDegMid"],
                    "R2_mean": d["R2_mean"],
                    "ACF1_abs_mean": d["ACF1_abs_mean"],
                    "Compliance_mean": d["Compliance_mean"],
                    "MB_Core": d["MB_Core"],
                    "MB_Plus": d["MB_Plus"],
                })

        overall = aggregate_overall(fam_scores)
        overall_rows.append({
            "Model": model_name,
            "Overall_MB_Core": overall["Overall_MB_Core"],
            "Overall_MB_Plus": overall["Overall_MB_Plus"],
            "Num_Families": overall["Num_Families"],
        })

    # DataFrames
    df_fam = pd.DataFrame(per_family_rows)
    df_overall = pd.DataFrame(overall_rows)

    # Sort nicely
    if not df_fam.empty:
        df_fam["Family_order"] = df_fam["Family"].map({FAMILY_PRETTY[k]: i for i, k in enumerate(FAMILY_ORDER)})
        df_fam = df_fam.sort_values(["Model","Family_order"]).drop(columns=["Family_order"])
    if not df_overall.empty:
        df_overall = df_overall.sort_values("Model")

    # Write CSVs
    per_family_path = os.path.join(args.outdir, "mathbode_scores_per_family.csv")
    overall_path   = os.path.join(args.outdir, "mathbode_scores_overall.csv")
    df_fam.to_csv(per_family_path, index=False)
    df_overall.to_csv(overall_path, index=False)

    # Print a compact summary table to stdout
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        print("\n=== Overall (MB-Core / MB-Plus) ===")
        print(df_overall.fillna("").to_string(index=False))
        print(f"\nWrote: {overall_path}")

        if not df_fam.empty:
            # Pivot for a quick paper-ready feel (Core only)
            pivot_core = df_fam.pivot_table(index="Model", columns="Family", values="MB_Core", aggfunc="mean")
            print("\n=== Per-Family MB-Core (higher is better) ===")
            print(pivot_core.round(3).to_string())
        print(f"\nWrote: {per_family_path}")
        print("\nDone.")

if __name__ == "__main__":
    main()
