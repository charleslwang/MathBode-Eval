# mathbode/plot_curves.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def _safe_unique(x):
    return pd.unique(x) if len(x) else []

def _group_stats(df, value_col, by):
    g = df.groupby(by, dropna=False)[value_col]
    mean = g.mean()
    std = g.std(ddof=1)
    n = g.count().astype(float)
    sem = std / np.sqrt(np.maximum(n, 1.0))
    out = pd.concat(
        [mean.rename("mean"), std.rename("std"), sem.rename("sem"), n.rename("n")],
        axis=1,
    ).reset_index()
    return out

def _ensure_sorted_freqs(df):
    if "frequency_cycles" in df.columns:
        return df.sort_values("frequency_cycles")
    return df

def plot_curves(summary: pd.DataFrame,
                model_tag: str,
                outdir: str,
                all_rows: Optional[pd.DataFrame] = None):
    """
    Primary visualization bundle.

    summary: per-(family, frequency) means (e.g., from summarize_gain_phase_means)
             must include: family, frequency_cycles, gain, phase_deg
             optional: r2_model, compliance_rate, acc_rate, h2_over_h1
    all_rows: optional detailed rows (from summarize_gain_phase_all) to enable:
              - phase stability across start phases (circular std @ mid-band)
              - accuracy vs frequency (exact, not just averaged twice)
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- basic checks
    need_cols = {"family", "frequency_cycles", "gain", "phase_deg"}
    missing = need_cols - set(summary.columns)
    if missing:
        raise ValueError(f"plot_curves: summary missing columns: {sorted(missing)}")

    # normalize types
    summary = summary.copy()
    summary["frequency_cycles"] = summary["frequency_cycles"].astype(int)

    fams = list(_safe_unique(summary["family"]))

    # ---- 1) Gain and Phase with error bands (mean ± sem) ----
    for metric, ylabel, fname, hline in [
        ("gain", "Gain (A_model / A_truth)", "gain", 1.0),
        ("phase_deg", "Phase (deg)", "phase", 0.0),
    ]:
        # If there are multiple per-(family,freq) rows in summary, we compute SEM here too.
        # Else, SEM=0 and it's just a line.
        plt.figure(figsize=(7.5, 4.2))
        for fam in fams:
            sub = summary[summary["family"] == fam]
            stats = _group_stats(sub, value_col=metric, by=["family", "frequency_cycles"])
            stats = _ensure_sorted_freqs(stats)
            xs = stats["frequency_cycles"].values
            ys = stats["mean"].values
            es = stats["sem"].values
            label = f"{fam}"
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=label)
            plt.fill_between(xs, ys - es, ys + es, alpha=0.18)

        plt.axhline(hline, color="k", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel("Frequency (cycles / 64 steps)")
        plt.ylabel(ylabel)
        plt.title(f"{model_tag} \u2022 {ylabel} vs frequency")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_{fname}_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 2) Mid-band bars: mean |G-1| and mean |phi| over {4,8} ----
    mid_freqs = {4, 8}
    mid = summary[summary["frequency_cycles"].isin(mid_freqs)].copy()
    if len(mid):
        mid["abs_g_err"] = (mid["gain"] - 1.0).abs()
        mid["abs_phi"] = mid["phase_deg"].abs()

        g_mid = mid.groupby("family")["abs_g_err"].mean().reset_index()
        p_mid = mid.groupby("family")["abs_phi"].mean().reset_index()

        fams_sorted = sorted(fams)
        x = np.arange(len(fams_sorted))
        width = 0.38

        g_map = dict(zip(g_mid["family"], g_mid["abs_g_err"]))
        p_map = dict(zip(p_mid["family"], p_mid["abs_phi"]))
        g_vals = np.array([g_map.get(f, np.nan) for f in fams_sorted])
        p_vals = np.array([p_map.get(f, np.nan) for f in fams_sorted])

        plt.figure(figsize=(7.5, 4.0))
        plt.bar(x - width/2, g_vals, width, label="mean |G-1| (mid-band)")
        plt.bar(x + width/2, p_vals, width, label="mean |phi| deg (mid-band)")
        plt.xticks(x, fams_sorted, rotation=0)
        plt.ylabel("Error / degrees")
        plt.title(f"{model_tag} \u2022 Mid-band summaries (freqs 4 & 8)")
        plt.legend(frameon=False)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_midband_bars.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 3) Fit quality R^2 vs frequency (mean ± sem) ----
    # Use r2_model if present.
    if "r2_model" in summary.columns:
        plt.figure(figsize=(7.5, 4.2))
        for fam in fams:
            sub = summary[summary["family"] == fam]
            stats = _group_stats(sub, value_col="r2_model", by=["family", "frequency_cycles"])
            stats = _ensure_sorted_freqs(stats)
            xs = stats["frequency_cycles"].values
            ys = stats["mean"].values
            es = stats["sem"].values
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=fam)
            plt.fill_between(xs, ys - es, ys + es, alpha=0.18)
        plt.ylim(0, 1.02)
        plt.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel("Frequency (cycles / 64 steps)")
        plt.ylabel("First-harmonic fit R^2 (model)")
        plt.title(f"{model_tag} \u2022 Fit quality vs frequency")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_r2_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 4) Compliance bars (if present) ----
    if "compliance_rate" in summary.columns:
        comp_fam = summary.groupby("family")["compliance_rate"].mean().reset_index()
        fams_sorted = sorted(_safe_unique(comp_fam["family"]))
        vals = [comp_fam[comp_fam["family"] == f]["compliance_rate"].values.mean() for f in fams_sorted]
        plt.figure(figsize=(7.0, 3.8))
        plt.bar(fams_sorted, vals)
        plt.ylim(0, 1.02)
        plt.ylabel("Compliance rate")
        plt.title(f"{model_tag} \u2022 Prompt compliance (higher is better)")
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_compliance.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 5) Accuracy vs frequency (requires all_rows) ----
    if all_rows is not None and "acc_rate" in all_rows.columns:
        acc = all_rows.groupby(["family", "frequency_cycles"])["acc_rate"].mean().reset_index()
        plt.figure(figsize=(7.5, 4.2))
        for fam in sorted(_safe_unique(acc["family"])):
            sub = acc[acc["family"] == fam].sort_values("frequency_cycles")
            plt.plot(sub["frequency_cycles"], sub["acc_rate"], marker="o", label=fam)
        plt.ylim(0, 1.02)
        plt.xlabel("Frequency (cycles / 64 steps)")
        plt.ylabel("Accuracy (tolerance-based)")
        plt.title(f"{model_tag} \u2022 Accuracy vs frequency")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_accuracy_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 6) Phase stability (circular std @ mid-band) (requires all_rows) ----
    if all_rows is not None and "phase_deg_model_minus_truth" in all_rows.columns:
        ar = all_rows.copy()
        ar = ar[ar["frequency_cycles"].isin({4, 8})]
        rows = []
        for fam, g in ar.groupby("family"):
            phi = g["phase_deg_model_minus_truth"].to_numpy()
            # reuse circular std in Python (inline to avoid import cycle)
            def _circ_std_deg(vals):
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return float("nan")
                rad = np.deg2rad(vals)
                C = np.nanmean(np.cos(rad)); S = np.nanmean(np.sin(rad))
                R = np.sqrt(C*C + S*S); R = np.clip(R, 1e-12, 1.0)
                return float(np.rad2deg(np.sqrt(np.maximum(0.0, -2.0*np.log(R)))))
            rows.append(dict(family=fam, phase_dispersion_deg=_circ_std_deg(phi)))
        if rows:
            tb = pd.DataFrame(rows).sort_values("family")
            plt.figure(figsize=(7.0, 3.8))
            plt.bar(tb["family"], tb["phase_dispersion_deg"])
            plt.ylabel("Phase dispersion (deg) @ mid-band")
            plt.title(f"{model_tag} \u2022 Phase stability across start phases")
            plt.tight_layout()
            path = os.path.join(outdir, f"{model_tag}_phase_stability.png")
            plt.savefig(path, dpi=160)
            plt.close()

    # ---- 7) H2/H1 vs frequency (optional if present) ----
    if "h2_over_h1" in summary.columns:
        plt.figure(figsize=(7.5, 4.2))
        for fam in fams:
            sub = summary[summary["family"] == fam]
            stats = _group_stats(sub, value_col="h2_over_h1", by=["family", "frequency_cycles"])
            stats = _ensure_sorted_freqs(stats)
            xs = stats["frequency_cycles"].values
            ys = stats["mean"].values
            es = stats["sem"].values
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=fam)
            plt.fill_between(xs, ys - es, ys + es, alpha=0.18)
        plt.xlabel("Frequency (cycles / 64 steps)")
        plt.ylabel("H2/H1 (second harmonic ratio)")
        plt.title(f"{model_tag} \u2022 Nonlinearity proxy (H2/H1)")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_h2h1_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()

if __name__ == "__main__":
    import argparse, os, sys

    def _read_any(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        elif ext in (".csv", ".tsv"):
            sep = "," if ext == ".csv" else "\t"
            return pd.read_csv(path, sep=sep)
        else:
            raise ValueError(f"Unsupported input format: {ext}")

    ap = argparse.ArgumentParser(
        description="Plot MathBode curves from summary (and optionally all_rows) tables."
    )
    ap.add_argument("--summary", required=True,
                    help="Path to summary CSV/Parquet (per-(family,freq) means).")
    ap.add_argument("--tag", required=True, help="Model tag for plot titles and filenames.")
    ap.add_argument("--outdir", default="results", help="Output directory (default: results)")
    ap.add_argument("--all-rows", default="",
                    help="Optional path to detailed all_rows CSV/Parquet to enable accuracy/phase-stability plots.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    try:
        summary = _read_any(args.summary)
    except Exception as e:
        print(f"[ERROR] Failed to load summary file: {e}", file=sys.stderr)
        sys.exit(2)

    all_rows = None
    if args.all_rows:
        try:
            all_rows = _read_any(args.all_rows)
        except Exception as e:
            print(f"[WARN] Failed to load all_rows file ({e}); continuing without it.", file=sys.stderr)
            all_rows = None

    plot_curves(summary, args.tag, args.outdir, all_rows=all_rows)
    print(f"[OK] plots written to: {args.outdir}")
