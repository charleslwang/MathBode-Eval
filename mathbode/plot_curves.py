# mathbode/plot_curves.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_curves(summary: pd.DataFrame, model_tag: str, outdir: str):
    """
    Expects 'summary' to contain at least:
      family, frequency_cycles, gain, phase_deg
    Optionally:
      r2, n_total, n_valid, compliance_rate
    May contain multiple rows per (family, frequency) when there are multiple base keys/phases.
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
        plt.figure(figsize=(7.5, 4.2))
        for fam in fams:
            sub = summary[summary["family"] == fam]
            stats = _group_stats(sub, value_col=metric, by=["family", "frequency_cycles"])
            stats = _ensure_sorted_freqs(stats)
            xs = stats["frequency_cycles"].values
            ys = stats["mean"].values
            es = stats["sem"].values
            label = f"{fam}"
            # line + band
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=label)
            plt.fill_between(xs, ys - es, ys + es, alpha=0.18)

        # reference line
        plt.axhline(hline, color="k", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel("Frequency (cycles / 64 steps)")
        plt.ylabel(ylabel)
        plt.title(f"{model_tag} • {ylabel} vs frequency")
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

        # per family averages
        g_mid = mid.groupby("family")["abs_g_err"].mean().reset_index()
        p_mid = mid.groupby("family")["abs_phi"].mean().reset_index()

        # bar plot side-by-side
        fams_sorted = sorted(fams)
        x = np.arange(len(fams_sorted))
        width = 0.38

        # reorder to shared family order
        g_map = dict(zip(g_mid["family"], g_mid["abs_g_err"]))
        p_map = dict(zip(p_mid["family"], p_mid["abs_phi"]))
        g_vals = np.array([g_map.get(f, np.nan) for f in fams_sorted])
        p_vals = np.array([p_map.get(f, np.nan) for f in fams_sorted])

        plt.figure(figsize=(7.5, 4.0))
        plt.bar(x - width/2, g_vals, width, label="mean |G-1| (mid-band)")
        plt.bar(x + width/2, p_vals, width, label="mean |phi| deg (mid-band)")
        plt.xticks(x, fams_sorted, rotation=0)
        plt.ylabel("Error / degrees")
        plt.title(f"{model_tag} • Mid-band summaries (freqs 4 & 8)")
        plt.legend(frameon=False)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_midband_bars.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 3) Fit quality R^2 vs frequency (if present) ----
    if "r2" in summary.columns:
        plt.figure(figsize=(7.5, 4.2))
        for fam in fams:
            sub = summary[summary["family"] == fam]
            stats = _group_stats(sub, value_col="r2", by=["family", "frequency_cycles"])
            stats = _ensure_sorted_freqs(stats)
            xs = stats["frequency_cycles"].values
            ys = stats["mean"].values
            es = stats["sem"].values
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=fam)
            plt.fill_between(xs, ys - es, ys + es, alpha=0.18)
        plt.ylim(0, 1.02)
        plt.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel("Frequency (cycles / 64 steps)")
        plt.ylabel("First-harmonic fit $R^2$")
        plt.title(f"{model_tag} • Fit quality vs frequency")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_r2_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()

    # ---- 4) Compliance bars (if present) ----
    # Accept either precomputed compliance_rate or (n_valid / n_total).
    have_comp = "compliance_rate" in summary.columns or (
        "n_valid" in summary.columns and "n_total" in summary.columns
    )
    if have_comp:
        comp = summary.copy()
        if "compliance_rate" not in comp.columns:
            comp["compliance_rate"] = comp["n_valid"] / comp["n_total"]
        # average per family across frequencies (and runs)
        comp_fam = comp.groupby("family")["compliance_rate"].mean().reset_index()
        fams_sorted = sorted(_safe_unique(comp_fam["family"]))
        vals = [comp_fam[comp_fam["family"] == f]["compliance_rate"].values.mean() for f in fams_sorted]
        plt.figure(figsize=(7.0, 3.8))
        plt.bar(fams_sorted, vals)
        plt.ylim(0, 1.02)
        plt.ylabel("Compliance rate")
        plt.title(f"{model_tag} • Prompt compliance (higher is better)")
        plt.tight_layout()
        path = os.path.join(outdir, f"{model_tag}_compliance.png")
        plt.savefig(path, dpi=160)
        plt.close()
