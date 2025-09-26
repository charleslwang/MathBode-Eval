# mathbode/plot_panels.py
# Generates publication-ready overlay plots from summary CSV/Parquet files.
# Outputs two bundles: <outdir>/main and <outdir>/appendix

import os, glob, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ----------------------------- Styling -----------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
LINEWIDTH = 2.0
MS = 5.5

# legend position (outside, centered below)
LEGEND_KW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False,
)

BAND_FACE = "#f2f2f2"  # light band fill

# Families in the desired order
FAMILY_ORDER = [
    "linear_solve",
    "ratio_saturation",
    "exponential_interest",
    "linear_system",
    "similar_triangles",
]

# Pretty names
FAMILY_PRETTY = {
    "linear_solve": "Linear Solve",
    "ratio_saturation": "Ratio Saturation",
    "exponential_interest": "Exponential Interest",
    "linear_system": "Linear System",
    "similar_triangles": "Similar Triangles",
}

MODEL_PRETTY_OVERRIDES = {
    r"openai.*gpt[-_]?4o": "GPT-4o",
    r"meta.*llama.*guard|llama.*scout.*guard": "Llama 4 Guard",
    r"meta.*llama.*": "Llama 4 Instruct",
    r"deepseek.*v3\.?1": "DeepSeek V3.1",
    r"mistral.*mixtral.*8x7b": "Mixtral 8×7B",
    r"qwen.*235b.*a22b.*instruct": "Qwen3 235B (Instruct)",
    r"qwen.*": "Qwen3",
}

# -------------------------- IO + Harmonization --------------------------

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv",):
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported summary format: {path}")

def _guess_model_name(path: str) -> str:
    stem = os.path.basename(path)
    stem = re.sub(r"^summary[_-]?", "", stem, flags=re.I)
    stem = re.sub(r"\.(csv|parquet|pq)$", "", stem, flags=re.I)
    raw = stem
    # Pretty mapping via regex overrides
    for pat, pretty in MODEL_PRETTY_OVERRIDES.items():
        if re.search(pat, raw, flags=re.I):
            return pretty
    # Fallback: squish and title-case lightly
    raw = raw.replace("_", " ").replace("-", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Accept either 'phase_deg' or a renamed column
    col_map = {}
    # many runs have 'h2_over_h1_model', expose as 'h2_over_h1'
    if "h2_over_h1" not in df.columns:
        if "h2_over_h1_model" in df.columns:
            col_map["h2_over_h1_model"] = "h2_over_h1"
    # for backward compat
    rename_candidates = {
        "phase_err_deg": "phase_deg",
        "phase_deg_model_minus_truth": "phase_deg",
    }
    for a, b in rename_candidates.items():
        if a in df.columns and "phase_deg" not in df.columns:
            col_map[a] = b

    if col_map:
        df = df.rename(columns=col_map)

    # types
    if "frequency_cycles" in df.columns:
        df["frequency_cycles"] = pd.to_numeric(df["frequency_cycles"], errors="coerce").astype("Int64")
    if "gain" in df.columns:
        df["gain"] = pd.to_numeric(df["gain"], errors="coerce")
    if "phase_deg" in df.columns:
        df["phase_deg"] = pd.to_numeric(df["phase_deg"], errors="coerce")
    if "r2_model" in df.columns:
        df["r2_model"] = pd.to_numeric(df["r2_model"], errors="coerce")
    if "compliance_rate" in df.columns:
        df["compliance_rate"] = pd.to_numeric(df["compliance_rate"], errors="coerce")
    for c in ("h2_over_h1", "res_rms_norm", "res_acf1"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # standardize family names
    if "family" in df.columns:
        df["family"] = df["family"].astype(str)
    return df

def load_summaries(summaries_dir: str) -> Dict[str, pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(summaries_dir, "summary*.csv"))) + \
            sorted(glob.glob(os.path.join(summaries_dir, "summary*.parquet"))) + \
            sorted(glob.glob(os.path.join(summaries_dir, "*.csv")))  # tolerate loose files
    models: Dict[str, pd.DataFrame] = {}
    for p in paths:
        try:
            df = _read_any(p)
        except Exception:
            continue
        # must contain columns we plot
        if "family" not in df.columns or "frequency_cycles" not in df.columns:
            continue
        df = _normalize_columns(df)
        pretty = _guess_model_name(p)
        models[pretty] = df
    return models

# ------------------------------ Utilities ------------------------------

def _safe_sem(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / max(np.sqrt(len(x)), 1.0))

def _band(ax, y0, y1):
    ax.axhspan(y0, y1, color=BAND_FACE, zorder=0)

def _prep_family_axes(ncols: int, nrows: int = 1, figsize=(15, 4.2)):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False)
    axs = np.array(axs).reshape(nrows, ncols)
    return fig, axs

def _family_iter():
    for fam in FAMILY_ORDER:
        yield fam, FAMILY_PRETTY.get(fam, fam.replace("_", " ").title())

def _collect_by_family(models: Dict[str, pd.DataFrame], metric: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for fam, _pretty in _family_iter():
        out[fam] = {}
        for model, df in models.items():
            if metric not in df.columns:
                continue
            sub = df[df["family"] == fam].copy()
            if sub.empty:
                continue
            sub = sub.dropna(subset=["frequency_cycles", metric])
            out[fam][model] = sub
    return out

def _plot_overlay_per_family(
    models: Dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    hline: float = None,
    ylim: Tuple[float, float] = None,
    out_path: str = "",
    title_suffix: str = "",
    panel_subset: List[str] = None,
    as_integer_xticks: bool = True,
):
    fam_to_models = _collect_by_family(models, metric)
    fams = [f for f, _ in _family_iter() if panel_subset is None or f in panel_subset]
    n = len(fams)
    if n == 0:
        return

    # layout: 1xN (wide) or 2x3 if N==5
    if n == 5:
        nrows, ncols, figsize = 1, 5, (18, 4.2)
    elif n <= 3:
        nrows, ncols, figsize = 1, n, (6*n, 4.2)
    else:
        nrows, ncols, figsize = 2, math.ceil(n/2), (6*ncols, 8.4)

    fig, axs = _prep_family_axes(ncols=ncols, nrows=nrows, figsize=figsize)
    axs = axs.flatten()

    model_list = sorted(models.keys())
    for i, fam in enumerate(fams):
        ax = axs[i]
        fam_pretty = FAMILY_PRETTY.get(fam, fam)
        # gentle band for readability (set per-metric)
        if metric == "gain":
            _band(ax, 1.0, 1.2)  # light band above unity
            _band(ax, 0.8, 1.0)  # and below
        elif metric == "phase_deg":
            _band(ax, -10, 10)
        elif metric == "r2_model":
            _band(ax, 0.9, 1.0)

        marker_i = 0
        for k, model in enumerate(model_list):
            dfm = fam_to_models.get(fam, {}).get(model, None)
            if dfm is None or dfm.empty:
                continue
            # group by frequency to get mean ± sem of the metric
            g = dfm.groupby("frequency_cycles")[metric]
            xs = g.mean().index.to_numpy(dtype=float)
            ys = g.mean().to_numpy(dtype=float)
            es = g.apply(_safe_sem).to_numpy(dtype=float)

            ax.plot(xs, ys,
                    marker=MARKERS[marker_i % len(MARKERS)],
                    linewidth=LINEWIDTH, markersize=MS, label=model)
            ax.fill_between(xs, ys-es, ys+es, alpha=0.18)
            marker_i += 1

        if hline is not None:
            ax.axhline(hline, color="k", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_title(fam_pretty)
        ax.set_xlabel("Frequency (cycles / 64 steps)")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        if as_integer_xticks:
            xt = sorted(dfm["frequency_cycles"].dropna().unique()) if dfm is not None else [1,2,4,8,16]
            ax.set_xticks(list(map(int, xt)))

    # global legend
    handles, labels = axs[0].get_legend_handles_labels()
    if len(handles):
        fig.legend(handles, labels, **LEGEND_KW)

    if title_suffix:
        fig.suptitle(title_suffix, y=1.02, fontsize=18)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# --------------------------- Appendix helpers ---------------------------

def _midband_table(models: Dict[str, pd.DataFrame], out_path: str):
    """Save a compact mid-band (freq 4&8) table image with mean |G-1| and |phi|."""
    rows = []
    for model, df in models.items():
        df2 = df[df["frequency_cycles"].isin([4,8])].copy()
        if df2.empty or "gain" not in df2.columns or "phase_deg" not in df2.columns:
            continue
        df2["abs_g_err"] = (df2["gain"] - 1.0).abs()
        df2["abs_phi"] = df2["phase_deg"].abs()
        for fam, g in df2.groupby("family"):
            rows.append({
                "Model": model,
                "Family": FAMILY_PRETTY.get(fam, fam),
                "|G-1| (mid)": g["abs_g_err"].mean(),
                "|Phase| deg (mid)": g["abs_phi"].mean(),
            })
    if not rows:
        return
    tb = pd.DataFrame(rows)
    tb = tb.pivot(index="Family", columns="Model", values="|G-1| (mid)").round(3).sort_index()
    tb2 = pd.DataFrame(rows).pivot(index="Family", columns="Model", values="|Phase| deg (mid)").round(2).sort_index()

    # render two small tables stacked
    fig, axs = plt.subplots(2, 1, figsize=(min(12, 4 + 1.1*len(tb.columns)), 5.6))
    for ax, title, data in zip(
        axs,
        ["Mean |G-1| (Freqs 4 & 8)", "Mean |Phase| (deg) (Freqs 4 & 8)"],
        [tb, tb2]
    ):
        ax.axis("off")
        tbl = ax.table(cellText=data.values,
                       rowLabels=data.index,
                       colLabels=data.columns,
                       cellLoc="center",
                       loc="center")
        tbl.scale(1.1, 1.2)
        ax.set_title(title, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _compliance_panels(models: Dict[str, pd.DataFrame], out_path: str):
    # bar per family with grouped models
    fams = [f for f, _ in _family_iter()]
    model_list = sorted(models.keys())
    vals = np.full((len(fams), len(model_list)), np.nan)

    for i, fam in enumerate(fams):
        for j, model in enumerate(model_list):
            df = models[model]
            if "compliance_rate" not in df.columns:
                continue
            sub = df[df["family"] == fam]["compliance_rate"].dropna()
            if not sub.empty:
                vals[i, j] = sub.mean()

    fig, axs = plt.subplots(1, len(fams), figsize=(18, 4.2), sharey=True)
    if len(fams) == 1:
        axs = [axs]
    x = np.arange(len(model_list))
    width = 0.8 / len(fams)

    for i, (fam, ax) in enumerate(zip(fams, axs)):
        for j, model in enumerate(model_list):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            ax.bar(j, v, width=0.8, label=None)
        ax.set_title(FAMILY_PRETTY.get(fam, fam))
        ax.set_xticks(x)
        ax.set_xticklabels(model_list, rotation=30, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Compliance")

    # single legend would be redundant; bars already labeled by xticks
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _overlay_metric(models: Dict[str, pd.DataFrame], metric: str, ylabel: str, out_path: str, hline=None, ylim=None):
    _plot_overlay_per_family(
        models=models, metric=metric, ylabel=ylabel,
        hline=hline, ylim=ylim, out_path=out_path,
        title_suffix="", panel_subset=None
    )

# ------------------------------- Entry --------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Make paper-ready overlay plots from MathBode summaries.")
    ap.add_argument("--summaries_dir", required=True, help="Folder with summary_*.csv/.parquet for each model.")
    ap.add_argument("--outdir", required=True, help="Output directory root.")
    args = ap.parse_args()

    models = load_summaries(args.summaries_dir)
    if not models:
        raise SystemExit(f"No usable summaries found in: {args.summaries_dir}")

    out_main = os.path.join(args.outdir, "main")
    out_appx = os.path.join(args.outdir, "appendix")
    os.makedirs(out_main, exist_ok=True)
    os.makedirs(out_appx, exist_ok=True)

    # ------------------ MAIN FIGURES ------------------

    # 1) Gain vs Frequency (all families)
    _plot_overlay_per_family(
        models=models,
        metric="gain",
        ylabel="Gain",
        hline=1.0,
        ylim=None,
        out_path=os.path.join(out_main, "fig1_gain_vs_frequency.png"),
        title_suffix="Gain vs Frequency",
    )

    # 2) Phase Error vs Frequency (only diagnostic families)
    _plot_overlay_per_family(
        models=models,
        metric="phase_deg",
        ylabel="Phase Error (deg)",
        hline=0.0,
        ylim=None,
        out_path=os.path.join(out_main, "fig2_phase_error_exponential_and_system.png"),
        title_suffix="Phase Error vs Frequency",
        panel_subset=["exponential_interest", "linear_system"],
    )

    # 3) R² vs Frequency (all families)
    if any("r2_model" in df.columns for df in models.values()):
        _plot_overlay_per_family(
            models=models,
            metric="r2_model",
            ylabel="Fit Quality (R²)",
            hline=1.0,
            ylim=(0.0, 1.02),
            out_path=os.path.join(out_main, "fig3_r2_vs_frequency.png"),
            title_suffix="Fit Quality vs Frequency",
        )

    # ------------------ APPENDIX ------------------

    # A1) Mid-band summary table image
    _midband_table(models, os.path.join(out_appx, "tabA1_midband_summary.png"))

    # A2) Compliance bars per family
    if any("compliance_rate" in df.columns for df in models.values()):
        _compliance_panels(models, os.path.join(out_appx, "figA2_compliance_by_family.png"))

    # A3) Nonlinearity proxy (H2/H1)
    if any(("h2_over_h1" in df.columns) or ("h2_over_h1_model" in df.columns) for df in models.values()):
        # ensure unified column name
        models_h = {m: (d.rename(columns={"h2_over_h1_model": "h2_over_h1"}) if "h2_over_h1_model" in d.columns else d)
                    for m, d in models.items()}
        _overlay_metric(models_h, "h2_over_h1", "H2 / H1", os.path.join(out_appx, "figA3_h2_over_h1_vs_frequency.png"))

    # A4) Residual diagnostics
    if any("res_rms_norm" in df.columns for df in models.values()):
        _overlay_metric(models, "res_rms_norm", "Residual RMS (normalized)", os.path.join(out_appx, "figA4_residual_rms.png"))
    if any("res_acf1" in df.columns for df in models.values()):
        _overlay_metric(models, "res_acf1", "Residual ACF(1)", os.path.join(out_appx, "figA5_residual_acf1.png"))

    # A6) (Optional) Geometry standalone: Gain & R² for Similar Triangles only
    # Gain
    _plot_overlay_per_family(
        models=models,
        metric="gain",
        ylabel="Gain",
        hline=1.0,
        out_path=os.path.join(out_appx, "figA6_geometry_gain.png"),
        title_suffix="Similar Triangles • Gain",
        panel_subset=["similar_triangles"],
    )
    # R²
    if any("r2_model" in df.columns for df in models.values()):
        _plot_overlay_per_family(
            models=models,
            metric="r2_model",
            ylabel="Fit Quality (R²)",
            hline=1.0,
            ylim=(0.98, 1.001),
            out_path=os.path.join(out_appx, "figA6_geometry_r2.png"),
            title_suffix="Similar Triangles • Fit Quality",
            panel_subset=["similar_triangles"],
        )

    print(f"[OK] Wrote main figures → {out_main}")
    print(f"[OK] Wrote appendix figures/tables → {out_appx}")

if __name__ == "__main__":
    main()
