# mathbode/plot_panels.py
# Generates publication-ready overlay plots from summary CSV/Parquet files.
# Outputs two bundles: <outdir>/main and <outdir>/appendix

import os, glob, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- Styling -----------------------------

# Modern, clean aesthetic with better typography and colors
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Professional color palette with better contrast
COLORS = [
    "#2E86AB",  # Deep blue
    "#A23B72",  # Rose
    "#F18F01",  # Orange
    "#C73E1D",  # Red
    "#6A994E",  # Green
    "#8338EC",  # Purple
    "#FB5607",  # Bright orange
    "#3A86FF",  # Bright blue
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h"]
LINEWIDTH = 2.2
MS = 7
MARKER_EDGE_WIDTH = 1.5
MARKER_EDGE_COLOR = "white"

# Enhanced legend styling
LEGEND_KW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=True,
    fancybox=True,
    shadow=True,
    borderpad=1,
    columnspacing=2,
    handlelength=2.5,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=0.95,
)

# Subtle band colors with gradients
BAND_FACE = "#f7f7f7"
BAND_EDGE = "#e0e0e0"
BAND_ALPHA = 0.6

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
    r"meta.*llama.*": "Llama 4 Instruct",
    r"deepseek.*v3\.?1": "DeepSeek V3.1",
    r"mistral.*mixtral.*8x7b": "Mixtral 8×7B",
    r"qwen.*235b.*a22b.*instruct": "Qwen3 235B Instruct",
    r"qwen.*": "Qwen3",
}

# Model to logo mapping (logos should be in logos/ directory)
MODEL_LOGOS = {
    "GPT-4o": "logos/oai.png",
    "Llama 4 Instruct": "logos/meta.png",
    "DeepSeek V3.1": "logos/deepseek.png",
    "Mixtral 8×7B": "logos/mixtral.png",
    "Qwen3 235B Instruct": "logos/qwen.png",
    "Qwen3": "logos/qwen.png",
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
    """Create a subtle gradient band for visual guidance"""
    # Create gradient effect with multiple alpha levels
    n_bands = 3
    alphas = np.linspace(BAND_ALPHA * 0.3, BAND_ALPHA, n_bands)
    y_range = y1 - y0
    for i, alpha in enumerate(alphas):
        y_start = y0 + (i * y_range / n_bands)
        y_end = y0 + ((i + 1) * y_range / n_bands)
        ax.axhspan(y_start, y_end, color=BAND_FACE, alpha=alpha, zorder=0, linewidth=0)
    # Add subtle edge lines
    ax.axhline(y0, color=BAND_EDGE, linewidth=0.5, alpha=0.3, zorder=1)
    ax.axhline(y1, color=BAND_EDGE, linewidth=0.5, alpha=0.3, zorder=1)

def _prep_family_axes(ncols: int, nrows: int = 1, figsize=(15, 4.2)):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False)
    axs = np.array(axs).reshape(nrows, ncols)
    # Add subtle background
    fig.patch.set_facecolor('white')
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

def _create_legend_with_logos(fig, handles, labels, logo_dir: Optional[str] = None):
    """Create enhanced legend with optional model provider logos"""
    if logo_dir and os.path.exists(logo_dir):
        # Try to add logos to legend
        new_handles = []
        for handle, label in zip(handles, labels):
            logo_path = MODEL_LOGOS.get(label, "")
            if logo_path:
                full_path = os.path.join(logo_dir, logo_path.replace("logos/", ""))
                if os.path.exists(full_path):
                    try:
                        # Create custom handle with logo
                        img = plt.imread(full_path)
                        # Keep original handle properties but add logo somehow
                        # For now, just use the regular handle
                        new_handles.append(handle)
                    except:
                        new_handles.append(handle)
                else:
                    new_handles.append(handle)
            else:
                new_handles.append(handle)
        handles = new_handles
    
    legend = fig.legend(handles, labels, **LEGEND_KW)
    # Enhance legend appearance
    for text in legend.get_texts():
        text.set_fontweight('medium')
    return legend

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
    logo_dir: Optional[str] = None,
):
    fam_to_models = _collect_by_family(models, metric)
    fams = [f for f, _ in _family_iter() if panel_subset is None or f in panel_subset]
    n = len(fams)
    if n == 0:
        return

    # layout: 1xN (wide) or 2x3 if N==5
    if n == 5:
        nrows, ncols, figsize = 1, 5, (20, 5)
    elif n <= 3:
        nrows, ncols, figsize = 1, n, (7*n, 5)
    else:
        nrows, ncols, figsize = 2, math.ceil(n/2), (7*ncols, 10)

    fig, axs = _prep_family_axes(ncols=ncols, nrows=nrows, figsize=figsize)
    axs = axs.flatten()

    model_list = sorted(models.keys())
    for i, fam in enumerate(fams):
        ax = axs[i]
        fam_pretty = FAMILY_PRETTY.get(fam, fam)
        
        # Enhanced bands with metric-specific styling
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

            color = COLORS[marker_i % len(COLORS)]
            marker = MARKERS[marker_i % len(MARKERS)]
            
            # Enhanced line and marker styling
            ax.plot(xs, ys,
                    marker=marker,
                    color=color,
                    linewidth=LINEWIDTH,
                    markersize=MS,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    markeredgecolor=MARKER_EDGE_COLOR,
                    label=model,
                    alpha=0.9,
                    zorder=10 + marker_i)
            
            # Enhanced error bands with gradient
            ax.fill_between(xs, ys-es, ys+es, 
                          color=color,
                          alpha=0.15,
                          zorder=5 + marker_i,
                          edgecolor='none')
            marker_i += 1

        if hline is not None:
            ax.axhline(hline, color="#666666", linestyle="--", linewidth=1.5, 
                      alpha=0.6, zorder=2, label='_nolegend_')

        # Enhanced title and labels
        ax.set_title(fam_pretty, pad=10, fontweight='bold')
        ax.set_xlabel("Frequency (cycles / 64 steps)", fontweight='medium')
        ax.set_ylabel(ylabel, fontweight='medium')
        
        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        if ylim:
            ax.set_ylim(*ylim)
        if as_integer_xticks:
            xt = sorted(dfm["frequency_cycles"].dropna().unique()) if dfm is not None else [1,2,4,8,16]
            ax.set_xticks(list(map(int, xt)))
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    # Enhanced global legend
    handles, labels = axs[0].get_legend_handles_labels()
    if len(handles):
        _create_legend_with_logos(fig, handles, labels, logo_dir)

    if title_suffix:
        fig.suptitle(title_suffix, y=1.02, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", facecolor='white', edgecolor='none')
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

    # Enhanced table rendering
    fig, axs = plt.subplots(2, 1, figsize=(min(14, 5 + 1.2*len(tb.columns)), 6.5))
    fig.patch.set_facecolor('white')
    
    for ax, title, data in zip(
        axs,
        ["Mean |G-1| (Frequencies 4 & 8)", "Mean |Phase| (degrees) (Frequencies 4 & 8)"],
        [tb, tb2]
    ):
        ax.axis("off")
        
        # Create table with enhanced styling
        cell_colors = []
        for _ in range(len(data.index)):
            row_colors = []
            for _ in range(len(data.columns)):
                row_colors.append('#f9f9f9')
            cell_colors.append(row_colors)
        
        tbl = ax.table(cellText=data.values,
                       rowLabels=data.index,
                       colLabels=data.columns,
                       cellLoc="center",
                       loc="center",
                       cellColours=cell_colors,
                       colColours=['#e8e8e8'] * len(data.columns),
                       rowColours=['#e8e8e8'] * len(data.index))
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.5)
        
        # Style the table
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor('#cccccc')
            cell.set_linewidth(0.5)
            if i == 0 or j == -1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
        
        ax.set_title(title, pad=15, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)

def _compliance_panels(models: Dict[str, pd.DataFrame], out_path: str):
    # Enhanced bar chart styling
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

    fig, axs = plt.subplots(1, len(fams), figsize=(20, 5), sharey=True)
    fig.patch.set_facecolor('white')
    
    if len(fams) == 1:
        axs = [axs]
    x = np.arange(len(model_list))

    for i, (fam, ax) in enumerate(zip(fams, axs)):
        for j, model in enumerate(model_list):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            color = COLORS[j % len(COLORS)]
            ax.bar(j, v, width=0.7, color=color, alpha=0.8, 
                  edgecolor='white', linewidth=2)
        
        ax.set_title(FAMILY_PRETTY.get(fam, fam), fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_list, rotation=35, ha="right", fontweight='medium')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Compliance Rate", fontweight='medium')
        
        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add percentage labels on bars
        for j, model in enumerate(model_list):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, v + 0.02, f'{v:.1%}', ha='center', va='bottom', 
                       fontsize=9, fontweight='medium')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)

def _overlay_metric(models: Dict[str, pd.DataFrame], metric: str, ylabel: str, out_path: str, 
                   hline=None, ylim=None, logo_dir=None):
    _plot_overlay_per_family(
        models=models, metric=metric, ylabel=ylabel,
        hline=hline, ylim=ylim, out_path=out_path,
        title_suffix="", panel_subset=None, logo_dir=logo_dir
    )

# ------------------------------- Entry --------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Make paper-ready overlay plots from MathBode summaries.")
    ap.add_argument("--summaries_dir", required=True, help="Folder with summary_*.csv/.parquet for each model.")
    ap.add_argument("--outdir", required=True, help="Output directory root.")
    ap.add_argument("--logo_dir", default=None, help="Optional directory containing model provider logos.")
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
        logo_dir=args.logo_dir,
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
        logo_dir=args.logo_dir,
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
            logo_dir=args.logo_dir,
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
        _overlay_metric(models_h, "h2_over_h1", "H2 / H1", 
                       os.path.join(out_appx, "figA3_h2_over_h1_vs_frequency.png"),
                       logo_dir=args.logo_dir)

    # A4) Residual diagnostics
    if any("res_rms_norm" in df.columns for df in models.values()):
        _overlay_metric(models, "res_rms_norm", "Residual RMS (normalized)", 
                       os.path.join(out_appx, "figA4_residual_rms.png"),
                       logo_dir=args.logo_dir)
    if any("res_acf1" in df.columns for df in models.values()):
        _overlay_metric(models, "res_acf1", "Residual ACF(1)", 
                       os.path.join(out_appx, "figA5_residual_acf1.png"),
                       logo_dir=args.logo_dir)

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
        logo_dir=args.logo_dir,
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
            logo_dir=args.logo_dir,
        )

    print(f"✨ [OK] Wrote main figures → {out_main}")
    print(f"✨ [OK] Wrote appendix figures/tables → {out_appx}")

if __name__ == "__main__":
    main()
    