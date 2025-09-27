# mathbode/plot_panels.py
# Generates publication-ready overlay plots from summary CSV/Parquet files.
# Outputs two bundles: <outdir>/main and <outdir>/appendix

import os, glob, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- Styling -----------------------------

# Ultra-modern aesthetic with clean gradients
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": 600,
    "axes.labelsize": 12,
    "axes.labelweight": 500,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.03,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.linewidth": 0,
    "axes.edgecolor": "none",
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
})

# Premium color palette with vibrant gradients
COLORS = [
    "#FF6B6B",  # Coral red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Sky blue
    "#96CEB4",  # Sage green
    "#FECA57",  # Golden yellow
    "#9B59B6",  # Purple
    "#FF8B94",  # Pink
    "#74B9FF",  # Light blue
]

# Gradient color pairs for fills
GRADIENT_PAIRS = [
    ("#FF6B6B", "#FF8E53"),
    ("#667EEA", "#764BA2"),
    ("#06BEB6", "#48B1BF"),
    ("#F093FB", "#F5576C"),
    ("#4FACFE", "#00F2FE"),
    ("#43E97B", "#38F9D7"),
    ("#FA709A", "#FEE140"),
    ("#30CFD0", "#330867"),
]

MARKERS = ["o", "s", "D", "^", "v", "p", "h", "*", "X"]
LINEWIDTH = 2.8
MS = 8
MARKER_EDGE_WIDTH = 2
MARKER_EDGE_COLOR = "white"

# Enhanced legend styling with gradient background
LEGEND_KW = dict(
    # Keep it centered, but give it an 80% wide anchor box:
    # (left=0.10, y=-0.14, width=0.80, height ignored)
    loc="upper center",
    bbox_to_anchor=(0.10, -0.14, 0.80, 0.0),
    mode="expand",          # spread entries to fill the anchor box width
    ncol=3,
    frameon=True,
    fancybox=True,
    shadow=False,
    borderpad=1.2,
    columnspacing=2.8,
    handlelength=3.0,
    facecolor="white",
    edgecolor="none",
    framealpha=1.0,
    borderaxespad=0.6,
)

# Gradient background colors
BG_GRADIENT_START = "#FFFFFF"
BG_GRADIENT_END = "#F8F9FA"

# Families in desired order
FAMILY_ORDER = [
    "linear_solve",
    "ratio_saturation",
    "exponential_interest",
    "linear_system",
    "similar_triangles",
]

# Pretty names with better typography
FAMILY_PRETTY = {
    "linear_solve": "Linear Solve",
    "ratio_saturation": "Ratio Saturation",
    "exponential_interest": "Exponential Interest",
    "linear_system": "Linear System",
    "similar_triangles": "Similar Triangles",
}

# Model pretty-name overrides
MODEL_PRETTY_OVERRIDES = {
    r"openai.*gpt[-_]?4o": "GPT-4o",
    r"meta.*llama.*": "Llama 4 Instruct",
    r"deepseek.*v3\.?1": "DeepSeek V3.1",
    r"mistral.*mixtral.*8x7b": "Mixtral 8×7B",
    r"qwen.*235b.*a22b.*instruct": "Qwen3 235B",
    r"qwen.*": "Qwen3",
}

# -------------------------- IO + Harmonization --------------------------

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported summary format: {path}")

def _guess_model_name(path: str) -> str:
    stem = os.path.basename(path)
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

    # Expose h2_over_h1 if the file uses _model suffix
    if "h2_over_h1" not in df.columns and "h2_over_h1_model" in df.columns:
        df = df.rename(columns={"h2_over_h1_model": "h2_over_h1"})

    # Phase column harmonization
    if "phase_deg" not in df.columns:
        for alt in ("phase_err_deg", "phase_deg_model_minus_truth"):
            if alt in df.columns:
                df = df.rename(columns={alt: "phase_deg"})
                break

    # Types
    if "frequency_cycles" in df.columns:
        df["frequency_cycles"] = pd.to_numeric(df["frequency_cycles"], errors="coerce").astype("Int64")
    for c in ("gain","phase_deg","r2_model","compliance_rate","h2_over_h1","res_rms_norm","res_acf1"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "family" in df.columns:
        df["family"] = df["family"].astype(str)

    return df

def load_summaries(summaries_dir: str) -> Dict[str, pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(summaries_dir, "summary*.csv"))) \
          + sorted(glob.glob(os.path.join(summaries_dir, "summary*.parquet"))) \
          + sorted(glob.glob(os.path.join(summaries_dir, "*.csv")))
    models: Dict[str, pd.DataFrame] = {}
    for p in paths:
        try:
            df = _read_any(p)
        except Exception:
            continue
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

def _add_gradient_background(ax):
    """Add a subtle gradient background to the plot"""
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    gradient = np.hstack((gradient, gradient))
    
    extent = [ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]]
    ax.imshow(gradient, extent=extent, aspect='auto', cmap='gray_r', 
              alpha=0.03, zorder=-10, interpolation='bilinear')

def _add_subtle_band(ax, y0, y1, color="#E8F4FD", alpha=0.3):
    """Add a subtle shaded band with soft edges"""
    rect = FancyBboxPatch((ax.get_xlim()[0], y0), 
                          ax.get_xlim()[1] - ax.get_xlim()[0], 
                          y1 - y0,
                          boxstyle="round,pad=0", 
                          facecolor=color,
                          edgecolor='none',
                          alpha=alpha,
                          zorder=0,
                          transform=ax.transData)
    ax.add_patch(rect)

def _prep_family_axes(ncols: int, nrows: int = 1, figsize=(15, 4.2)):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False)
    axs = np.array(axs).reshape(nrows, ncols)
    fig.patch.set_facecolor('#FAFAFA')
    return fig, axs

def _family_iter():
    for fam in FAMILY_ORDER:
        yield fam, FAMILY_PRETTY.get(fam, fam.replace("_", " ").title())

def _collect_by_family(models: Dict[str, pd.DataFrame], metric: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for fam, _ in _family_iter():
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

def _create_premium_legend(fig, handles, labels):
    """Create a premium-looking legend with better styling"""
    legend = fig.legend(handles, labels, **LEGEND_KW)
    
    # Style the legend text
    for text in legend.get_texts():
        text.set_fontweight(500)
        text.set_fontsize(27)
    
    # Add subtle shadow to legend box
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
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

    # Layout with adjusted spacing
    if n == 5:
        nrows, ncols, figsize = 1, 5, (22, 5.5)
    elif n <= 3:
        nrows, ncols, figsize = 1, n, (8*n, 5.5)
    else:
        nrows, ncols, figsize = 2, math.ceil(n/2), (8*ncols, 11)

    fig, axs = _prep_family_axes(ncols=ncols, nrows=nrows, figsize=figsize)
    axs = axs.flatten()

    model_list = sorted(models.keys())
    
    for i, fam in enumerate(fams):
        ax = axs[i]
        fam_pretty = FAMILY_PRETTY.get(fam, fam)
        
        # Add gradient background
        _add_gradient_background(ax)
        
        # Add subtle guidance bands based on metric
        if metric == "gain":
            _add_subtle_band(ax, 0.9, 1.1, "#E8F4FD", 0.15)
        elif metric == "phase_deg":
            _add_subtle_band(ax, -5, 5, "#FFF4E6", 0.12)
        elif metric == "r2_model":
            _add_subtle_band(ax, 0.95, 1.0, "#E8F5E9", 0.15)
        elif metric == "res_acf1":
            _add_subtle_band(ax, -0.1, 0.1, "#FCE4EC", 0.10)

        marker_i = 0
        for k, model in enumerate(model_list):
            dfm = fam_to_models.get(fam, {}).get(model, None)
            if dfm is None or dfm.empty:
                continue
            
            g = dfm.groupby("frequency_cycles")[metric]
            xs = g.mean().index.to_numpy(dtype=float)
            ys = g.mean().to_numpy(dtype=float)
            es = g.apply(_safe_sem).to_numpy(dtype=float)

            color = COLORS[marker_i % len(COLORS)]
            marker = MARKERS[marker_i % len(MARKERS)]
            
            # Main line with enhanced styling
            line = ax.plot(xs, ys,
                    marker=marker,
                    color=color,
                    linewidth=LINEWIDTH,
                    markersize=MS,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    markeredgecolor=MARKER_EDGE_COLOR,
                    markerfacecolor=color,
                    label=model,
                    alpha=0.95,
                    zorder=20 + marker_i,
                    solid_capstyle='round',
                    solid_joinstyle='round')
            
            # Add subtle line glow effect
            ax.plot(xs, ys,
                   color=color,
                   linewidth=LINEWIDTH + 3,
                   alpha=0.15,
                   zorder=19 + marker_i)
            
            # Error bands with gradient effect
            ax.fill_between(xs, ys-es, ys+es,
                          color=color,
                          alpha=0.12,
                          zorder=10 + marker_i,
                          edgecolor='none',
                          interpolate=True)
            
            marker_i += 1

        # Reference line if specified
        if hline is not None:
            ax.axhline(hline, color="#B0B0B0", linestyle="--", 
                      linewidth=1.2, alpha=0.5, zorder=5, label='_nolegend_')

        # Enhanced title styling
        ax.set_title(fam_pretty, pad=12, fontsize=14, fontweight=600, color="#2C3E50")
        
        # Axis labels with better typography
        ax.set_xlabel("Frequency (cycles / 64 steps)", fontsize=11, fontweight=500, color="#546E7A")
        ax.set_ylabel(ylabel, fontsize=11, fontweight=500, color="#546E7A")
        
        # Grid styling
        ax.grid(True, alpha=0.08, linewidth=0.5, color='#E0E0E0')
        ax.set_axisbelow(True)
        
        # Remove all spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Style tick labels
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#607D8B', 
                      length=0, pad=8)
        
        if ylim:
            ax.set_ylim(*ylim)
        
        if as_integer_xticks:
            xt = sorted(dfm["frequency_cycles"].dropna().unique()) if dfm is not None else [1,2,4,8,16]
            ax.set_xticks(list(map(int, xt)))
            ax.set_xticklabels(list(map(str, map(int, xt))), fontweight=500)

    # Hide unused subplots
    for j in range(n, len(axs)):
        axs[j].set_visible(False)

    # Premium legend
    handles, labels = axs[0].get_legend_handles_labels()
    if len(handles):
        _create_premium_legend(fig, handles, labels)

    # Title with enhanced styling
    if title_suffix:
        fig.suptitle(title_suffix, y=1.03, fontsize=17, fontweight=600, color="#1A237E")

    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", facecolor='#FAFAFA', 
                   edgecolor='none', pad_inches=0.3)
    plt.close(fig)

# --------------------------- Appendix helpers ---------------------------

def _midband_table(models: Dict[str, pd.DataFrame], out_path: str):
    """Save a premium-styled mid-band table"""
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

    fig, axs = plt.subplots(2, 1, figsize=(min(15, 6 + 1.3*len(tb.columns)), 7))
    fig.patch.set_facecolor('#FAFAFA')

    for ax, title, data in zip(
        axs,
        ["Mean |G−1| at Mid-Frequencies (4 & 8 cycles)", 
         "Mean |Phase Error| at Mid-Frequencies (4 & 8 cycles)"],
        [tb, tb2]
    ):
        ax.axis("off")
        
        # Create gradient-styled cells
        cell_colors = []
        for _ in range(len(data.index)):
            row_colors = ['#FFFFFF'] * len(data.columns)
            cell_colors.append(row_colors)
        
        tbl = ax.table(cellText=data.values,
                       rowLabels=data.index,
                       colLabels=data.columns,
                       cellLoc="center",
                       loc="center",
                       cellColours=cell_colors,
                       colColours=['#E3F2FD'] * len(data.columns),
                       rowColours=['#E3F2FD'] * len(data.index))
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.3, 1.8)
        
        # Premium table styling
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(0.5)
            if i == 0 or j == -1:
                cell.set_text_props(weight=600, color='#1A237E')
                cell.set_facecolor('#E3F2FD')
            else:
                cell.set_text_props(weight=400)
        
        ax.set_title(title, pad=20, fontsize=14, fontweight=600, color='#1A237E')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor='#FAFAFA', edgecolor='none')
    plt.close(fig)

def _compliance_panels(models: Dict[str, pd.DataFrame], out_path: str):
    """Create premium bar charts for compliance rates"""
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

    fig, axs = plt.subplots(1, len(fams), figsize=(22, 5.5), sharey=True)
    fig.patch.set_facecolor('#FAFAFA')
    
    if len(fams) == 1:
        axs = [axs]
    
    x = np.arange(len(model_list))

    for i, (fam, ax) in enumerate(zip(fams, axs)):
        _add_gradient_background(ax)
        
        for j, model in enumerate(model_list):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            
            color = COLORS[j % len(COLORS)]
            
            # Main bar with gradient effect
            bar = ax.bar(j, v, width=0.65, color=color, alpha=0.85,
                        edgecolor='white', linewidth=2.5, zorder=10)
            
            # Add glow effect
            ax.bar(j, v, width=0.68, color=color, alpha=0.2, 
                  edgecolor='none', zorder=5)
            
            # Value label on top
            if v > 0:
                ax.text(j, v + 0.02, f'{v:.0%}', ha='center', va='bottom',
                       fontsize=10, fontweight=600, color=color)
        
        ax.set_title(FAMILY_PRETTY.get(fam, fam), fontweight=600, pad=12, 
                    fontsize=13, color='#2C3E50')
        ax.set_xticks(x)
        ax.set_xticklabels(model_list, rotation=30, ha="right", fontweight=500, 
                          fontsize=10, color='#546E7A')
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Compliance Rate" if i == 0 else "", fontweight=500, 
                     fontsize=11, color='#546E7A')
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(True, alpha=0.08, axis='y', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, pad=8, colors='#607D8B')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor='#FAFAFA', edgecolor='none')
    plt.close(fig)

def _overlay_metric(models: Dict[str, pd.DataFrame], metric: str, ylabel: str, 
                   out_path: str, hline=None, ylim=None, logo_dir=None):
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

    # 2) Phase Error vs Frequency (ALL families)
    _plot_overlay_per_family(
        models=models,
        metric="phase_deg",
        ylabel="Phase Error (deg)",
        hline=0.0,
        ylim=None,
        out_path=os.path.join(out_main, "fig2_phase_error_vs_frequency.png"),
        title_suffix="Phase Error vs Frequency",
        panel_subset=None,
        logo_dir=args.logo_dir,
    )

    # 3) Residual ACF(1) vs Frequency (ALL families)
    if any("res_acf1" in df.columns for df in models.values()):
        _plot_overlay_per_family(
            models=models,
            metric="res_acf1",
            ylabel="Residual ACF(1)",
            hline=0.0,
            ylim=None,
            out_path=os.path.join(out_main, "fig3_residual_acf1_vs_frequency.png"),
            title_suffix="Residual Autocorrelation vs Frequency",
            panel_subset=None,
            logo_dir=args.logo_dir,
        )

    # 4) R² vs Frequency (all families)
    if any("r2_model" in df.columns for df in models.values()):
        _plot_overlay_per_family(
            models=models,
            metric="r2_model",
            ylabel="Fit Quality (R²)",
            hline=1.0,
            ylim=(0.0, 1.02),
            out_path=os.path.join(out_main, "fig4_r2_vs_frequency.png"),
            title_suffix="Model Fit Quality vs Frequency",
            logo_dir=args.logo_dir,
        )

    # ------------------ APPENDIX ------------------

    # A1) Mid-band summary table
    _midband_table(models, os.path.join(out_appx, "tabA1_midband_summary.png"))

    # A2) Compliance bars per family
    if any("compliance_rate" in df.columns for df in models.values()):
        _compliance_panels(models, os.path.join(out_appx, "figA2_compliance_by_family.png"))

    # A3) Nonlinearity proxy (H2/H1)
    if any("h2_over_h1" in df.columns for df in models.values()):
        _overlay_metric(models, "h2_over_h1", "H₂/H₁ Ratio",
                        os.path.join(out_appx, "figA3_h2_over_h1_vs_frequency.png"),
                        logo_dir=args.logo_dir)

    # A4) Residual RMS
    if any("res_rms_norm" in df.columns for df in models.values()):
        _overlay_metric(models, "res_rms_norm", "Residual RMS (normalized)",
                        os.path.join(out_appx, "figA4_residual_rms.png"),
                        logo_dir=args.logo_dir)

    # A5) Residual ACF(1) appendix version
    if any("res_acf1" in df.columns for df in models.values()):
        _overlay_metric(models, "res_acf1", "Residual ACF(1)",
                        os.path.join(out_appx, "figA5_residual_acf1.png"),
                        logo_dir=args.logo_dir)

    # A6) Geometry-only focus
    _plot_overlay_per_family(
        models=models,
        metric="gain",
        ylabel="Gain",
        hline=1.0,
        out_path=os.path.join(out_appx, "figA6_geometry_gain.png"),
        title_suffix="Similar Triangles • Gain Analysis",
        panel_subset=["similar_triangles"],
        logo_dir=args.logo_dir,
    )
    
    if any("r2_model" in df.columns for df in models.values()):
        _plot_overlay_per_family(
            models=models,
            metric="r2_model",
            ylabel="Fit Quality (R²)",
            hline=1.0,
            ylim=(0.98, 1.001),
            out_path=os.path.join(out_appx, "figA6_geometry_r2.png"),
            title_suffix="Similar Triangles • Fit Quality Analysis",
            panel_subset=["similar_triangles"],
            logo_dir=args.logo_dir,
        )

    print(f"✨ [OK] Wrote main figures → {out_main}")
    print(f"✨ [OK] Wrote appendix figures/tables → {out_appx}")

if __name__ == "__main__":
    main()
    