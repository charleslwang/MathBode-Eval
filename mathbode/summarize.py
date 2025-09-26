import argparse
import math
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ------------------------ Robust helpers ------------------------

_NUM_RE = re.compile(r"[-+]?(\d+(\.\d+)?|\.\d+)([eE][-+]?\d+)?")

def _extract_last_number(s: str) -> Optional[float]:
    if not isinstance(s, str):
        return None
    m = list(_NUM_RE.finditer(s))
    if not m:
        return None
    try:
        return float(m[-1].group(0))
    except Exception:
        return None

def _prefer_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _ensure_numeric_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    return x.apply(_extract_last_number)

def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi

def _fit_first_harmonic(t: np.ndarray, y: np.ndarray, omega: float) -> Optional[Dict[str, float]]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 16:
        return None
    X = np.c_[np.ones(mask.sum()), np.sin(omega * t[mask]), np.cos(omega * t[mask])]
    try:
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    except Exception:
        return None
    b0, bs, bc = map(float, beta)
    A = math.hypot(bs, bc)
    phi = math.atan2(bc, bs)
    yhat = X @ beta
    ss_res = float(np.sum((y[mask] - yhat) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return dict(A=A, phi=phi, r2=r2)

# NEW: small helper to get H1..HK amplitudes by least-squares projection
def _harmonics(t: np.ndarray, y: np.ndarray, omega: float, K: int = 3) -> Dict[str, float]:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 16:
        return {}
    out: Dict[str, float] = {}
    for k in range(1, K + 1):
        Xk = np.c_[np.sin(k * omega * t[m]), np.cos(k * omega * t[m])]
        try:
            beta, *_ = np.linalg.lstsq(Xk, y[m], rcond=None)
        except Exception:
            return {}
        bs, bc = map(float, beta)
        out[f"H{k}"] = float(np.hypot(bs, bc))
    return out

# ------------------------ Core summarize ------------------------

def _summarize_all_rows(enriched: pd.DataFrame) -> pd.DataFrame:
    need = {
        "row_id", "family", "frequency_cycles", "phase_deg",
        "time_step", "ground_truth", "y_hat", "question_id"
    }
    missing = need - set(enriched.columns)
    if missing:
        raise ValueError(f"[summarize] enriched missing columns: {sorted(missing)}")

    df = enriched.copy()

    for col in ["frequency_cycles", "phase_deg", "time_step"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["family", "frequency_cycles", "phase_deg", "time_step"])
    df["frequency_cycles"] = df["frequency_cycles"].astype(int)
    df["phase_deg"] = df["phase_deg"].astype(int)

    df["ground_truth"] = _ensure_numeric_series(df["ground_truth"])
    df["y_hat"] = _ensure_numeric_series(df["y_hat"])

    has_amp = "amplitude_scale" in df.columns
    group_cols = ["family", "frequency_cycles", "phase_deg", "question_id"] + (["amplitude_scale"] if has_amp else [])

    rows: List[Dict] = []

    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("time_step")
        T = int(pd.Series(g["time_step"].to_numpy()).nunique())
        freq = int(g["frequency_cycles"].iloc[0])

        t = g["time_step"].to_numpy(dtype=float)
        y_true = pd.to_numeric(g["ground_truth"], errors="coerce").to_numpy(dtype=float)
        y_model = pd.to_numeric(g["y_hat"], errors="coerce").to_numpy(dtype=float)

        is_valid = np.isfinite(y_model)
        n_total = int(len(y_model))
        n_valid = int(np.count_nonzero(is_valid))
        comp = (n_valid / n_total) if n_total > 0 else 0.0

        if T < 8 or np.isfinite(y_true).sum() < 16 or np.isfinite(y_model).sum() < 16:
            out = {
                "family": keys[0],
                "frequency_cycles": freq,
                "phase_deg": keys[2],
                "question_id": keys[3 if not has_amp else 3],
                "gain": float("nan"),
                "phase_deg_model_minus_truth": float("nan"),
                "A_truth": float("nan"),
                "A_model": float("nan"),
                "r2_model": float("nan"),
                "compliance_rate": comp,
                "steps_per_sweep": T,
                # NEW: keep schema stable with NaNs for new fields
                "h2_over_h1_model": float("nan"),
                "h3_over_h1_model": float("nan"),
                "nonlin_index_model": float("nan"),
                "h2_over_h1_truth": float("nan"),
                "res_rms_norm": float("nan"),
                "res_acf1": float("nan"),
            }
            if has_amp:
                out["amplitude_scale"] = keys[4]
            rows.append(out)
            continue

        omega = 2 * math.pi * freq / max(T, 1)

        ft = _fit_first_harmonic(t, y_true, omega)
        fm = _fit_first_harmonic(t, y_model, omega)

        # Prepare defaults for NEW metrics
        h2_over_h1_model = float("nan")
        h3_over_h1_model = float("nan")
        nonlin_index_model = float("nan")
        h2_over_h1_truth = float("nan")
        res_rms_norm = float("nan")
        res_acf1 = float("nan")

        if ft and fm and (ft["A"] > 0):
            # NEW: harmonic spectra (truth & model)
            spec_t = _harmonics(t, y_true, omega, K=3)
            spec_m = _harmonics(t, y_model, omega, K=3)
            H1t = spec_t.get("H1", np.nan)
            H2t = spec_t.get("H2", np.nan)
            H1m = spec_m.get("H1", np.nan)
            H2m = spec_m.get("H2", np.nan)
            H3m = spec_m.get("H3", np.nan)

            if np.isfinite(H1m) and H1m > 0 and np.isfinite(H2m):
                h2_over_h1_model = float(H2m / H1m)
            if np.isfinite(H1m) and H1m > 0 and np.isfinite(H3m):
                h3_over_h1_model = float(H3m / H1m)

            # NEW: simple nonlinearity index: 1 - fundamental energy / total (H1..H3)
            if np.isfinite(H1m):
                total_e = (H1m ** 2) + (0 if not np.isfinite(H2m) else H2m ** 2) + (0 if not np.isfinite(H3m) else H3m ** 2)
                if total_e > 0:
                    nonlin_index_model = float(1.0 - (H1m ** 2) / total_e)

            if np.isfinite(H1t) and H1t > 0 and np.isfinite(H2t):
                h2_over_h1_truth = float(H2t / H1t)

            # NEW: residual metrics against first-harmonic *shape* (demeaned)
            # Reconstruct the first-harmonic model (zero-mean) from (A,phi):
            A_m = float(fm["A"])
            phi_m = float(fm["phi"])
            y_model_dm = y_model - np.nanmean(y_model[np.isfinite(y_model)])
            yhat = A_m * np.sin(omega * t + phi_m)  # zero-mean harmonic approx
            res = y_model_dm - yhat
            res = res[np.isfinite(res)]
            if res.size:
                res_rms = float(np.sqrt(np.mean(res ** 2)))
                res_rms_norm = float(res_rms / (float(ft["A"]) if ft["A"] > 0 else np.nan))
                if res.size >= 3:
                    # lag-1 autocorrelation
                    res_acf1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])

        if not ft or not fm or (ft["A"] <= 0):
            out = {
                "family": keys[0],
                "frequency_cycles": freq,
                "phase_deg": keys[2],
                "question_id": keys[3 if not has_amp else 3],
                "gain": float("nan"),
                "phase_deg_model_minus_truth": float("nan"),
                "A_truth": float("nan") if not ft else float(ft["A"]),
                "A_model": float("nan") if not fm else float(fm["A"]),
                "r2_model": float("nan") if not fm else float(fm["r2"]),
                "compliance_rate": comp,
                "steps_per_sweep": T,
                # NEW fields (NaN if unavailable)
                "h2_over_h1_model": h2_over_h1_model,
                "h3_over_h1_model": h3_over_h1_model,
                "nonlin_index_model": nonlin_index_model,
                "h2_over_h1_truth": h2_over_h1_truth,
                "res_rms_norm": res_rms_norm,
                "res_acf1": res_acf1,
            }
            if has_amp:
                out["amplitude_scale"] = keys[4]
            rows.append(out)
            continue

        gain = float(fm["A"] / ft["A"])
        dphi = float(_wrap_pi(fm["phi"] - ft["phi"]))

        out = {
            "family": keys[0],
            "frequency_cycles": freq,
            "phase_deg": keys[2],
            "question_id": keys[3 if not has_amp else 3],
            "gain": gain,
            "phase_deg_model_minus_truth": math.degrees(dphi),
            "A_truth": float(ft["A"]),
            "A_model": float(fm["A"]),
            "r2_model": float(fm["r2"]),
            "compliance_rate": comp,
            "steps_per_sweep": T,
            # NEW metrics
            "h2_over_h1_model": h2_over_h1_model,
            "h3_over_h1_model": h3_over_h1_model,
            "nonlin_index_model": nonlin_index_model,
            "h2_over_h1_truth": h2_over_h1_truth,
            "res_rms_norm": res_rms_norm,
            "res_acf1": res_acf1,
        }
        if has_amp:
            out["amplitude_scale"] = keys[4]
        rows.append(out)

    cols = [
        "family", "frequency_cycles", "phase_deg", "question_id"
    ] + (["amplitude_scale"] if has_amp else []) + [
        "gain", "phase_deg_model_minus_truth",
        "A_truth", "A_model", "r2_model",
        "compliance_rate", "steps_per_sweep",
        # NEW columns in per-row table
        "h2_over_h1_model", "h3_over_h1_model", "nonlin_index_model",
        "h2_over_h1_truth", "res_rms_norm", "res_acf1",
    ]
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    cols = [c for c in cols if c in res.columns]
    return res[cols].sort_values(
        ["family", "question_id"] + (["amplitude_scale"] if "amplitude_scale" in res.columns else []) + ["frequency_cycles", "phase_deg"]
    )

def _summarize_means(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame(columns=["family", "frequency_cycles", "gain", "phase_deg", "r2_model", "compliance_rate"])

    df = all_rows.copy()

    # Keep start phase and phase error distinct
    if "phase_deg_model_minus_truth" in df.columns:
        df = df.rename(columns={"phase_deg_model_minus_truth": "phase_err_deg"})  # <- no collision

    # Coerce numeric where present (guard if dup labels ever appear)
    for c in [
        "gain", "phase_err_deg", "r2_model", "compliance_rate", "frequency_cycles",
        # NEW aggregated fields:
        "h2_over_h1_model", "h3_over_h1_model", "nonlin_index_model",
        "h2_over_h1_truth", "res_rms_norm", "res_acf1",
    ]:
        if c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):  # duplicate labels safeguard
                col = col.iloc[:, 0]
            df[c] = pd.to_numeric(col, errors="coerce")

    by = ["family", "frequency_cycles"]
    agg = {
        "gain": "mean",
        "r2_model": "mean",
        "compliance_rate": "mean",
        # NEW aggregated means:
        "h2_over_h1_model": "mean",
        "h3_over_h1_model": "mean",
        "nonlin_index_model": "mean",
        "h2_over_h1_truth": "mean",
        "res_rms_norm": "mean",
        "res_acf1": "mean",
    }
    if "phase_err_deg" in df.columns:
        agg["phase_err_deg"] = "mean"

    out = df.groupby(by, dropna=False).agg(agg).reset_index()
    out["frequency_cycles"] = out["frequency_cycles"].astype(int)

    # For backward-compat with plot_curves(), expose phase error under 'phase_deg'
    if "phase_err_deg" in out.columns:
        out = out.rename(columns={"phase_err_deg": "phase_deg"})

    return out.sort_values(by)


# ------------------------ IO & main ------------------------

def _derive_tag_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem.replace("preds_", "")

def main():
    ap = argparse.ArgumentParser(
        description="Summarize MathBode inference: one input parquet -> one summary CSV."
    )
    ap.add_argument("--inference", required=True, help="Parquet with predictions (must include row_id).")
    ap.add_argument("--outdir", required=True, help="Directory to write summary CSV.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    preds = pd.read_parquet(args.inference)
    if "row_id" not in preds.columns:
        raise ValueError("[summarize] Inference parquet must include 'row_id'.")

    pred_candidates = ["y_hat", "final_answer", "answer", "response", "output", "raw", "text", "message", "content"]
    pred_col = _prefer_column(preds, pred_candidates)
    if pred_col is None:
        raise ValueError(f"[summarize] Could not find a prediction column among: {pred_candidates}")
    preds_clean = preds[["row_id", pred_col]].rename(columns={pred_col: "y_hat"})

    # Load dataset rows for overlap (FALLBACK families list if ALL_FAMILIES not exported)
    try:
        from mathbode.data import load_mathbode, ALL_FAMILIES  # type: ignore
        families = list(ALL_FAMILIES) if isinstance(ALL_FAMILIES, (list, tuple, set)) and len(ALL_FAMILIES) else [
            "linear_solve", "ratio_saturation", "exponential_interest", "linear_system", "similar_triangles"
        ]
    except Exception:
        # absolute fallback
        from mathbode.data import load_mathbode  # type: ignore
        families = ["linear_solve", "ratio_saturation", "exponential_interest", "linear_system", "similar_triangles"]

    df_src = load_mathbode(families)
    row_ids = set(pd.to_numeric(preds_clean["row_id"], errors="coerce").dropna().astype(int).tolist())
    df_src = df_src[df_src["row_id"].isin(row_ids)].copy()
    if df_src.empty:
        raise RuntimeError("[summarize] No overlap between inference row_id and dataset; check inputs.")

    enriched = df_src.merge(preds_clean, on="row_id", how="left")

    all_rows = _summarize_all_rows(enriched)
    summary = _summarize_means(all_rows)

    tag = _derive_tag_from_path(args.inference)
    out_csv = os.path.join(args.outdir, f"summary_{tag}.csv")
    summary.to_csv(out_csv, index=False)

    # Diagnostics
    total_rows = len(enriched)
    have_pred = int(enriched["y_hat"].notna().sum())
    parsed_numeric = int(_ensure_numeric_series(enriched["y_hat"]).notna().sum())
    print(f"[OK] wrote summary CSV → {out_csv}")
    print(f"[diag] joined rows: {total_rows} | with any y_hat: {have_pred} | numeric-parsable y_hat: {parsed_numeric}")
    if summary[["gain", "phase_deg"]].isna().all().all():
        print("[diag] All gain/phase are NaN. Likely the predictions could not be parsed to numbers or series too sparse.")

if __name__ == "__main__":
    main()
