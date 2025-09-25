# mathbode/summarize.py
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

# ---------- Core helpers ----------

def wrap_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi

def _fit_sin_cos(t: np.ndarray, y: np.ndarray, omega: float) -> Optional[Dict[str, float]]:
    """
    OLS fit onto [1, sin(ωt), cos(ωt)]; returns amplitude A, phase phi, and R^2.
    Requires at least ~16 valid points.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 16:
        return None
    X = np.c_[np.ones(mask.sum()), np.sin(omega * t[mask]), np.cos(omega * t[mask])]
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    b0, bs, bc = map(float, beta)
    A = math.hypot(bs, bc)
    phi = math.atan2(bc, bs)
    yhat = X @ beta
    ss_res = float(np.sum((y[mask] - yhat) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return dict(A=A, phi=phi, r2=r2)

def _detect_T(time_steps: np.ndarray) -> int:
    """
    Detect steps per sweep T from a group's time_step values.
    Assumes each (freq, phase) subgroup has the same count; use the modal count.
    """
    ts = np.asarray(time_steps)
    # Count unique steps per (freq, phase) inside the group after we split by them.
    # Here, caller ensures we pass a single (freq,phase) slice or provides per-slice counts.
    # As a fallback, use number of unique time steps in this array.
    return int(pd.Series(ts).nunique())

def _to_float(arr: pd.Series) -> np.ndarray:
    """Best-effort numeric conversion; non-parsable -> NaN."""
    # If y_hat already numeric, astype will be fast; otherwise coerce
    try:
        return arr.astype(float).to_numpy()
    except Exception:
        return pd.to_numeric(arr, errors="coerce").to_numpy()

# ---------- Public API ----------

def summarize_gain_phase_all(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Return ALL results, one row per (family, frequency_cycles, question_id, amplitude_scale?, phase_deg).
    Columns:
      family, frequency_cycles, phase_deg, question_id, amplitude_scale (if present),
      gain, phase_deg, A_truth, A_model, r2_truth, r2_model,
      n_total, n_valid, compliance_rate, steps_per_sweep
    """
    need = {"family", "frequency_cycles", "phase_deg", "time_step", "ground_truth", "y_hat", "question_id"}
    missing = need - set(preds.columns)
    if missing:
        raise ValueError(f"summarize_gain_phase_all: preds missing columns: {sorted(missing)}")

    df = preds.copy()

    # normalize dtypes
    df["frequency_cycles"] = df["frequency_cycles"].astype(int)
    df["phase_deg"] = df["phase_deg"].astype(int)

    has_amp = "amplitude_scale" in df.columns
    group_cols = ["family", "frequency_cycles", "phase_deg", "question_id"] + (["amplitude_scale"] if has_amp else [])

    rows: List[Dict] = []

    # group per base key + phase + frequency
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("time_step")
        # detect T for this slice
        T = _detect_T(g["time_step"].to_numpy())
        if T < 8:
            continue  # too short to fit meaningfully

        freq = int(g["frequency_cycles"].iloc[0])
        omega = 2 * math.pi * freq / T

        t = g["time_step"].to_numpy()
        y_true = _to_float(g["ground_truth"])
        y_model = _to_float(g["y_hat"])

        # compliance: valid numeric predictions (finite y_hat)
        is_valid = np.isfinite(y_model)
        n_total = int(len(y_model))
        n_valid = int(np.count_nonzero(is_valid))
        compliance_rate = (n_valid / n_total) if n_total > 0 else 0.0

        ft = _fit_sin_cos(t, y_true, omega)
        fm = _fit_sin_cos(t, y_model, omega)

        # If the model series is too sparse, still record compliance but skip gain/phase
        if not ft or not fm or ft["A"] <= 0:
            out = dict(
                family=keys[0],
                frequency_cycles=freq,
                phase_deg=keys[2],
                question_id=keys[3 if not has_amp else 3],
                gain=float("nan"),
                phase_deg_model_minus_truth=float("nan"),
                A_truth=float("nan") if not ft else ft["A"],
                A_model=float("nan") if not fm else fm["A"],
                r2_truth=float("nan") if not ft else ft["r2"],
                r2_model=float("nan") if not fm else fm["r2"],
                n_total=n_total,
                n_valid=n_valid,
                compliance_rate=compliance_rate,
                steps_per_sweep=T,
            )
            if has_amp:
                out["amplitude_scale"] = keys[4]
            rows.append(out)
            continue

        gain = fm["A"] / ft["A"]
        dphi = wrap_pi(fm["phi"] - ft["phi"])

        out = dict(
            family=keys[0],
            frequency_cycles=freq,
            phase_deg=keys[2],
            question_id=keys[3 if not has_amp else 3],
            gain=gain,
            phase_deg_model_minus_truth=math.degrees(dphi),
            A_truth=ft["A"],
            A_model=fm["A"],
            r2_truth=ft["r2"],
            r2_model=fm["r2"],
            n_total=n_total,
            n_valid=n_valid,
            compliance_rate=compliance_rate,
            steps_per_sweep=T,
        )
        if has_amp:
            out["amplitude_scale"] = keys[4]
        rows.append(out)

    cols = [
        "family", "frequency_cycles", "phase_deg", "question_id"
    ] + (["amplitude_scale"] if has_amp else []) + [
        "gain", "phase_deg_model_minus_truth",
        "A_truth", "A_model", "r2_truth", "r2_model",
        "n_total", "n_valid", "compliance_rate",
        "steps_per_sweep",
    ]
    res = pd.DataFrame(rows)[cols].sort_values(
        ["family", "question_id"] + (["amplitude_scale"] if has_amp else []) + ["frequency_cycles", "phase_deg"]
    )
    return res

def summarize_gain_phase_means(all_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up per-(family, frequency) means (for error bands / quick curves),
    given the detailed output of summarize_gain_phase_all.
    """
    df = all_rows.copy()
    # harmonize column name for plotting compatibility
    if "phase_deg_model_minus_truth" in df.columns:
        df = df.rename(columns={"phase_deg_model_minus_truth": "phase_deg"})
    by = ["family", "frequency_cycles"]
    agg = {
        "gain": "mean",
        "phase_deg": "mean",
        "r2_model": "mean",
        "A_truth": "mean",
        "A_model": "mean",
        "compliance_rate": "mean",
    }
    out = df.groupby(by, dropna=False).agg(agg).reset_index()
    return out.sort_values(by)
