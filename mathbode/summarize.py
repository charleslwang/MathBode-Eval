# mathbode/summarize.py
import math
import numpy as np
import pandas as pd

def fit_sin_cos(t, y, omega):
    t=np.asarray(t,dtype=float); y=np.asarray(y,dtype=float)
    X=np.c_[np.ones_like(t), np.sin(omega*t), np.cos(omega*t)]
    mask=np.isfinite(y); X,y=X[mask],y[mask]
    if len(y)<16: return None
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    b0,bs,bc=beta
    A=math.hypot(bs,bc)
    phi=math.atan2(bc,bs)
    yhat=X@beta
    ss_res=float(np.sum((y-yhat)**2))
    ss_tot=float(np.sum((y-np.mean(y))**2)) or 1.0
    r2=1.0 - ss_res/ss_tot
    return dict(A=A,phi=phi,r2=r2)

def wrap_pi(x): return (x+math.pi)%(2*math.pi)-math.pi

def get_steps_per_sweep(time_steps):
    """Detect steps_per_sweep from time_steps by finding the period of the most common difference."""
    if len(time_steps) < 2:
        return 1
    diffs = np.diff(np.sort(time_steps))
    if len(diffs) == 0:
        return 1
    # Find the most common non-zero difference
    unique, counts = np.unique(diffs, return_counts=True)
    if len(unique) == 0:
        return 1
    most_common_diff = unique[counts.argmax()]
    if most_common_diff <= 0:
        return 1
    # Calculate steps_per_sweep as the period
    steps_per_sweep = 1.0 / (most_common_diff * 1e-9)  # Convert ns to seconds
    return max(1, int(round(steps_per_sweep)))

def summarize_gain_phase(preds:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for fam, g1 in preds.groupby("family"):
        for freq, g2 in g1.groupby("frequency_cycles"):
            t_truth=[]; y_truth=[]; y_model=[]
            all_time_steps = []
            
            # First pass: collect all time steps to detect steps_per_sweep
            for _, sweep in g2.groupby(["question_id","phase_deg","amplitude_scale"]):
                s = sweep.sort_values("time_step")
                all_time_steps.extend(s["time_step"].to_numpy())
            
            # Detect steps_per_sweep from the data
            steps_per_sweep = get_steps_per_sweep(all_time_steps) if all_time_steps else 1
            
            # Second pass: process the data with the detected steps_per_sweep
            for _, sweep in g2.groupby(["question_id","phase_deg","amplitude_scale"]):
                s = sweep.sort_values("time_step")
                t_truth.append(s["time_step"].to_numpy())
                y_truth.append(s["ground_truth"].to_numpy(dtype=float))
                y_model.append(s["y_hat"].to_numpy(dtype=float))
                
            if not t_truth: continue
            t_truth = np.concatenate(t_truth)
            y_truth = np.concatenate(y_truth)
            y_model = np.concatenate(y_model)
            if len(y_truth) < 32 or len(y_model) < 32: continue
            
            # Use detected steps_per_sweep for omega calculation
            omega = 2 * math.pi * int(freq) / steps_per_sweep
            ft=fit_sin_cos(t_truth, y_truth, omega)
            fm=fit_sin_cos(t_truth, y_model, omega)
            if not ft or not fm: continue
            gain=fm["A"]/ft["A"] if ft["A"]>0 else float("nan")
            dphi=wrap_pi(fm["phi"]-ft["phi"])
            rows.append({
                "family":fam,
                "frequency_cycles":int(freq),
                "gain":gain,
                "phase_deg":math.degrees(dphi),
                "r2_truth":ft["r2"],
                "r2_model":fm["r2"],
                "A_truth":ft["A"],
                "A_model":fm["A"],
                "steps_per_sweep": steps_per_sweep
            })
    return pd.DataFrame(rows).sort_values(["family","frequency_cycles"])
