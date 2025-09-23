# mathbode/data.py
from typing import List, Optional
import pandas as pd
from datasets import load_dataset

ALL_FAMILIES=["linear_solve","ratio_saturation","exponential_interest","linear_system","similar_triangles"]

def load_mathbode(families:List[str], max_rows:Optional[int]=None)->pd.DataFrame:
    frames=[]
    for fam in families:
        ds=load_dataset("cognitive-metrology-lab/MathBode", fam)
        df=pd.DataFrame(ds["train"]); df["family"]=fam
        frames.append(df)
    full=pd.concat(frames, ignore_index=True).reset_index(drop=True)
    full["row_id"]=full.index  # stable id for checkpoints
    if max_rows: full=full.head(max_rows)
    return full

def stratified_subset(df:pd.DataFrame, frequencies:List[int], phases:List[int], max_sweeps_per_freq:int)->pd.DataFrame:
    import numpy as np
    # Filter first
    df = df[df["frequency_cycles"].isin(frequencies)]
    df = df[df["phase_deg"].isin(phases)]
    
    # Reset index to ensure we have clean 0-based indexing
    df = df.reset_index(drop=True)
    
    keep_idx = []
    for (fam, freq), g in df.groupby(["family","frequency_cycles"]):
        # Reset index again for the group to ensure proper indexing
        g = g.reset_index(drop=True)
        keys = g[["question_id", "phase_deg", "amplitude_scale"]].drop_duplicates()
        if len(keys) == 0:
            continue
            
        keys = keys.sample(n=min(max_sweeps_per_freq, len(keys)), random_state=42)
        # Use indicator=True to see which rows are being kept
        merged = g.merge(keys, on=["question_id", "phase_deg", "amplitude_scale"], how="inner")
        keep_idx.extend(merged.index.tolist())
    
    if not keep_idx:
        raise ValueError("No matching rows found for the given frequencies and phases")
        
    out = df.iloc[sorted(set(keep_idx))].copy().reset_index(drop=True)
    return out
