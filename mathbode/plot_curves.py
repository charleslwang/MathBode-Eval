# mathbode/plot_curves.py
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_curves(summary:pd.DataFrame, model_tag:str, outdir:str):
    os.makedirs(outdir, exist_ok=True)
    fams=summary["family"].unique().tolist()
    for metric,ylabel,fname in [("gain","Gain (A_model / A_truth)","gain"),
                                ("phase_deg","Phase (deg)","phase")]:
        plt.figure(figsize=(7,4))
        for fam in fams:
            sub=summary[summary["family"]==fam]
            xs=sub["frequency_cycles"].values
            ys=sub[metric].values
            plt.plot(xs, ys, marker="o", label=fam)
        plt.xlabel("Frequency (cycles / 256 steps)")
        plt.ylabel(ylabel)
        plt.title(f"{model_tag} • {ylabel} vs frequency")
        plt.legend(frameon=False)
        plt.tight_layout()
        path=os.path.join(outdir, f"{model_tag}_{fname}_curves.png")
        plt.savefig(path, dpi=160)
        plt.close()
