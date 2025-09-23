#!/usr/bin/env bash
set -euo pipefail

# Make local 'mathbode' package importable (expects mathbode/{__init__.py,clients,utils,data,infer,summarize,plot_curves}.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# =======================
# Config: models & presets
# =======================

# Edit these to taste (comment out OR leave arrays empty to skip a provider)
OPENAI_MODELS=()                         # e.g., ("gpt-4o-mini")
GEMINI_MODELS=("gemini-2.5-flash")       # example
ANTHROPIC_MODELS=()                      # e.g., ("claude-3-7-sonnet-20250219")

# Presets:
#   SMOKE    : ~5.1k (freqs 4,8; phase 0; 2 sweeps)
#   MVP      : ~7.7k (freqs 4,8,16; phase 0; 2 sweeps)
#   MVP_PLUS : ~13.4k (freqs 1,2,4,8,16; phase 0 for {1,2,16}, tri-phase for {4,8}; 2 sweeps)
#   FULL     : ~38.4k (freqs 1,2,4,8,16; tri-phase all; 2 sweeps)
CONFIG="${CONFIG:-MVP}"  # override: CONFIG=MVP_PLUS ./run_matrix.sh

OUTDIR="${OUTDIR:-results}"
WORKERS="${WORKERS:-4}"
TEMP="${TEMP:-0.0}"
MAXTOK="${MAXTOK:-32}"
API_BASE="${API_BASE:-}"     # optional OpenAI-compatible base URL

# Families to include (all 5 by default)
FAMILIES=("linear_solve" "ratio_saturation" "exponential_interest" "linear_system" "similar_triangles")

# =======================
# Derived parameters
# =======================
MB_MODE="STANDARD"
case "$CONFIG" in
  SMOKE)
    FREQS=(4 8)
    PHASES=(0)
    SWEEPS=2
    ;;
  MVP)
    FREQS=(4 8 16)
    PHASES=(0)
    SWEEPS=2
    ;;
  MVP_PLUS)
    # Frequencies like FULL, but custom per-frequency phase selection (done in Python below)
    FREQS=(1 2 4 8 16)
    PHASES=(0)    # placeholder (actual per-freq phases handled in Python)
    SWEEPS=2
    MB_MODE="MVP_PLUS"
    ;;
  FULL)
    FREQS=(1 2 4 8 16)
    PHASES=(0 120 240)
    SWEEPS=2
    ;;
  *)
    echo "Unknown CONFIG='$CONFIG' (use SMOKE|MVP|MVP_PLUS|FULL)"; exit 1;;
esac

mkdir -p "$OUTDIR" cache

# Helpful print
echo "=== MathBode Run Matrix ==="
echo "CONFIG: $CONFIG"
echo "Families: ${FAMILIES[*]}"
echo "Freqs: ${FREQS[*]}   Phases: ${PHASES[*]}   Sweeps/freq: $SWEEPS"
echo "Workers: $WORKERS  Temp: $TEMP  MaxTokens: $MAXTOK"
echo "Mode: $MB_MODE"
echo "Outdir: $OUTDIR"
echo

# =======================
# Step 1: Build a shared subset (row_ids) so all models see the exact same rows
# =======================
SUBTAG="f$(IFS=-; echo "${FAMILIES[*]}")__fr$(IFS=-; echo "${FREQS[*]}")__ph$(IFS=-; echo "${PHASES[*]}")__k${SWEEPS}__${MB_MODE}"
ROWID_FILE="cache/row_ids__${SUBTAG}.txt"

# Export array params for the Python block (must be BEFORE we call python)
export MB_FAMILIES="$(IFS=,; echo "${FAMILIES[*]}")"
export MB_FREQS="$(IFS=,; echo "${FREQS[*]}")"
export MB_PHASES="$(IFS=,; echo "${PHASES[*]}")"
export MB_SWEEPS="$SWEEPS"
export MB_ROWIDS="$ROWID_FILE"
export MB_MODE

if [[ ! -f "$ROWID_FILE" ]]; then
  echo "Generating shared subset row_ids → $ROWID_FILE"
  python - <<'PY'
import os, random
import pandas as pd
from mathbode.data import load_mathbode

families = os.environ["MB_FAMILIES"].split(",")
freqs    = [int(x) for x in os.environ["MB_FREQS"].split(",")]
# PHASES env is only used in STANDARD/FULL/SMOKE/MVP; MVP_PLUS does custom mapping
phases_env = [int(x) for x in os.environ["MB_PHASES"].split(",")]
k        = int(os.environ["MB_SWEEPS"])
rowid_path = os.environ["MB_ROWIDS"]
mode = os.environ.get("MB_MODE","STANDARD")

random.seed(42)

df = load_mathbode(families)

row_ids = []

if mode != "MVP_PLUS":
    # Standard behavior: same phases for all freqs
    df = df[df["frequency_cycles"].isin(freqs)]
    df = df[df["phase_deg"].isin(phases_env)]
    # sample K sweeps per (family, freq, phase)
    for (fam, f, ph), g in df.groupby(["family","frequency_cycles","phase_deg"]):
        keys = g[["question_id","amplitude_scale"]].drop_duplicates()
        keys = keys.sample(n=min(k, len(keys)), random_state=42)
        merged = g.merge(keys, on=["question_id","amplitude_scale"], how="inner")
        row_ids.extend(merged["row_id"].tolist())
else:
    # MVP_PLUS logic:
    # - For freqs {4,8}: phases {0,120,240}
    # - For freqs {1,2,16}: phase {0}
    tri = {4,8}
    for fam in df["family"].unique():
        fam_df = df[df["family"]==fam]
        for f in freqs:
            allowed_phases = [0,120,240] if f in tri else [0]
            sub = fam_df[(fam_df["frequency_cycles"]==f) & (fam_df["phase_deg"].isin(allowed_phases))]
            for ph, g in sub.groupby("phase_deg"):
                # sample K sweeps per (fam,f,ph) based on unique (question_id, amplitude_scale)
                keys = g[["question_id","amplitude_scale"]].drop_duplicates()
                keys = keys.sample(n=min(k, len(keys)), random_state=42)
                merged = g.merge(keys, on=["question_id","amplitude_scale"], how="inner")
                row_ids.extend(merged["row_id"].tolist())

row_ids = sorted(set(row_ids))
with open(rowid_path, "w") as f:
    for rid in row_ids:
        f.write(str(rid) + "\n")

print(f"Wrote {len(row_ids)} row_ids to {rowid_path}")
PY
else
  echo "Using existing subset row_ids → $ROWID_FILE"
fi

# =======================
# Step 2: function to run one model (inference -> summarize -> plots); resumes automatically
# =======================
run_one_model () {
  local provider="$1"
  local model="$2"
  local outdir="$3"
  local workers="$4"
  local temp="$5"
  local maxtok="$6"
  local api_base="$7"

  echo
  echo ">>> $provider :: $model"
  # Sanity: keys
  if [[ "$provider" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "   [SKIP] Missing OPENAI_API_KEY"; return 0
  fi
  if [[ "$provider" == "gemini" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "   [SKIP] Missing GOOGLE_API_KEY"; return 0
  fi
  if [[ "$provider" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "   [SKIP] Missing ANTHROPIC_API_KEY"; return 0
  fi

  # Run inference on the fixed subset of row_ids, with resume
  python - <<'PY' || { echo "   [ERROR] Inference failed"; return 1; }
import os, pandas as pd
from mathbode.data import load_mathbode
from mathbode.infer import run_inference

provider = os.environ["MB_PROVIDER"]
model    = os.environ["MB_MODEL"]
outdir   = os.environ["MB_OUTDIR"]
workers  = int(os.environ["MB_WORKERS"])
temp     = float(os.environ["MB_TEMP"])
maxtok   = int(os.environ["MB_MAXTOK"])
api_base = os.environ.get("MB_API_BASE") or None
rowid_path = os.environ["MB_ROWIDS"]
families = os.environ["MB_FAMILIES"].split(",")

# load all, then filter to subset row_ids
df = load_mathbode(families)
row_ids = [int(x.strip()) for x in open(rowid_path).read().splitlines() if x.strip()]
df = df[df["row_id"].isin(row_ids)].copy().reset_index(drop=True)

# Run (has checkpointed resume)
pred_path = run_inference(
    df=df,
    provider=provider,
    model=model,
    outdir=outdir,
    temperature=temp,
    max_tokens=maxtok,
    api_base=api_base,
    workers=workers,
    provider_rps=0.0
)
print("Preds:", pred_path)
PY

  # Summarize & plot
  python - <<'PY' || { echo "   [ERROR] Summarize failed"; return 1; }
import os, pandas as pd
from mathbode.summarize import summarize_gain_phase
from mathbode.plot_curves import plot_curves

provider = os.environ["MB_PROVIDER"]
model    = os.environ["MB_MODEL"]
outdir   = os.environ["MB_OUTDIR"]
tag = f"{provider}_{model}".replace("/","_")
pred_path = os.path.join(outdir, f"preds_{tag}.parquet")

preds = pd.read_parquet(pred_path)
summary = summarize_gain_phase(preds)
sum_path = os.path.join(outdir, f"summary_{tag}.csv")
summary.to_csv(sum_path, index=False)
print("Summary:", sum_path)

# Mid-band quick stats (still use 4 & 8)
mid = summary[summary["frequency_cycles"].isin([4,8])]
print("mean |G-1|:", (mid["gain"]-1).abs().mean())
print("mean |phi| (deg):", mid["phase_deg"].abs().mean())

plot_curves(summary, tag, outdir)
print("Plots saved for", tag)
PY
}

# =======================
# Step 3: loop over model matrices
# =======================
# Common env for the Python blocks
export MB_OUTDIR="$OUTDIR"
export MB_WORKERS="$WORKERS"
export MB_TEMP="$TEMP"
export MB_MAXTOK="$MAXTOK"
export MB_API_BASE="$API_BASE"

# OpenAI
if (( ${#OPENAI_MODELS[@]} )); then
  for m in "${OPENAI_MODELS[@]}"; do
    export MB_PROVIDER="openai"
    export MB_MODEL="$m"
    run_one_model "openai" "$m" "$OUTDIR" "$WORKERS" "$TEMP" "$MAXTOK" "$API_BASE" || true
  done
fi

# Gemini
if (( ${#GEMINI_MODELS[@]} )); then
  for m in "${GEMINI_MODELS[@]}"; do
    export MB_PROVIDER="gemini"
    export MB_MODEL="$m"
    run_one_model "gemini" "$m" "$OUTDIR" "$WORKERS" "$TEMP" "$MAXTOK" "" || true
  done
fi

# Anthropic
if (( ${#ANTHROPIC_MODELS[@]} )); then
  for m in "${ANTHROPIC_MODELS[@]}"; do
    export MB_PROVIDER="anthropic"
    export MB_MODEL="$m"
    run_one_model "anthropic" "$m" "$OUTDIR" "$WORKERS" "$TEMP" "$MAXTOK" "" || true
  done
fi

echo
echo "✅ Done. Results in: $OUTDIR"
