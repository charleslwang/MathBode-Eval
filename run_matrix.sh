#!/usr/bin/env bash
set -euo pipefail

# Make local 'mathbode' package importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# =======================
# Config: models & presets
# =======================

# Edit these to taste (comment out arrays to skip a provider)
GEMINI_MODELS=()                                 # e.g. ("gemini-2.5-flash")
ANTHROPIC_MODELS=()                              # e.g. ("claude-3-7-sonnet-20250219")
TOGETHER_MODELS=("openai/gpt-oss-120b")
# TOGETHER_MODELS=("deepseek-ai/DeepSeek-V3.1")
# TOGETHER_MODELS=("meta-llama/Llama-4-Scout-17B-16E-Instruct")
OPENAI_MODELS=()                                 # e.g. ("gpt-4o-mini")

# Presets:
#   SMOKE    : freqs {4,8}, phase {0},    K=2 base keys/family
#   MVP      : freqs {4,8,16}, phase {0}, K=2
#   MVP_PLUS : freqs {1,2,4,8,16}, tri-phase only for {4,8}, K=2
#   FULL     : freqs {1,2,4,8,16}, tri-phase for all, K=2
CONFIG="${CONFIG:-MVP_PLUS}"              # override like: CONFIG=FULL ./run_matrix_infer_only.sh

OUTDIR="${OUTDIR:-results}"
WORKERS="${WORKERS:-4}"
TEMP="${TEMP:-0.0}"
MAXTOK=${MAXTOK:-1028}  # max tokens per response
API_BASE="${API_BASE:-}"                  # optional OpenAI-compatible base URL
SIGNAL="${SIGNAL:-sinusoid}"              # dataset drive type
SWEEPS="${SWEEPS:-2}"                     # number of base keys per family; or "ALL"
BASE_KEYS_FILE="${BASE_KEYS_FILE:-}"      # optional CSV: family,question_id,amplitude_scale[,p0]

# Families to include (all 5 by default)
FAMILIES=("linear_solve" "ratio_saturation" "exponential_interest" "linear_system" "similar_triangles")

# =======================
# Derived parameters
# =======================
MB_MODE="STANDARD"
case "$CONFIG" in
  SMOKE)     FREQS=(4 8);           PHASES=(0);           MB_MODE="SMOKE" ;;
  MVP)       FREQS=(4 8 16);        PHASES=(0);           MB_MODE="MVP" ;;
  MVP_PLUS)  FREQS=(1 2 4 8 16);    PHASES=(0);           MB_MODE="MVP_PLUS" ;;  # tri-phase handled in builder
  FULL)      FREQS=(1 2 4 8 16);    PHASES=(0 120 240);   MB_MODE="FULL" ;;
  *) echo "Unknown CONFIG='$CONFIG' (use SMOKE|MVP|MVP_PLUS|FULL)"; exit 1;;
esac

mkdir -p "$OUTDIR" cache subsets

echo "=== MathBode Inference Matrix (inference only) ==="
echo "CONFIG: $CONFIG | Mode: $MB_MODE | Signal: $SIGNAL"
echo "Families: ${FAMILIES[*]}"
echo "Freqs: ${FREQS[*]}   Phases: ${PHASES[*]}   Base-keys per family (SWEEPS): $SWEEPS"
echo "Workers: $WORKERS  Temp: $TEMP  MaxTokens: $MAXTOK"
echo "Outdir: $OUTDIR"
echo

# =======================
# Step 1: Build a shared subset (row_ids) for fairness across models
# =======================
SUBTAG="f$(IFS=-; echo "${FAMILIES[*]}")__fr$(IFS=-; echo "${FREQS[*]}")__ph$(IFS=-; echo "${PHASES[*]}")__k${SWEEPS}__${MB_MODE}__sig${SIGNAL}"
ROWID_FILE="cache/row_ids__${SUBTAG}.txt"

export MB_FAMILIES="$(IFS=,; echo "${FAMILIES[*]}")"
export MB_FREQS="$(IFS=,; echo "${FREQS[*]}")"
export MB_PHASES="$(IFS=,; echo "${PHASES[*]}")"
export MB_SIGNAL="$SIGNAL"
export MB_SWEEPS="$SWEEPS"
export MB_ROWIDS="$ROWID_FILE"
export MB_MODE="$MB_MODE"
export MB_BASE_KEYS_FILE="$BASE_KEYS_FILE"

if [[ ! -f "$ROWID_FILE" ]]; then
  echo "Generating shared subset row_ids → $ROWID_FILE"
  python - <<'PY'
import os, pandas as pd, random
from mathbode.data import load_mathbode

families = os.environ["MB_FAMILIES"].split(",")
freqs    = [int(x) for x in os.environ["MB_FREQS"].split(",")]
phases_env = [int(x) for x in os.environ["MB_PHASES"].split(",")]
mode     = os.environ.get("MB_MODE","STANDARD")
k_env    = os.environ["MB_SWEEPS"]         # "2" or "ALL"
rowid_path = os.environ["MB_ROWIDS"]
signal   = os.environ.get("MB_SIGNAL","sinusoid").lower()
base_keys_file = os.environ.get("MB_BASE_KEYS_FILE","")

random.seed(42)
df = load_mathbode(families)

if "signal_type" in df.columns:
    df = df[df["signal_type"].str.lower() == signal].copy()
else:
    raise RuntimeError("Dataset missing 'signal_type' column.")

def phases_for_freq(f):
    if mode == "MVP_PLUS":
        return [0,120,240] if f in {4,8} else [0]
    return phases_env

required_pairs = [(f, ph) for f in freqs for ph in phases_for_freq(f)]
allowed = pd.DataFrame(required_pairs, columns=["frequency_cycles","phase_deg"]).drop_duplicates()

base_cols = ["family","question_id","amplitude_scale"]
if "p0" in df.columns and df["p0"].nunique() > 1:
    base_cols.append("p0")
base_df = df[base_cols].drop_duplicates()

def load_base_keys_file(path):
    keys = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts=[p.strip() for p in line.split(",")]
            if len(parts) < 3: continue
            rec = {"family":parts[0], "question_id":parts[1], "amplitude_scale":float(parts[2])}
            if len(parts) >= 4:
                rec["p0"] = float(parts[3])
            keys.append(rec)
    return pd.DataFrame(keys)

if base_keys_file and os.path.exists(base_keys_file):
    base_df = load_base_keys_file(base_keys_file)
    base_df = base_df[base_df["family"].isin(families)]
elif k_env.upper() != "ALL":
    K = int(k_env)
    picks=[]
    for fam, g in base_df.groupby("family"):
        g2 = g.sort_values(["question_id","amplitude_scale"]).reset_index(drop=True)
        picks.append(g2.head(K))
    base_df = pd.concat(picks, ignore_index=True)

keep = []
T_mode = None

for _, bk in base_df.iterrows():
    mask = (
        (df["family"]==bk["family"]) &
        (df["question_id"]==bk["question_id"]) &
        (df["amplitude_scale"]==bk["amplitude_scale"])
    )
    if "p0" in bk and "p0" in df.columns:
        mask &= (df["p0"]==bk["p0"])
    sub = df[mask]
    if sub.empty: 
        continue

    sub_req = sub.merge(allowed, on=["frequency_cycles","phase_deg"], how="inner")
    if sub_req.empty:
        continue

    counts = sub_req.groupby(["frequency_cycles","phase_deg"])["time_step"].nunique()
    if len(counts) != len(allowed):
        continue
    Ts = set(counts.values)
    if len(Ts) != 1:
        continue
    T_here = counts.iloc[0]
    if T_mode is None:
        T_mode = T_here
    if T_here != T_mode:
        continue

    keep.extend(sub_req["row_id"].tolist())

keep = sorted(set(keep))
with open(rowid_path,"w") as f:
    for rid in keep: f.write(str(rid)+"\n")

print(f"Wrote {len(keep)} row_ids to {rowid_path} | signal={signal} | T={T_mode}")
PY
else
  echo "Using existing subset row_ids → $ROWID_FILE"
fi

# =======================
# Step 2: function to run one model (INFERENCE ONLY)
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
  # API keys sanity
  if [[ "$provider" == "gemini"    && -z "${GOOGLE_API_KEY:-}"    ]]; then echo "   [SKIP] Missing GOOGLE_API_KEY"; return 0; fi
  if [[ "$provider" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then echo "   [SKIP] Missing ANTHROPIC_API_KEY"; return 0; fi
  if [[ "$provider" == "together"  && -z "${TOGETHER_API_KEY:-}"  ]]; then echo "   [SKIP] Missing TOGETHER_API_KEY"; return 0; fi
  if [[ "$provider" == "openai"    && -z "${OPENAI_API_KEY:-}"    ]]; then echo "   [SKIP] Missing OPENAI_API_KEY"; return 0; fi

  # Inference on fixed row_ids; run_inference writes preds_{tag}.parquet
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

src = load_mathbode(families)
row_ids = [int(x.strip()) for x in open(rowid_path).read().splitlines() if x.strip()]
src = src[src["row_id"].isin(row_ids)].copy().reset_index(drop=True)

pred_path = run_inference(
    df=src[["row_id","prompt"]],
    provider=provider,
    model=model,
    outdir=outdir,
    temperature=temp,
    max_tokens=maxtok,
    api_base=api_base,
    workers=workers,
    provider_rps=0.0
)
print("Preds parquet written:", pred_path)
PY
}

# =======================
# Step 3: loop over model matrices (inference only)
# =======================
export MB_OUTDIR="$OUTDIR"
export MB_WORKERS="$WORKERS"
export MB_TEMP="$TEMP"
export MB_MAXTOK="$MAXTOK"
export MB_API_BASE="$API_BASE"

# Gemini
if (( ${#GEMINI_MODELS[@]} )); then
  for m in "${GEMINI_MODELS[@]}"; do
    export MB_PROVIDER="gemini"
    export MB_MODEL="$m"
    run_one_model "gemini" "$m" "$OUTDIR" "$WORKERS" "$TEMP" "$MAXTOK" "" || true
  done
fi

# OpenAI
if (( ${#OPENAI_MODELS[@]} )); then
  for m in "${OPENAI_MODELS[@]}"; do
    export MB_PROVIDER="openai"
    export MB_MODEL="$m"
    run_one_model "openai" "$m" "$OUTDIR" "$WORKERS" "$TEMP" "$MAXTOK" "$API_BASE" || true
  done
fi



# Together
if (( ${#TOGETHER_MODELS[@]} )); then
  for m in "${TOGETHER_MODELS[@]}"; do
    export MB_PROVIDER="together"
    export MB_MODEL="$m"
    run_one_model "together" "$m" "$OUTDIR" "$WORKERS" "$TEMP" "$MAXTOK" "" || true
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
echo "✅ Done. Inference parquets are in: $OUTDIR"
