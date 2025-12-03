# MathBode-Eval: Dynamic Fingerprints of LLM Mathematical Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2509.23143-b31b1b.svg)](https://arxiv.org/abs/2509.23143)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MathBode-Eval** is a framework for analyzing the *dynamic* mathematical-reasoning behavior of Large Language Models (LLMs). Borrowing from Bode analysis in control theory, we probe models with parametric math problems whose key parameter is **sinusoidally modulated**. We then measure the **gain** and **phase shift** of the model’s answers relative to ground truth, along with harmonic distortion, to produce a model’s **dynamic fingerprint**—revealing stability and failure modes that static accuracy misses.

This repository contains the complete toolkit: data loading, parallelized model inference, signal processing, summarization, and publication-quality plotting.

---

## Contents

- [Concept](#concept)
- [Getting Started](#-getting-started)
  - [Installation](#1-installation)
  - [API Keys](#2-api-key-configuration)
  - [Run Evaluations](#3-running-an-evaluation)
- [Workflow](#-workflow)
- [Repository Structure](#-repository-structure)
- [Codebase Deep Dive](#-codebase-deep-dive)
- [Configuration & Customization](#-configuration--customization)
  - [Evaluation Presets](#evaluation-presets)
  - [Adding a New Model](#adding-a-new-model)
  - [Model Config Example](#model-config-example)
  - [Reproducibility](#reproducibility)
- [Outputs](#-outputs)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## Concept

We treat an LLM solving a parametric math task as a **dynamic system**. For a fixed problem family, we sweep a single parameter with a sinusoid at frequency \( f \) and record the model’s sequence of answers.

- **Gain**: amplitude ratio \( \mathrm{A}_{\text{model}} / \mathrm{A}_{\text{truth}} \).  
  - 1.0 → perfectly scaled; >1.0 → over-reaction; <1.0 → under-reaction.
- **Phase shift**: lag in degrees between model and truth.  
  - 0° → perfectly in sync; positive → lag.
- **Harmonic distortion**: deviation from a pure sinusoidal response, e.g., `h2_over_h1` (second harmonic ratio) and `res_acf1` (residual autocorrelation).

Sweeping **frequencies** yields Bode-style plots (gain/phase vs. frequency) characterizing dynamic reasoning across problem families.

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/your-repo/MathBode-Eval.git
cd MathBode-Eval
pip install -r requirements.txt
````

> Python 3.10+ is recommended.

### 2. API Key Configuration

Export keys for providers you plan to evaluate:

```bash
export OPENAI_API_KEY='your-openai-key'
export GOOGLE_API_KEY='your-google-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export TOGETHER_API_KEY='your-together-key'
```

> You only need keys for providers you’re actually using.

### 3. Running an Evaluation

`run_matrix.sh` is the main entry point. Configure with environment variables.

**Smoke test (quick):**

```bash
CONFIG=SMOKE ./run_matrix.sh
```

**Extended run with concurrency:**

```bash
CONFIG=MVP_PLUS WORKERS=8 ./run_matrix.sh
```

Results stream into `results/` (raw predictions), `summaries/` (aggregated metrics), and `plots/`.

---

## 📊 Workflow

1. **Data Loading — `mathbode/data.py`**
   Loads `cognitive-metrology-lab/MathBode` from Hugging Face and performs stratified sampling per **family × frequency × phase** according to `CONFIG`. Caches selected row IDs in `cache/` for reproducibility.

2. **Inference — `mathbode/infer.py`**
   Runs model calls with a `ThreadPoolExecutor`, internal rate-limiters (RPM/TPM), and robust checkpointing to Parquet in `results/`. Skips already-completed rows to allow resumption.

3. **Summarization — `mathbode/summarize.py`**
   Fits the **first harmonic** via least squares for truth and predictions; computes:

   * `gain = A_model / A_truth`
   * `phase_deg_model_minus_truth`
   * `r2_model`
   * Harmonics (`h2_over_h1_model`, etc.)
   * Residual diagnostics (`res_acf1`)
     Writes `summary_*.csv` in `summaries/`.

4. **Plotting — `mathbode/plot_panels.py`**
   Generates main/appendix panels (Bode plots, overlays per family) using a custom Matplotlib style. Saves to `plots/main/` and `plots/appendix/`.

---

## 📂 Repository Structure

```
MathBode-Eval/
├── mathbode/
│   ├── clients.py        # Base & provider-specific API clients + RPM/TPM limiters
│   ├── data.py           # Load/sample MathBode dataset from Hugging Face
│   ├── infer.py          # Parallel, resumable inference → results/*.parquet
│   ├── plot_panels.py    # Publication-quality figures
│   ├── summarize.py      # Signal processing: gain, phase, harmonics, residuals
│   └── utils.py          # Parsing, formatting, helpers (e.g., fixed-decimal coercion)
├── results/              # Raw model predictions (parquet)
├── summaries/            # Aggregated metrics per model (CSV)
├── plots/
│   ├── main/             # Core figures for paper
│   └── appendix/         # Supplementary figures
├── cache/                # Cached row IDs for reproducible subsets
├── run_matrix.sh         # Main orchestration script
├── model.json            # Model configuration
├── requirements.txt      # Dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

---

## 💻 Codebase Deep Dive

### `mathbode/clients.py`

* `BaseClient` + specific clients: `OpenAIClient`, `GeminiClient`, `AnthropicClient`, `TogetherClient`.
* Built-in **RequestRateLimiter** (RPM) & **TokenRateLimiter** (TPM).
* Abstracts provider SDK differences and retry/backoff logic.

### `mathbode/infer.py`

* Orchestrates end-to-end inference via `run_inference`.
* Parallelization via `ThreadPoolExecutor` (controlled by `WORKERS`).
* Periodic batch writes to `results/*.parquet` for durability.
* Robust numeric parsing via `utils.coerce_to_fixed_decimals`.

### `mathbode/summarize.py`

* `_fit_first_harmonic`: least-squares sinusoid fit to truth and predictions.
* `_harmonics`: amplitudes of H1, H2, H3 (distortion).
* Outputs per-sweep metrics: `gain`, `phase_deg_model_minus_truth`, `h2_over_h1_model`, `res_acf1`, `r2_model`.

### `mathbode/plot_panels.py`

* One-stop plotting engine with consistent styles, labels, and legends.
* `\_plot_overlay_per_family` creates multi-panel overlays across families.
* Produces both main Bode plots (gain/phase) and auxiliary figures (e.g., compliance, residuals).

---

## 🛠️ Configuration & Customization

### Evaluation Presets

`CONFIG` in `run_matrix.sh` selects the sweep scope:

| Preset     | Frequencies | Phases                 | Sweeps/Freq | Description               |
| :--------- | :---------- | :--------------------- | :---------- | :------------------------ |
| `SMOKE`    | 4, 8        | 0°                     | 2           | Minimal pipeline check.   |
| `MVP`      | 4, 8, 16    | 0°                     | 2           | Balanced key frequencies. |
| `MVP_PLUS` | 1–16        | Mixed (0° & tri-phase) | 2           | Extended coverage.        |
| `FULL`     | 1–16        | 0°, 120°, 240°         | 2           | Comprehensive evaluation. |

> You can customize these in `run_matrix.sh` (frequencies, phases, per-freq sweeps, and sampling seeds).

### Adding a New Model

1. **Edit `model.json`** — add your model under the appropriate provider key.
2. **New provider?** — implement a client in `mathbode/clients.py` inheriting `BaseClient`.
3. **Run** — `run_matrix.sh` will pick up the new entry.

### Model Config Example

```json
{
  "openai":    ["gpt-4o-mini", "gpt-4.1-mini"],
  "anthropic": ["claude-3-5-sonnet-20240620"],
  "google":    ["gemini-1.5-pro"],
  "together":  ["meta-llama/Meta-Llama-3-70B-Instruct-Turbo"]
}
```

> Use provider-specific IDs accepted by each API.

### Reproducibility

* **Dataset splits** (selected row IDs) are cached under `cache/` per `CONFIG`.
* Set `PY_SEED` and `NP_SEED` (if exposed) to lock sampling and plotting jitter.
* Inference is **idempotent** per (model, sweep) row—completed rows are skipped.

---

## 📦 Outputs

* **Raw predictions**: `results/*.parquet`
  Columns typically include identifiers (family, frequency, phase), truth `y`, predictions `y_hat`, and metadata (timestamps, model id).

* **Summaries**: `summaries/summary_<model>_<config>.csv`
  Per-sweep metrics: `gain`, `phase_deg_model_minus_truth`, `r2_model`, `h2_over_h1_model`, `res_acf1`, etc.

* **Plots**:

  * `plots/main/`: Bode-style gain/phase vs. frequency, per-family overlays.
  * `plots/appendix/`: supplementary panels (e.g., compliance, residual patterns).

---

## 🧰 Troubleshooting

* **Rate limits / 429s**

  * Lower `WORKERS`, or adjust internal RPM/TPM caps (client constructors).
  * Ensure only the needed providers are enabled in `model.json`.

* **Partial runs / resuming**

  * Safe to re-run the same command; completed rows are skipped; checkpoints append.

* **Parsing errors**

  * Improve your model’s answer formatting instructions or adjust `coerce_to_fixed_decimals` tolerance in `utils.py`.

* **Empty plots / summaries**

  * Verify `results/*.parquet` exist and that `summarize.py` produced non-empty CSVs.
  * Confirm your `CONFIG` actually sampled rows for the targeted families.

---

## 📖 Citation

If you use **MathBode-Eval** in your research, please cite:

```bibtex
@misc{wang2025mathbodeeval,
  title         = {MathBode: Measuring the Stability of LLM Reasoning using Frequency Response},
  author        = {Charles L. Wang},
  year          = {2025},
  eprint        = {2509.23143},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2509.23143}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
