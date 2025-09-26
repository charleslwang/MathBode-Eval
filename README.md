# MathBode-Eval: Dynamic Fingerprints of LLM Mathematical Reasoning

**MathBode-Eval** is a sophisticated framework for analyzing the dynamic reasoning capabilities of Large Language Models (LLMs). Inspired by Bode analysis in control systems engineering, this project evaluates how well models track and solve mathematical problems where parameters vary sinusoidally over time. By measuring the **gain** and **phase shift** of a model's response relative to the ground truth, we can create a unique "dynamic fingerprint" that reveals deeper insights into its reasoning stability and failure modes beyond simple static accuracy.

This repository contains the complete toolkit for running these evaluations, from data loading and model inference to advanced signal processing and plotting.

The central idea is to treat an LLM solving a parametric math problem as a dynamic system. We feed it a sequence of problems where a single parameter is modulated by a sine wave at a specific frequency. We then analyze the model's sequence of answers to see how its output signal compares to the true solution's signal.

-   **Gain**: The ratio of the model's output amplitude to the true amplitude. A gain of 1.0 means the model's reasoning is perfectly scaled. A gain > 1.0 indicates over-reaction, while a gain < 1.0 indicates under-reaction.
-   **Phase Shift**: The lag (in degrees) between the model's response and the true response. A phase shift of 0° means the model is perfectly in sync. A positive phase shift indicates a lag in reasoning.
-   **Harmonic Distortion**: Measures how much the model's output deviates from a pure sine wave, indicating non-linear reasoning errors. This is quantified by metrics like `h2_over_h1` (second harmonic ratio) and `res_acf1` (residual autocorrelation).

By sweeping through different frequencies, we can generate Bode plots that characterize a model's dynamic performance across a spectrum of problem variations.

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/your-repo/MathBode-Eval.git
cd MathBode-Eval
pip install -r requirements.txt
```

### 2. API Key Configuration

Export your API keys for the LLM providers you wish to evaluate:

```bash
export OPENAI_API_KEY='your-openai-key'
export GOOGLE_API_KEY='your-google-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export TOGETHER_API_KEY='your-together-key'
```

### 3. Running an Evaluation

The `run_matrix.sh` script is the main entry point for running evaluations. It can be configured with environment variables.

**Example: Run a quick "smoke test" evaluation**

```bash
CONFIG=SMOKE ./run_matrix.sh
```

This will run a small subset of the evaluation on the models defined in `model.json`.

**Example: Run a more comprehensive evaluation with 8 workers**

```bash
CONFIG=MVP_PLUS WORKERS=8 ./run_matrix.sh
```

## Workflow

The evaluation process follows a clear, multi-stage pipeline:

1.  **Data Loading (`mathbode/data.py`)**: The script loads the `cognitive-metrology-lab/MathBode` dataset from the Hugging Face Hub. Based on the chosen `CONFIG`, it performs stratified sampling to select a subset of problem sweeps across different families, frequencies, and phases.

2.  **Inference (`mathbode/infer.py`)**: For each model, the script runs inference on the selected prompts. It uses a `ThreadPoolExecutor` for concurrent API calls, manages rate limiting, and robustly checkpoints results to a Parquet file in the `results/` directory. This allows the process to be resumed if interrupted.

3.  **Summarization (`mathbode/summarize.py`)**: After inference is complete, this script processes the raw model outputs (`y_hat`). It performs a least-squares fit to a sine wave to calculate the core metrics for each sweep: `gain`, `phase_deg_model_minus_truth`, and `r2_model`. It also computes advanced harmonic distortion and residual metrics.

4.  **Plotting (`mathbode/plot_panels.py`)**: Finally, this script reads the summary files and generates the publication-quality Bode plots and other visualizations, saving them to the `plots/` directory.

## 📂 Repository Structure

```
MathBode-Eval/
├── mathbode/           # Core Python package for the evaluation framework
│   ├── clients.py      # API clients for OpenAI, Gemini, Anthropic, and Together
│   ├── data.py         # Loads and samples the MathBode dataset from Hugging Face
│   ├── infer.py        # Manages parallelized, resumable model inference
│   ├── plot_panels.py  # Generates all publication-quality plots and figures
│   ├── summarize.py    # Performs signal processing to compute gain, phase, and other metrics
│   └── utils.py        # Helper functions, including robust answer parsing
├── results/            # Default output directory for all generated files
│   ├── *.parquet       # Raw model predictions (from infer.py)
├── summaries/          # Aggregated metrics per model
│   └── summary_*.csv   # CSV files with gain/phase data (from summarize.py)
├── plots/              # Generated visualizations
│   ├── main/           # Core figures for the main paper
│   └── appendix/       # Supplementary figures for the appendix
├── cache/              # Caches row IDs for reproducible dataset subsets
├── run_matrix.sh       # Main entry point script for running evaluations
├── model.json          # JSON configuration for models to be evaluated
├── LICENSE             # MIT License
└── README.md           # This file
```

## 💻 Codebase Deep Dive

### `mathbode/clients.py`

-   Implements a `BaseClient` and specific subclasses (`OpenAIClient`, `GeminiClient`, `AnthropicClient`, `TogetherClient`) for interacting with various LLM APIs.
-   Includes robust, self-contained `RequestRateLimiter` (RPM) and `TokenRateLimiter` (TPM) classes to avoid API errors.
-   Handles different SDKs and API calling conventions (e.g., legacy vs. new Google GenAI SDKs).

### `mathbode/infer.py`

-   The heart of the inference pipeline.
-   `run_inference` orchestrates the process, skipping already completed rows to allow for easy resumption.
-   Uses `ThreadPoolExecutor` to run inference jobs in parallel, controlled by the `WORKERS` environment variable.
-   Saves results in batches, ensuring progress is not lost during long runs.
-   Calls `coerce_to_fixed_decimals` from `utils.py` for robust parsing of the model's numerical answer.

### `mathbode/summarize.py`

-   Performs the core signal processing and analysis.
-   `_fit_first_harmonic` uses least-squares regression to fit a sine wave to both the ground truth and the model's output, extracting amplitude (`A`) and phase (`phi`).
-   `_harmonics` calculates the amplitude of the first three harmonics (`H1`, `H2`, `H3`) to measure non-linear distortion.
-   Calculates key metrics:
    -   `gain`: `A_model / A_truth`
    -   `phase_deg_model_minus_truth`: Phase difference in degrees.
    -   `h2_over_h1_model`: Ratio of the second harmonic to the fundamental, a measure of non-linearity.
    -   `res_acf1`: Lag-1 autocorrelation of the residuals, which detects systematic, unmodeled patterns in the model's errors.

### `mathbode/plot_panels.py`

-   A powerful, self-contained plotting engine for generating all figures.
-   Uses a highly customized, modern `matplotlib` stylesheet for a premium aesthetic.
-   `_plot_overlay_per_family` is the main function, which generates a panel of plots, one for each problem family.
-   Intelligently handles model naming, color palettes, and marker styles.
-   Generates both the main figures (e.g., Gain vs. Frequency) and appendix figures (e.g., Compliance Rate, Residuals).

## 🛠️ Configuration & Customization

### Evaluation Presets

The `CONFIG` environment variable in `run_matrix.sh` controls the scope of the evaluation. Presets are defined in the script itself.

| Preset     | Frequencies | Phases                 | Sweeps/Freq | Description                                       |
| :--------- | :---------- | :--------------------- | :---------- | :------------------------------------------------ |
| `SMOKE`    | 4, 8        | 0°                     | 2           | A minimal run to verify the pipeline works.       |
| `MVP`      | 4, 8, 16    | 0°                     | 2           | A balanced set of key frequencies.                |
| `MVP_PLUS` | 1-16        | Mixed (0° and tri-phase) | 2           | An extended evaluation with more frequency coverage. |
| `FULL`     | 1-16        | 0°, 120°, 240°         | 2           | The most comprehensive evaluation.                |

### Adding a New Model

1.  **Add to `model.json`**: Add your model's identifier under the correct provider key (e.g., `"openai": ["gpt-4o-mini"]`).
2.  **Implement Client (if new provider)**: If you are adding a model from a new provider, create a new client class in `mathbode/clients.py` that inherits from `BaseClient`.
3.  **Run the evaluation**: The `run_matrix.sh` script will automatically pick up the new model.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
