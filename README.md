# MathBode: Dynamic Fingerprints of LLM Reasoning via Gain and Phase

A dynamic evaluation framework for analyzing how language models track and respond to time-varying mathematical problems. This tool generates Bode plots to visualize model performance across different frequencies of parameter variation.

## 📊 Overview

MathBode evaluates LLMs on mathematical problems where parameters vary sinusoidally over time. It measures:
- **Gain**: How much the model amplifies/attenuates the input signal
- **Phase Shift**: How much the model lags behind the true solution
- **R²**: Goodness of fit to the expected sinusoidal response

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages: `pip install -r requirements.txt`

### Basic Usage

1. **Configure Models**
   Edit `model.json` to specify which models to evaluate:
   ```json
   {
     "gemini": ["gemini-2.5-pro"],
     "openai": ["gpt-5"],
     "anthropic": ["claude-sonnet-4-20250514"]
   }
   ```

2. **Set API Keys**
   ```bash
   export OPENAI_API_KEY='your-key-here'
   export GOOGLE_API_KEY='your-key-here'
   export ANTHROPIC_API_KEY='your-key-here'
   ```

3. **Run Evaluation**
   ```bash
   # Basic run with default settings
   ./run_matrix.sh
   
   # Custom configuration
   CONFIG=MVP_PLUS WORKERS=3 ./run_matrix.sh
   ```

## 🗂 Repository Structure

```
MathBode-Eval/
├── mathbode/               # Core Python package
│   ├── __init__.py         # Package initialization
│   ├── clients.py          # API clients for different LLM providers
│   ├── data.py             # Data loading and processing
│   ├── infer.py            # Model inference logic
│   ├── plot_curves.py      # Visualization utilities
│   ├── summarize.py        # Analysis and metrics calculation
│   └── utils.py            # Helper functions
├── cache/                  # Cached data (e.g., row IDs)
│   └── row_ids_*.txt       # Pre-computed row indices for reproducibility
├── results/                # Output directory
│   ├── preds_*.parquet     # Raw model predictions
│   ├── summary_*.parquet   # Aggregated metrics
│   └── plots/              # Generated visualizations
├── run_matrix.sh           # Main entry point for running evaluations
├── model.json              # Model configuration
└── requirements.txt        # Python dependencies
```

### Key Scripts and Their Functions

#### 1. `run_matrix.sh`
- **Purpose**: Main entry point for running evaluations
- **Key Features**:
  - Handles configuration presets (SMOKE, MVP, MVP_PLUS, FULL)
  - Manages parallel execution of model evaluations
  - Coordinates data loading, inference, and analysis

#### 2. `mathbode/clients.py`
- **Purpose**: Implements API clients for different LLM providers
- **Supported Providers**:
  - OpenAI (GPT models)
  - Google Gemini
  - Anthropic Claude
- **Features**:
  - Handles API authentication
  - Implements retry logic and rate limiting
  - Standardizes response formats

#### 3. `mathbode/data.py`
- **Purpose**: Data loading and preprocessing
- **Key Functions**:
  - Loads the MathBode dataset from Hugging Face
  - Filters data by problem families, frequencies, and phases
  - Implements stratified sampling for evaluation subsets

#### 4. `mathbode/infer.py`
- **Purpose**: Core inference logic
- **Features**:
  - Manages concurrent API requests
  - Handles batching and rate limiting
  - Saves intermediate results
  - Implements resumable evaluation

#### 5. `mathbode/summarize.py`
- **Purpose**: Analysis of model performance
- **Key Metrics**:
  - Gain and phase shift calculations
  - R² goodness of fit
  - Statistical analysis of model responses

#### 6. `mathbode/plot_curves.py`
- **Purpose**: Visualization of results
- **Outputs**:
  - Bode plots (gain/phase vs frequency)
  - Time series comparisons
  - Statistical summaries

## 🛠 Configuration

### Environment Variables
- `WORKERS`: Number of parallel API calls (default: 4)
- `TEMP`: Sampling temperature (default: 0.0)
- `MAXTOK`: Maximum tokens to generate (default: 32)
- `API_BASE`: Custom API base URL (for proxy/self-hosted models)
- `CONFIG`: Configuration preset (see below)
- `OUTDIR`: Output directory (default: `results/`)
- `MB_MODE`: Internal mode flag (auto-set based on config)

### Configuration Presets

MathBode provides several configuration presets that control the evaluation scope:

| Preset   | Frequencies | Phases | Sweeps | Total Samples | Description |
|----------|-------------|--------|--------|--------------|-------------|
| `SMOKE`  | 4, 8        | 0°     | 2      | ~5.1k        | Quick test with minimal samples |
| `MVP`    | 4, 8, 16    | 0°     | 2      | ~7.7k        | Balanced evaluation with key frequencies |
| `MVP_PLUS` | 1-16      | Mixed* | 2      | ~13.4k       | Enhanced with more frequencies and phases |
| `FULL`   | 1-16        | 0°, 120°, 240° | 2 | ~38.4k | Comprehensive evaluation with all phases |

*MVP_PLUS uses custom phase selection: 0° for frequencies 1,2,16 and tri-phase (0°,120°,240°) for 4,8

### Problem Families

By default, all 5 problem families are included:
1. `linear_solve`
2. `ratio_saturation`
3. `exponential_interest`
4. `linear_system`
5. `similar_triangles`

To modify which families are included, edit the `FAMILIES` array in `run_matrix.sh`.



## 📊 Outputs

Results are saved in the `results/` directory:
- `preds_*.parquet`: Raw model predictions
- `summary_*.parquet`: Aggregated metrics
- `plots/`: Visualizations (Bode plots)

## 🔧 Customization

### Adding New Models
1. Add model name to `model.json` under the appropriate provider
2. Implement any custom logic in `mathbode/clients.py` if needed

### Modifying Problem Sets
Edit `mathbode/data.py` to adjust:
- Problem families
- Frequency ranges
- Phase offsets
- Amplitude scaling

## 📈 Analysis

### Key Metrics
- **Gain**: `1.0` is ideal (no amplification/attenuation)
- **Phase Shift**: `0°` is ideal (no delay)
- **R²**: `1.0` is perfect fit

### Interpreting Results
- **High Gain (>1)**: Model overreacts to input changes
- **Low Gain (<1)**: Model underreacts to input changes
- **Positive Phase**: Model lags behind true solution
- **Negative Phase**: Model anticipates changes (rare)

## 📝 Citation

If you use MathBode in your research, please cite:

```bibtex
soon
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
