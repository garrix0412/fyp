# LORSTransformerDRL: Deep Reinforcement Learning for Intelligent Stock Trading

**A Novel Framework Combining Lee Oscillator Retrograde Signal (LORS) with Transformer Architecture for Direct Trading Decision Optimization**

## Overview

This repository contains the implementation of **LORSTransformerDRL**, an end-to-end intelligent stock trading system that integrates:

- **Lee Oscillator Retrograde Signal (LORS)**: Chaotic neural dynamics for capturing nonlinear market patterns
- **Transformer Architecture**: Self-attention mechanism for identifying critical market moments
- **Deep Reinforcement Learning (DQN)**: Direct optimization of trading decisions (buy/hold/sell)

Unlike traditional "forecast-then-decide" approaches, this system directly optimizes trading actions, avoiding error accumulation from prediction pipelines.

## Key Features

- ✅ **Direct Action Optimization**: DRL-based end-to-end trading strategy learning
- ✅ **Chaotic Dynamics Modeling**: LORS mechanism for nonlinear market behavior
- ✅ **Attention-Based Pattern Recognition**: Transformer architecture for temporal dependencies
- ✅ **Anti-Leakage Protocol**: Strict chronological split with proper validation
- ✅ **Comprehensive Baselines**: 9 model implementations for comparison
- ✅ **Reproducible Experiments**: Multi-seed experiments with statistical analysis

## Repository Structure

```
├── src/                    # Source code modules
│   ├── models/            # Model implementations
│   ├── agents/            # RL agents (DQN)
│   ├── environment/       # Trading environment simulator
│   ├── utils/             # Utilities (config, data, metrics)
│   └── visualization/     # Plotting and analysis tools
├── scripts/               # Execution scripts
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── fetch_data.py     # Data fetching
├── data/                  # Market data
├── checkpoints/           # Trained model weights
├── logs/                  # Training logs
├── results/               # Experiment results
├── figures/               # Generated visualizations
├── Paper/                 # Research paper
└── requirements.txt       # Dependencies
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd FYP_2025_QF2_G2

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch>=1.12.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities
- `yfinance>=0.1.70` - Financial data fetching
- `ta>=0.10.0` - Technical indicators
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical plotting
- `tqdm>=4.62.0` - Progress bars

## Quick Start

### 1. Data Preparation

Download and process financial data:

```bash
python scripts/fetch_data.py
```

This downloads Dow Jones Industrial Average data (2016-2025) and computes technical indicators (RSI, MACD, Bollinger Bands, etc.).

### 2. Quick Test

Run a quick test with the main model:

```bash
python scripts/train.py --model LORSTransformerDRL --episodes 5
```

### 3. Full Training

Train a single model:

```bash
python scripts/train.py --model LORSTransformerDRL --episodes 100
```

Train all models with multiple seeds:

```bash
python scripts/train.py --all --full --episodes 100
```

### 4. Visualization

Generate figures from experiment results:

```bash
python src/visualization/plot_results.py results/<experiment_file>.json
```

## Models

The following models are implemented for comparison:

| Model | Description |
|-------|-------------|
| **LORSTransformerDRL** | Main model: LSTM + Transformer + LORS + DQN |
| Transformer_DQN | Standard Transformer without LORS |
| AttentionLSTM_DQN | LSTM with attention mechanism |
| ChaoticRNN_DQN | Chaotic dynamics without retrograde signaling |
| LSTM_DQN | Basic LSTM baseline |
| GRU_DQN | Basic GRU baseline |
| CNN_DQN | Convolutional baseline |
| DQN_MLP | Multi-layer perceptron baseline |
| LORSTransformer | LORS + Transformer (ablation study) |
| Random | Random action baseline |

## Methodology

### Data Split Protocol

Following strict anti-leakage protocol:
- **Train**: 60% (2016-01-04 to 2021-05-25)
- **Validation**: 20% (2021-05-26 to 2023-03-14)
- **Test**: 20% (2023-03-15 to 2024-12-31)

Key safeguards:
- Chronological split without shuffling
- Preprocessors fit only on training data
- Model selection based on validation Sharpe ratio
- Single-pass test evaluation
- Transaction costs (10 bps) applied before reward

### Trading Environment

- **State**: 120-day sliding window of 10 features
- **Actions**: Buy (0), Hold (1), Sell (2)
- **Reward**: Portfolio value change after transaction costs
- **Initial Capital**: $1,000,000

### LORS Mechanism

The Lee Oscillator Retrograde Signal introduces controlled chaotic dynamics:

```
E(t+1) = Sig[a1·LORS(t) + a2·E(t) - a3·I(t) + a4·S(t)]
I(t+1) = Sig[b1·LORS(t) - b2·E(t) - b3·I(t) + b4·S(t)]
LORS(t) = [E(t) - I(t)] · exp(-k·S²(t)) + Ω(t)
```

Eight LORS configurations with different bifurcation parameters are combined via attention weighting.

## Evaluation Metrics

### Risk-Return Metrics
- **CR (%)**: Cumulative Return = (V_T / V_0 - 1) × 100%
- **Sharpe**: Sharpe Ratio = (mean return) / (std return)
- **MDD (%)**: Maximum Drawdown
- **Win (%)**: Win Rate of trades

### Trading Behavior Metrics
- Average holding time (days)
- Transaction frequency (times/month)
- Average per-trade return (%)

## Experimental Results

Expected results from the paper:

| Model | CR (%) | Sharpe | MDD (%) | Win (%) |
|-------|--------|--------|---------|---------|
| **LORSTransformerDRL** | **214.9** | **0.216** | **42.3** | **58.3** |
| CNN_DQN | 197.1 | 0.162 | 47.9 | 54.9 |
| DQN_MLP | 133.9 | 0.186 | 42.3 | 54.1 |
| Transformer_DQN | 135.1 | 0.107 | 51.2 | 55.9 |
| Random | -73.4 | 0.114 | 51.2 | 54.9 |

Multi-seed results (mean ± std, n=5):
- **LORSTransformerDRL**: CR = 216.8 ± 8.5%, Sharpe = 0.215 ± 0.009

## Configuration

Key hyperparameters can be modified in `src/utils/config.py`:

```python
# Training
max_episodes = 100
patience = 15          # Early stopping patience
val_freq = 5           # Validation frequency

# DQN
lr = 0.0001
gamma = 0.99
epsilon_decay = 0.995
buffer_size = 20000
batch_size = 64

# Model (LORSTransformerDRL)
embed_dim = 128
num_heads = 8
n_layers = 3
dropout = 0.15
```

## Project Timeline

**Experiment Period**: July 20 - July 25, 2025
- **Total Experiments**: 45 (9 models × 5 seeds)
- **Training Duration**: ~3 hours per experiment
- **Total Runtime**: ~135 hours

## File Descriptions

| File/Directory | Description |
|----------------|-------------|
| `src/utils/config.py` | Centralized hyperparameter management |
| `src/utils/data_utils.py` | Data split and anti-leakage preprocessing |
| `src/utils/metrics.py` | Evaluation metrics implementation |
| `src/models/` | Model implementations |
| `src/agents/` | DQN agent with experience replay |
| `src/environment/` | Trading environment simulator |
| `scripts/train.py` | Training and evaluation script |
| `results/` | Experimental results (JSON format) |
| `logs/` | Training logs with timestamps |
| `figures/` | Generated figures and tables |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ma2025lorstransformerdrl,
  title={LORSTransformerDRL: A Novel Deep Reinforcement Learning Framework for Intelligent Stock Trading with Chaotic Oscillators and Attention Mechanisms},
  author={Ma, Pengyue and Zeng, Zihang and Lu, Ziqian and Lee, Raymond S. T.},
  journal={},
  year={2025}
}
```

## License

This project is for academic research purposes only.

## Acknowledgments

- Guangdong Provincial/Zhuhai Key Laboratory of Interdisciplinary Research and Application for Data Science
- Beijing Normal-Hong Kong Baptist University

---

**Note**: GPU acceleration is automatically detected (CUDA/MPS/CPU). Training 100 episodes takes approximately 1-2 hours on CPU, 10-20 minutes on GPU.
