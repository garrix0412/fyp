# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research implementation of **LORSTransformerDRL**, a deep reinforcement learning framework for stock trading that combines:
- **Lee Oscillator Retrograde Signal (LORS)**: Chaotic neural dynamics with 8 bifurcation configurations
- **Transformer Architecture**: Multi-head self-attention for temporal pattern recognition
- **Deep Q-Network (DQN)**: Direct trading action optimization (buy/hold/sell)

The system implements end-to-end trading decision learning, avoiding the error accumulation of traditional "forecast-then-decide" pipelines.

## Development Commands

### Data Preparation
```bash
# Download and process market data (Dow Jones 2016-2025)
python scripts/fetch_data.py

# Ensures data/DJI_2016_2025.csv exists with technical indicators
```

### Training

```bash
# Quick test (5 episodes, single model)
python scripts/train.py --model LORSTransformerDRL --episodes 5

# Train single model (100 episodes)
python scripts/train.py --model LORSTransformerDRL --episodes 100

# Train all baseline models
python scripts/train.py --all --episodes 100

# Multi-seed experiment (for statistical significance)
python scripts/train.py --model LORSTransformerDRL --seeds 42,43,44,45,46 --episodes 100

# Full experiment mode (all models × 5 seeds = 45 experiments)
python scripts/train.py --all --full --episodes 100
```

### Testing and Visualization

```bash
# Generate paper figures from results
python src/visualization/plot_results.py

# Produces figures/figure2.png (multi-metric radar)
#          figures/figure3.png (risk-return scatter)
#          figures/figure4a.png, figure4b.png (performance dynamics)
```

### Device Detection

Training automatically detects and uses available hardware:
- **MPS** (Apple Silicon) > **CUDA** (NVIDIA) > **CPU**
- No manual configuration required

## Code Architecture

### Core Data Flow

The system follows a strict chronological data pipeline to prevent information leakage:

```
Raw Data (OHLCV + Indicators)
    ↓
ChronologicalSplitter (60/20/20 split, no shuffling)
    ↓
AntiLeakagePreprocessor (fit only on train, transform val/test)
    ↓
TradingEnv (120-day sliding window)
    ↓
DQN Agent (epsilon-greedy policy, experience replay)
    ↓
Model Forward Pass (LSTM → Transformer+LORS → Q-values)
    ↓
Action Selection (0=buy, 1=hold, 2=sell)
    ↓
Environment Step (apply transaction costs, calculate reward)
```

### Critical Anti-Leakage Protocol

**Data splitting** ([src/utils/data_utils.py:11-82](src/utils/data_utils.py#L11-L82)):
- Strict chronological ordering without shuffling
- Train: t₁ to t₀.₆T
- Validation: t₀.₆T+₁ to t₀.₈T
- Test: t₀.₈T+₁ to tT
- All preprocessing (scalers) fit only on training data

**Trading environment** ([src/environment/trading_env.py:18-178](src/environment/trading_env.py#L18-L178)):
- Transaction costs deducted BEFORE reward calculation
- Reward = net portfolio value change (no additional penalties)
- Full position trading (all-in or all-out strategy)
- No future information in observation windows

### Model Architecture Components

**LORSTransformerDRL** ([src/models/lors_transformer_drl.py:11-139](src/models/lors_transformer_drl.py#L11-L139)):

1. **Embedding Layer**: Projects 10 input features to 128-dim space
2. **LSTM Temporal Processing**: Captures sequential dependencies
3. **Transformer Layers** (×3): Each layer includes:
   - Multi-head self-attention (8 heads)
   - LORS mechanism with 8 bifurcation configurations
   - Attention-weighted LORS combination
   - Residual connections + LayerNorm
4. **Dual Pooling**: 0.5 × avg_pool + 0.5 × max_pool
5. **Output Network**: FC(128) → ReLU → FC(3 actions)

**LORS Dynamics** ([src/models/lors_transformer_drl.py:102-130](src/models/lors_transformer_drl.py#L102-L130)):
```
E(t+1) = tanh(s · [a₁·LORS(t) + a₂·E(t) - a₃·I(t) + a₄·S(t)])
I(t+1) = tanh(s · [b₁·LORS(t) - b₂·E(t) - b₃·I(t) + b₄·S(t)])
LORS(t) = [E(t) - I(t)] · exp(-α·k·S²(t)) + c·W(t)
```
- Eight parameter sets create diverse chaotic behaviors
- Attention mechanism learns optimal configuration weighting

### DQN Agent Architecture

**Training loop** ([src/agents/dqn_agent.py:17-206](src/agents/dqn_agent.py#L17-L206)):
- **Policy network**: Q(s,a) predictions for action selection
- **Target network**: Stabilizes training with periodic updates (every 10 episodes)
- **Experience replay**: Breaks temporal correlations (buffer size 20,000)
- **Epsilon-greedy exploration**: Starts at 1.0, decays by 0.995/episode to min 0.01
- **Gradient clipping**: max_norm=1.0 for stability

**Training protocol** ([scripts/train.py:133-207](scripts/train.py#L133-L207)):
- Validation every 5 episodes
- Model selection based on validation Sharpe ratio
- Early stopping with patience=15 episodes
- Best model restored after training

### Configuration System

**Centralized hyperparameters** ([src/utils/config.py](src/utils/config.py)):
- `DATA_CONFIG`: Features, split ratios, file paths
- `ENV_CONFIG`: Window size (120), transaction fee (10 bps), initial capital ($1M)
- `DQN_CONFIG`: Learning rate (1e-4), gamma (0.99), buffer size (20K), batch size (64)
- `MODEL_CONFIG`: Architecture specifications for all 9 models
- `EXPERIMENT_CONFIG`: Quick test vs. full experiment modes

Access configs via:
```python
from src.utils import get_config
config = get_config('full_experiment')  # or 'quick_test'
```

### Model Registry

**Available models** ([src/models/registry.py](src/models/registry.py)):
- `LORSTransformerDRL`: Main model (LSTM + Transformer + LORS + DQN)
- `LORSTransformer`: Ablation study version (identical architecture)
- `Transformer_DQN`: Standard Transformer without LORS
- `AttentionLSTM_DQN`: LSTM with attention mechanism
- `ChaoticRNN_DQN`: Chaotic dynamics without retrograde signaling
- `LSTM_DQN`, `GRU_DQN`: Basic RNN baselines
- `CNN_DQN`: Convolutional baseline
- `DQN_MLP`: Feedforward baseline
- `Random`: Random action baseline

Access via:
```python
from src.models import get_model, MODEL_REGISTRY
model = get_model('LORSTransformerDRL', **config['models']['LORSTransformerDRL'])
```

## Key Implementation Details

### Transaction Cost Application

**CRITICAL**: Transaction costs are deducted BEFORE reward calculation ([src/environment/trading_env.py:94-153](src/environment/trading_env.py#L94-L153)):

```python
# Calculate transaction cost (10 basis points)
cost = self.transaction_fee * abs(delta_N) * current_price

# Update cash and shares (cost deducted first)
self.cash = self.cash - delta_N * current_price - cost
self.shares = self.shares + delta_N

# Calculate portfolio value
portfolio_value = self.cash + self.shares * current_price

# Reward = net value change (includes cost impact)
reward = portfolio_value - self.prev_portfolio_value
```

### Multi-Seed Reproducibility

For statistical significance, experiments use seeds `{42, 43, 44, 45, 46}`:
- Set seeds for `torch`, `numpy`, `random` before each experiment
- Results aggregated as mean ± std across 5 runs
- Expected paper results: LORSTransformerDRL achieves CR=216.8±8.5%, Sharpe=0.215±0.009

### Evaluation Metrics

**TradingMetrics class** ([src/utils/metrics.py](src/utils/metrics.py)):
- **CR** (Cumulative Return): (V_final / V_initial - 1) × 100%
- **Sharpe Ratio**: mean(returns) / std(returns)
- **MDD** (Maximum Drawdown): max percentage loss from peak
- **Win Rate**: percentage of profitable trades
- **Trading behavior**: Avg holding time, transaction frequency, per-trade return

## Important Constraints

### Research Integrity Requirements

1. **No data leakage**: Never fit preprocessors on validation/test data
2. **Single-pass test evaluation**: Test set evaluated only once with best model
3. **Chronological ordering**: No shuffling of time-series data
4. **Transaction costs**: Always apply 10 bps fee before reward calculation
5. **Model selection**: Based on validation Sharpe ratio, not test metrics

### Performance Expectations

- **Training time**: ~1-2 hours on CPU, ~10-20 minutes on GPU for 100 episodes
- **Full experiment** (45 runs): ~135 hours total runtime
- **Memory usage**: ~2-4GB RAM for single experiment
- **Expected test performance**: CR ~200%, Sharpe ~0.2, MDD ~40%

## File Organization

```
src/
├── models/
│   ├── lors_transformer_drl.py  # Main model + LORSTransformer ablation
│   ├── baselines.py             # 7 baseline model implementations
│   └── registry.py              # Model factory and registry
├── agents/
│   └── dqn_agent.py             # DQN with experience replay
├── environment/
│   └── trading_env.py           # Trading simulator with transaction costs
├── utils/
│   ├── config.py                # Centralized hyperparameter management
│   ├── data_utils.py            # ChronologicalSplitter, AntiLeakagePreprocessor
│   ├── metrics.py               # TradingMetrics computation
│   └── __init__.py              # Convenience imports
└── visualization/
    └── plot_results.py          # Paper figure generation

scripts/
├── train.py                     # Training script with validation
├── fetch_data.py                # Data download and indicator computation
└── evaluate.py                  # (if exists) Separate evaluation script

data/                            # CSV files with OHLCV + indicators
logs/                            # Training logs with timestamps
results/                         # JSON files with experiment results
figures/                         # Generated visualizations (PNG/PDF)
checkpoints/                     # Saved model states
```

## Common Modification Patterns

### Adding a New Model

1. Implement model class in `src/models/baselines.py` or new file
2. Register in `src/models/registry.py`:
   ```python
   MODEL_REGISTRY['NewModel'] = NewModelClass
   ```
3. Add config to `src/utils/config.py`:
   ```python
   MODEL_CONFIG['NewModel'] = {...}
   ```
4. Train with: `python scripts/train.py --model NewModel --episodes 100`

### Modifying Hyperparameters

Edit `src/utils/config.py` sections:
- DQN parameters: learning rate, epsilon decay, buffer size
- Model architecture: embed_dim, num_heads, n_layers, dropout
- Training: max_episodes, patience, val_freq
- Environment: window_size, transaction_fee

### Changing Data Source

1. Modify `scripts/fetch_data.py` to fetch different ticker/date range
2. Update `DATA_CONFIG['ticker']`, `DATA_CONFIG['date_range']` in config.py
3. Ensure features list matches available indicators
4. Re-run data preparation: `python scripts/fetch_data.py`

## Troubleshooting

### Common Issues

**"No such file: data/DJI_2016_2025.csv"**
→ Run `python scripts/fetch_data.py` first

**"CUDA out of memory"**
→ Reduce `batch_size` in `DQN_CONFIG` or use CPU/MPS

**"Validation Sharpe always negative"**
→ Increase `max_episodes` or adjust learning rate; market data may be challenging

**Results differ from paper**
→ Verify random seed, check data split dates match paper exactly, ensure transaction costs applied correctly

### Debugging Training

Add verbosity to training loop:
```python
# In scripts/train.py, modify train_one_episode()
if step % 100 == 0:
    print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Epsilon: {agent.epsilon:.4f}")
```

Check replay buffer:
```python
print(f"Replay buffer size: {len(agent.memory)}/{agent.memory.maxlen}")
```

Monitor Q-values:
```python
q_values = agent.model(state.unsqueeze(0))
print(f"Q-values: {q_values.detach().cpu().numpy()}")
```

## Research Paper Alignment

This implementation strictly follows the methodology in the research paper:
- Section 2.3: LORS mechanism with 8 bifurcation configurations
- Section 3.1: Data split protocol (60/20/20, chronological)
- Section 3.3: Trading environment design (transaction costs, reward function)
- Section 3.4: DRL training protocol (DQN, experience replay, epsilon-greedy)
- Section 4.1: Experimental setup (hyperparameters, evaluation metrics)

When modifying the code, maintain alignment with the paper's methodology to ensure reproducibility.
