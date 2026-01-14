---

# Paper Topic Proposal (Revised)

## Title

**Graph-Enhanced LORS-Transformer for Cross-Asset Trading: A Dynamic Correlation Approach with Interpretable Attention**

> Abbreviation: **Graph-LORS Trader**

---

## 1. Background & Motivation

The existing project **LORSTransformerDRL** embeds LORS (Lee Oscillator Retrograde Signal) into Transformer attention and uses DQN to directly optimize buy/hold/sell decisions, achieving end-to-end decision learning with strict time-series anti-leakage protocols and consistent transaction cost handling. Experiments show strong risk-return performance on DJI daily data (e.g., Sharpe and CR outperform multiple baselines).

However, in real-world trading scenarios, single-asset models relying solely on their own historical features are often insufficient:

* **Cross-market linkages** (USD, commodities, interest rates, risk appetite, sector rotation) significantly affect target asset movements
* **Single-asset models** cannot capture systemic risk transmission and inter-market dependencies
* **Lack of interpretability** in existing models limits their practical deployment and regulatory compliance

Therefore, this paper proposes a **focused extension** to the existing LORS-Transformer framework with two core enhancements:

1. **Dynamic Correlation Graph**: Explicitly model multi-asset relationships as trading decision context
2. **Dual-layer Interpretability**: Graph attention weights + LORS configuration weights for explainable decisions

---

## 2. Research Questions

**RQ1 (Cross-Asset Context):** Does dynamic correlation graph context significantly improve LORS-Transformer's Sharpe ratio, MDD, and stability compared to single-asset models?

**RQ2 (Interpretability):** Can graph attention weights and LORS configuration weights provide consistent economic explanations (e.g., risk asset nodes gain importance during market stress)?

**RQ3 (Generalization):** Does the Graph-LORS framework generalize across different markets (US, Japan, Hong Kong)?

---

## 3. Expected Contributions

* **C1:** Propose Graph-LORS Trader: a **Dynamic Correlation Graph + LORS-Transformer** trading framework, evaluated under strict chronological split and cost-consistent protocols.

* **C2:** Provide dual-layer interpretability: **Graph attention weights** (cross-asset influence) + **LORS 8-configuration weights** (market regime adaptation).

* **C3:** Cross-market generalization validation: extend from single DJI to multiple international indices (^GSPC, ^N225, ^HSI).

---

## 4. Method Overview

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Graph-LORS Trader                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ Multi-Asset   │    │ Target Asset │                   │
│  │ Prices (5     │    │ Sequence     │                   │
│  │ nodes)        │    │ (120-day)    │                   │
│  └──────┬───────┘    └──────┬───────┘                   │
│         │                    │                           │
│         ▼                    ▼                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ Dynamic Corr │    │LORS-Transformer│  ← Reuse        │
│  │ Graph A(t)   │    │   Encoder     │                  │
│  └──────┬───────┘    └──────┬───────┘                   │
│         │                    │                           │
│         ▼                    │                           │
│  ┌──────────────┐            │                           │
│  │  GAT Encoder │            │                           │
│  │  (2 layers)  │            │                           │
│  └──────┬───────┘            │                           │
│         │    g_t             │  h_t                      │
│         ▼                    ▼                           │
│  ┌─────────────────────────────────────┐                │
│  │      Gated Fusion                    │                │
│  │  h' = h_t + gate ⊙ W·g_t            │                │
│  └──────────────────┬──────────────────┘                │
│                     │                                    │
│                     ▼                                    │
│              ┌──────────────┐                           │
│              │   DQN Head   │  ← Reuse                  │
│              │ (buy/hold/sell)│                          │
│              └──────────────┘                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Data & Features

**(A) Price & Technical Indicators (Existing)**

Continue using the existing 10 features: OHLCV + RSI/MACD/Bollinger/Volatility.

**(B) Dynamic Graph Construction (New)**

Graph nodes (5 assets, manageable complexity):
```python
GRAPH_ASSETS = {
    'target': '^DJI',           # Target asset
    'context': [
        '^GSPC',                # S&P 500 (US broad market)
        '^VIX',                 # Volatility index (risk sentiment)
        'GC=F',                 # Gold (safe haven)
        'DX-Y.NYB',             # USD index (macro proxy)
    ]
}
```

Dynamic adjacency matrix construction:
* For each trading day t, compute correlation matrix using historical window [t-60, t-1]
* Apply top-k sparsification to get adjacency matrix A(t)
* **Critical**: Only use data before time t to prevent information leakage

---

## 5. Model Architecture

### 5.1 Target Asset Encoder: LORS-Transformer (Reuse Existing)

Directly reuse the existing LORSTransformerDRL structure:
- Embedding → LSTM → Transformer layers → LORS module (8 configurations) → dual pooling
- LORS 8-configuration weights learned via attention (inherently interpretable)

### 5.2 Graph Context Encoder: GAT (New)

Input: Graph G(t) = (V, E, A(t)), node features are asset statistics over the window.

```python
class GraphAttentionEncoder(nn.Module):
    """2-layer GAT with attention weight output"""

    def __init__(self, in_dim=10, hidden_dim=32, out_dim=64, heads=4):
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, out_dim, heads=1)

    def forward(self, x, edge_index):
        h, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        h = F.elu(h)
        out, attn2 = self.gat2(h, edge_index, return_attention_weights=True)
        return out, (attn1, attn2)  # Return attention for interpretability
```

Output: Market context vector g_t (target node's neighborhood aggregation).

### 5.3 Gated Fusion (New)

Combine target asset representation h_t with graph context g_t:

```python
gate = sigmoid(W1 @ concat([h_t, g_t]))
h_prime = h_t + gate * (W2 @ g_t)
```

Then feed h_prime into Q-head for action selection.

### 5.4 Decision Head (Reuse Existing)

* DQN: Q(s,a) for 3 actions (buy/hold/sell)
* Experience replay, target network, epsilon-greedy (all existing)

---

## 6. Experiment Design

### 6.1 Datasets & Tasks

| Dataset | Purpose | Notes |
|---------|---------|-------|
| ^DJI | Main experiment | Primary validation |
| ^GSPC | Cross-market | US broad market |
| ^N225 | Cross-market | Japan |
| ^HSI | Cross-market | Hong Kong |

### 6.2 Training/Validation/Test Protocol (Reuse & Enforce)

* Chronological split 60/20/20, no shuffling
* Scaler fit only on train, transform val/test
* Validation Sharpe for model selection + early stopping
* Test single-pass, no parameter tuning on test
* Multi-seed (5 seeds: 42, 43, 44, 45, 46) for statistical significance

### 6.3 Evaluation Metrics

**Reuse existing:**
* CR, Sharpe, MDD, Win Rate
* Behavioral metrics: holding time, turnover frequency, per-trade return

**Additional (for robustness):**
* Cost sensitivity: transaction_fee ∈ {5, 10, 20} bps

### 6.4 Baselines

**Reuse existing model registry:**
* LORSTransformerDRL (no graph baseline)
* Transformer_DQN, CNN_DQN, DQN_MLP
* LSTM_DQN, GRU_DQN
* Random

**New baselines:**
* Graph-Transformer (no LORS) - isolate graph contribution
* Static-Graph-LORS (fixed correlation) - isolate dynamic graph contribution

### 6.5 Ablation Studies (Focused)

| Ablation | Variants | Purpose |
|----------|----------|---------|
| Graph Type | No graph / Static graph / Dynamic graph | Validate dynamic graph value |
| Graph Sparsity | Top-3 / Top-5 / Full correlation | Optimal sparsity level |
| Fusion Method | Concatenation / Addition / Gated | Best fusion strategy |

### 6.6 Interpretability Analysis

**Graph Attention Analysis:**
* Visualize edge attention weights α_ij(t) over time
* Compare attention patterns during: normal periods vs. crisis periods (e.g., COVID crash)
* Statistical analysis: which nodes gain importance during high volatility?

**LORS Configuration Analysis:**
* Output LORS 8-configuration weights w_i(t) distribution
* Correlate with market volatility/trend strength
* Identify which bifurcation configurations activate under different regimes

---

## 7. Implementation Plan

### 7.1 Reuse Existing Modules (No Changes)

* Training entry & validation: `scripts/train.py`
* Anti-leakage pipeline: `src/utils/data_utils.py`
* Trading environment: `src/environment/trading_env.py`
* Metrics: `src/utils/metrics.py`
* Model registry: `src/models/registry.py`

### 7.2 New Modules

| Module | Location | LOC (est.) |
|--------|----------|------------|
| Dynamic graph construction | `src/utils/graph_utils.py` | ~80 |
| GAT encoder | `src/models/graph_encoder.py` | ~100 |
| Graph-LORS Trader | `src/models/graph_lors_trader.py` | ~150 |
| Interpretability utils | `src/utils/interpret_utils.py` | ~100 |
| **Total new code** | | **~430 lines** |

### 7.3 Implementation Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| Phase 1 | Graph construction + data preparation | 2 weeks |
| Phase 2 | GAT encoder + fusion module | 2 weeks |
| Phase 3 | Training debugging + main experiments | 2 weeks |
| Phase 4 | Ablation + cross-market validation | 2 weeks |
| Phase 5 | Interpretability analysis + paper writing | 2 weeks |
| **Total** | | **10 weeks** |

---

## 8. Risk Assessment & Mitigation

| Risk | Mitigation |
|------|------------|
| **Data leakage in graph** | Strictly use [t-window, t-1] for correlation; add unit tests |
| **Training instability** | Start with simple 2-layer GAT; use validation Sharpe early stopping |
| **Graph too sparse/dense** | Ablation on top-k values; default k=3 |
| **Cross-market data gaps** | Use only assets with complete daily data; forward-fill holidays |

---

## 9. Paper Structure (Outline)

```
1. Introduction
   - Motivation: Single-asset models miss cross-market dynamics
   - Contribution: Graph-LORS + dual-layer interpretability

2. Related Work
   - 2.1 LORS and chaotic networks in finance
   - 2.2 Graph neural networks for financial markets
   - 2.3 Interpretable trading systems

3. Method
   - 3.1 Problem formulation
   - 3.2 Dynamic correlation graph construction
   - 3.3 Graph-LORS Trader architecture
   - 3.4 Interpretability outputs

4. Experiments
   - 4.1 Data and anti-leakage protocol
   - 4.2 Main results (DJI)
   - 4.3 Ablation: graph contribution
   - 4.4 Cross-market generalization

5. Interpretability Analysis
   - 5.1 Graph attention visualization
   - 5.2 LORS configuration weight analysis
   - 5.3 Case study: COVID-19 market crash

6. Conclusion & Future Work
   - Summary of contributions
   - Future: Multi-modal signals, strategy switching (Bandit/MoE)
```

---

## 10. Future Work (Deferred from Original Proposal)

The following extensions are valuable but deferred to maintain focus:

1. **Multi-Modal External Signals**: News sentiment (FinBERT), macro factors
2. **Strategy Switching**: Expert strategy library + Contextual Bandit/MoE switcher
3. **RL Algorithm Comparison**: SAC/PPO vs DQN
4. **Continuous Position Sizing**: Replace discrete actions with continuous allocation

These can form the basis of follow-up papers or thesis extensions.

---

## 11. Conclusion

This revised proposal focuses on a **single, well-defined contribution**: integrating dynamic cross-asset correlation graphs into the LORS-Transformer trading framework with interpretable attention mechanisms. By narrowing the scope from 4 innovations to 2 core contributions, we ensure:

1. **Feasibility**: ~430 lines of new code, 10-week timeline
2. **Rigor**: Strict anti-leakage, multi-seed validation, cross-market testing
3. **Novelty**: Graph-enhanced LORS with dual-layer interpretability
4. **Extensibility**: Clear future work directions for follow-up research

---
