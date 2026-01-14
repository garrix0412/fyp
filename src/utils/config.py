"""
Configuration Module - Hyperparameter Settings
Strictly aligned with Paper Section 4.1: Experimental Setup

Extended for Graph-LORS Trader:
- Graph configuration for dynamic correlation graphs
- New model configurations for graph-enhanced models
"""

# ============================================================================
# Data Configuration
# ============================================================================
DATA_CONFIG = {
    'ticker': '^DJI',
    'csv_path': 'data/DJI_2016_2025.csv',
    'date_range': ('2016-01-01', '2025-01-01'),

    # Feature list (Paper Section 3.3)
    'features': [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'Volatility_20'
    ],

    # Data split ratio (Paper Section 3.1: 60/20/20)
    'split_ratio': {
        'train': 0.6,
        'val': 0.2,
        'test': 0.2
    },
}

# ============================================================================
# Graph Configuration (New for Graph-LORS Trader)
# ============================================================================
GRAPH_CONFIG = {
    # Asset configuration
    'target': '^DJI',
    'context_assets': ['^GSPC', '^VIX', 'GC=F', 'DX-Y.NYB'],

    # Data paths
    'price_matrix_path': 'data/price_matrix_2016_2025.csv',
    'multi_asset_path': 'data/multi_asset_2016_2025.csv',
    'graph_target_path': 'data/DJI_graph_2016_2025.csv',

    # Graph construction parameters
    'correlation_window': 60,          # Rolling window for correlation
    'top_k': 3,                        # Sparsification parameter

    # Node feature columns (10 features from GraphNodeFeatureBuilder)
    'node_features': [
        'Return_Mean', 'Return_Std', 'Return_Skew',           # 3
        'Momentum_5d', 'Momentum_20d', 'Momentum_60d',        # 3
        'Volatility_10d', 'Volatility_20d',                   # 2
        'Relative_Strength', 'Trend_vs_MA'                    # 2
    ],

    # Graph encoder parameters
    'node_feature_dim': 10,            # Number of node features (must match GraphNodeFeatureBuilder)
    'graph_hidden_dim': 32,            # GAT hidden dimension
    'context_dim': 64,                 # Graph context vector dimension
    'graph_heads': 4,                  # Number of GAT attention heads
}

# ============================================================================
# Trading Environment Configuration
# ============================================================================
ENV_CONFIG = {
    'window_size': 120,                # Observation window size
    'initial_cash': 1000000,           # Initial capital (Paper: 1,000,000)
    'transaction_fee': 0.001,          # Transaction cost (Paper: 10 bps = 0.001)
    'slippage': 0.0,                   # Slippage (Paper: φ_t = 0)
}

# ============================================================================
# DQN Training Configuration
# ============================================================================
DQN_CONFIG = {
    # Optimizer
    'lr': 0.0001,                      # Learning rate (Paper original value)
    'weight_decay': 1e-4,              # L2 regularization

    # Reinforcement Learning parameters
    'gamma': 0.99,                     # Discount factor
    'epsilon_start': 1.0,              # Initial exploration rate
    'epsilon_min': 0.01,               # Minimum exploration rate
    'epsilon_decay': 0.995,            # Exploration rate decay (Paper original)

    # Experience replay
    'buffer_size': 20000,              # Replay buffer size (Paper original)
    'batch_size': 64,                  # Batch size (Paper original)

    # Target network
    'target_update_freq': 10,          # Target network update frequency

    # Gradient clipping
    'max_grad_norm': 1.0,              # Gradient clipping (Paper mentioned)
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAIN_CONFIG = {
    # Training epochs
    'max_episodes': 100,               # Maximum training rounds (Paper original)
    'patience': 15,                    # Early stopping patience (Paper original)
    'val_freq': 5,                     # Validation frequency (every 5 rounds)

    # Random seeds (Paper Section 4.4b: seeds = {42, 43, 44, 45, 46})
    'seeds': [42],                     # Single seed for quick test, 5 for full experiment

    # Save paths
    'checkpoint_dir': 'checkpoints',
    'results_dir': 'results',
    'figures_dir': 'figures',
    'logs_dir': 'logs',
}

# ============================================================================
# Model Architecture Configuration
# ============================================================================
MODEL_CONFIG = {
    # LORSTransformerDRL (Paper main model)
    'LORSTransformerDRL': {
        'input_dim': 10,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'output_dim': 3,
        'n_layers': 3,
        'dropout': 0.15,
    },

    # LORSTransformer (Simplified version)
    'LORSTransformer': {
        'input_dim': 10,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'output_dim': 3,
        'n_layers': 3,
        'dropout': 0.15,
    },

    # LSTM_DQN
    'LSTM_DQN': {
        'input_dim': 10,
        'hidden_dim': 128,
        'output_dim': 3,
        'n_layers': 2,
        'dropout': 0.15,
    },

    # GRU_DQN
    'GRU_DQN': {
        'input_dim': 10,
        'hidden_dim': 128,
        'output_dim': 3,
        'n_layers': 2,
        'dropout': 0.15,
    },

    # CNN_DQN
    'CNN_DQN': {
        'input_dim': 10,
        'window_size': 120,
        'output_dim': 3,
    },

    # Transformer_DQN
    'Transformer_DQN': {
        'input_dim': 10,
        'embed_dim': 128,
        'num_heads': 4,
        'hidden_dim': 256,
        'output_dim': 3,
        'n_layers': 2,
        'dropout': 0.15,
    },

    # AttentionLSTM_DQN
    'AttentionLSTM_DQN': {
        'input_dim': 10,
        'hidden_dim': 128,
        'output_dim': 3,
        'n_layers': 2,
        'dropout': 0.15,
    },

    # ChaoticRNN_DQN
    'ChaoticRNN_DQN': {
        'input_dim': 10,
        'hidden_dim': 128,
        'output_dim': 3,
        'n_layers': 2,
        'dropout': 0.15,
    },

    # DQN_MLP
    'DQN_MLP': {
        'input_dim': 10,
        'window_size': 120,
        'output_dim': 3,
    },

    # =========================================================================
    # Graph-Enhanced Models (New)
    # =========================================================================

    # GraphLORSTrader (Main model for Graph-LORS paper)
    'GraphLORSTrader': {
        # Target encoder params (same as LORSTransformerDRL)
        'input_dim': 10,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'n_layers': 3,
        'dropout': 0.15,
        # Graph encoder params
        'n_assets': 5,
        'node_feature_dim': 10,
        'graph_hidden_dim': 32,
        'context_dim': 64,
        'graph_heads': 4,
        # Output
        'output_dim': 3,
    },

    # GraphTransformer (Ablation: Graph + Transformer, no LORS)
    'GraphTransformer': {
        'input_dim': 10,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'n_layers': 3,
        'dropout': 0.15,
        'n_assets': 5,
        'node_feature_dim': 10,
        'graph_hidden_dim': 32,
        'context_dim': 64,
        'graph_heads': 4,
        'output_dim': 3,
    },

    # StaticGraphLORS (Ablation: Static graph, isolate dynamic graph contribution)
    'StaticGraphLORS': {
        'input_dim': 10,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'n_layers': 3,
        'dropout': 0.15,
        'n_assets': 5,
        'node_feature_dim': 10,
        'graph_hidden_dim': 32,
        'context_dim': 64,
        'graph_heads': 4,
        'output_dim': 3,
    },
}

# ============================================================================
# Experiment Mode Configuration
# ============================================================================
EXPERIMENT_CONFIG = {
    # Quick test mode (for debugging)
    'quick_test': {
        'max_episodes': 5,
        'models': ['DQN_MLP', 'LSTM_DQN'],
        'seeds': [42],
    },

    # Full experiment mode (original baselines)
    'full_experiment': {
        'max_episodes': 100,
        'models': [
            'LORSTransformerDRL',
            'LORSTransformer',
            'LSTM_DQN',
            'GRU_DQN',
            'CNN_DQN',
            'Transformer_DQN',
            'AttentionLSTM_DQN',
            'ChaoticRNN_DQN',
            'DQN_MLP',
        ],
        'seeds': [42, 43, 44, 45, 46],
    },

    # Graph model quick test
    'graph_quick_test': {
        'max_episodes': 5,
        'models': ['GraphLORSTrader'],
        'seeds': [42],
        'use_graph': True,
    },

    # Graph model full experiment
    'graph_full_experiment': {
        'max_episodes': 100,
        'models': [
            'GraphLORSTrader',      # Main model
            'GraphTransformer',     # Ablation: no LORS
            'StaticGraphLORS',      # Ablation: static graph
            'LORSTransformerDRL',   # Baseline: no graph
        ],
        'seeds': [42, 43, 44, 45, 46],
        'use_graph': True,
    },

    # Ablation study: Graph type comparison
    'graph_ablation': {
        'max_episodes': 100,
        'models': [
            'LORSTransformerDRL',   # No graph
            'StaticGraphLORS',      # Static graph
            'GraphLORSTrader',      # Dynamic graph
        ],
        'seeds': [42, 43, 44, 45, 46],
        'use_graph': True,
    },
}

# ============================================================================
# Visualization Configuration
# ============================================================================
VIS_CONFIG = {
    'dpi': 300,
    'figure_size': (12, 6),
    'save_format': 'png',

    # Paper figures
    'generate_figures': {
        'figure2': True,   # Multi-metric radar
        'figure3': True,   # Risk-return scatter
        'figure4': True,   # Performance dynamics
        'table1': True,    # Results table
        'table2': True,    # Trading behavior
    },
}

# ============================================================================
# Helper Functions
# ============================================================================
def get_config(mode='quick_test'):
    """
    Get complete configuration for specified mode

    Args:
        mode: One of:
            - 'quick_test': Quick debugging (original models)
            - 'full_experiment': Full experiment (original models)
            - 'graph_quick_test': Quick debugging (graph models)
            - 'graph_full_experiment': Full experiment (graph models)
            - 'graph_ablation': Ablation study for graph contribution

    Returns:
        Complete configuration dictionary
    """
    if mode not in EXPERIMENT_CONFIG:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(EXPERIMENT_CONFIG.keys())}")

    exp_config = EXPERIMENT_CONFIG[mode]

    config = {
        'data': DATA_CONFIG,
        'env': ENV_CONFIG,
        'dqn': DQN_CONFIG,
        'train': {**TRAIN_CONFIG, **exp_config},
        'models': MODEL_CONFIG,
        'vis': VIS_CONFIG,
        'graph': GRAPH_CONFIG,
    }

    return config


def get_graph_config():
    """
    Get graph-specific configuration

    Returns:
        Graph configuration dictionary
    """
    return GRAPH_CONFIG.copy()


def print_config(config):
    """Print configuration information"""
    print("\n" + "="*70)
    print("Experiment Configuration")
    print("="*70)

    for section, params in config.items():
        if section == 'models':
            print(f"\n{section.upper()}: {len(params)} models")
        else:
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 50:
                    print(f"  {key}: {type(value).__name__}")
                else:
                    print(f"  {key}: {value}")

    print("="*70)


if __name__ == "__main__":
    # Test configuration
    print("Configuration Module Test")

    # Quick test mode
    config_quick = get_config('quick_test')
    print_config(config_quick)

    # Full experiment mode
    config_full = get_config('full_experiment')
    print("\nModel list:", config_full['train']['models'])
    print("Seed list:", config_full['train']['seeds'])
