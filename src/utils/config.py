"""
Configuration Module - Hyperparameter Settings
Strictly aligned with Paper Section 4.1: Experimental Setup
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

    # Full experiment mode
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
        mode: 'quick_test' or 'full_experiment'

    Returns:
        Complete configuration dictionary
    """
    if mode == 'quick_test':
        exp_config = EXPERIMENT_CONFIG['quick_test']
    elif mode == 'full_experiment':
        exp_config = EXPERIMENT_CONFIG['full_experiment']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    config = {
        'data': DATA_CONFIG,
        'env': ENV_CONFIG,
        'dqn': DQN_CONFIG,
        'train': {**TRAIN_CONFIG, **exp_config},
        'models': MODEL_CONFIG,
        'vis': VIS_CONFIG,
    }

    return config


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
