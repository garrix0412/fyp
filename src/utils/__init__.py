"""
Utilities Module

This module provides utility functions and classes for:
- Data splitting and preprocessing (data_utils)
- Trading performance metrics (metrics)
- Configuration management (config)
"""

from .data_utils import ChronologicalSplitter, AntiLeakagePreprocessor, verify_leakage_guards
from .metrics import TradingMetrics, format_table1_latex
from .config import (
    DATA_CONFIG, ENV_CONFIG, DQN_CONFIG, TRAIN_CONFIG,
    MODEL_CONFIG, EXPERIMENT_CONFIG, VIS_CONFIG,
    get_config, print_config
)

__all__ = [
    # Data utilities
    'ChronologicalSplitter',
    'AntiLeakagePreprocessor',
    'verify_leakage_guards',

    # Metrics
    'TradingMetrics',
    'format_table1_latex',

    # Configuration
    'DATA_CONFIG',
    'ENV_CONFIG',
    'DQN_CONFIG',
    'TRAIN_CONFIG',
    'MODEL_CONFIG',
    'EXPERIMENT_CONFIG',
    'VIS_CONFIG',
    'get_config',
    'print_config',
]
