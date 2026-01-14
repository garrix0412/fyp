"""
Utilities Module

This module provides utility functions and classes for:
- Data splitting and preprocessing (data_utils)
- Trading performance metrics (metrics)
- Configuration management (config)
- Dynamic graph construction (graph_utils) - NEW for Graph-LORS Trader
- Interpretability analysis (interpret_utils) - NEW for Graph-LORS Trader
"""

from .data_utils import ChronologicalSplitter, AntiLeakagePreprocessor, verify_leakage_guards
from .metrics import TradingMetrics, format_table1_latex
from .config import (
    DATA_CONFIG, ENV_CONFIG, DQN_CONFIG, TRAIN_CONFIG,
    MODEL_CONFIG, EXPERIMENT_CONFIG, VIS_CONFIG, GRAPH_CONFIG,
    get_config, get_graph_config, print_config
)
from .graph_utils import (
    DynamicGraphBuilder,
    GraphNodeFeatureBuilder,
    verify_no_future_leakage,
    load_price_matrix
)
from .interpret_utils import (
    GraphAttentionAnalyzer,
    LORSConfigAnalyzer,
    InterpretabilityAnalyzer
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
    'GRAPH_CONFIG',
    'get_config',
    'get_graph_config',
    'print_config',

    # Graph utilities (NEW for Graph-LORS Trader)
    'DynamicGraphBuilder',
    'GraphNodeFeatureBuilder',
    'verify_no_future_leakage',
    'load_price_matrix',

    # Interpretability utilities (NEW for Graph-LORS Trader)
    'GraphAttentionAnalyzer',
    'LORSConfigAnalyzer',
    'InterpretabilityAnalyzer',
]
