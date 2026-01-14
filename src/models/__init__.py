"""
Models Module

This module provides all model implementations:
- LORSTransformerDRL: Main model with LORS mechanism
- Baseline models: LSTM, GRU, CNN, MLP, Transformer, etc.
- Graph-enhanced models: GraphLORSTrader, GraphTransformer, StaticGraphLORS
- Model registry for easy access
"""

from .lors_transformer_drl import LORSTransformerDRL, LORSTransformer
from .baselines import (
    LSTM_DQN,
    GRU_DQN,
    CNN_DQN,
    DQN_MLP,
    Transformer_DQN,
    AttentionLSTM_DQN,
    ChaoticRNN_DQN,
)
from .registry import (
    MODEL_REGISTRY, get_model, list_models,
    GRAPH_MODELS, GRAPH_MODELS_AVAILABLE, is_graph_model
)

# Import graph models if available
if GRAPH_MODELS_AVAILABLE:
    from .graph_lors_trader import GraphLORSTrader, GraphTransformer, StaticGraphLORS
    from .graph_encoder import GraphAttentionEncoder, GraphContextModule

__all__ = [
    # Main models
    'LORSTransformerDRL',
    'LORSTransformer',

    # Baseline models
    'LSTM_DQN',
    'GRU_DQN',
    'CNN_DQN',
    'DQN_MLP',
    'Transformer_DQN',
    'AttentionLSTM_DQN',
    'ChaoticRNN_DQN',

    # Graph-enhanced models (if available)
    'GraphLORSTrader',
    'GraphTransformer',
    'StaticGraphLORS',
    'GraphAttentionEncoder',
    'GraphContextModule',

    # Registry functions
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
    'GRAPH_MODELS',
    'GRAPH_MODELS_AVAILABLE',
    'is_graph_model',
]
