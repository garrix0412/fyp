"""
Models Module

This module provides all model implementations:
- LORSTransformerDRL: Main model with LORS mechanism
- Baseline models: LSTM, GRU, CNN, MLP, Transformer, etc.
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
from .registry import MODEL_REGISTRY, get_model, list_models

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

    # Registry functions
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
]
