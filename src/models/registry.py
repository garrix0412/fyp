"""
Model Registry

Centralized registry for all models.
Provides easy access to model classes and their configurations.

Extended for Graph-LORS Trader:
- GraphLORSTrader: Main model with dynamic correlation graph
- GraphTransformer: Ablation without LORS
- StaticGraphLORS: Ablation with static graph
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

# Import graph models (lazy import to avoid dependency issues)
try:
    from .graph_lors_trader import GraphLORSTrader, GraphTransformer, StaticGraphLORS
    GRAPH_MODELS_AVAILABLE = True
except ImportError:
    GRAPH_MODELS_AVAILABLE = False
    print("Warning: Graph models not available. Install torch_geometric to enable.")


# Model Registry - maps model names to classes
MODEL_REGISTRY = {
    # Original models
    'LORSTransformerDRL': LORSTransformerDRL,
    'LORSTransformer': LORSTransformer,
    'LSTM_DQN': LSTM_DQN,
    'GRU_DQN': GRU_DQN,
    'CNN_DQN': CNN_DQN,
    'DQN_MLP': DQN_MLP,
    'Transformer_DQN': Transformer_DQN,
    'AttentionLSTM_DQN': AttentionLSTM_DQN,
    'ChaoticRNN_DQN': ChaoticRNN_DQN,
}

# Add graph models if available
if GRAPH_MODELS_AVAILABLE:
    MODEL_REGISTRY.update({
        'GraphLORSTrader': GraphLORSTrader,
        'GraphTransformer': GraphTransformer,
        'StaticGraphLORS': StaticGraphLORS,
    })

# List of models that require graph input
GRAPH_MODELS = ['GraphLORSTrader', 'GraphTransformer', 'StaticGraphLORS']


def get_model(model_name, **kwargs):
    """
    Get model instance by name

    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        **kwargs: Model-specific arguments (input_dim, hidden_dim, etc.)

    Returns:
        Model instance

    Raises:
        ValueError: If model_name not found in registry

    Example:
        >>> model = get_model('LORSTransformerDRL', input_dim=10, embed_dim=128)
        >>> model = get_model('LSTM_DQN', input_dim=10, hidden_dim=128)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_models():
    """
    List all available models

    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info():
    """
    Get information about all available models

    Returns:
        Dictionary with model names and their descriptions
    """
    info = {
        # Original models
        'LORSTransformerDRL': {
            'description': 'Main model: LSTM + Transformer + LORS + DQN',
            'category': 'Main',
            'paper_section': 'Section 2.3',
            'requires_graph': False,
        },
        'LORSTransformer': {
            'description': 'LORS + Transformer (ablation study)',
            'category': 'Ablation',
            'paper_section': 'Section 4.3',
            'requires_graph': False,
        },
        'Transformer_DQN': {
            'description': 'Standard Transformer without LORS',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        'AttentionLSTM_DQN': {
            'description': 'LSTM with attention mechanism',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        'ChaoticRNN_DQN': {
            'description': 'Chaotic dynamics without retrograde signaling',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        'LSTM_DQN': {
            'description': 'Basic LSTM baseline',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        'GRU_DQN': {
            'description': 'Basic GRU baseline',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        'CNN_DQN': {
            'description': 'Convolutional neural network baseline',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        'DQN_MLP': {
            'description': 'Multi-layer perceptron baseline',
            'category': 'Baseline',
            'paper_section': 'Section 4.2',
            'requires_graph': False,
        },
        # Graph-enhanced models (new)
        'GraphLORSTrader': {
            'description': 'Graph-LORS: Dynamic correlation graph + LORS-Transformer',
            'category': 'Graph-Main',
            'paper_section': 'Section 5',
            'requires_graph': True,
        },
        'GraphTransformer': {
            'description': 'Graph + Transformer (ablation: no LORS)',
            'category': 'Graph-Ablation',
            'paper_section': 'Section 6.5',
            'requires_graph': True,
        },
        'StaticGraphLORS': {
            'description': 'Static graph + LORS (ablation: no dynamic graph)',
            'category': 'Graph-Ablation',
            'paper_section': 'Section 6.5',
            'requires_graph': True,
        },
    }
    return info


def is_graph_model(model_name: str) -> bool:
    """
    Check if a model requires graph input

    Args:
        model_name: Name of the model

    Returns:
        True if model requires graph data
    """
    return model_name in GRAPH_MODELS


def print_model_info():
    """
    Print formatted information about all available models
    """
    info = get_model_info()

    print("\n" + "="*70)
    print("Available Models")
    print("="*70)

    # Group by category
    categories = {}
    for model_name, model_info in info.items():
        category = model_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((model_name, model_info))

    # Print by category
    for category in ['Main', 'Graph-Main', 'Ablation', 'Graph-Ablation', 'Baseline']:
        if category in categories:
            print(f"\n{category} Models:")
            for model_name, model_info in categories[category]:
                graph_tag = " [Graph]" if model_info.get('requires_graph', False) else ""
                print(f"  - {model_name:<20} {model_info['description']}{graph_tag}")

    print("="*70)


if __name__ == "__main__":
    """Test registry"""
    print("Model Registry Test")
    print_model_info()

    print("\nAvailable models:", list_models())
