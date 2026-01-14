"""
Graph-LORS Trader: Dynamic Correlation Graph + LORS-Transformer

Main model integrating:
1. LORS-Transformer encoder for target asset temporal patterns
2. GAT encoder for multi-asset graph context
3. Gated fusion mechanism for combining representations
4. DQN head for trading decisions

Paper Section 5: Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional

from .graph_encoder import GraphContextModule


class LORSTransformerEncoder(nn.Module):
    """
    LORS-Transformer Encoder (extracted from LORSTransformerDRL)

    Encodes target asset sequence using LSTM + Transformer + LORS mechanism.
    Returns both the encoding and LORS configuration weights for interpretability.
    """

    def __init__(self, input_dim: int = 10, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 256,
                 n_layers: int = 3, dropout: float = 0.15):
        super().__init__()

        self.embed_dim = embed_dim

        # Embedding and positional encoding
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))

        # Temporal processing (LSTM)
        self.temporal = nn.LSTM(embed_dim, embed_dim, batch_first=True)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # LORS parameters - 8 configurations with different bifurcation parameters
        self.lors_params = nn.ParameterList([
            nn.Parameter(torch.tensor([0.6, 0.6, 0.6, 0.6, 0.5, -0.6, -0.6, -0.6, 1.2, 60.0, 1.2])),
            nn.Parameter(torch.tensor([0.55, 0.55, 0.55, 0.55, 0.4, -0.55, -0.55, -0.55, 1.1, 55.0, 1.1])),
            nn.Parameter(torch.tensor([0.7, 0.7, 0.7, 0.7, 0.6, -0.7, -0.7, -0.7, 1.3, 65.0, 1.3])),
            nn.Parameter(torch.tensor([0.45, 0.45, 0.45, 0.45, 0.3, -0.45, -0.45, -0.45, 1.0, 50.0, 1.0])),
            nn.Parameter(torch.tensor([0.65, 0.65, 0.65, 0.65, 0.55, -0.65, -0.65, -0.65, 1.25, 62.0, 1.25])),
            nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.45, -0.5, -0.5, -0.5, 1.15, 58.0, 1.15])),
            nn.Parameter(torch.tensor([0.75, 0.75, 0.75, 0.75, 0.65, -0.75, -0.75, -0.75, 1.35, 68.0, 1.35])),
            nn.Parameter(torch.tensor([0.4, 0.4, 0.4, 0.4, 0.35, -0.4, -0.4, -0.4, 0.95, 48.0, 0.95]))
        ])

        # Attention mechanism for LORS combination
        self.lors_attention = nn.Linear(embed_dim, 8)
        self.alpha = nn.Parameter(torch.tensor(1.2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through LORS-Transformer encoder

        Args:
            x: Input sequence (batch, seq_len, input_dim)

        Returns:
            encoding: Encoded representation (batch, embed_dim)
            lors_info: Dict with LORS attention weights
        """
        # Embedding + positional encoding
        x = self.embedding(x)
        seq_len = x.size(1)
        x = self.dropout(x + self.pos_encoding[:, :seq_len, :])

        # Temporal processing with LSTM
        x, _ = self.temporal(x)

        # Store LORS attention weights for interpretability
        all_lors_weights = []

        # Transformer layers with LORS mechanism
        for layer in self.layers:
            attention_output = layer(x)
            sim = attention_output

            # LORS module - 8 configurations
            lors_outputs = []
            for params in self.lors_params:
                a1, a2, a3, a4, b1, b2, b3, b4, c, k, s = params

                # Initialize LORS state variables
                u, v, w, z = [torch.zeros_like(sim) for _ in range(4)]

                # LORS dynamics
                tempu = a1 * z + a2 * u - a3 * v + a4 * sim
                tempv = b1 * z - b2 * u - b3 * v + b4 * sim
                u = torch.tanh(s * tempu)
                v = torch.tanh(s * tempv)
                w = torch.tanh(s * sim)
                z = (u - v) * torch.exp(-self.alpha * k * sim ** 2) + c * w

                lors_outputs.append(z)

            # Attention-weighted combination of LORS outputs
            lors_outputs = torch.stack(lors_outputs, dim=2)  # (batch, seq, 8, embed)
            attn_weights = F.softmax(self.lors_attention(sim), dim=-1)  # (batch, seq, 8)
            all_lors_weights.append(attn_weights.detach())

            lors_combined = (lors_outputs * attn_weights.unsqueeze(-1)).sum(dim=2)

            # Residual connection + LayerNorm
            x = self.layer_norm(attention_output + 0.5 * lors_combined)

        # Dual pooling strategy
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        encoding = 0.5 * avg_pool + 0.5 * max_pool

        # Aggregate LORS weights across layers
        lors_info = {
            'lors_weights': all_lors_weights,  # List of (batch, seq, 8) per layer
            'avg_lors_weights': torch.stack(all_lors_weights).mean(dim=0).mean(dim=1)  # (batch, 8)
        }

        return encoding, lors_info


class GatedFusion(nn.Module):
    """
    Gated Fusion Module

    Combines target asset representation with graph context using
    learned gating mechanism.

    h' = h_t + gate * (W_g * g_t)
    gate = sigmoid(W_1 * [h_t; g_t])
    """

    def __init__(self, target_dim: int, context_dim: int, output_dim: int):
        """
        Initialize Gated Fusion

        Args:
            target_dim: Dimension of target representation
            context_dim: Dimension of graph context
            output_dim: Dimension of fused output
        """
        super().__init__()

        self.target_dim = target_dim
        self.context_dim = context_dim

        # Project context to target dimension
        self.context_proj = nn.Linear(context_dim, target_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(target_dim + target_dim, target_dim),
            nn.Sigmoid()
        )

        # Output projection (optional, if output_dim != target_dim)
        if output_dim != target_dim:
            self.output_proj = nn.Linear(target_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, h_t: torch.Tensor, g_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated fusion

        Args:
            h_t: Target representation (batch, target_dim)
            g_t: Graph context (batch, context_dim)

        Returns:
            h_prime: Fused representation (batch, output_dim)
        """
        # Project context
        g_proj = self.context_proj(g_t)

        # Compute gate
        concat = torch.cat([h_t, g_proj], dim=-1)
        gate_value = self.gate(concat)

        # Gated fusion
        h_prime = h_t + gate_value * g_proj

        # Output projection
        return self.output_proj(h_prime)


class GraphLORSTrader(nn.Module):
    """
    Graph-LORS Trader: Main Model

    Integrates dynamic correlation graph with LORS-Transformer for trading.

    Architecture:
        Target Asset Sequence → LORS-Transformer Encoder → h_t
        Multi-Asset Graph → GAT Encoder → g_t
        [h_t, g_t] → Gated Fusion → h'
        h' → DQN Head → Q(s, a)

    Outputs attention weights from both components for interpretability.
    """

    def __init__(self,
                 # Target encoder params
                 input_dim: int = 10,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.15,
                 # Graph encoder params
                 n_assets: int = 5,
                 node_feature_dim: int = 8,
                 graph_hidden_dim: int = 32,
                 context_dim: int = 64,
                 graph_heads: int = 4,
                 # Output params
                 output_dim: int = 3):
        """
        Initialize Graph-LORS Trader

        Args:
            input_dim: Target asset feature dimension
            embed_dim: LORS-Transformer embedding dimension
            num_heads: Number of attention heads in Transformer
            hidden_dim: Transformer feedforward dimension
            n_layers: Number of Transformer layers
            dropout: Dropout probability
            n_assets: Number of assets in graph
            node_feature_dim: Graph node feature dimension
            graph_hidden_dim: GAT hidden dimension
            context_dim: Graph context vector dimension
            graph_heads: Number of GAT attention heads
            output_dim: Number of actions (3: buy/hold/sell)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.context_dim = context_dim

        # 1. Target Asset Encoder (LORS-Transformer)
        self.target_encoder = LORSTransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )

        # 2. Graph Context Encoder (GAT)
        self.graph_encoder = GraphContextModule(
            n_assets=n_assets,
            node_feature_dim=node_feature_dim,
            hidden_dim=graph_hidden_dim,
            context_dim=context_dim,
            heads=graph_heads,
            dropout=dropout
        )

        # 3. Gated Fusion
        self.fusion = GatedFusion(
            target_dim=embed_dim,
            context_dim=context_dim,
            output_dim=embed_dim
        )

        # 4. DQN Head
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, target_seq: torch.Tensor,
                graph_data: Data,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through Graph-LORS Trader

        Args:
            target_seq: Target asset sequence (batch, seq_len, input_dim)
            graph_data: PyG Data object with graph structure
            return_attention: Whether to return attention weights

        Returns:
            q_values: Action Q-values (batch, output_dim)
            attention_info: Dict with all attention weights (if return_attention=True)
        """
        # 1. Encode target asset sequence
        h_t, lors_info = self.target_encoder(target_seq)

        # 2. Encode graph context
        g_t, graph_info = self.graph_encoder(graph_data, return_attention=return_attention)

        # Handle single sample vs batch
        if g_t.dim() == 1:
            g_t = g_t.unsqueeze(0)
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)

        # Ensure batch dimensions match
        if g_t.size(0) != h_t.size(0):
            # Repeat graph context for batch
            g_t = g_t.expand(h_t.size(0), -1)

        # 3. Gated fusion
        h_prime = self.fusion(h_t, g_t)

        # 4. Q-value output
        q_values = self.fc_out(h_prime)

        if return_attention:
            attention_info = {
                'lors_info': lors_info,
                'graph_info': graph_info,
                'h_t': h_t.detach(),
                'g_t': g_t.detach(),
                'h_prime': h_prime.detach()
            }
            return q_values, attention_info

        return q_values, None

    def get_interpretability_outputs(self, target_seq: torch.Tensor,
                                     graph_data: Data) -> Dict:
        """
        Get detailed interpretability outputs

        Returns all attention weights and intermediate representations
        for analysis.

        Args:
            target_seq: Target asset sequence
            graph_data: Graph data

        Returns:
            Dict with:
                - lors_config_weights: LORS 8-configuration weights
                - graph_attention: GAT attention weights
                - node_importance: Importance of each node to target
                - representations: Intermediate representations
        """
        q_values, attention_info = self.forward(
            target_seq, graph_data, return_attention=True
        )

        return {
            'q_values': q_values,
            'lors_config_weights': attention_info['lors_info']['avg_lors_weights'],
            'graph_attention': attention_info['graph_info']['gat_attention'],
            'node_embeddings': attention_info['graph_info']['node_embeddings'],
            'target_encoding': attention_info['h_t'],
            'graph_context': attention_info['g_t'],
            'fused_representation': attention_info['h_prime']
        }


class GraphTransformer(nn.Module):
    """
    Graph-Transformer Baseline (No LORS)

    Ablation model to isolate graph contribution.
    Uses standard Transformer without LORS mechanism.
    """

    def __init__(self,
                 input_dim: int = 10,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.15,
                 n_assets: int = 5,
                 node_feature_dim: int = 8,
                 graph_hidden_dim: int = 32,
                 context_dim: int = 64,
                 graph_heads: int = 4,
                 output_dim: int = 3):
        super().__init__()

        self.embed_dim = embed_dim

        # Standard Transformer encoder (no LORS)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        self.temporal = nn.LSTM(embed_dim, embed_dim, batch_first=True)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # Graph encoder
        self.graph_encoder = GraphContextModule(
            n_assets=n_assets,
            node_feature_dim=node_feature_dim,
            hidden_dim=graph_hidden_dim,
            context_dim=context_dim,
            heads=graph_heads,
            dropout=dropout
        )

        # Fusion
        self.fusion = GatedFusion(embed_dim, context_dim, embed_dim)

        # Output
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, target_seq: torch.Tensor,
                graph_data: Data,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Encode target
        x = self.embedding(target_seq)
        seq_len = x.size(1)
        x = self.dropout(x + self.pos_encoding[:, :seq_len, :])
        x, _ = self.temporal(x)
        x = self.transformer(x)

        h_t = 0.5 * x.mean(dim=1) + 0.5 * x.max(dim=1)[0]

        # Encode graph
        g_t, graph_info = self.graph_encoder(graph_data, return_attention)

        if g_t.dim() == 1:
            g_t = g_t.unsqueeze(0)
        if g_t.size(0) != h_t.size(0):
            g_t = g_t.expand(h_t.size(0), -1)

        # Fusion and output
        h_prime = self.fusion(h_t, g_t)
        q_values = self.fc_out(h_prime)

        return q_values, None


class StaticGraphLORS(nn.Module):
    """
    Static Graph LORS Baseline

    Ablation model to isolate dynamic graph contribution.
    Uses fixed (average) correlation matrix instead of time-varying graph.
    """

    def __init__(self,
                 input_dim: int = 10,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.15,
                 n_assets: int = 5,
                 node_feature_dim: int = 8,
                 graph_hidden_dim: int = 32,
                 context_dim: int = 64,
                 graph_heads: int = 4,
                 output_dim: int = 3):
        super().__init__()

        # Use GraphLORSTrader as base
        self.model = GraphLORSTrader(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            n_assets=n_assets,
            node_feature_dim=node_feature_dim,
            graph_hidden_dim=graph_hidden_dim,
            context_dim=context_dim,
            graph_heads=graph_heads,
            output_dim=output_dim
        )

        # Static graph will be set during training
        self.static_edge_index = None
        self.static_edge_attr = None

    def set_static_graph(self, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """Set the static graph structure (computed from training data average)"""
        self.static_edge_index = edge_index
        self.static_edge_attr = edge_attr

    def forward(self, target_seq: torch.Tensor,
                graph_data: Data,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Override graph_data edge_index with static version
        if self.static_edge_index is not None:
            graph_data.edge_index = self.static_edge_index.to(target_seq.device)
            graph_data.edge_attr = self.static_edge_attr.to(target_seq.device)

        return self.model(target_seq, graph_data, return_attention)


if __name__ == "__main__":
    # Test the Graph-LORS Trader
    print("Graph-LORS Trader Module Test")
    print("="*70)

    # Test parameters
    batch_size = 4
    seq_len = 120
    input_dim = 10
    n_assets = 5
    node_feature_dim = 8

    # Create dummy data
    target_seq = torch.randn(batch_size, seq_len, input_dim)

    # Create dummy graph
    x = torch.randn(n_assets, node_feature_dim)
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 2, 3],
        [1, 2, 3, 4, 2, 3, 4, 4]
    ], dtype=torch.long)
    graph_data = Data(x=x, edge_index=edge_index)
    graph_data.target_idx = 0

    # Test GraphLORSTrader
    print("\n1. Testing GraphLORSTrader")
    model = GraphLORSTrader(
        input_dim=input_dim,
        embed_dim=128,
        n_assets=n_assets,
        node_feature_dim=node_feature_dim
    )
    q_values, attn_info = model(target_seq, graph_data, return_attention=True)
    print(f"   Input: target_seq {target_seq.shape}, graph ({n_assets} nodes)")
    print(f"   Output: Q-values {q_values.shape}")
    print(f"   LORS weights shape: {attn_info['lors_info']['avg_lors_weights'].shape}")

    # Test interpretability outputs
    print("\n2. Testing Interpretability Outputs")
    interp = model.get_interpretability_outputs(target_seq, graph_data)
    print(f"   Keys: {interp.keys()}")
    print(f"   LORS config weights: {interp['lors_config_weights'].shape}")

    # Test GraphTransformer baseline
    print("\n3. Testing GraphTransformer (No LORS)")
    baseline = GraphTransformer(
        input_dim=input_dim,
        n_assets=n_assets,
        node_feature_dim=node_feature_dim
    )
    q_values_baseline, _ = baseline(target_seq, graph_data)
    print(f"   Output: Q-values {q_values_baseline.shape}")

    print("\n" + "="*70)
    print("All tests passed!")
