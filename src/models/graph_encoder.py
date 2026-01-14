"""
Graph Attention Network (GAT) Encoder for Graph-LORS Trader

Implements 2-layer GAT with attention weight output for interpretability.
Aggregates multi-asset context information into market context vector.

Paper Section 5.2: Graph Context Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Dict, Optional


class GraphAttentionEncoder(nn.Module):
    """
    2-Layer Graph Attention Network Encoder

    Encodes multi-asset graph structure into context vector.
    Returns attention weights for interpretability analysis.

    Architecture:
        Input: Node features (n_nodes, in_dim)
        GAT Layer 1: Multi-head attention (heads=4)
        ELU activation
        GAT Layer 2: Single-head attention
        Output: Node embeddings (n_nodes, out_dim)

    The target node's embedding is used as market context vector.
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 32,
                 out_dim: int = 64, heads: int = 4, dropout: float = 0.1):
        """
        Initialize GAT Encoder

        Args:
            in_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension (per head)
            out_dim: Output embedding dimension
            heads: Number of attention heads in first layer
            dropout: Dropout probability
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads

        # First GAT layer: multi-head attention
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True  # Concatenate head outputs
        )

        # Second GAT layer: single-head attention
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

        self.dropout = nn.Dropout(dropout)

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through GAT encoder

        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Graph connectivity (2, n_edges)
            edge_attr: Edge weights (n_edges,) - optional
            return_attention: Whether to return attention weights

        Returns:
            out: Node embeddings (n_nodes, out_dim)
            attention_dict: Dict with attention weights from both layers
        """
        attention_dict = {}

        # First GAT layer
        if return_attention:
            h, (edge_index_1, attn_weights_1) = self.gat1(
                x, edge_index,
                return_attention_weights=True
            )
            attention_dict['layer1_edge_index'] = edge_index_1
            attention_dict['layer1_weights'] = attn_weights_1
        else:
            h = self.gat1(x, edge_index)

        h = F.elu(h)
        h = self.norm1(h)
        h = self.dropout(h)

        # Second GAT layer
        if return_attention:
            out, (edge_index_2, attn_weights_2) = self.gat2(
                h, edge_index,
                return_attention_weights=True
            )
            attention_dict['layer2_edge_index'] = edge_index_2
            attention_dict['layer2_weights'] = attn_weights_2
        else:
            out = self.gat2(h, edge_index)

        out = self.norm2(out)

        return out, attention_dict

    def get_target_embedding(self, x: torch.Tensor, edge_index: torch.Tensor,
                             target_idx: int = 0,
                             edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Get embedding for target node specifically

        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Graph connectivity
            target_idx: Index of target node
            edge_attr: Edge weights (optional)

        Returns:
            target_embedding: Embedding of target node (out_dim,)
            attention_dict: Attention weights for interpretability
        """
        out, attention_dict = self.forward(x, edge_index, edge_attr)
        target_embedding = out[target_idx]

        return target_embedding, attention_dict


class GraphContextModule(nn.Module):
    """
    Complete Graph Context Module

    Wraps GAT encoder with preprocessing and provides clean interface
    for integration with LORS-Transformer.

    Handles:
    - Node feature normalization
    - Graph encoding via GAT
    - Target node extraction
    - Attention weight collection for interpretability
    """

    def __init__(self, n_assets: int = 5, node_feature_dim: int = 8,
                 hidden_dim: int = 32, context_dim: int = 64,
                 heads: int = 4, dropout: float = 0.1):
        """
        Initialize Graph Context Module

        Args:
            n_assets: Number of assets (graph nodes)
            node_feature_dim: Dimension of node features
            hidden_dim: GAT hidden dimension
            context_dim: Output context vector dimension
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_assets = n_assets
        self.node_feature_dim = node_feature_dim
        self.context_dim = context_dim

        # Node feature normalization
        self.node_norm = nn.LayerNorm(node_feature_dim)

        # GAT encoder
        self.gat_encoder = GraphAttentionEncoder(
            in_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            out_dim=context_dim,
            heads=heads,
            dropout=dropout
        )

        # Optional: projection layer for context vector
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, graph_data: Data,
                return_attention: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass to get market context vector

        Args:
            graph_data: PyG Data object with x, edge_index, target_idx
            return_attention: Whether to return attention weights

        Returns:
            context_vector: Market context (context_dim,)
            attention_info: Dict with attention weights and analysis
        """
        # Normalize node features
        x = self.node_norm(graph_data.x)

        # Get GAT embeddings
        node_embeddings, attention_dict = self.gat_encoder(
            x, graph_data.edge_index,
            return_attention=return_attention
        )

        # Extract target node embedding
        target_idx = getattr(graph_data, 'target_idx', 0)
        target_embedding = node_embeddings[target_idx]

        # Project to context space
        context_vector = self.context_proj(target_embedding)

        # Collect attention info
        attention_info = {
            'node_embeddings': node_embeddings,
            'gat_attention': attention_dict,
            'target_idx': target_idx
        }

        return context_vector, attention_info

    def forward_batch(self, batch: Batch,
                      return_attention: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for batched graphs

        Args:
            batch: PyG Batch object containing multiple graphs
            return_attention: Whether to return attention weights

        Returns:
            context_vectors: (batch_size, context_dim)
            attention_info: Batched attention information
        """
        # Normalize node features
        x = self.node_norm(batch.x)

        # Get GAT embeddings for all nodes in batch
        node_embeddings, attention_dict = self.gat_encoder(
            x, batch.edge_index,
            return_attention=return_attention
        )

        # Extract target embeddings from each graph in batch
        # This requires knowing how to index into batched graphs
        batch_size = batch.num_graphs
        context_vectors = []

        ptr = batch.ptr  # Pointer to graph boundaries
        for i in range(batch_size):
            start_idx = ptr[i].item()
            target_idx = getattr(batch, 'target_idx', 0)
            if isinstance(target_idx, torch.Tensor):
                target_idx = target_idx[i].item() if target_idx.dim() > 0 else target_idx.item()
            node_idx = start_idx + target_idx
            context_vectors.append(node_embeddings[node_idx])

        context_vectors = torch.stack(context_vectors)
        context_vectors = self.context_proj(context_vectors)

        attention_info = {
            'node_embeddings': node_embeddings,
            'gat_attention': attention_dict,
            'batch': batch
        }

        return context_vectors, attention_info


class EdgeAttentionAnalyzer:
    """
    Utility class for analyzing GAT attention weights

    Provides methods to extract and visualize attention patterns
    for interpretability analysis.
    """

    @staticmethod
    def extract_attention_matrix(attention_dict: Dict,
                                 n_nodes: int,
                                 layer: str = 'layer2') -> torch.Tensor:
        """
        Convert edge attention weights to attention matrix

        Args:
            attention_dict: Dict from GAT forward pass
            n_nodes: Number of nodes
            layer: Which layer's attention ('layer1' or 'layer2')

        Returns:
            Attention matrix (n_nodes, n_nodes)
        """
        edge_index = attention_dict[f'{layer}_edge_index']
        weights = attention_dict[f'{layer}_weights']

        # Initialize attention matrix
        attn_matrix = torch.zeros(n_nodes, n_nodes, device=weights.device)

        # Fill in attention weights
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # Handle multi-head attention (average across heads if needed)
        if weights.dim() > 1:
            weights = weights.mean(dim=-1)

        for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
            attn_matrix[dst, src] = weights[i]

        return attn_matrix

    @staticmethod
    def get_node_importance(attention_dict: Dict,
                            target_idx: int,
                            n_nodes: int) -> torch.Tensor:
        """
        Get importance scores of each node to target node

        Args:
            attention_dict: Dict from GAT forward pass
            target_idx: Index of target node
            n_nodes: Number of nodes

        Returns:
            Importance scores (n_nodes,)
        """
        attn_matrix = EdgeAttentionAnalyzer.extract_attention_matrix(
            attention_dict, n_nodes, layer='layer2'
        )

        # Get attention weights from target node's perspective
        importance = attn_matrix[target_idx]

        return importance


if __name__ == "__main__":
    # Test the GAT encoder
    print("GAT Encoder Module Test")
    print("="*70)

    # Create dummy graph data
    n_nodes = 5
    in_dim = 8
    n_edges = 12

    # Random node features
    x = torch.randn(n_nodes, in_dim)

    # Random edges (fully connected minus self-loops)
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
        [1, 2, 3, 4, 0, 2, 3, 0, 4, 1, 4, 2]
    ], dtype=torch.long)

    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    data.target_idx = 0

    # Test GraphAttentionEncoder
    print("\n1. Testing GraphAttentionEncoder")
    encoder = GraphAttentionEncoder(in_dim=in_dim, hidden_dim=32, out_dim=64, heads=4)
    out, attn_dict = encoder(x, edge_index)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention keys: {attn_dict.keys()}")

    # Test GraphContextModule
    print("\n2. Testing GraphContextModule")
    context_module = GraphContextModule(
        n_assets=n_nodes,
        node_feature_dim=in_dim,
        hidden_dim=32,
        context_dim=64
    )
    context_vec, attn_info = context_module(data)
    print(f"   Context vector shape: {context_vec.shape}")
    print(f"   Attention info keys: {attn_info.keys()}")

    # Test attention analysis
    print("\n3. Testing EdgeAttentionAnalyzer")
    attn_matrix = EdgeAttentionAnalyzer.extract_attention_matrix(
        attn_dict, n_nodes, layer='layer2'
    )
    print(f"   Attention matrix shape: {attn_matrix.shape}")
    print(f"   Attention matrix:\n{attn_matrix}")

    importance = EdgeAttentionAnalyzer.get_node_importance(attn_dict, target_idx=0, n_nodes=n_nodes)
    print(f"   Node importance to target: {importance}")

    print("\n" + "="*70)
    print("All tests passed!")
