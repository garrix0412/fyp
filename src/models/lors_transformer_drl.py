"""
LORSTransformerDRL Model Implementation
Main model combining LSTM, Transformer, and Lee Oscillator Retrograde Signal (LORS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LORSTransformerDRL(nn.Module):
    """
    Main model: LSTM + Transformer + LORS + DQN

    Architecture:
    - Embedding layer for input features
    - Positional encoding
    - LSTM for temporal dynamics
    - Multi-layer Transformer with LORS mechanism
    - Dual pooling (avg + max)
    - Output layer for Q-values

    Paper Section 2.3: LORS Mechanism
    Eight LORS configurations with different bifurcation parameters
    Attention-weighted combination of LORS outputs
    """

    def __init__(self, input_dim=10, embed_dim=128, num_heads=8, hidden_dim=256,
                 output_dim=3, n_layers=3, dropout=0.15):
        super().__init__()

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

        # Output network
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # LORS parameters - 8 configurations with different bifurcation parameters
        # Parameters: [a1, a2, a3, a4, b1, b2, b3, b4, c, k, s]
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

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch, sequence, features)

        Returns:
            Q-values (batch, actions)
        """
        # Embedding + positional encoding
        x = self.embedding(x)
        seq_len = x.size(1)
        x = self.dropout(x + self.pos_encoding[:, :seq_len, :])

        # Temporal processing with LSTM
        x, _ = self.temporal(x)

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
                # Paper formula:
                # E(t+1) = Sig[a1·LORS(t) + a2·E(t) - a3·I(t) + a4·S(t)]
                # I(t+1) = Sig[b1·LORS(t) - b2·E(t) - b3·I(t) + b4·S(t)]
                # LORS(t) = [E(t) - I(t)] · exp(-k·S²(t)) + Ω(t)
                tempu = a1 * z + a2 * u - a3 * v + a4 * sim
                tempv = b1 * z - b2 * u - b3 * v + b4 * sim
                u = torch.tanh(s * tempu)
                v = torch.tanh(s * tempv)
                w = torch.tanh(s * sim)
                z = (u - v) * torch.exp(-self.alpha * k * sim ** 2) + c * w

                lors_outputs.append(z)

            # Attention-weighted combination of LORS outputs
            lors_outputs = torch.stack(lors_outputs, dim=2)  # (batch, seq, 8, embed)
            attn_weights = F.softmax(self.lors_attention(sim), dim=-1).unsqueeze(-1)  # (batch, seq, 8, 1)
            lors_combined = (lors_outputs * attn_weights).sum(dim=2)  # (batch, seq, embed)

            # Residual connection + LayerNorm
            x = self.layer_norm(attention_output + 0.5 * lors_combined)

        # Dual pooling strategy
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        pooled = 0.5 * avg_pool + 0.5 * max_pool

        # Output Q-values
        return self.fc_out(pooled)


class LORSTransformer(nn.Module):
    """
    LORS + Transformer (ablation study version)

    Same structure as LORSTransformerDRL but used in ablation experiments
    to demonstrate the contribution of DRL optimization.

    This version has identical architecture but is referred to separately
    in experiments to distinguish it from the full DRL-optimized version.
    """

    def __init__(self, input_dim=10, embed_dim=128, num_heads=8, hidden_dim=256,
                 output_dim=3, n_layers=3, dropout=0.15):
        super().__init__()

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

        # Output network
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # LORS parameters - 8 configurations (LORS#0-7)
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

    def forward(self, x):
        """Forward pass - identical to LORSTransformerDRL"""
        # Embedding + positional encoding
        x = self.embedding(x)
        seq_len = x.size(1)
        x = self.dropout(x + self.pos_encoding[:, :seq_len, :])

        # Temporal processing with LSTM
        x, _ = self.temporal(x)

        # Transformer layers with LORS mechanism
        for layer in self.layers:
            attention_output = layer(x)
            sim = attention_output

            # LORS module
            lors_outputs = []
            for params in self.lors_params:
                a1, a2, a3, a4, b1, b2, b3, b4, c, k, s = params
                u, v, w, z = [torch.zeros_like(sim) for _ in range(4)]

                tempu = a1 * z + a2 * u - a3 * v + a4 * sim
                tempv = b1 * z - b2 * u - b3 * v + b4 * sim
                u = torch.tanh(s * tempu)
                v = torch.tanh(s * tempv)
                w = torch.tanh(s * sim)
                z = (u - v) * torch.exp(-self.alpha * k * sim ** 2) + c * w

                lors_outputs.append(z)

            # Attention-weighted combination
            lors_outputs = torch.stack(lors_outputs, dim=2)
            attn_weights = F.softmax(self.lors_attention(sim), dim=-1).unsqueeze(-1)
            lors_combined = (lors_outputs * attn_weights).sum(dim=2)

            # Residual connection + LayerNorm
            x = self.layer_norm(attention_output + 0.5 * lors_combined)

        # Dual pooling
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        pooled = 0.5 * avg_pool + 0.5 * max_pool

        # Output
        return self.fc_out(pooled)
