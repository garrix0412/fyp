"""
Baseline Models for Comparison

This module contains all baseline model implementations used for comparison:
- LSTM_DQN: Basic LSTM with DQN
- GRU_DQN: Basic GRU with DQN
- CNN_DQN: Convolutional neural network with DQN
- DQN_MLP: Multi-layer perceptron with DQN
- Transformer_DQN: Standard Transformer without LORS
- AttentionLSTM_DQN: LSTM with attention mechanism
- ChaoticRNN_DQN: RNN with chaotic dynamics (no retrograde signaling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_DQN(nn.Module):
    """
    Basic LSTM baseline model

    Architecture:
    - Multi-layer LSTM
    - Batch normalization
    - Fully connected output layers
    """

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=3, n_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1])  # Take last time step
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class GRU_DQN(nn.Module):
    """
    Basic GRU baseline model

    Architecture:
    - Multi-layer GRU
    - Batch normalization
    - Fully connected output layers
    """

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=3, n_layers=2, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = self.dropout(gru_out[:, -1])  # Take last time step
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class CNN_DQN(nn.Module):
    """
    Convolutional Neural Network baseline

    Architecture:
    - 2-layer CNN with batch normalization
    - Max pooling
    - Fully connected output layers

    Useful for capturing local patterns in time series
    """

    def __init__(self, input_dim=10, window_size=120, output_dim=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (window_size // 2), 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # Transpose for Conv1d: (batch, features, sequence)
        x = x.transpose(1, 2)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


class DQN_MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline

    Architecture:
    - Flatten input
    - 3-layer MLP with batch normalization
    - Simple but effective baseline

    This model has no temporal structure awareness
    """

    def __init__(self, input_dim=10, window_size=120, output_dim=3):
        super().__init__()
        self.flatten_dim = input_dim * window_size
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class Transformer_DQN(nn.Module):
    """
    Standard Transformer model (no LORS)

    Used to compare the contribution of LORS mechanism

    Architecture:
    - Embedding → LSTM → Transformer Layers → Pooling → Output
    - Same backbone as LORSTransformerDRL but without LORS modules
    """

    def __init__(self, input_dim=10, embed_dim=128, num_heads=4, hidden_dim=256,
                 output_dim=3, n_layers=2, dropout=0.15):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))

        # Temporal processing layer (consistent with LORSTransformerDRL)
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

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Embedding + positional encoding
        x = self.embedding(x)
        seq_len = x.size(1)
        x = self.dropout(x + self.pos_encoding[:, :seq_len, :])

        # LSTM temporal processing
        x, _ = self.temporal(x)

        # Transformer layers (no LORS)
        for layer in self.layers:
            x = layer(x)
            x = self.layer_norm(x)

        # Dual pooling strategy (consistent with LORSTransformerDRL)
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        pooled = 0.5 * avg_pool + 0.5 * max_pool

        return self.fc_out(pooled)


class AttentionLSTM_DQN(nn.Module):
    """
    LSTM + Attention Mechanism

    Combines temporal modeling with attention
    Attention helps identify important time steps
    """

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=3, n_layers=2, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)

        # Attention mechanism
        self.attention_fc = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Calculate attention weights
        attn_hidden = torch.tanh(self.attention_fc(lstm_out))  # (batch, seq_len, hidden_dim)
        attn_scores = self.attention_score(attn_hidden).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim)

        x = self.dropout(context)
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class ChaoticRNN_DQN(nn.Module):
    """
    Chaotic RNN Model

    Introduces chaotic dynamics but without retrograde signaling
    Uses simplified Lee oscillator (no retrograde feedback)

    Used to demonstrate the importance of retrograde signaling in LORS
    """

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=3, n_layers=2, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Base RNN layer
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)

        # Chaotic oscillator parameters (simplified Lee oscillator, no retrograde)
        # Parameters: [a1, a2, a3, a4, c, k, s]
        self.chaos_params = nn.Parameter(torch.tensor([0.6, 0.6, 0.6, 0.6, 1.2, 60.0, 1.2]))
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def chaotic_transform(self, x):
        """
        Simplified chaotic transformation (no retrograde signaling)

        Compares the contribution of full LORS in the paper
        """
        a1, a2, a3, a4, c, k, s = self.chaos_params

        # Simplified chaotic dynamics: only forward propagation, no feedback
        u = torch.tanh(s * (a1 * x + a2 * x))
        v = torch.tanh(s * (a3 * x - a4 * x))
        w = torch.tanh(s * x)

        # Chaotic output (no retrograde signal z feedback)
        z = (u - v) * torch.exp(-self.alpha * k * x ** 2) + c * w

        return z

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)

        # Apply chaotic transformation
        chaotic_out = self.chaotic_transform(rnn_out)

        # Residual connection + LayerNorm
        x = self.layer_norm(rnn_out + 0.5 * chaotic_out)

        # Take last time step
        x = self.dropout(x[:, -1])
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)
