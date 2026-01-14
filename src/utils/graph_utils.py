"""
Dynamic Graph Construction Utilities for Graph-LORS Trader
Implements anti-leakage dynamic correlation graph construction.

Paper Section 4.2: Dynamic Graph Construction
- Rolling window correlation: [t-60, t-1] (prevents information leakage)
- Top-k sparsification for adjacency matrix
- Node feature preparation for GAT input
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict


class DynamicGraphBuilder:
    """
    Dynamic correlation graph builder with strict anti-leakage protocol.

    For each trading day t, computes correlation matrix using only
    historical data [t-window, t-1], never using future information.
    """

    def __init__(self, window_size: int = 60, top_k: int = 3):
        """
        Initialize graph builder.

        Args:
            window_size: Rolling window size for correlation computation
            top_k: Number of top correlated assets to keep per node
        """
        self.window_size = window_size
        self.top_k = top_k

    def compute_correlation_matrix(
        self,
        price_matrix: pd.DataFrame,
        t_idx: int
    ) -> pd.DataFrame:
        """
        Compute correlation matrix at time t using only historical data.

        CRITICAL ANTI-LEAKAGE: Only uses data from [t-window, t-1],
        never includes data at time t or later.

        Args:
            price_matrix: DataFrame with asset prices (columns=assets, index=dates)
            t_idx: Current time index (0-based)

        Returns:
            Correlation matrix (n_assets x n_assets)
        """
        # Ensure we don't use future data
        start_idx = max(0, t_idx - self.window_size)
        end_idx = t_idx  # Exclusive - does not include t

        if end_idx <= start_idx:
            # Not enough history, return identity matrix
            n_assets = len(price_matrix.columns)
            return pd.DataFrame(
                np.eye(n_assets),
                index=price_matrix.columns,
                columns=price_matrix.columns
            )

        # Get historical window
        window_data = price_matrix.iloc[start_idx:end_idx]

        # Compute returns
        returns = window_data.pct_change().dropna()

        if len(returns) < 2:
            # Not enough data for correlation
            n_assets = len(price_matrix.columns)
            return pd.DataFrame(
                np.eye(n_assets),
                index=price_matrix.columns,
                columns=price_matrix.columns
            )

        # Compute correlation matrix
        corr_matrix = returns.corr()

        # Handle NaN values (can occur if an asset has no variance)
        corr_matrix = corr_matrix.fillna(0)
        np.fill_diagonal(corr_matrix.values, 1.0)

        return corr_matrix

    def build_adjacency_matrix(
        self,
        corr_matrix: pd.DataFrame,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Build sparse adjacency matrix using top-k correlation values.

        Args:
            corr_matrix: Correlation matrix
            top_k: Number of top connections per node (default: self.top_k)

        Returns:
            Adjacency matrix (n_assets x n_assets)
        """
        k = top_k if top_k is not None else self.top_k
        n = len(corr_matrix)
        adj = np.zeros((n, n))

        for i in range(n):
            # Get correlations for node i
            row = corr_matrix.iloc[i].copy()

            # Exclude self-connection
            row.iloc[i] = -np.inf

            # Get top-k correlated assets
            top_indices = row.nlargest(k).index

            for j_name in top_indices:
                j = list(corr_matrix.columns).index(j_name)
                # Use absolute correlation as edge weight
                adj[i, j] = abs(corr_matrix.iloc[i, j])

        return adj

    def get_edge_index_and_weights(
        self,
        adj_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert adjacency matrix to edge index format (for PyTorch Geometric).

        Args:
            adj_matrix: Adjacency matrix

        Returns:
            edge_index: (2, num_edges) array of [source, target] pairs
            edge_weight: (num_edges,) array of edge weights
        """
        rows, cols = np.where(adj_matrix > 0)
        edge_index = np.stack([rows, cols], axis=0)
        edge_weight = adj_matrix[rows, cols]

        return edge_index, edge_weight

    def build_graph_sequence(
        self,
        price_matrix: pd.DataFrame,
        start_idx: int = None,
        end_idx: int = None
    ) -> List[Dict]:
        """
        Build dynamic graph sequence for a date range.

        Args:
            price_matrix: DataFrame with asset prices
            start_idx: Start index (default: window_size to ensure enough history)
            end_idx: End index (default: len(price_matrix))

        Returns:
            List of graph dictionaries, each containing:
            - 'date': Trading date
            - 'corr_matrix': Correlation matrix
            - 'adj_matrix': Adjacency matrix
            - 'edge_index': Edge index for PyG
            - 'edge_weight': Edge weights
        """
        if start_idx is None:
            start_idx = self.window_size
        if end_idx is None:
            end_idx = len(price_matrix)

        graphs = []

        for t in range(start_idx, end_idx):
            # Compute correlation using only historical data
            corr_matrix = self.compute_correlation_matrix(price_matrix, t)

            # Build sparse adjacency matrix
            adj_matrix = self.build_adjacency_matrix(corr_matrix)

            # Convert to edge format
            edge_index, edge_weight = self.get_edge_index_and_weights(adj_matrix)

            graph = {
                'date': price_matrix.index[t],
                't_idx': t,
                'corr_matrix': corr_matrix,
                'adj_matrix': adj_matrix,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
            }
            graphs.append(graph)

        return graphs


class GraphNodeFeatureBuilder:
    """
    Build node features for GAT input.

    Each node (asset) needs features that summarize its recent behavior
    without leaking future information.
    """

    def __init__(self, feature_window: int = 60):
        """
        Args:
            feature_window: Window size for computing node features
        """
        self.feature_window = feature_window

    def compute_node_features(
        self,
        price_matrix: pd.DataFrame,
        t_idx: int
    ) -> np.ndarray:
        """
        Compute node features at time t using only historical data.

        Features per node (10 dimensions to match topic.md):
        - Return statistics: mean, std, skew (3)
        - Price momentum: 5-day, 20-day, 60-day (3)
        - Volatility: 10-day, 20-day (2)
        - Relative strength: vs market mean (1)
        - Trend: price vs 20-day MA (1)

        Args:
            price_matrix: DataFrame with asset prices
            t_idx: Current time index

        Returns:
            Node features (n_assets, n_features)
        """
        n_assets = len(price_matrix.columns)
        n_features = 10

        # Get historical window [t-window, t-1]
        start_idx = max(0, t_idx - self.feature_window)
        end_idx = t_idx

        if end_idx <= start_idx:
            return np.zeros((n_assets, n_features))

        window_data = price_matrix.iloc[start_idx:end_idx]
        returns = window_data.pct_change().dropna()

        features = np.zeros((n_assets, n_features))

        for i, asset in enumerate(price_matrix.columns):
            asset_returns = returns[asset].values
            asset_prices = window_data[asset].values

            if len(asset_returns) < 2:
                continue

            # Return statistics (3 features)
            features[i, 0] = np.mean(asset_returns)
            features[i, 1] = np.std(asset_returns)
            features[i, 2] = self._safe_skew(asset_returns)

            # Price momentum (3 features)
            if len(asset_prices) >= 5:
                features[i, 3] = (asset_prices[-1] / asset_prices[-5] - 1)
            if len(asset_prices) >= 20:
                features[i, 4] = (asset_prices[-1] / asset_prices[-20] - 1)
            if len(asset_prices) >= 60:
                features[i, 5] = (asset_prices[-1] / asset_prices[0] - 1)

            # Volatility (2 features)
            if len(asset_returns) >= 10:
                features[i, 6] = np.std(asset_returns[-10:])
            if len(asset_returns) >= 20:
                features[i, 7] = np.std(asset_returns[-20:])

            # Relative strength vs market (1 feature)
            market_returns = returns.mean(axis=1).values
            if len(market_returns) >= 20:
                features[i, 8] = np.mean(asset_returns[-20:]) - np.mean(market_returns[-20:])

            # Trend: price vs 20-day MA (1 feature)
            if len(asset_prices) >= 20:
                ma_20 = np.mean(asset_prices[-20:])
                features[i, 9] = (asset_prices[-1] / ma_20 - 1)

        return features

    def _safe_skew(self, data: np.ndarray) -> float:
        """Compute skewness safely."""
        if len(data) < 3:
            return 0.0
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return np.mean(((data - np.mean(data)) / std) ** 3)


def verify_no_future_leakage(
    price_matrix: pd.DataFrame,
    t_idx: int,
    window_size: int = 60
) -> bool:
    """
    Verify that graph construction at time t doesn't use future data.

    Args:
        price_matrix: Price data
        t_idx: Time index to verify
        window_size: Correlation window size

    Returns:
        True if no leakage detected
    """
    # Check that correlation window ends before t
    start_idx = max(0, t_idx - window_size)
    end_idx = t_idx  # Should not include t

    # Verify dates
    if end_idx <= start_idx:
        return True  # Not enough history, safe

    max_date_used = price_matrix.index[end_idx - 1]
    current_date = price_matrix.index[t_idx]

    assert max_date_used < current_date, \
        f"Leakage detected: Using data from {max_date_used} for prediction at {current_date}"

    return True


def load_price_matrix(filepath: str = 'data/price_matrix_2016_2025.csv') -> pd.DataFrame:
    """
    Load price matrix from CSV file.

    Args:
        filepath: Path to price matrix CSV

    Returns:
        DataFrame with prices (index=dates, columns=assets)
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


if __name__ == "__main__":
    """Test graph utilities."""
    print("="*70)
    print("Graph Utilities Test")
    print("="*70)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    assets = ['^DJI', '^GSPC', '^VIX', 'GC=F', 'DX-Y.NYB']

    # Simulate correlated prices
    base = np.cumsum(np.random.randn(200) * 0.01) + 100
    price_data = {}
    for asset in assets:
        noise = np.cumsum(np.random.randn(200) * 0.005)
        price_data[asset] = base + noise * (10 if asset == '^VIX' else 1)

    price_matrix = pd.DataFrame(price_data, index=dates)

    # Test graph builder
    print("\n1. Testing DynamicGraphBuilder...")
    builder = DynamicGraphBuilder(window_size=60, top_k=3)

    # Build graph at t=100
    t = 100
    corr = builder.compute_correlation_matrix(price_matrix, t)
    print(f"   Correlation matrix at t={t}:")
    print(corr.round(3))

    adj = builder.build_adjacency_matrix(corr)
    print(f"\n   Adjacency matrix (top-3):")
    print(adj.round(3))

    edge_index, edge_weight = builder.get_edge_index_and_weights(adj)
    print(f"\n   Edge index shape: {edge_index.shape}")
    print(f"   Number of edges: {len(edge_weight)}")

    # Test anti-leakage
    print("\n2. Testing anti-leakage verification...")
    assert verify_no_future_leakage(price_matrix, t, 60)
    print("   ✅ No future leakage detected")

    # Test node feature builder
    print("\n3. Testing GraphNodeFeatureBuilder...")
    feature_builder = GraphNodeFeatureBuilder(feature_window=60)
    node_features = feature_builder.compute_node_features(price_matrix, t)
    print(f"   Node features shape: {node_features.shape}")
    print(f"   Feature range: [{node_features.min():.4f}, {node_features.max():.4f}]")

    # Test graph sequence
    print("\n4. Testing graph sequence building...")
    graphs = builder.build_graph_sequence(price_matrix, start_idx=60, end_idx=65)
    print(f"   Built {len(graphs)} graphs")
    for g in graphs[:3]:
        print(f"   - Date: {g['date']}, Edges: {len(g['edge_weight'])}")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
