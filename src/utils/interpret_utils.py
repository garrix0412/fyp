"""
Interpretability Analysis Utilities for Graph-LORS Trader

Provides dual-layer interpretability:
1. Graph Attention Analysis: Cross-asset influence patterns
2. LORS Configuration Analysis: Market regime adaptation

Paper Section 6.5: Interpretability Analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class GraphAttentionAnalyzer:
    """
    Analyzer for Graph Attention Weights

    Extracts and visualizes cross-asset attention patterns from GAT layers.
    Identifies which context assets influence trading decisions.
    """

    def __init__(self, asset_names: List[str], target_idx: int = 0):
        """
        Initialize Graph Attention Analyzer

        Args:
            asset_names: List of asset ticker names
            target_idx: Index of target asset
        """
        self.asset_names = asset_names
        self.target_idx = target_idx
        self.n_assets = len(asset_names)

        # Storage for attention history
        self.attention_history = []
        self.timestamps = []

    def record_attention(self, attention_info: Dict, timestamp):
        """
        Record attention weights for later analysis

        Args:
            attention_info: Dict from model forward pass
            timestamp: Associated timestamp
        """
        self.attention_history.append(attention_info)
        self.timestamps.append(timestamp)

    def get_node_importance_series(self) -> pd.DataFrame:
        """
        Get time series of node importance to target

        Returns:
            DataFrame with importance scores over time
        """
        importance_data = []

        for t, attn_info in zip(self.timestamps, self.attention_history):
            gat_attn = attn_info.get('gat_attention', {})

            # Extract layer 2 attention (final layer)
            if 'layer2_weights' in gat_attn:
                weights = gat_attn['layer2_weights']
                edge_index = gat_attn['layer2_edge_index']

                # Build importance from attention to target
                importance = self._extract_importance_to_target(
                    edge_index, weights
                )

                row = {'timestamp': t}
                for i, name in enumerate(self.asset_names):
                    row[name] = importance[i] if i < len(importance) else 0.0

                importance_data.append(row)

        if not importance_data:
            return pd.DataFrame()

        df = pd.DataFrame(importance_data)
        df.set_index('timestamp', inplace=True)
        return df

    def _extract_importance_to_target(self, edge_index: torch.Tensor,
                                      weights: torch.Tensor) -> np.ndarray:
        """Extract importance scores from edges pointing to target"""
        importance = np.zeros(self.n_assets)

        edge_index = edge_index.cpu().numpy()
        weights = weights.cpu().numpy()

        # Average across heads if multi-head
        if weights.ndim > 1:
            weights = weights.mean(axis=-1)

        # Find edges pointing to target
        for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            if dst == self.target_idx:
                importance[src] = weights[i]

        return importance

    def analyze_crisis_periods(self, crisis_dates: List[Tuple],
                               normal_dates: List[Tuple]) -> Dict:
        """
        Compare attention patterns during crisis vs normal periods

        Args:
            crisis_dates: List of (start, end) tuples for crisis periods
            normal_dates: List of (start, end) tuples for normal periods

        Returns:
            Dict with statistical comparison
        """
        importance_df = self.get_node_importance_series()

        if importance_df.empty:
            return {}

        # Filter by periods
        crisis_mask = pd.Series(False, index=importance_df.index)
        normal_mask = pd.Series(False, index=importance_df.index)

        for start, end in crisis_dates:
            crisis_mask |= (importance_df.index >= start) & (importance_df.index <= end)

        for start, end in normal_dates:
            normal_mask |= (importance_df.index >= start) & (importance_df.index <= end)

        crisis_data = importance_df[crisis_mask]
        normal_data = importance_df[normal_mask]

        # Statistical comparison
        results = {
            'crisis_mean': crisis_data.mean().to_dict() if not crisis_data.empty else {},
            'normal_mean': normal_data.mean().to_dict() if not normal_data.empty else {},
            'crisis_std': crisis_data.std().to_dict() if not crisis_data.empty else {},
            'normal_std': normal_data.std().to_dict() if not normal_data.empty else {},
        }

        # Compute difference
        if results['crisis_mean'] and results['normal_mean']:
            results['difference'] = {
                k: results['crisis_mean'].get(k, 0) - results['normal_mean'].get(k, 0)
                for k in self.asset_names
            }

        return results

    def plot_attention_heatmap(self, save_path: Optional[str] = None):
        """
        Plot attention heatmap over time

        Args:
            save_path: Path to save figure (optional)
        """
        importance_df = self.get_node_importance_series()

        if importance_df.empty:
            print("No attention data to plot")
            return

        # Exclude target from heatmap (it's always 0)
        context_cols = [c for c in importance_df.columns if c != self.asset_names[self.target_idx]]

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(importance_df[context_cols].T, cmap='RdYlBu_r',
                    ax=ax, cbar_kws={'label': 'Attention Weight'})
        ax.set_xlabel('Time')
        ax.set_ylabel('Context Asset')
        ax.set_title('Graph Attention Weights Over Time')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")

        plt.close()


class LORSConfigAnalyzer:
    """
    Analyzer for LORS Configuration Weights

    Tracks which LORS bifurcation configurations are activated
    under different market conditions.
    """

    def __init__(self, n_configs: int = 8):
        """
        Initialize LORS Config Analyzer

        Args:
            n_configs: Number of LORS configurations (default: 8)
        """
        self.n_configs = n_configs
        self.config_names = [f'LORS#{i}' for i in range(n_configs)]

        # Storage
        self.config_history = []
        self.timestamps = []
        self.market_conditions = []

    def record_config_weights(self, lors_weights: torch.Tensor,
                              timestamp, market_condition: Optional[Dict] = None):
        """
        Record LORS configuration weights

        Args:
            lors_weights: Weights tensor (batch, 8) or (8,)
            timestamp: Associated timestamp
            market_condition: Optional dict with market metrics
        """
        if lors_weights.dim() > 1:
            weights = lors_weights.mean(dim=0)  # Average across batch
        else:
            weights = lors_weights

        self.config_history.append(weights.cpu().numpy())
        self.timestamps.append(timestamp)
        self.market_conditions.append(market_condition or {})

    def get_config_series(self) -> pd.DataFrame:
        """
        Get time series of configuration weights

        Returns:
            DataFrame with config weights over time
        """
        if not self.config_history:
            return pd.DataFrame()

        data = np.array(self.config_history)
        df = pd.DataFrame(data, index=self.timestamps, columns=self.config_names)
        return df

    def correlate_with_volatility(self, volatility_series: pd.Series) -> pd.DataFrame:
        """
        Correlate LORS config weights with market volatility

        Args:
            volatility_series: Market volatility time series

        Returns:
            Correlation DataFrame
        """
        config_df = self.get_config_series()

        if config_df.empty:
            return pd.DataFrame()

        # Align indices
        common_idx = config_df.index.intersection(volatility_series.index)
        config_aligned = config_df.loc[common_idx]
        vol_aligned = volatility_series.loc[common_idx]

        # Compute correlations
        correlations = {}
        for col in config_df.columns:
            correlations[col] = config_aligned[col].corr(vol_aligned)

        return pd.Series(correlations)

    def identify_regime_configs(self, regime_labels: pd.Series) -> Dict:
        """
        Identify which configs activate under different market regimes

        Args:
            regime_labels: Series with regime labels (e.g., 'bull', 'bear', 'sideways')

        Returns:
            Dict mapping regime to dominant configs
        """
        config_df = self.get_config_series()

        if config_df.empty:
            return {}

        # Align indices
        common_idx = config_df.index.intersection(regime_labels.index)

        results = {}
        for regime in regime_labels.unique():
            mask = regime_labels.loc[common_idx] == regime
            regime_configs = config_df.loc[common_idx][mask]

            if not regime_configs.empty:
                mean_weights = regime_configs.mean()
                dominant_config = mean_weights.idxmax()

                results[regime] = {
                    'mean_weights': mean_weights.to_dict(),
                    'dominant_config': dominant_config,
                    'dominant_weight': mean_weights.max()
                }

        return results

    def plot_config_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of LORS configuration weights

        Args:
            save_path: Path to save figure (optional)
        """
        config_df = self.get_config_series()

        if config_df.empty:
            print("No config data to plot")
            return

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, col in enumerate(config_df.columns):
            ax = axes[i]
            config_df[col].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(col)
            ax.set_xlabel('Weight')
            ax.set_ylabel('Frequency')

        plt.suptitle('LORS Configuration Weight Distributions', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved config distribution to {save_path}")

        plt.close()

    def plot_config_over_time(self, save_path: Optional[str] = None):
        """
        Plot configuration weights over time as stacked area

        Args:
            save_path: Path to save figure (optional)
        """
        config_df = self.get_config_series()

        if config_df.empty:
            print("No config data to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # Stacked area plot
        config_df.plot.area(ax=ax, alpha=0.7, linewidth=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Configuration Weight')
        ax.set_title('LORS Configuration Weights Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved config timeline to {save_path}")

        plt.close()


class InterpretabilityAnalyzer:
    """
    Combined Interpretability Analyzer

    Integrates Graph Attention and LORS Config analysis
    for comprehensive interpretability reporting.
    """

    def __init__(self, asset_names: List[str], target_idx: int = 0):
        """
        Initialize Interpretability Analyzer

        Args:
            asset_names: List of asset names
            target_idx: Target asset index
        """
        self.graph_analyzer = GraphAttentionAnalyzer(asset_names, target_idx)
        self.lors_analyzer = LORSConfigAnalyzer()
        self.asset_names = asset_names

        # Action history for decision analysis
        self.action_history = []
        self.decision_contexts = []

    def record_step(self, attention_info: Dict, lors_weights: torch.Tensor,
                    timestamp, action: int, market_condition: Optional[Dict] = None):
        """
        Record a single trading step for analysis

        Args:
            attention_info: Graph attention info from model
            lors_weights: LORS configuration weights
            timestamp: Current timestamp
            action: Trading action taken (0=buy, 1=hold, 2=sell)
            market_condition: Optional market metrics
        """
        self.graph_analyzer.record_attention(attention_info, timestamp)
        self.lors_analyzer.record_config_weights(lors_weights, timestamp, market_condition)
        self.action_history.append(action)

        self.decision_contexts.append({
            'timestamp': timestamp,
            'action': action,
            'market_condition': market_condition
        })

    def generate_report(self, save_dir: str = 'figures'):
        """
        Generate comprehensive interpretability report

        Args:
            save_dir: Directory to save figures
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*70)
        print("Generating Interpretability Report")
        print("="*70)

        # 1. Graph Attention Analysis
        print("\n1. Graph Attention Analysis")
        importance_df = self.graph_analyzer.get_node_importance_series()
        if not importance_df.empty:
            print(f"   Recorded {len(importance_df)} timesteps")
            print(f"   Mean importance by asset:")
            for col in importance_df.columns:
                print(f"     {col}: {importance_df[col].mean():.4f}")

            self.graph_analyzer.plot_attention_heatmap(
                f"{save_dir}/graph_attention_heatmap.png"
            )

        # 2. LORS Configuration Analysis
        print("\n2. LORS Configuration Analysis")
        config_df = self.lors_analyzer.get_config_series()
        if not config_df.empty:
            print(f"   Recorded {len(config_df)} timesteps")
            print(f"   Mean config weights:")
            for col in config_df.columns:
                print(f"     {col}: {config_df[col].mean():.4f}")

            self.lors_analyzer.plot_config_distribution(
                f"{save_dir}/lors_config_distribution.png"
            )
            self.lors_analyzer.plot_config_over_time(
                f"{save_dir}/lors_config_timeline.png"
            )

        # 3. Action Distribution
        print("\n3. Action Distribution")
        if self.action_history:
            actions = np.array(self.action_history)
            action_names = ['Buy', 'Hold', 'Sell']
            for i, name in enumerate(action_names):
                count = (actions == i).sum()
                pct = count / len(actions) * 100
                print(f"   {name}: {count} ({pct:.1f}%)")

        print("\n" + "="*70)
        print(f"Report saved to {save_dir}/")
        print("="*70)

    def case_study_covid(self, start_date='2020-02-15', end_date='2020-04-15'):
        """
        Case study: COVID-19 market crash

        Analyzes model behavior during the COVID crash period.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            Dict with analysis results
        """
        print("\n" + "="*70)
        print("Case Study: COVID-19 Market Crash")
        print(f"Period: {start_date} to {end_date}")
        print("="*70)

        importance_df = self.graph_analyzer.get_node_importance_series()
        config_df = self.lors_analyzer.get_config_series()

        if importance_df.empty or config_df.empty:
            print("Insufficient data for case study")
            return {}

        # Filter to COVID period
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        covid_mask = (importance_df.index >= start) & (importance_df.index <= end)

        covid_importance = importance_df[covid_mask]
        covid_config = config_df.loc[covid_importance.index] if not covid_importance.empty else pd.DataFrame()

        results = {}

        if not covid_importance.empty:
            print("\n1. Graph Attention During Crisis:")
            print(f"   VIX importance: {covid_importance.get('^VIX', pd.Series([0])).mean():.4f}")
            print(f"   Gold importance: {covid_importance.get('GC=F', pd.Series([0])).mean():.4f}")

            results['graph_attention'] = covid_importance.mean().to_dict()

        if not covid_config.empty:
            print("\n2. LORS Configuration Activation:")
            dominant = covid_config.mean().idxmax()
            print(f"   Dominant config: {dominant}")
            print(f"   Config weights: {covid_config.mean().to_dict()}")

            results['lors_config'] = covid_config.mean().to_dict()

        return results


if __name__ == "__main__":
    # Test the interpretability utilities
    print("Interpretability Utilities Test")
    print("="*70)

    # Setup
    assets = ['^DJI', '^GSPC', '^VIX', 'GC=F', 'DX-Y.NYB']
    analyzer = InterpretabilityAnalyzer(assets, target_idx=0)

    # Generate dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')

    for i, date in enumerate(dates):
        # Dummy attention info
        attention_info = {
            'gat_attention': {
                'layer2_edge_index': torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]]),
                'layer2_weights': torch.rand(4, 4)
            }
        }

        # Dummy LORS weights
        lors_weights = torch.softmax(torch.randn(8), dim=0)

        # Random action
        action = np.random.choice([0, 1, 2])

        analyzer.record_step(attention_info, lors_weights, date, action)

    # Generate report
    analyzer.generate_report('figures/test_interpret')

    print("\nTest completed!")
