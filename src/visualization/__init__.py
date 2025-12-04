"""
Visualization Module

This module provides visualization functions for paper figures:
- Figure 2: Multi-metric grouped bar chart
- Figure 3: Risk-return trade-off scatter plot
- Figure 4a: Cumulative return trajectories
- Figure 4b: Drawdown trajectories
"""

from .plot_results import (
    load_results,
    plot_figure2_bar,
    plot_figure3_risk_return,
    plot_figure4a_cumulative_returns,
    plot_figure4b_drawdown,
)

__all__ = [
    'load_results',
    'plot_figure2_bar',
    'plot_figure3_risk_return',
    'plot_figure4a_cumulative_returns',
    'plot_figure4b_drawdown',
]
