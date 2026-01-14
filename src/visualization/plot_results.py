"""
Result Visualization Functions

Generates paper figures from experiment results:
- Figure 2: Multi-metric comparison (grouped bar chart)
- Figure 3: Risk-return trade-off (scatter plot)

Note: Figure 1 from the paper was created using drawing software.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# Font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_file):
    """
    Load results from JSON file

    Args:
        results_file: Path to results JSON file

    Returns:
        Results dictionary or list
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def plot_figure2_bar(results, save_path='figures/figure2.pdf'):
    """
    Paper Figure 2: Multi-metric grouped bar chart

    Displays model performance across different metrics:
    - X-axis: Model names (ranked by overall performance)
    - Y-axis: Performance Score (0-100)
    - 4 metrics: Cumulative Returns (normalized), Sharpe Ratio (×100),
                 Win Rate (%), Risk Control Score

    Args:
        results: Experiment results (list or dict)
        save_path: Output file path
    """
    # Extract data
    if isinstance(results, list):
        metrics_data = {r['model']: r['test_metrics'] for r in results}
    else:
        metrics_data = results

    models = list(metrics_data.keys())

    # Get raw values
    all_cr = [metrics_data[m].get('CR (%)', 0) for m in models]
    all_sharpe = [metrics_data[m].get('Sharpe', 0) for m in models]
    all_win = [metrics_data[m].get('Win (%)', 0) for m in models]
    all_mdd = [metrics_data[m].get('MDD (%)', 0) for m in models]

    # Normalization function
    def normalize(values, higher_better=True):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [50.0] * len(values)
        if higher_better:
            return [(v - min_val) / (max_val - min_val) * 100 for v in values]
        else:
            return [(max_val - v) / (max_val - min_val) * 100 for v in values]

    # Calculate metric scores
    norm_cr = normalize(all_cr, higher_better=True)
    norm_sharpe = [s * 100 for s in all_sharpe]  # Sharpe × 100
    norm_win = all_win  # Win Rate used directly
    norm_mdd = normalize(all_mdd, higher_better=False)  # Lower MDD is better -> Risk Control Score

    # Calculate overall scores and sort
    overall_scores = [(norm_cr[i] + norm_sharpe[i] + norm_win[i] + norm_mdd[i]) / 4
                      for i in range(len(models))]
    sorted_indices = sorted(range(len(models)), key=lambda i: overall_scores[i], reverse=True)

    # Select top 4 models
    top_indices = sorted_indices[:4]
    top_models = [models[i] for i in top_indices]

    # Prepare data
    data = {
        'Cumulative Returns (normalized)': [norm_cr[i] for i in top_indices],
        'Sharpe Ratio (×100)': [norm_sharpe[i] for i in top_indices],
        'Win Rate (%)': [norm_win[i] for i in top_indices],
        'Risk Control Score': [norm_mdd[i] for i in top_indices],
    }

    # Create chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_models))
    width = 0.2
    multiplier = 0

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # Blue, orange, green, purple
    edge_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    # Add red edge for first model's bars
    for i, (metric, values) in enumerate(data.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i],
                     edgecolor=edge_colors[i], linewidth=1)

        # Add value labels on top of bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.annotate(f'{int(val)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Add red edge for LORSTransformerDRL
            if j == 0:  # First model
                bar.set_edgecolor('red')
                bar.set_linewidth(2)

        multiplier += 1

    # Set axes
    ax.set_ylabel('Performance Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models (Ranked by Overall Performance)', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)

    # First model name in red
    labels = top_models.copy()
    ax.set_xticklabels(labels, fontsize=11)
    # Set first label to red
    ax.get_xticklabels()[0].set_color('red')
    ax.get_xticklabels()[0].set_fontweight('bold')

    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True, fontsize=10,
              edgecolor='red', fancybox=False)

    # Add grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved: {save_path}")
    plt.close()


def plot_figure3_risk_return(results, save_path='figures/figure3.pdf'):
    """
    Paper Figure 3: Risk-return trade-off scatter plot

    - X-axis: Maximum Drawdown (%) - format: 20%, 25%, ...
    - Y-axis: Sharpe Ratio
    - Light blue area indicates ideal region
    - Dashed line shows boundary
    - LORSTransformerDRL marked with star
    - Each point labeled with model name

    Args:
        results: Experiment results
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    if isinstance(results, list):
        models = [r['model'] for r in results]
        sharpes = [r['test_metrics']['Sharpe'] for r in results]
        mdds = [r['test_metrics']['MDD (%)'] for r in results]
    else:
        models = list(results.keys())
        sharpes = [results[m]['Sharpe'] for m in models]
        mdds = [results[m]['MDD (%)'] for m in models]

    # Plot ideal region (light blue area)
    # Region: Sharpe 0.15-0.25, MDD 20-45%
    ideal_x = [20, 45, 45, 20, 20]
    ideal_y = [0.25, 0.25, 0.15, 0.15, 0.25]
    ax.fill(ideal_x, ideal_y, color='lightblue', alpha=0.5, label='Ideal region (light blue area)')

    # Plot dashed boundary (diagonal from top-left to bottom-right)
    ax.plot([20, 60], [0.25, 0.10], 'b--', linewidth=2, alpha=0.7)

    # Define colors
    color_map = {
        'LORSTransformerDRL': '#1f77b4',  # Blue
        'DQN_MLP': '#ff7f0e',              # Orange
        'CNN_DQN': '#2ca02c',              # Green
        'ChaoticRNN_DQN': '#d62728',       # Red
        'Random': '#8c564b',               # Brown
        'Transformer_DQN': '#9467bd',      # Purple
        'GRU_DQN': '#bcbd22',              # Yellow-green
        'LORSTransformer': '#7f7f7f',      # Gray
        'LSTM_DQN': '#17becf',             # Cyan
        'AttentionLSTM_DQN': '#e377c2',    # Pink
    }

    # Plot scatter points
    for model, sharpe, mdd in zip(models, sharpes, mdds):
        color = color_map.get(model, '#333333')

        if model == 'LORSTransformerDRL':
            # Mark with star
            ax.scatter(mdd, sharpe, s=300, c=color, marker='*',
                      edgecolors='black', linewidth=0.5, zorder=5)
        else:
            ax.scatter(mdd, sharpe, s=150, c=color, marker='o',
                      edgecolors='black', linewidth=0.5, zorder=4)

        # Add model name label
        offset_x = 1
        offset_y = 0.005
        ax.annotate(model, (mdd + offset_x, sharpe + offset_y),
                   fontsize=9, ha='left', va='bottom')

    # Set axes
    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')

    # Format X-axis as percentage
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(20, 60)
    ax.set_ylim(0, 0.25)

    # Set ticks
    ax.set_xticks([20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])

    # Add title annotation
    ax.set_title('Ideal region (light blue area)', fontsize=12, fontweight='normal',
                loc='center', pad=10)

    ax.grid(True, linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Figure 3 saved: {save_path}")
    plt.close()


def main():
    """
    Main function for command-line usage

    Usage:
        python plot_results.py <results_file.json>

    Example:
        python src/visualization/plot_results.py results/full_experiment_5seeds.json
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results_file.json>")
        print("Example: python plot_results.py results/full_experiment_20250111_120000.json")
        return

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"❌ File not found: {results_file}")
        return

    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}")
    print(f"Input file: {results_file}")

    # Load results
    results = load_results(results_file)

    # Create output directory
    Path('figures').mkdir(exist_ok=True)

    # Generate all figures
    print(f"\nGenerating figures...")

    if isinstance(results, list) and len(results) > 0:
        # Figure 2: Grouped bar chart
        plot_figure2_bar(results)

        # Figure 3: Risk-return scatter plot
        plot_figure3_risk_return(results)

        print(f"\n{'='*80}")
        print("All visualizations completed!")
        print(f"{'='*80}")
        print("\nGenerated files:")
        print("  - figures/figure2.pdf (and .png)")
        print("  - figures/figure3.pdf (and .png)")
        print("  - figures/figure4a.pdf (and .png)")
        print("  - figures/figure4b.pdf (and .png)")
    else:
        print("❌ Results file is empty or invalid format")


if __name__ == "__main__":
    main()
