"""
Trading Evaluation Metrics System
Strictly follows Paper Section 3.7: Model Comparison and Evaluation Framework
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class TradingMetrics:
    """
    Paper evaluation metrics implementation

    Paper quote:
    "Decision-centric metrics. All metrics are computed on the test period only,
     under the chronological, anti-leakage protocol; transaction costs are
     applied before reward."
    """

    @staticmethod
    def calculate_returns(portfolio_values: List[float]) -> pd.Series:
        """
        Calculate daily returns series

        Args:
            portfolio_values: Portfolio value sequence

        Returns:
            Daily returns series r_t = V_t/V_{t-1} - 1
        """
        portfolio_series = pd.Series(portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        return returns

    @staticmethod
    def cumulative_return(portfolio_values: List[float]) -> float:
        """
        Cumulative Return (Paper Table I metric)

        Paper formula:
        CR = (V_T / V_0 - 1) × 100%

        Args:
            portfolio_values: Portfolio value sequence

        Returns:
            Cumulative return (%)
        """
        if len(portfolio_values) < 2:
            return 0.0

        V0 = portfolio_values[0]
        VT = portfolio_values[-1]

        CR = (VT / V0 - 1) * 100

        return CR

    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
        """
        Sharpe Ratio (Paper Table I core metric)

        Paper formula:
        SR = (r̄ - r_f) / σ(r)

        where r̄ and σ(r) are sample mean and std of daily returns
        Paper setting: r_f = 0 for daily index data

        Args:
            returns: Daily returns series
            rf: Risk-free rate (paper sets to 0)

        Returns:
            Sharpe ratio (unitless)
        """
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        sharpe = (mean_return - rf) / std_return

        return sharpe

    @staticmethod
    def maximum_drawdown(portfolio_values: List[float]) -> float:
        """
        Maximum Drawdown (Paper Table I risk metric)

        Paper formula:
        MDD = max_{1≤t≤T} [1 - V_t / max_{1≤s≤t} V_s] × 100%

        Args:
            portfolio_values: Portfolio value sequence

        Returns:
            Maximum drawdown (%), returned as positive value
        """
        if len(portfolio_values) < 2:
            return 0.0

        cumulative = pd.Series(portfolio_values)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        MDD = drawdown.min() * 100  # Convert to percentage

        return abs(MDD)  # Return positive value

    @staticmethod
    def win_rate(actions: List[int], rewards: List[float]) -> float:
        """
        Win Rate (Paper Table I metric)

        Paper definition:
        WR = #{profitable trades} / #{all trades} × 100%

        Only counts trading actions (non-hold)

        Args:
            actions: Action sequence (0=buy, 1=hold, 2=sell)
            rewards: Reward sequence

        Returns:
            Win rate (%)
        """
        if len(actions) != len(rewards):
            raise ValueError("Action and reward sequences must have same length")

        # Only count trading actions (non-hold)
        trading_actions = [(a, r) for a, r in zip(actions, rewards) if a != 1]

        if len(trading_actions) == 0:
            return 0.0

        wins = sum(1 for _, r in trading_actions if r > 0)
        win_rate = wins / len(trading_actions) * 100

        return win_rate

    @staticmethod
    def average_holding_time(actions: List[int]) -> float:
        """
        Average Holding Time (Paper Table II metric)

        Args:
            actions: Action sequence

        Returns:
            Average holding time (days)
        """
        if len(actions) == 0:
            return 0.0

        holding_periods = []
        current_hold = 0

        for action in actions:
            if action == 1:  # Hold
                current_hold += 1
            else:
                if current_hold > 0:
                    holding_periods.append(current_hold)
                current_hold = 0

        # Handle last holding period
        if current_hold > 0:
            holding_periods.append(current_hold)

        if len(holding_periods) == 0:
            return 0.0

        return np.mean(holding_periods)

    @staticmethod
    def trading_frequency(actions: List[int], total_days: int) -> float:
        """
        Trading Frequency (Paper Table II metric)

        Paper unit: times/month

        Args:
            actions: Action sequence
            total_days: Total trading days

        Returns:
            Trading frequency (times/month)
        """
        if total_days == 0:
            return 0.0

        # Count non-hold actions
        trades = sum(1 for a in actions if a != 1)

        # Convert to monthly frequency (assuming 20 trading days per month)
        trades_per_month = trades / (total_days / 20)

        return trades_per_month

    @staticmethod
    def average_per_trade_return(actions: List[int], rewards: List[float]) -> float:
        """
        Average Per-Trade Return (Paper Table II)

        Args:
            actions: Action sequence
            rewards: Reward sequence

        Returns:
            Average per-trade return (%)
        """
        trading_rewards = [r for a, r in zip(actions, rewards) if a != 1]

        if len(trading_rewards) == 0:
            return 0.0

        return np.mean(trading_rewards) * 100  # Convert to percentage

    @classmethod
    def comprehensive_report(cls, portfolio_values: List[float],
                            actions: List[int],
                            rewards: List[float]) -> Dict[str, float]:
        """
        Generate comprehensive evaluation report - corresponds to paper Table I and Table II

        Args:
            portfolio_values: Portfolio value sequence
            actions: Action sequence
            rewards: Reward sequence

        Returns:
            Dictionary containing all metrics
        """
        returns = cls.calculate_returns(portfolio_values)

        report = {
            # Table I metrics
            'CR (%)': cls.cumulative_return(portfolio_values),
            'Sharpe': cls.sharpe_ratio(returns),
            'MDD (%)': cls.maximum_drawdown(portfolio_values),
            'Win (%)': cls.win_rate(actions, rewards),

            # Table II metrics
            'Avg_holding (days)': cls.average_holding_time(actions),
            'Transactions (times/month)': cls.trading_frequency(actions, len(portfolio_values)),
            'Avg_per_trade (%)': cls.average_per_trade_return(actions, rewards),

            # Additional statistics
            'Total_trades': sum(1 for a in actions if a != 1),
            'Final_value': portfolio_values[-1] if len(portfolio_values) > 0 else 0,
        }

        return report

    @staticmethod
    def print_report(report: Dict[str, float], title: str = "Evaluation Report"):
        """
        Format and print evaluation report

        Args:
            report: Evaluation metrics dictionary
            title: Report title
        """
        print("\n" + "="*70)
        print(f"📊 {title}")
        print("="*70)

        # Main metrics (Table I)
        print("\nCore Metrics (Paper Table I):")
        print(f"  Cumulative Return (CR):   {report['CR (%)']:>10.2f}%")
        print(f"  Sharpe Ratio:             {report['Sharpe']:>10.3f}")
        print(f"  Maximum Drawdown (MDD):   {report['MDD (%)']:>10.2f}%")
        print(f"  Win Rate:                 {report['Win (%)']:>10.2f}%")

        # Trading behavior (Table II)
        print("\nTrading Behavior (Paper Table II):")
        print(f"  Average Holding Time:     {report['Avg_holding (days)']:>10.2f} days")
        print(f"  Trading Frequency:        {report['Transactions (times/month)']:>10.2f} times/month")
        print(f"  Average Per-Trade Return: {report['Avg_per_trade (%)']:>10.4f}%")

        # Additional statistics
        print("\nAdditional Statistics:")
        print(f"  Total Trades:             {report['Total_trades']:>10.0f}")
        print(f"  Final Value:              {report['Final_value']:>10,.2f}")

        print("="*70)

    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models' results - generate paper Table I format

        Args:
            results: {model_name: metrics_dict}

        Returns:
            DataFrame format comparison table
        """
        df = pd.DataFrame(results).T

        # Select core metrics from paper
        core_metrics = ['CR (%)', 'Sharpe', 'MDD (%)', 'Win (%)']
        df_core = df[core_metrics]

        # Sort by Sharpe (paper sorts by performance)
        df_core = df_core.sort_values('Sharpe', ascending=False)

        return df_core


def format_table1_latex(results: Dict[str, Dict[str, float]]) -> str:
    """
    Generate LaTeX code for paper Table I

    Args:
        results: {model_name: metrics_dict}

    Returns:
        LaTeX table code
    """
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Consolidated returns \\& risk on the DJI test set}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\hline\n"
    latex += "Model & CR (\\%) & Sharpe & MDD (\\%) & Win (\\%) \\\\\n"
    latex += "\\hline\n"

    # Sort by Sharpe
    sorted_models = sorted(results.items(),
                          key=lambda x: x[1]['Sharpe'],
                          reverse=True)

    for model_name, metrics in sorted_models:
        latex += f"{model_name} & "
        latex += f"{metrics['CR (%)']:.1f} & "
        latex += f"{metrics['Sharpe']:.3f} & "
        latex += f"{metrics['MDD (%)']:.1f} & "
        latex += f"{metrics['Win (%)']:.1f} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\end{table}"

    return latex


if __name__ == "__main__":
    """Test code"""
    print("Evaluation Metrics Module")
    print("Usage: from src.utils.metrics import TradingMetrics")

    # Simple test
    test_portfolio = [1000000, 1010000, 1005000, 1020000, 1015000]
    test_actions = [0, 1, 2, 0, 1]
    test_rewards = [0.01, 0, -0.005, 0.015, 0]

    report = TradingMetrics.comprehensive_report(
        test_portfolio, test_actions, test_rewards
    )

    TradingMetrics.print_report(report, "Test Report")
