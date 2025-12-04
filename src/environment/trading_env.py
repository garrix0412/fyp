"""
Trading Environment for Reinforcement Learning

Implements market simulator following paper Section 3.3:
Design of the Trading Environment

Key features:
- Transaction costs deducted BEFORE reward calculation
- Reward = net portfolio value change
- Full position trading (all-in or all-out strategy)
- Strict chronological order (no future information leakage)
"""

import torch
import numpy as np


class TradingEnv:
    """
    Trading Environment Class

    Paper Section 3.3: Design of the Trading Environment

    Key corrections from paper:
    1. Transaction costs deducted before reward calculation
    2. Reward = portfolio value change (no additional penalties)
    3. Trade 1 unit of stock per transaction

    State space: Sliding window of normalized features
    Action space: {0: Buy, 1: Hold, 2: Sell}
    Reward: Net portfolio value change after transaction costs
    """

    def __init__(self, data, original_close, window_size=120, transaction_fee=0.001, device='cpu'):
        """
        Initialize trading environment

        Args:
            data: Normalized feature data (numpy array)
            original_close: Original close prices (pandas Series, for calculating values)
            window_size: Observation window size (default: 120 days)
            transaction_fee: Transaction cost rate (default: 0.001 = 10 bps)
            device: torch device for tensors
        """
        self.data = data
        self.original_close = original_close
        self.window_size = window_size
        self.n_steps = len(data) - window_size
        self.transaction_fee = transaction_fee
        self.device = device

        # Trading state
        self.current_step = 0
        self.cash = 1000000  # Initial capital: $1,000,000
        self.shares = 0
        self.actions = [0, 1, 2]  # 0=Buy, 1=Hold, 2=Sell
        self.prev_portfolio_value = self.cash

        # Trading history
        self.portfolio_history = [self.cash]
        self.action_history = []
        self.reward_history = []

    def reset(self):
        """
        Reset environment to initial state

        Returns:
            Initial state tensor
        """
        self.current_step = 0
        self.cash = 1000000
        self.shares = 0
        self.prev_portfolio_value = self.cash

        self.portfolio_history = [self.cash]
        self.action_history = []
        self.reward_history = []

        return self._get_state()

    def _get_state(self):
        """
        Get current state (observation window)

        Returns:
            State tensor (window_size, features)
        """
        start = self.current_step
        end = self.current_step + self.window_size
        window = self.data[start:end]
        return torch.tensor(window, dtype=torch.float32, device=self.device)

    def step(self, action):
        """
        Execute action and transition to next state

        Paper formula (Section 3.3):
        Cost_t = c·|ΔN_t|·P^close_t
        C_t = C_{t-1} - ΔN_t·P^close_t - Cost_t
        N_t = N_{t-1} + ΔN_t
        V_t = C_t + N_t·P^close_t
        r_t = V_t - V_{t-1}

        Args:
            action: Action to take (0=buy, 1=hold, 2=sell)

        Returns:
            next_state: Next observation (or None if episode ended)
            reward: Reward received
            done: Whether episode ended
        """
        # Get current price
        current_price = self.original_close.iloc[
            self.current_step + self.window_size - 1
        ]

        # Calculate ΔN_t (full position trading strategy)
        delta_N = 0
        if action == 0:  # Buy - use all cash to buy
            max_shares = int(self.cash / (current_price * (1 + self.transaction_fee)))
            if max_shares > 0:
                delta_N = max_shares
        elif action == 2:  # Sell - sell all shares
            if self.shares > 0:
                delta_N = -self.shares

        # Calculate transaction cost (paper formula)
        cost = self.transaction_fee * abs(delta_N) * current_price

        # Update cash and shares (paper order)
        self.cash = self.cash - delta_N * current_price - cost
        self.shares = self.shares + delta_N

        # Calculate portfolio value
        portfolio_value = self.cash + self.shares * current_price

        # Reward = net value change (paper Eq. 2)
        reward = portfolio_value - self.prev_portfolio_value

        self.prev_portfolio_value = portfolio_value
        self.current_step += 1

        # Record history
        self.portfolio_history.append(portfolio_value)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # Check if episode ended
        done = self.current_step >= self.n_steps
        next_state = self._get_state() if not done else None

        return next_state, reward, done

    def get_portfolio_value(self):
        """Get current portfolio value"""
        if self.current_step >= self.window_size:
            current_price = self.original_close.iloc[
                self.current_step + self.window_size - 1
            ]
        else:
            current_price = self.original_close.iloc[self.window_size - 1]

        return self.cash + self.shares * current_price

    def get_state_info(self):
        """
        Get current state information

        Returns:
            Dictionary with current state info
        """
        return {
            'step': self.current_step,
            'cash': self.cash,
            'shares': self.shares,
            'portfolio_value': self.get_portfolio_value(),
        }
