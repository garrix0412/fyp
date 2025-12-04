"""
Deep Q-Network (DQN) Agent

Implements DQN algorithm with:
- Experience replay buffer
- Target network with periodic updates
- Epsilon-greedy exploration
- Gradient clipping for stability
"""

import torch
import torch.nn.functional as F
import random
from collections import deque


class DQNAgent:
    """
    DQN Agent with experience replay and target network

    Paper reference: Section 3.4 - DRL Training Protocol

    Key components:
    - Policy network: Q-value predictions
    - Target network: Stabilize training
    - Experience replay: Break temporal correlations
    - Epsilon-greedy: Balance exploration and exploitation
    """

    def __init__(self, model, env, config, device):
        """
        Initialize DQN agent

        Args:
            model: Q-network (policy network)
            env: Trading environment
            config: Configuration dictionary with 'dqn', 'model_args', 'train' sections
            device: torch device (cuda/mps/cpu)
        """
        self.device = device
        self.model = model.to(device)

        # Create target network (same architecture as policy network)
        model_class = type(model)
        self.target_model = model_class(**config['model_args']).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.env = env
        self.config = config

        # Optimizer
        dqn_config = config['dqn']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=dqn_config['lr'],
            weight_decay=dqn_config['weight_decay']
        )

        # RL parameters
        self.gamma = dqn_config['gamma']  # Discount factor
        self.epsilon = dqn_config['epsilon_start']  # Initial exploration rate
        self.epsilon_min = dqn_config['epsilon_min']  # Minimum exploration rate
        self.epsilon_decay = dqn_config['epsilon_decay']  # Decay rate per episode

        # Experience replay buffer
        self.memory = deque(maxlen=dqn_config['buffer_size'])
        self.batch_size = dqn_config['batch_size']

    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state tensor
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Action index (0=buy, 1=hold, 2=sell)
        """
        # Exploration: random action with probability epsilon
        if training and random.random() < self.epsilon:
            return random.choice(self.env.actions)

        # Exploitation: action with highest Q-value
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if episode ended)
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Perform one training step using batch from replay buffer

        Returns:
            Loss value (float) or None if insufficient samples
        """
        # Need enough samples in buffer
        if len(self.memory) < self.batch_size:
            return None

        # Sample random batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Handle terminal states (next_state = None)
        valid_next_states = [ns for ns in next_states if ns is not None]
        if len(valid_next_states) == 0:
            return None

        valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
        next_states = torch.stack(valid_next_states)
        dones = torch.tensor([dones[i] for i in valid_indices], device=self.device, dtype=torch.float32)
        rewards = rewards[valid_indices]

        # Compute Q(s, a) for selected actions
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_values = q_values[valid_indices]

        # Compute target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss (MSE between Q-values and targets)
        loss = F.mse_loss(q_values, targets)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['dqn']['max_grad_norm']
        )

        self.optimizer.step()

        # Note: Epsilon is decayed per episode, not per step
        # See train_one_episode() in training loop

        return loss.item()

    def update_target_model(self):
        """
        Update target network with policy network weights

        Paper: Target network updated periodically to stabilize training
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """
        Decay epsilon (called once per episode)

        Epsilon decay schedule:
        - Start: epsilon_start (e.g., 1.0)
        - Decay: multiply by epsilon_decay (e.g., 0.995) each episode
        - Min: epsilon_min (e.g., 0.01)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """
        Save agent state

        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        """
        Load agent state

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
