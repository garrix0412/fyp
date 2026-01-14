"""
Training Script for Graph-LORS Trader

Extended training pipeline for graph-enhanced models:
- Supports both graph models and original baselines
- Integrates dynamic correlation graph construction
- Provides interpretability analysis outputs

Usage:
    # Quick test
    python scripts/train_graph.py --model GraphLORSTrader --episodes 5

    # Full experiment
    python scripts/train_graph.py --mode graph_full_experiment --episodes 100

    # Ablation study
    python scripts/train_graph.py --mode graph_ablation --episodes 100
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from torch_geometric.data import Data

# Import project modules
from src.models import get_model, MODEL_REGISTRY, is_graph_model, GRAPH_MODELS_AVAILABLE
from src.agents import DQNAgent
from src.environment import TradingEnv
from src.utils import (
    ChronologicalSplitter,
    AntiLeakagePreprocessor,
    TradingMetrics,
    get_config,
    get_graph_config,
    print_config,
)
from src.utils.graph_utils import DynamicGraphBuilder, GraphNodeFeatureBuilder

# Create necessary directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device('cpu')
    print("Using CPU")


class GraphTradingEnv:
    """
    Extended Trading Environment with Graph Support

    Wraps the base TradingEnv and provides graph data at each step.
    """

    def __init__(self, base_env, price_matrix, graph_builder, node_feature_builder,
                 assets, target_idx=0, device='cpu'):
        """
        Initialize Graph Trading Environment

        Args:
            base_env: Base TradingEnv instance
            price_matrix: Price matrix for all assets
            graph_builder: DynamicGraphBuilder instance
            node_feature_builder: GraphNodeFeatureBuilder instance
            assets: List of asset names
            target_idx: Index of target asset
            device: Torch device
        """
        self.base_env = base_env
        self.price_matrix = price_matrix
        self.graph_builder = graph_builder
        self.node_feature_builder = node_feature_builder
        self.assets = assets
        self.target_idx = target_idx
        self.device = device
        self.n_assets = len(assets)

        # Map base_env attributes
        self.actions = base_env.actions
        self.window_size = base_env.window_size

        # Offset for aligning base_env step with price_matrix index
        self.data_offset = 0

    def set_data_offset(self, offset):
        """Set offset for aligning with price matrix"""
        self.data_offset = offset

    def reset(self):
        """Reset environment"""
        state = self.base_env.reset()
        return state

    def step(self, action):
        """Execute action and return next state with graph data"""
        next_state, reward, done = self.base_env.step(action)
        return next_state, reward, done

    def get_graph_data(self, t_idx):
        """
        Get graph data for current time step

        Args:
            t_idx: Time index in price matrix

        Returns:
            PyG Data object with graph structure and node features
        """
        # Build dynamic graph (only uses historical data)
        corr_matrix = self.graph_builder.compute_correlation_matrix(
            self.price_matrix, t_idx
        )
        adj_matrix = self.graph_builder.build_adjacency_matrix(corr_matrix)
        edge_index, edge_weight = self.graph_builder.get_edge_index_and_weights(adj_matrix)

        # Get node features
        node_features = self.node_feature_builder.compute_node_features(
            self.price_matrix, t_idx
        )

        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32, device=self.device),
            edge_index=torch.tensor(edge_index, dtype=torch.long, device=self.device),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float32, device=self.device),
        )
        data.target_idx = self.target_idx

        return data

    @property
    def current_step(self):
        return self.base_env.current_step

    @property
    def portfolio_history(self):
        return self.base_env.portfolio_history

    @property
    def action_history(self):
        return self.base_env.action_history

    @property
    def reward_history(self):
        return self.base_env.reward_history


class GraphDQNAgent:
    """
    Extended DQN Agent for Graph Models

    Handles models that require both sequence and graph inputs.
    """

    def __init__(self, model, env, config, device, use_graph=True):
        """
        Initialize Graph DQN Agent

        Args:
            model: Q-network model
            env: GraphTradingEnv instance
            config: Configuration dictionary
            device: Torch device
            use_graph: Whether model uses graph input
        """
        self.device = device
        self.model = model.to(device)
        self.use_graph = use_graph

        # Create target network
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
        self.gamma = dqn_config['gamma']
        self.epsilon = dqn_config['epsilon_start']
        self.epsilon_min = dqn_config['epsilon_min']
        self.epsilon_decay = dqn_config['epsilon_decay']

        # Experience replay
        from collections import deque
        self.memory = deque(maxlen=dqn_config['buffer_size'])
        self.batch_size = dqn_config['batch_size']

    def act(self, state, graph_data=None, training=True):
        """Select action"""
        if training and random.random() < self.epsilon:
            return random.choice(self.env.actions)

        with torch.no_grad():
            if self.use_graph and graph_data is not None:
                q_values, _ = self.model(state.unsqueeze(0), graph_data)
            else:
                q_values = self.model(state.unsqueeze(0))
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done, graph_data=None, next_graph_data=None):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done, graph_data, next_graph_data))

    def train_step(self):
        """Training step"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, graph_datas, next_graph_datas = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Handle terminal states
        valid_next_states = [ns for ns in next_states if ns is not None]
        if len(valid_next_states) == 0:
            return None

        valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
        next_states_tensor = torch.stack(valid_next_states)
        dones_tensor = torch.tensor([dones[i] for i in valid_indices], device=self.device, dtype=torch.float32)
        rewards = rewards[valid_indices]

        # Compute Q-values
        if self.use_graph:
            # For graph models, we need to handle graph data
            # Use the first valid graph as representative (simplified)
            valid_graphs = [graph_datas[i] for i in valid_indices if graph_datas[i] is not None]
            if valid_graphs:
                representative_graph = valid_graphs[0]
                q_values, _ = self.model(states, representative_graph)
            else:
                return None
        else:
            q_values = self.model(states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_values = q_values[valid_indices]

        # Compute target Q-values
        with torch.no_grad():
            if self.use_graph and valid_graphs:
                next_q_values, _ = self.target_model(next_states_tensor, representative_graph)
            else:
                next_q_values = self.target_model(next_states_tensor)
            next_q_values = next_q_values.max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones_tensor)

        # Compute loss and optimize
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['dqn']['max_grad_norm'])
        self.optimizer.step()

        return loss.item()

    def update_target_model(self):
        """Update target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save agent state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def train_one_episode_graph(agent, env, price_matrix_indices):
    """Train one episode with graph data"""
    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done:
        # Get current time index for graph
        t_idx = price_matrix_indices[env.current_step] if env.current_step < len(price_matrix_indices) else price_matrix_indices[-1]

        # Get graph data
        graph_data = env.get_graph_data(t_idx)

        # Select action
        action = agent.act(state, graph_data, training=True)

        # Execute action
        next_state, reward, done = env.step(action)

        # Get next graph data
        if not done:
            next_t_idx = price_matrix_indices[env.current_step] if env.current_step < len(price_matrix_indices) else price_matrix_indices[-1]
            next_graph_data = env.get_graph_data(next_t_idx)
        else:
            next_graph_data = None

        # Store experience
        agent.remember(state, action, reward, next_state, done, graph_data, next_graph_data)

        # Train
        agent.train_step()

        state = next_state
        total_reward += reward
        step += 1

    agent.decay_epsilon()
    return total_reward


def evaluate_graph(agent, env, price_matrix_indices):
    """Evaluate agent with graph data"""
    agent.model.eval()

    state = env.reset()
    done = False

    while not done:
        t_idx = price_matrix_indices[env.current_step] if env.current_step < len(price_matrix_indices) else price_matrix_indices[-1]
        graph_data = env.get_graph_data(t_idx)

        action = agent.act(state, graph_data, training=False)
        next_state, reward, done = env.step(action)
        state = next_state

    metrics = TradingMetrics.comprehensive_report(
        env.portfolio_history,
        env.action_history,
        env.reward_history
    )

    agent.model.train()
    return metrics


def run_graph_experiment(model_name, seed, config):
    """Run single experiment for graph model"""
    print(f"\n{'='*70}")
    print(f"Running Graph Experiment: {model_name} | Seed: {seed}")
    print(f"{'='*70}")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load configurations
    graph_config = config['graph']
    data_config = config['data']

    # Check if model requires graph
    requires_graph = is_graph_model(model_name)

    # Load price matrix for graph construction
    if requires_graph:
        try:
            price_matrix = pd.read_csv(
                graph_config['price_matrix_path'],
                index_col=0, parse_dates=True
            )
            print(f"Loaded price matrix: {price_matrix.shape}")
        except FileNotFoundError:
            print("Price matrix not found. Run fetch_multi_asset_data.py first.")
            return None

        # Initialize graph builders
        graph_builder = DynamicGraphBuilder(
            window_size=graph_config['correlation_window'],
            top_k=graph_config['top_k']
        )
        node_feature_builder = GraphNodeFeatureBuilder(
            feature_window=graph_config['correlation_window']
        )

        # Use graph-aligned target data
        data_path = graph_config['graph_target_path']
    else:
        price_matrix = None
        graph_builder = None
        node_feature_builder = None
        data_path = data_config['csv_path']

    # Load and split data
    print(f"\nLoading data from: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if 'Ticker' in data.columns:
        data = data[data['Ticker'] == data_config['ticker']].drop(columns=['Ticker'])

    # Chronological split
    splitter = ChronologicalSplitter(
        train_ratio=data_config['split_ratio']['train'],
        val_ratio=data_config['split_ratio']['val'],
        test_ratio=data_config['split_ratio']['test']
    )
    train_data, val_data, test_data = splitter.split(data)

    # Keep original prices
    train_close = train_data['Close'].copy()
    val_close = val_data['Close'].copy()
    test_close = test_data['Close'].copy()

    # Preprocess
    features = data_config['features']
    preprocessor = AntiLeakagePreprocessor()

    train_scaled = preprocessor.fit_transform(train_data, features)
    val_scaled = preprocessor.transform(val_data, features)
    test_scaled = preprocessor.transform(test_data, features)

    # Create environments
    env_config = config['env']

    train_base_env = TradingEnv(
        train_scaled, train_close,
        window_size=env_config['window_size'],
        transaction_fee=env_config['transaction_fee'],
        device=device
    )

    val_base_env = TradingEnv(
        val_scaled, val_close,
        window_size=env_config['window_size'],
        transaction_fee=env_config['transaction_fee'],
        device=device
    )

    test_base_env = TradingEnv(
        test_scaled, test_close,
        window_size=env_config['window_size'],
        transaction_fee=env_config['transaction_fee'],
        device=device
    )

    # Wrap with graph environment if needed
    if requires_graph:
        assets = [graph_config['target']] + graph_config['context_assets']

        # Create index mappings for price matrix
        train_indices = [price_matrix.index.get_loc(d) for d in train_data.index if d in price_matrix.index]
        val_indices = [price_matrix.index.get_loc(d) for d in val_data.index if d in price_matrix.index]
        test_indices = [price_matrix.index.get_loc(d) for d in test_data.index if d in price_matrix.index]

        train_env = GraphTradingEnv(
            train_base_env, price_matrix, graph_builder, node_feature_builder,
            assets, target_idx=0, device=device
        )
        val_env = GraphTradingEnv(
            val_base_env, price_matrix, graph_builder, node_feature_builder,
            assets, target_idx=0, device=device
        )
        test_env = GraphTradingEnv(
            test_base_env, price_matrix, graph_builder, node_feature_builder,
            assets, target_idx=0, device=device
        )
    else:
        train_env = train_base_env
        val_env = val_base_env
        test_env = test_base_env
        train_indices = list(range(len(train_data)))
        val_indices = list(range(len(val_data)))
        test_indices = list(range(len(test_data)))

    # Create model
    model_config = config['models'][model_name]
    model = get_model(model_name, **model_config)

    # Create agent
    agent_config = {
        'model_args': model_config,
        'dqn': config['dqn'],
        'train': config['train'],
    }

    if requires_graph:
        agent = GraphDQNAgent(model, train_env, agent_config, device, use_graph=True)
    else:
        agent = DQNAgent(model, train_env, agent_config, device)

    # Training loop
    train_config = config['train']
    max_episodes = train_config['max_episodes']
    patience = train_config['patience']
    val_freq = train_config['val_freq']

    best_val_sharpe = -np.inf
    best_model_state = None
    patience_counter = 0

    print(f"\nStarting training: {max_episodes} episodes, patience={patience}")

    for episode in range(max_episodes):
        # Train one episode
        if requires_graph:
            train_reward = train_one_episode_graph(agent, train_env, train_indices)
        else:
            state = train_env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state, training=True)
                next_state, reward, done = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                total_reward += reward
            agent.decay_epsilon()
            train_reward = total_reward

        # Validation
        if (episode + 1) % val_freq == 0:
            if requires_graph:
                val_metrics = evaluate_graph(agent, val_env, val_indices)
            else:
                agent.model.eval()
                state = val_env.reset()
                done = False
                while not done:
                    action = agent.act(state, training=False)
                    next_state, reward, done = val_env.step(action)
                    state = next_state
                val_metrics = TradingMetrics.comprehensive_report(
                    val_env.portfolio_history,
                    val_env.action_history,
                    val_env.reward_history
                )
                agent.model.train()

            val_sharpe = val_metrics['Sharpe']

            print(f"Episode {episode+1}: Train reward={train_reward:.2f}, Val Sharpe={val_sharpe:.4f}, Epsilon={agent.epsilon:.4f}")

            if val_sharpe > best_val_sharpe:
                print(f"  New best! (prev: {best_val_sharpe:.4f})")
                best_val_sharpe = val_sharpe
                best_model_state = agent.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at episode {episode+1}")
                    break

        # Update target network
        if (episode + 1) % config['dqn']['target_update_freq'] == 0:
            agent.update_target_model()

    # Restore best model
    if best_model_state is not None:
        agent.model.load_state_dict(best_model_state)

    # Test evaluation
    print(f"\n{'='*70}")
    print("Test Evaluation")
    print(f"{'='*70}")

    if requires_graph:
        test_metrics = evaluate_graph(agent, test_env, test_indices)
    else:
        agent.model.eval()
        state = test_env.reset()
        done = False
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done = test_env.step(action)
            state = next_state
        test_metrics = TradingMetrics.comprehensive_report(
            test_env.portfolio_history,
            test_env.action_history,
            test_env.reward_history
        )

    TradingMetrics.print_report(test_metrics, "Test Results")

    results = {
        'model': model_name,
        'seed': seed,
        'best_val_sharpe': best_val_sharpe,
        'test_metrics': test_metrics,
        'requires_graph': requires_graph,
        'timestamp': datetime.now().isoformat(),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Graph-LORS Trader models')
    parser.add_argument('--model', type=str, default='GraphLORSTrader',
                       help='Model name')
    parser.add_argument('--mode', type=str, default='graph_quick_test',
                       help='Experiment mode')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override max episodes')
    parser.add_argument('--seeds', type=str, default=None,
                       help='Override seeds (comma-separated)')

    args = parser.parse_args()

    # Check graph models available
    if not GRAPH_MODELS_AVAILABLE:
        print("Error: Graph models require torch_geometric. Please install it first.")
        print("  pip install torch_geometric")
        return

    # Load config
    config = get_config(args.mode)

    # Override if specified
    if args.episodes:
        config['train']['max_episodes'] = args.episodes
    if args.seeds:
        config['train']['seeds'] = [int(s) for s in args.seeds.split(',')]

    # Get models and seeds
    if args.model != 'all':
        models = [args.model]
    else:
        models = config['train']['models']

    seeds = config['train']['seeds']

    print_config(config)

    # Run experiments
    all_results = []

    print(f"\n{'='*70}")
    print(f"Running {len(models)} models x {len(seeds)} seeds = {len(models)*len(seeds)} experiments")
    print(f"{'='*70}")

    for model_name in models:
        for seed in seeds:
            results = run_graph_experiment(model_name, seed, config)
            if results:
                all_results.append(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/graph_experiment_{args.mode}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to: {results_file}")
    print("Experiments completed!")


if __name__ == "__main__":
    main()
