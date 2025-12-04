"""
Training Script for LORSTransformerDRL

Complete training pipeline for paper reproduction:
- Model training with validation
- Model evaluation on test set
- Result saving and logging

Usage:
    # Train single model
    python scripts/train.py --model LORSTransformerDRL --episodes 100

    # Train all models
    python scripts/train.py --all --episodes 100

    # Multi-seed experiments
    python scripts/train.py --model LORSTransformerDRL --seeds 42,43,44,45,46

    # Full experiment (all models, all seeds)
    python scripts/train.py --all --full --episodes 100
"""

import sys
import os
# Add project root to path
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

# Import project modules
from src.models import get_model, MODEL_REGISTRY
from src.agents import DQNAgent
from src.environment import TradingEnv
from src.utils import (
    ChronologicalSplitter,
    AntiLeakagePreprocessor,
    TradingMetrics,
    get_config,
    print_config,
)

# Create necessary directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Device configuration: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("🚀 Using Apple Silicon GPU (MPS) for training acceleration")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("🚀 Using NVIDIA GPU (CUDA) for training acceleration")
else:
    device = torch.device('cpu')
    print("⚠️  Using CPU for training (slower)")


def train_one_episode(agent, env):
    """
    Train for one episode

    Args:
        agent: DQN agent
        env: Trading environment

    Returns:
        total_reward: Episode total reward
    """
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state, training=True)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        loss = agent.train_step()

        state = next_state
        total_reward += reward

    # Decay epsilon once per episode
    agent.decay_epsilon()

    return total_reward


def evaluate(agent, env):
    """
    Evaluate agent (no exploration)

    Args:
        agent: DQN agent
        env: Trading environment

    Returns:
        metrics: Comprehensive evaluation metrics
    """
    # Set model to evaluation mode
    agent.model.eval()

    state = env.reset()
    done = False

    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done = env.step(action)
        state = next_state

    # Calculate all metrics
    metrics = TradingMetrics.comprehensive_report(
        env.portfolio_history,
        env.action_history,
        env.reward_history
    )

    # Set model back to training mode
    agent.model.train()

    return metrics


def train_with_validation(agent, train_env, val_env, config):
    """
    Training loop with validation and early stopping

    Paper: "Hyperparameters and early-stopping are selected by validation Sharpe"

    Args:
        agent: DQN agent
        train_env: Training environment
        val_env: Validation environment
        config: Configuration dictionary

    Returns:
        Dictionary with training history
    """
    train_config = config['train']
    max_episodes = train_config['max_episodes']
    patience = train_config['patience']
    val_freq = train_config['val_freq']

    best_val_sharpe = -np.inf
    best_model_state = None
    patience_counter = 0

    train_rewards = []
    val_sharpes = []

    print(f"\n{'='*70}")
    print(f"Starting Training - Max {max_episodes} episodes, Early stop patience={patience}")
    print(f"{'='*70}")

    for episode in range(max_episodes):
        # Train
        train_reward = train_one_episode(agent, train_env)
        train_rewards.append(train_reward)

        # Periodic validation
        if (episode + 1) % val_freq == 0:
            val_metrics = evaluate(agent, val_env)
            val_sharpe = val_metrics['Sharpe']
            val_sharpes.append((episode, val_sharpe))

            print(f"\nEpisode {episode+1}/{max_episodes}:")
            print(f"  Training reward: {train_reward:.2f}")
            print(f"  Validation Sharpe: {val_sharpe:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")

            # Model selection
            if val_sharpe > best_val_sharpe:
                print(f"  ✅ New best model! (Previous: {best_val_sharpe:.4f})")
                best_val_sharpe = val_sharpe
                best_model_state = agent.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  ⚠️  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered!")
                    break

        # Update target network
        if (episode + 1) % config['dqn']['target_update_freq'] == 0:
            agent.update_target_model()

    # Restore best model
    if best_model_state is not None:
        agent.model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model (Val Sharpe: {best_val_sharpe:.4f})")

    return {
        'train_rewards': train_rewards,
        'val_sharpes': val_sharpes,
        'best_val_sharpe': best_val_sharpe,
    }


def run_single_experiment(model_name, seed, config):
    """
    Run single experiment for one model with one seed

    Args:
        model_name: Model name
        seed: Random seed
        config: Configuration dictionary

    Returns:
        results: Experiment results dictionary
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"\nRandom seed: {seed}")

    # Load and split data
    print(f"\n{'='*70}")
    print("Loading and Splitting Data")
    print(f"{'='*70}")

    data = pd.read_csv(config['data']['csv_path'], index_col=0)

    # Handle data with or without Ticker column
    if 'Ticker' in data.columns:
        data = data[data['Ticker'] == config['data']['ticker']].drop(columns=['Ticker'])
    # If no Ticker column, assume data is already for the target ticker

    # Chronological split
    splitter = ChronologicalSplitter(
        train_ratio=config['data']['split_ratio']['train'],
        val_ratio=config['data']['split_ratio']['val'],
        test_ratio=config['data']['split_ratio']['test']
    )
    train_data, val_data, test_data = splitter.split(data)

    # Keep original prices
    train_close_orig = train_data['Close'].copy()
    val_close_orig = val_data['Close'].copy()
    test_close_orig = test_data['Close'].copy()

    # Preprocess
    features = config['data']['features']
    preprocessor = AntiLeakagePreprocessor()

    train_scaled = preprocessor.fit_transform(train_data, features)
    val_scaled = preprocessor.transform(val_data, features)
    test_scaled = preprocessor.transform(test_data, features)

    # Create environments
    train_env = TradingEnv(train_scaled, train_close_orig,
                          window_size=config['env']['window_size'],
                          transaction_fee=config['env']['transaction_fee'],
                          device=device)

    val_env = TradingEnv(val_scaled, val_close_orig,
                        window_size=config['env']['window_size'],
                        transaction_fee=config['env']['transaction_fee'],
                        device=device)

    test_env = TradingEnv(test_scaled, test_close_orig,
                         window_size=config['env']['window_size'],
                         transaction_fee=config['env']['transaction_fee'],
                         device=device)

    # Create model
    print(f"\n{'='*70}")
    print(f"Training Model: {model_name}")
    print(f"{'='*70}")

    model_config = config['models'][model_name]
    model = get_model(model_name, **model_config)

    # Create agent
    agent_config = {
        'model_args': model_config,
        'dqn': config['dqn'],
        'train': config['train'],
    }

    agent = DQNAgent(model, train_env, agent_config, device)

    # Train
    training_results = train_with_validation(agent, train_env, val_env, config)

    # Test evaluation
    print(f"\n{'='*70}")
    print("Test Set Evaluation (single pass)")
    print(f"{'='*70}")

    test_metrics = evaluate(agent, test_env)
    TradingMetrics.print_report(test_metrics, "Test Results")

    # Save results
    results = {
        'model': model_name,
        'seed': seed,
        'config': config['train']['max_episodes'],  # Store max episodes for reference
        'training': training_results,
        'test_metrics': test_metrics,
        'best_val_sharpe': training_results['best_val_sharpe'],
        'timestamp': datetime.now().isoformat(),
    }

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train LORSTransformerDRL models')
    parser.add_argument('--model', type=str, default='LORSTransformerDRL',
                       help='Model name (see MODEL_REGISTRY)')
    parser.add_argument('--all', action='store_true',
                       help='Train all models')
    parser.add_argument('--full', action='store_true',
                       help='Full experiment mode (5 seeds)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Max training episodes')
    parser.add_argument('--seeds', type=str, default='42',
                       help='Random seeds (comma-separated)')

    args = parser.parse_args()

    # Determine experiment mode
    if args.full:
        mode = 'full_experiment'
        models = list(MODEL_REGISTRY.keys()) if args.all else [args.model]
        seeds = [42, 43, 44, 45, 46]
    else:
        mode = 'quick_test'
        models = list(MODEL_REGISTRY.keys()) if args.all else [args.model]
        seeds = [int(s) for s in args.seeds.split(',')]

    # Load config
    config = get_config(mode if not args.full else 'full_experiment')
    config['train']['max_episodes'] = args.episodes
    print_config(config)

    # Run experiments
    all_results = []

    print(f"\n{'='*80}")
    print(f"Running Experiments")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(models) * len(seeds)}")

    for model_name in models:
        for seed in seeds:
            print(f"\n{'#'*80}")
            print(f"Experiment: {model_name} | Seed: {seed}")
            print(f"{'#'*80}")

            results = run_single_experiment(model_name, seed, config)
            all_results.append(results)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/experiment_{mode}_{timestamp}.json"

    # Convert numpy types for JSON serialization
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

    print(f"\n✅ Results saved to: {results_file}")
    print(f"\n{'='*80}")
    print("Experiments completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
