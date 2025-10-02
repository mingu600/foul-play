#!/usr/bin/env python
"""
Train evaluation function weights from MCTS training data.

Usage:
    python scripts/train_eval_weights.py \
        --input training_data.jsonl \
        --output learned_weights.json \
        --target mcts_value  # or 'game_outcome'
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fp.search.feature_extraction import (
    extract_features_from_state_string,
    FEATURE_NAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default weights from poke-engine/src/genx/evaluate.rs
DEFAULT_WEIGHTS = {
    'pokemon_alive': 30.0,
    'pokemon_hp': 100.0,
    'attack_boost_pos': 30.0,
    'attack_boost_neg': -30.0,
    'defense_boost_pos': 15.0,
    'defense_boost_neg': -15.0,
    'special_attack_boost_pos': 30.0,
    'special_attack_boost_neg': -30.0,
    'special_defense_boost_pos': 15.0,
    'special_defense_boost_neg': -15.0,
    'speed_boost_pos': 30.0,
    'speed_boost_neg': -30.0,
    'burned': -25.0,
    'frozen': -40.0,
    'asleep': -25.0,
    'paralyzed': -25.0,
    'toxic': -30.0,
    'poisoned': -10.0,
    'leech_seed': -30.0,
    'substitute': 40.0,
    'confusion': -20.0,
    'reflect': 20.0,
    'light_screen': 20.0,
    'aurora_veil': 40.0,
    'safeguard': 5.0,
    'tailwind': 7.0,
    'healing_wish': 30.0,
    'stealth_rock': -10.0,
    'spikes': -7.0,
    'toxic_spikes': -7.0,
    'sticky_web': -25.0,
    'used_tera': -75.0,
    'has_item': 10.0,
}


def load_training_data(input_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load training data from JSONL file.

    Returns:
        (turn_records, game_end_records)
    """
    turn_records = []
    game_end_records = []

    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get('type') == 'game_end':
                game_end_records.append(record)
            else:
                turn_records.append(record)

    logger.info(f"Loaded {len(turn_records)} turn records from {len(game_end_records)} games")
    return turn_records, game_end_records


def prepare_training_data(
    turn_records: List[Dict],
    game_end_records: List[Dict],
    target: str = 'mcts_value',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X (features) and y (targets) for training.

    Args:
        turn_records: Turn data with states and MCTS values
        game_end_records: Game outcomes
        target: 'mcts_value', 'game_outcome', or 'hybrid'

    Returns:
        (X, y) where X is (n_samples, n_features) and y is (n_samples,)
    """
    # Build game outcome lookup
    game_outcomes = {
        record['battle_id']: record['outcome']
        for record in game_end_records
    }

    X_list = []
    y_list = []

    for record in turn_records:
        try:
            # Extract features
            features = extract_features_from_state_string(record['state'])
            X_list.append(features)

            # Get target value
            if target == 'mcts_value':
                y_val = record['mcts_value']

            elif target == 'game_outcome':
                outcome = game_outcomes.get(record['battle_id'], 0.0)
                # Map from {-1, 1} to [0, 1]
                y_val = (outcome + 1.0) / 2.0

            elif target == 'hybrid':
                # Exponential decay towards game outcome
                mcts_val = record['mcts_value']
                outcome = game_outcomes.get(record['battle_id'], 0.0)
                outcome_normalized = (outcome + 1.0) / 2.0

                # Weight MCTS more early in game, outcome more late in game
                # This is a simple heuristic - can be improved
                mcts_weight = 0.7
                y_val = mcts_weight * mcts_val + (1 - mcts_weight) * outcome_normalized

            else:
                raise ValueError(f"Unknown target: {target}")

            y_list.append(y_val)

        except Exception as e:
            logger.warning(f"Error processing record: {e}")
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Prepared {len(X)} training examples with target={target}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")

    return X, y


def train_weights(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 10.0,
    method: str = 'ridge',
) -> np.ndarray:
    """
    Train weights using linear regression.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        regularization: Regularization strength (higher = stay closer to defaults)
        method: 'ridge' (L2) or 'lasso' (L1)

    Returns:
        Learned weight vector (n_features,)
    """
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    if method == 'ridge':
        model = Ridge(alpha=regularization, fit_intercept=False)
    elif method == 'lasso':
        model = Lasso(alpha=regularization, fit_intercept=False, max_iter=10000)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Training {method} regression with alpha={regularization}")
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    logger.info(f"Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
    logger.info(f"Val MSE: {val_mse:.6f}, R²: {val_r2:.4f}")

    return model.coef_


def compare_weights(default_weights: Dict[str, float], learned_weights: np.ndarray):
    """Print comparison of default vs learned weights."""
    logger.info("\n" + "="*80)
    logger.info("Weight Comparison (Default -> Learned)")
    logger.info("="*80)

    for i, name in enumerate(FEATURE_NAMES):
        default = default_weights.get(name, 0.0)
        learned = learned_weights[i]
        change = learned - default
        pct_change = (change / default * 100) if default != 0 else float('inf')

        logger.info(
            f"{name:30s}: {default:8.2f} -> {learned:8.2f} "
            f"(Δ {change:+7.2f}, {pct_change:+6.1f}%)"
        )


def save_weights(weights: np.ndarray, output_path: Path):
    """Save learned weights to JSON file."""
    weight_dict = {
        name: float(weights[i])
        for i, name in enumerate(FEATURE_NAMES)
    }

    with open(output_path, 'w') as f:
        json.dump(weight_dict, f, indent=2)

    logger.info(f"Saved weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train eval weights from MCTS data')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to training data JSONL file')
    parser.add_argument('--output', type=str, default='learned_weights.json',
                        help='Path to output learned weights')
    parser.add_argument('--target', type=str, default='mcts_value',
                        choices=['mcts_value', 'game_outcome', 'hybrid'],
                        help='Target to train on')
    parser.add_argument('--regularization', type=float, default=10.0,
                        help='Regularization strength (higher = stay closer to defaults)')
    parser.add_argument('--method', type=str, default='ridge',
                        choices=['ridge', 'lasso'],
                        help='Regression method')

    args = parser.parse_args()

    # Load data
    turn_records, game_end_records = load_training_data(Path(args.input))

    if len(turn_records) == 0:
        logger.error("No training data found!")
        return

    # Prepare training data
    X, y = prepare_training_data(turn_records, game_end_records, args.target)

    # Train weights
    learned_weights = train_weights(X, y, args.regularization, args.method)

    # Compare to defaults
    compare_weights(DEFAULT_WEIGHTS, learned_weights)

    # Save
    save_weights(learned_weights, Path(args.output))

    logger.info("\nTraining complete!")
    logger.info(f"Next steps:")
    logger.info(f"1. Review weight changes above")
    logger.info(f"2. Update poke-engine/src/genx/evaluate.rs with new weights")
    logger.info(f"3. Rebuild poke-engine and test in real games")


if __name__ == '__main__':
    main()
