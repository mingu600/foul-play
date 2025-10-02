# Evaluation Function Weight Training

This document explains how to use MCTS data to improve the evaluation function weights in poke-engine.

## Overview

The evaluation function in `poke-engine/src/genx/evaluate.rs` is a linear combination of features:

```rust
score = POKEMON_HP * (hp/maxhp) + POKEMON_ALIVE * 1.0 + LEECH_SEED * has_leech_seed + ...
```

We can improve these weights by:
1. Collecting MCTS evaluations during games
2. Training a linear model to predict MCTS values from features
3. Using the learned weights to update the eval function

## Why This Works

- MCTS with 4M iterations is much smarter than the static eval
- By training eval to match MCTS values, we transfer MCTS's "search knowledge" to the eval
- This is similar to AlphaZero's policy iteration
- **Key**: We train on game outcomes (ground truth) to avoid circular reasoning

## Step-by-Step Guide

### Step 1: Collect Training Data

Integrate the training logger into your battle loop:

```python
from fp.search.training_logger import TrainingDataLogger
from fp.search.main import set_training_logger, find_best_move

# Create logger
with TrainingDataLogger("data/training_data.jsonl") as logger:
    # Enable data collection
    set_training_logger(logger)

    # Run games normally
    for game_idx in range(1000):
        battle = create_new_battle()
        battle.battle_id = f"game_{game_idx}"

        while not battle.is_over():
            # find_best_move automatically logs training data now
            move = find_best_move(battle)
            battle.apply_move(move)

        # Log game outcome (important!)
        outcome = 1.0 if battle.user_won() else -1.0
        logger.log_game_end(battle.battle_id, outcome)

    # Disable when done
    set_training_logger(None)
```

**What gets logged:**
- State representation (before each move)
- MCTS value (aggregated across determinizations)
- Policy entropy (how certain MCTS was)
- Game outcome (win/loss)

**How much data:**
- ~30 turns per game
- 1000 games = ~30,000 training points
- Recommend: 10,000+ training points (300+ games)

### Step 2: Train Weights

Run the training script:

```bash
python scripts/train_eval_weights.py \
    --input data/training_data.jsonl \
    --output data/learned_weights.json \
    --target game_outcome \
    --regularization 10.0
```

**Arguments:**
- `--target`: What to train on
  - `game_outcome`: Train on actual wins/losses (recommended, no circular dependency)
  - `mcts_value`: Train on MCTS evaluations (faster learning, but risky)
  - `hybrid`: Mix of both
- `--regularization`: How much to trust old weights (higher = more conservative)
  - `1.0`: Allow large changes
  - `10.0`: Moderate changes (recommended)
  - `100.0`: Stay very close to defaults
- `--method`:
  - `ridge`: L2 regularization (recommended)
  - `lasso`: L1 regularization (sparse weights)

**Output:**
```
Weight Comparison (Default -> Learned)
================================================================================
pokemon_alive           :    30.00 ->    32.45 (Δ  +2.45,  +8.2%)
pokemon_hp              :   100.00 ->   105.32 (Δ  +5.32,  +5.3%)
substitute              :    40.00 ->    52.18 (Δ +12.18, +30.5%)
sticky_web              :   -25.00 ->   -31.22 (Δ  -6.22, +24.9%)
...

Train MSE: 0.023456, R²: 0.7834
Val MSE: 0.025123, R²: 0.7691
```

**What to look for:**
- Validation R² > 0.5 (model is learning something useful)
- Weight changes < 50% (model isn't going crazy)
- Changes make intuitive sense

### Step 3: Validate Weights

Before deploying, test the new weights:

```python
# Test in a few games manually
# Compare win rate with old vs new weights

# You can also look at specific positions:
from fp.search.feature_extraction import extract_features_from_state, FEATURE_NAMES
import numpy as np
import json

# Load learned weights
with open('data/learned_weights.json') as f:
    learned = json.load(f)

# Evaluate a position
state = battle_to_poke_engine_state(battle)
features = extract_features_from_state(state)
old_eval = sum(DEFAULT_WEIGHTS[name] * features[i] for i, name in enumerate(FEATURE_NAMES))
new_eval = sum(learned[name] * features[i] for i, name in enumerate(FEATURE_NAMES))

print(f"Old eval: {old_eval:.1f}")
print(f"New eval: {new_eval:.1f}")
```

**Validation criteria:**
- Play 100+ games with new weights
- Win rate should be ≥ baseline
- If win rate drops, increase regularization or collect more data

### Step 4: Deploy to Rust

Update `poke-engine/src/genx/evaluate.rs`:

```rust
// Before
const POKEMON_ALIVE: f32 = 30.0;
const POKEMON_HP: f32 = 100.0;
const SUBSTITUTE: f32 = 40.0;

// After (example - use your learned values)
const POKEMON_ALIVE: f32 = 32.45;
const POKEMON_HP: f32 = 105.32;
const SUBSTITUTE: f32 = 52.18;
```

Rebuild:
```bash
cd poke-engine-dev
cargo build --release
```

Test in foul-play to ensure performance is maintained or improved.

## Iteration

This process can be repeated:

1. Collect data with weights_v1
2. Train → get weights_v2
3. Deploy weights_v2
4. Collect data with weights_v2
5. Train → get weights_v3
6. ...

This is **policy iteration** - proven to converge in theory.

**In practice:**
- Diminishing returns after 2-3 iterations
- Most gains come from first iteration
- Monitor for divergence (weights becoming unreasonable)

## Troubleshooting

### "Validation R² is negative or very low"
- Not enough training data - collect more games
- Target is too noisy - try `--target hybrid` instead of `game_outcome`
- Increase regularization

### "Weights changed dramatically (>100%)"
- Regularization too low - increase to 50 or 100
- Training data is biased (e.g., only vs weak opponents)
- Features might be correlated - inspect training data

### "New weights perform worse in real games"
- Overfitting - increase regularization
- Training data not representative - collect more diverse games
- MCTS values might be misleading - train on `game_outcome` instead

### "Model shows good R² but weights seem wrong"
- Check feature extraction - make sure signs are correct
- Verify state string parsing is working
- Inspect a few examples manually to debug

## Advanced: Active Learning

Collect data from positions where MCTS is most uncertain:

```python
# In your battle loop
if _training_logger is not None:
    policy_entropy = compute_policy_entropy(mcts_results)

    # Only log if MCTS was uncertain (high entropy = multiple good moves)
    if policy_entropy > 1.5:  # Threshold
        _training_logger.log_turn(battle, mcts_results)
```

This focuses training on "hard" positions where better eval would help most.

## Safety Measures

To prevent degenerate behavior:

1. **Always validate**: Test new weights in real games before deploying
2. **Use regularization**: Keep weights anchored to hand-tuned defaults
3. **Train on outcomes**: Game wins/losses are objective ground truth
4. **Monitor weight changes**: Flag changes > 100% for manual review
5. **Gradual deployment**: Mix old and new weights initially:
   ```rust
   const WEIGHT: f32 = 0.8 * OLD + 0.2 * NEW;
   ```

## Files

- `fp/search/training_logger.py`: Collects data during games
- `fp/search/training_data.py`: Extracts MCTS values from results
- `fp/search/feature_extraction.py`: Converts states to feature vectors
- `scripts/train_eval_weights.py`: Trains weights from collected data
- `scripts/collect_training_data.py`: Example data collection integration

## References

- AlphaZero paper: Policy iteration with self-play
- Temporal Difference Learning: Bootstrapping from later positions
- Policy Iteration: Proven convergence guarantees
