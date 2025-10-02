"""
Logger for collecting training data from MCTS games.

Usage:
    # At the start of your program
    training_logger = TrainingDataLogger("training_data.jsonl")

    # In find_best_move, after computing mcts_results
    training_logger.log_turn(battle, mcts_results)

    # When game ends
    training_logger.log_game_end(battle.battle_id, outcome=1.0 if won else -1.0)

    # Clean up
    training_logger.close()
"""

import json
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime

from poke_engine import MctsResult
from fp.battle import Battle
from fp.search.poke_engine_helpers import battle_to_poke_engine_state
from fp.search.training_data import compute_mcts_value_from_results, compute_policy_entropy

logger = logging.getLogger(__name__)


class TrainingDataLogger:
    """Logs training data from MCTS search during games."""

    def __init__(self, output_path: str, enabled: bool = True):
        """
        Args:
            output_path: Path to write training data (JSONL format)
            enabled: Whether to actually log data (set False to disable)
        """
        self.output_path = Path(output_path)
        self.enabled = enabled
        self.file_handle = None
        self.games_logged = 0
        self.turns_logged = 0

        if self.enabled:
            # Create output directory if needed
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Open file in append mode
            self.file_handle = open(self.output_path, 'a')
            logger.info(f"Training data logger initialized: {self.output_path}")

    def log_turn_determinization(
        self,
        battle: Battle,
        mcts_result: MctsResult,
        sample_chance: float,
        determinization_index: int,
        metadata: Optional[dict] = None
    ):
        """
        Log a single determinization's training data.

        Args:
            battle: The determinized battle state
            mcts_result: The MCTS result for this determinization
            sample_chance: The probability weight of this determinization
            determinization_index: Index of this determinization
            metadata: Optional additional metadata to log
        """
        if not self.enabled or self.file_handle is None:
            return

        try:
            # Get state representation
            state = battle_to_poke_engine_state(battle)
            state_string = state.to_string()

            # Compute MCTS value for this determinization
            # Use the visit-weighted average of move values
            if mcts_result.total_visits > 0:
                mcts_value = sum(
                    (opt.total_score / opt.visits) * (opt.visits / mcts_result.total_visits)
                    for opt in mcts_result.side_one
                    if opt.visits > 0
                )
            else:
                mcts_value = 0.5  # Default if no visits

            # Compute policy entropy for this determinization
            if mcts_result.total_visits > 0:
                import math
                probs = [
                    opt.visits / mcts_result.total_visits
                    for opt in mcts_result.side_one
                    if opt.visits > 0
                ]
                policy_entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
            else:
                policy_entropy = 0.0

            # Build record
            record = {
                'timestamp': datetime.now().isoformat(),
                'battle_id': getattr(battle, 'battle_id', 'unknown'),
                'turn': battle.turn,
                'determinization_index': determinization_index,
                'sample_chance': sample_chance,
                'state': state_string,
                'mcts_value': mcts_value,
                'policy_entropy': policy_entropy,
                'total_iterations': mcts_result.total_visits,
                'battle_type': battle.battle_type.value if hasattr(battle.battle_type, 'value') else str(battle.battle_type),
            }

            if metadata:
                record['metadata'] = metadata

            # Write as JSONL (one JSON object per line)
            self.file_handle.write(json.dumps(record) + '\n')
            self.file_handle.flush()  # Ensure written to disk

            self.turns_logged += 1

            if self.turns_logged % 100 == 0:
                logger.info(f"Logged {self.turns_logged} determinizations across {self.games_logged} games")

        except Exception as e:
            logger.error(f"Error logging determinization data: {e}", exc_info=True)

    def log_game_end(self, battle_id: str, outcome: float, metadata: Optional[dict] = None):
        """
        Log the outcome of a game.

        Args:
            battle_id: Unique identifier for this game
            outcome: 1.0 if won, -1.0 if lost, 0.0 if draw
            metadata: Optional additional metadata
        """
        if not self.enabled or self.file_handle is None:
            return

        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'battle_id': battle_id,
                'type': 'game_end',
                'outcome': outcome,
            }

            if metadata:
                record['metadata'] = metadata

            self.file_handle.write(json.dumps(record) + '\n')
            self.file_handle.flush()

            self.games_logged += 1

        except Exception as e:
            logger.error(f"Error logging game end: {e}", exc_info=True)

    def close(self):
        """Close the logger and flush all data."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            logger.info(
                f"Training data logger closed. Logged {self.turns_logged} turns "
                f"across {self.games_logged} games"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
