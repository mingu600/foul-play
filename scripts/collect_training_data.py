#!/usr/bin/env python
"""
Collect training data by running games with MCTS.

This script shows how to enable training data collection during games.
Integrate this with your existing battle loop.

Usage:
    python scripts/collect_training_data.py \
        --num-games 1000 \
        --output data/training_data.jsonl
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fp.search.training_logger import TrainingDataLogger
from fp.search.main import set_training_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data(num_games: int, output_path: Path):
    """
    Collect training data from games.

    This is a template - you'll need to integrate with your actual battle loop.
    """

    logger.info(f"Collecting training data from {num_games} games")
    logger.info(f"Output: {output_path}")

    # Create training data logger
    with TrainingDataLogger(str(output_path)) as training_logger:

        # Enable training data collection in the MCTS module
        set_training_logger(training_logger)

        logger.info("Training logger enabled - data will be collected during games")

        # =========================================================================
        # YOUR BATTLE LOOP GOES HERE
        # =========================================================================
        #
        # Example integration:
        #
        # for game_idx in range(num_games):
        #     battle = create_new_battle()
        #     battle.battle_id = f"training_game_{game_idx}"
        #
        #     while not battle.is_over():
        #         # This will automatically log training data
        #         move = find_best_move(battle)
        #         battle.apply_move(move)
        #
        #     # Log game outcome
        #     outcome = 1.0 if battle.user_won() else -1.0
        #     training_logger.log_game_end(battle.battle_id, outcome)
        #
        #     if (game_idx + 1) % 10 == 0:
        #         logger.info(f"Completed {game_idx + 1}/{num_games} games")
        #
        # =========================================================================

        logger.warning(
            "Data collection template created. "
            "You need to integrate this with your actual battle loop!"
        )

        logger.info("Example integration code:")
        logger.info("""
        from fp.search.training_logger import TrainingDataLogger
        from fp.search.main import set_training_logger, find_best_move

        # At start of your program
        with TrainingDataLogger("training_data.jsonl") as logger:
            set_training_logger(logger)

            # Run your battles normally
            for game in range(num_games):
                battle = create_battle()
                battle.battle_id = f"game_{game}"

                while not battle.is_over():
                    move = find_best_move(battle)  # Automatically logs data
                    battle.apply_move(move)

                # Log final outcome
                outcome = 1.0 if battle.won() else -1.0
                logger.log_game_end(battle.battle_id, outcome)

        # Data is now saved to training_data.jsonl
        """)

        # Disable training logger when done
        set_training_logger(None)

    logger.info(f"Data collection complete")
    logger.info(f"Training data saved to: {output_path}")
    logger.info(f"Next step: Train weights with:")
    logger.info(f"  python scripts/train_eval_weights.py --input {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect training data from games')
    parser.add_argument('--num-games', type=int, default=1000,
                        help='Number of games to collect')
    parser.add_argument('--output', type=str, default='data/training_data.jsonl',
                        help='Output path for training data')

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collect_data(args.num_games, output_path)


if __name__ == '__main__':
    main()
