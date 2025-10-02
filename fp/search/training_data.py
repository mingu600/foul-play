"""
Training data collection for eval weight optimization.

This module extracts MCTS values from search results and collects
training data for improving the evaluation function weights.
"""

import logging
from typing import List, Tuple
from poke_engine import MctsResult

logger = logging.getLogger(__name__)


def compute_mcts_value_from_results(
    mcts_results: List[Tuple[MctsResult, float, int]]
) -> float:
    """
    Compute a single MCTS value for the state from multiple determinization results.

    This aggregates MCTS values across determinizations weighted by their likelihood.
    We use the expected value of the move distribution as the state value.

    Args:
        mcts_results: List of (MctsResult, sample_chance, index) tuples

    Returns:
        A value in [0, 1] representing MCTS's evaluation of the state
        (0 = guaranteed loss, 1 = guaranteed win, 0.5 = even)
    """
    # Build aggregated policy across all determinizations (same as select_move_from_mcts_results)
    final_policy = {}
    for mcts_result, sample_chance, index in mcts_results:
        for s1_option in mcts_result.side_one:
            if s1_option.visits > 0:
                avg_score = s1_option.total_score / s1_option.visits
                visit_percentage = s1_option.visits / mcts_result.total_visits
                weighted_contribution = sample_chance * visit_percentage

                if s1_option.move_choice not in final_policy:
                    final_policy[s1_option.move_choice] = {
                        'visit_weight': 0.0,
                        'value': 0.0,
                        'value_sum': 0.0,
                    }

                final_policy[s1_option.move_choice]['visit_weight'] += weighted_contribution
                final_policy[s1_option.move_choice]['value_sum'] += weighted_contribution * avg_score

    # Compute value-weighted average across all moves
    total_weight = sum(p['visit_weight'] for p in final_policy.values())

    if total_weight == 0:
        logger.warning("No visits found in MCTS results, returning 0.5")
        return 0.5

    # Compute expected value of position = sum of (move_probability * move_value)
    mcts_value = sum(
        p['value_sum'] for p in final_policy.values()
    ) / total_weight

    return mcts_value


def compute_policy_entropy(mcts_results: List[Tuple[MctsResult, float, int]]) -> float:
    """
    Compute entropy of the aggregated move policy.

    High entropy = MCTS is uncertain about best move
    Low entropy = MCTS has a clear preference

    This can be used to filter training data or weight examples.
    """
    import math

    final_policy = {}
    for mcts_result, sample_chance, index in mcts_results:
        for s1_option in mcts_result.side_one:
            if s1_option.visits > 0:
                visit_percentage = s1_option.visits / mcts_result.total_visits
                weighted_contribution = sample_chance * visit_percentage
                final_policy[s1_option.move_choice] = (
                    final_policy.get(s1_option.move_choice, 0.0) + weighted_contribution
                )

    total_weight = sum(final_policy.values())
    if total_weight == 0:
        return 0.0

    # Normalize to probabilities
    probs = [p / total_weight for p in final_policy.values()]

    # Compute entropy: -sum(p * log(p))
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)

    return entropy
