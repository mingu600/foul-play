"""
Feature extraction from poke-engine state strings.

This module parses state strings and extracts features that correspond
to the evaluation function weights in poke-engine.
"""

import numpy as np
from poke_engine import State as PokeEngineState
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


# Feature names matching eval weights in poke-engine
FEATURE_NAMES = [
    # Per-pokemon features (summed across team)
    'pokemon_alive',
    'pokemon_hp',

    # Boosts (only for active pokemon)
    'attack_boost_pos',
    'attack_boost_neg',
    'defense_boost_pos',
    'defense_boost_neg',
    'special_attack_boost_pos',
    'special_attack_boost_neg',
    'special_defense_boost_pos',
    'special_defense_boost_neg',
    'speed_boost_pos',
    'speed_boost_neg',

    # Status conditions
    'burned',
    'frozen',
    'asleep',
    'paralyzed',
    'toxic',
    'poisoned',

    # Volatile statuses
    'leech_seed',
    'substitute',
    'confusion',

    # Side conditions
    'reflect',
    'light_screen',
    'aurora_veil',
    'safeguard',
    'tailwind',
    'healing_wish',

    # Hazards
    'stealth_rock',
    'spikes',
    'toxic_spikes',
    'sticky_web',

    # Tera
    'used_tera',

    # Items
    'has_item',
]


def extract_features_from_state(state: PokeEngineState) -> np.ndarray:
    """
    Extract feature vector from a poke-engine state.

    Features are extracted for side_one, representing the bot's side.
    The difference between side_one features and side_two features
    should match the eval score.

    Returns:
        numpy array of shape (num_features,) with feature values
    """
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    feature_idx = 0

    # Side one (bot's side) - positive contribution
    s1_features = _extract_side_features(state.side_one)
    s1_hazards = _extract_hazard_features(state.side_one)

    # Side two (opponent's side) - negative contribution
    s2_features = _extract_side_features(state.side_two)
    s2_hazards = _extract_hazard_features(state.side_two)

    # Take difference (side_one - side_two)
    features = s1_features - s2_features

    # Hazards affect the opposite side
    features[FEATURE_NAMES.index('stealth_rock')] = -(s2_hazards[0] - s1_hazards[0])
    features[FEATURE_NAMES.index('spikes')] = -(s2_hazards[1] - s1_hazards[1])
    features[FEATURE_NAMES.index('toxic_spikes')] = -(s2_hazards[2] - s1_hazards[2])
    features[FEATURE_NAMES.index('sticky_web')] = -(s2_hazards[3] - s1_hazards[3])

    return features


def _extract_side_features(side) -> np.ndarray:
    """Extract features for one side."""
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Pokemon features
    alive_count = 0
    total_hp = 0.0
    has_items = 0
    used_tera = 0

    for pkmn in side.pokemon:
        if pkmn.hp > 0:
            alive_count += 1
            total_hp += pkmn.hp / pkmn.maxhp

            if pkmn.item != 0:  # ITEMS::NONE = 0
                has_items += 1

            if pkmn.terastallized:
                used_tera = 1

    features[FEATURE_NAMES.index('pokemon_alive')] = alive_count
    features[FEATURE_NAMES.index('pokemon_hp')] = total_hp
    features[FEATURE_NAMES.index('has_item')] = has_items
    features[FEATURE_NAMES.index('used_tera')] = used_tera

    # Active pokemon boosts (separate positive and negative for better learning)
    active_pkmn = side.pokemon[side.active_index]

    if side.attack_boost > 0:
        features[FEATURE_NAMES.index('attack_boost_pos')] = side.attack_boost
    else:
        features[FEATURE_NAMES.index('attack_boost_neg')] = abs(side.attack_boost)

    if side.defense_boost > 0:
        features[FEATURE_NAMES.index('defense_boost_pos')] = side.defense_boost
    else:
        features[FEATURE_NAMES.index('defense_boost_neg')] = abs(side.defense_boost)

    if side.special_attack_boost > 0:
        features[FEATURE_NAMES.index('special_attack_boost_pos')] = side.special_attack_boost
    else:
        features[FEATURE_NAMES.index('special_attack_boost_neg')] = abs(side.special_attack_boost)

    if side.special_defense_boost > 0:
        features[FEATURE_NAMES.index('special_defense_boost_pos')] = side.special_defense_boost
    else:
        features[FEATURE_NAMES.index('special_defense_boost_neg')] = abs(side.special_defense_boost)

    if side.speed_boost > 0:
        features[FEATURE_NAMES.index('speed_boost_pos')] = side.speed_boost
    else:
        features[FEATURE_NAMES.index('speed_boost_neg')] = abs(side.speed_boost)

    # Status conditions (only for active)
    # Mapping status enum values - these might need adjustment based on actual enum
    status_map = {
        1: 'burned',
        2: 'frozen',
        3: 'paralyzed',
        4: 'poisoned',
        5: 'toxic',
        6: 'asleep',
    }

    if active_pkmn.status in status_map:
        features[FEATURE_NAMES.index(status_map[active_pkmn.status])] = 1.0

    # Volatile statuses (active pokemon only)
    for vs in side.volatile_statuses:
        vs_str = str(vs).lower()
        if 'leechseed' in vs_str:
            features[FEATURE_NAMES.index('leech_seed')] = 1.0
        elif 'substitute' in vs_str:
            features[FEATURE_NAMES.index('substitute')] = 1.0
        elif 'confusion' in vs_str:
            features[FEATURE_NAMES.index('confusion')] = 1.0

    # Side conditions (with durations)
    features[FEATURE_NAMES.index('reflect')] = side.side_conditions.reflect
    features[FEATURE_NAMES.index('light_screen')] = side.side_conditions.light_screen
    features[FEATURE_NAMES.index('aurora_veil')] = side.side_conditions.aurora_veil
    features[FEATURE_NAMES.index('safeguard')] = side.side_conditions.safeguard
    features[FEATURE_NAMES.index('tailwind')] = side.side_conditions.tailwind
    features[FEATURE_NAMES.index('healing_wish')] = side.side_conditions.healing_wish

    return features


def _extract_hazard_features(side) -> np.ndarray:
    """Extract hazard features (stored separately as they affect opponent)."""
    hazards = np.zeros(4, dtype=np.float32)
    hazards[0] = side.side_conditions.stealth_rock
    hazards[1] = side.side_conditions.spikes
    hazards[2] = side.side_conditions.toxic_spikes
    hazards[3] = side.side_conditions.sticky_web
    return hazards


def extract_features_from_state_string(state_string: str) -> np.ndarray:
    """
    Convenience function to extract features from state string.

    Args:
        state_string: String representation of state from to_string()

    Returns:
        Feature vector as numpy array
    """
    state = PokeEngineState.from_string(state_string)
    return extract_features_from_state(state)


def features_to_dict(features: np.ndarray) -> Dict[str, float]:
    """Convert feature vector to named dictionary for debugging."""
    return {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)}
