"""Odds conversions used to turn sportsbook prices into probabilities."""

import numpy as np


def american_to_prob(odds):
    """Implied probability of American odds, including the book's margin.

    -110 -> 110 / 210 = 0.5238; +150 -> 100 / 250 = 0.40.
    """
    odds = np.asarray(odds, dtype=float)
    prob = np.full(odds.shape, np.nan)
    fav = odds < 0
    dog = odds > 0
    prob[fav] = -odds[fav] / (-odds[fav] + 100)
    prob[dog] = 100 / (odds[dog] + 100)
    return prob


def no_vig(odds_a, odds_b):
    """Remove the margin from a two-way market by normalising.

    Returns the fair probability of side A. Two -110 sides -> 0.5 each.
    """
    p_a = american_to_prob(odds_a)
    p_b = american_to_prob(odds_b)
    return p_a / (p_a + p_b)


def prob_to_american(prob):
    """Fair American odds for a probability (0.25 -> +300, 0.6 -> -150)."""
    prob = np.asarray(prob, dtype=float)
    return np.where(prob >= 0.5, -100 * prob / (1 - prob), 100 * (1 - prob) / prob)


def parlay_payout_multiplier(leg_odds):
    """Decimal payout of a parlay priced as if its legs were independent.

    Two -110 legs pay 1.909 x 1.909 = 3.645 times the stake (about +264).
    """
    leg_odds = np.asarray(leg_odds, dtype=float)
    decimal = np.where(leg_odds < 0, 1 + 100 / -leg_odds, 1 + leg_odds / 100)
    return float(np.prod(decimal))
