"""Regression check on real data: the 2025 season, refit every four weeks."""

import pandas as pd
import pytest

from markets.nfl.data import load, load_injuries, player_games
from markets.nfl.evaluate import summarize, walk_forward
from markets.nfl.features import build_features, model_rows


@pytest.fixture(scope="module")
def summary_2025():
    games, players = load()
    rows = model_rows(build_features(player_games(games, players), load_injuries()))
    return summarize(walk_forward(rows, [2025])).set_index("stat")


def test_every_team_and_starting_qb_is_matched(summary_2025):
    # 32 teams x 17 games, minus each QB's first three games of history.
    assert summary_2025.loc["passing_yards", "player_games"] > 450


def test_model_beats_recent_average_baseline(summary_2025):
    for stat in ["passing_yards", "passing_tds", "skill_tds"]:
        assert summary_2025.loc[stat, "brier"] < summary_2025.loc[stat, "brier_naive"]


def test_saved_summary_matches_its_columns():
    saved = pd.read_csv("outputs/nfl/backtest/summary.csv")
    assert set(saved.stat) == {
        "receiving_yards",
        "receptions",
        "rushing_yards",
        "passing_yards",
        "passing_tds",
        "skill_tds",
    }
    assert (saved.brier_improvement_pct > 0).all()
