import numpy as np
import pandas as pd
import pytest

from markets.nfl.data import team_games, upcoming_rows


def schedule():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": 2026,
            "week": [1, 2],
            "home_team": ["SD", "AAA"],
            "away_team": ["AAA", "CCC"],
            "home_score": [20, np.nan],
            "result": [3, np.nan],
            "spread_line": [3.0, -7.0],
            "total_line": [45.0, 41.0],
            "home_moneyline": [-150, 250],
            "away_moneyline": [130, -300],
            "home_qb_id": ["q_lac", "q_new"],
            "away_qb_id": ["q_aaa", "q_ccc"],
            "home_qb_name": ["LAC QB", "New QB"],
            "away_qb_name": ["AAA QB", "CCC QB"],
        }
    )


def test_team_games_implied_totals_win_probability_and_aliases():
    teams = team_games(schedule()).set_index(["game_id", "team"])
    assert teams.loc[("g1", "LAC"), "implied_total"] == 24.0
    assert teams.loc[("g1", "AAA"), "implied_total"] == 21.0
    # Away team favored by 7 in a 41 total: 24 vs 17.
    assert teams.loc[("g2", "CCC"), "implied_total"] == 24.0
    assert teams.groupby("game_id").p_win.sum().to_numpy() == pytest.approx([1, 1])


def test_upcoming_rows_use_recent_roster_and_listed_qb():
    players = pd.DataFrame(
        {
            "player_id": ["wr1", "moved", "q_new"],
            "player_display_name": ["WR One", "Moved", "New QB"],
            "position": ["WR", "WR", "QB"],
            "season": [2026, 2026, 2025],
            "week": [1, 1, 17],
            "game_id": ["g1", "g1", "old"],
            "team": ["AAA", "AAA", "AAA"],
        }
    )
    # "moved" later appears for another team.
    later = pd.DataFrame(
        {
            "player_id": ["moved"],
            "player_display_name": ["Moved"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "game_id": ["gx"],
            "team": ["ZZZ"],
        }
    )
    rows = upcoming_rows(schedule(), pd.concat([players, later]), 2026, 2)
    aaa = rows[rows.team == "AAA"].set_index("player_id")
    assert "wr1" in aaa.index
    assert "moved" not in aaa.index
    assert aaa.loc["q_new", "is_starting_qb"]
