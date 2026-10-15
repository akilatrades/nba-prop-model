import numpy as np
import pytest

from nba_model.features import build_features, model_rows
from tests.synthetic import league


def row(frame, game, player):
    return frame[(frame.game_id == game) & (frame.player_id == player)].iloc[0]


def test_features_use_only_earlier_games():
    players, games = league()
    before = build_features(players, games)
    changed = players.copy()
    # Change everything about game 30; features for game 30 must not move.
    hit = changed.game_id == 30
    changed.loc[hit, ["points", "minutes", "rebounds"]] *= 3
    after = build_features(changed, games)
    columns = ["ew_minutes", "ew_points", "rate_points", "ew_team_points", "n_prior"]
    a, b = row(before, 30, 100), row(after, 30, 100)
    for column in columns:
        assert a[column] == pytest.approx(b[column])
    # ...but the next game's features do.
    assert row(before, 31, 100).ew_points != pytest.approx(
        row(after, 31, 100).ew_points
    )


def test_regular_sitting_out_vacates_his_share():
    players, games = league(sit={(100, 30)})
    rows = build_features(players, games)
    teammate = row(rows, 30, 101)  # same team, same position group (G)
    other = row(rows, 30, 102)  # same team, forward
    assert teammate.vacated_share_minutes > 0.1
    assert teammate.vacated_same_share_minutes == pytest.approx(
        teammate.vacated_share_minutes
    )
    assert other.vacated_same_share_minutes == 0
    assert other.vacated_share_minutes == pytest.approx(teammate.vacated_share_minutes)
    # The other team is unaffected.
    assert row(rows, 30, 200).vacated_share_minutes == 0
    # Next game he is back: his teammates see a returning regular, he does not.
    back = row(rows, 31, 101)
    assert back.returning_share_minutes > 0.1
    assert row(rows, 31, 100).returning_share_minutes == 0
    assert row(rows, 31, 100).games_missed == 1


def test_betting_line_gives_implied_points():
    players, games = league()
    rows = build_features(players, games)
    home = rows[(rows.game_id == 10) & rows.home].iloc[0]
    away = rows[(rows.game_id == 10) & ~rows.home].iloc[0]
    assert home.implied_points == pytest.approx(111.5)
    assert away.implied_points == pytest.approx(108.5)
    assert home.team_spread == -3.0


def test_model_rows_need_history_and_minutes():
    players, games = league()
    rows = model_rows(build_features(players, games))
    assert rows.n_prior.min() >= 5
    assert rows.ew_minutes.min() >= 10
    assert np.isfinite(rows.rate_points).all()
