import numpy as np
import pandas as pd
import pytest

from markets.nfl.features import HALF_LIFE, build_features, prior_ewm


def games_for(player_id, targets, yards, team="AAA", position="WR", season=2024):
    n = len(targets)
    return pd.DataFrame(
        {
            "player_id": player_id,
            "player_display_name": player_id,
            "position": position,
            "season": season,
            "week": range(1, n + 1),
            "game_id": [f"{season}_{w:02d}_{team}" for w in range(1, n + 1)],
            "team": team,
            "opp": "BBB",
            "targets": targets,
            "receptions": [t * 0.6 for t in targets],
            "receiving_yards": yards,
            "receiving_tds": 0,
            "carries": 0,
            "rushing_yards": 0,
            "rushing_tds": 0,
            "attempts": 0,
            "passing_yards": 0,
            "passing_tds": 0,
            "skill_tds": 0,
            "target_share": [t / 30 for t in targets],
            "carry_share": 0.0,
            "att_share": 0.0,
            "team_targets": 30,
            "team_carries": 25,
            "team_attempts": 33,
            "team_passing_yards": 230,
            "team_rushing_yards": 110,
            "implied_total": 22.0,
            "p_win": 0.5,
        }
    )


def test_prior_ewm_uses_only_earlier_games():
    frame = pd.DataFrame({"player_id": "A", "x": [10.0, 20.0, 30.0]})
    ew = prior_ewm(frame, "x")
    assert np.isnan(ew.iloc[0])
    assert ew.iloc[1] == 10.0
    # Weights 1 (game 2) and 0.5^(1/HALF_LIFE) (game 1).
    w = 0.5 ** (1 / HALF_LIFE)
    assert ew.iloc[2] == pytest.approx((20 + 10 * w) / (1 + w))


def test_features_ignore_the_game_being_predicted():
    base = games_for("A", [5, 6, 7, 8, 9], [50, 60, 70, 80, 90])
    changed = base.copy()
    changed.loc[4, ["targets", "receiving_yards"]] = [30, 300]
    a = build_features(base).iloc[4]
    b = build_features(changed).iloc[4]
    for column in ["ew_targets", "ew_target_share", "ypt", "ew_receiving_yards"]:
        assert a[column] == pytest.approx(b[column])


def test_efficiency_is_shrunk_toward_the_position_average():
    # Two players so the WR average is defined; check player A's 4th game.
    a = games_for("A", [10, 10, 10, 10], [200, 200, 200, 200])
    b = games_for("B", [10, 10, 10, 10], [0, 0, 0, 0])
    frame = build_features(pd.concat([a, b], ignore_index=True))
    row = frame[(frame.player_id == "A") & (frame.week == 4)].iloc[0]
    mu = row.mu_ypt  # first season falls back to its own average: 10 yds/target
    assert mu == pytest.approx(10.0)
    # 30 earlier targets for 600 yards, pulled toward 10 with strength 30.
    assert row.ypt == pytest.approx((600 + 30 * mu) / (30 + 30))


def test_defense_allowed_is_from_earlier_games():
    a = games_for("A", [5, 5, 5], [50, 50, 50])
    frame = build_features(a)
    # Opponent BBB allowed 230 passing yards in each earlier game.
    assert np.isnan(frame.def_pass_allowed.iloc[0])
    assert frame.def_pass_allowed.iloc[2] == pytest.approx(230)


def test_vacated_usage_counts_only_players_newly_ruled_out():
    wr1 = games_for("WR1", [9, 9, 9], [90, 90, 90])
    wr2 = games_for("WR2", [6, 6, 6], [60, 60, 60])
    te = games_for("TE1", [3, 3, 3], [30, 30, 30], position="TE")
    # A receiver who last played in week 1 and stays out.
    gone = games_for("OLD", [6], [60])
    frame = pd.concat([wr1, wr2, te, gone], ignore_index=True)
    # Week 3: WR1 played week 2 (30% of targets) and is now Out.
    injuries = pd.DataFrame(
        {
            "season": 2024,
            "week": [3, 3],
            "team": "AAA",
            "player_id": ["WR1", "OLD"],
            "report_status": ["Out", "Out"],
        }
    )
    frame = frame[~((frame.player_id == "WR1") & (frame.week == 3))]
    built = build_features(frame, injuries)
    week3 = built[built.week == 3].set_index("player_id")
    assert week3.loc["WR2", "vacated_targets"] == pytest.approx(9 / 30)
    assert week3.loc["WR2", "vacated_targets_same"] == pytest.approx(9 / 30)
    assert week3.loc["TE1", "vacated_targets_same"] == 0
    # Without an injury report nothing is vacated.
    assert (build_features(frame).vacated_targets == 0).all()
