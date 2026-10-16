import pandas as pd
import pytest

from markets.nba.ingame import game_states, minutes_until, on_court_intervals


def plays():
    # Player 1 starts (first seen leaving), player 2 enters at 6:00 of Q1.
    rows = [
        (
            1,
            "Jump Shot",
            True,
            2,
            1,
            None,
            None,
            1,
            11,
            0,
            2,
            0,
            "2026-01-01T00:01:00Z",
        ),
        (
            2,
            "Substitution",
            False,
            0,
            2,
            1,
            None,
            1,
            6,
            0,
            2,
            0,
            "2026-01-01T00:10:00Z",
        ),
        (3, "Jump Shot", True, 3, 2, 3, None, 1, 5, 0, 5, 0, "2026-01-01T00:12:00Z"),
        (
            4,
            "Defensive Rebound",
            False,
            0,
            1,
            None,
            None,
            2,
            10,
            0,
            5,
            0,
            "2026-01-01T00:30:00Z",
        ),
        (
            5,
            "Personal Foul",
            False,
            0,
            2,
            None,
            None,
            2,
            9,
            0,
            5,
            0,
            "2026-01-01T00:31:00Z",
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "sequence_number",
            "type_text",
            "scoring_play",
            "score_value",
            "athlete_id_1",
            "athlete_id_2",
            "athlete_id_3",
            "period_number",
            "clock_minutes",
            "clock_seconds",
            "home_score",
            "away_score",
            "wallclock",
        ],
    )
    period = frame.period_number
    left = frame.clock_minutes * 60 + frame.clock_seconds
    frame["elapsed"] = (period - 1) * 720.0 + (720.0 - left)
    frame["ts"] = pd.to_datetime(frame.wallclock).map(lambda t: int(t.timestamp()))
    return frame


def test_lineups_from_substitutions():
    iv = on_court_intervals(plays(), [1, 2, 3])
    # Q1 start until subbed out at 6:00; back on for Q2 (seen before any sub)
    assert iv[1] == [(0.0, 360.0), (720.0, 1440.0)]
    assert iv[2][0] == (360.0, 720.0)  # in at 6:00 to end of Q1
    assert minutes_until(iv[1], 600.0) == (6.0, False)
    assert minutes_until(iv[1], 1000.0) == pytest.approx((6.0 + 280 / 60, True))


def test_game_state_counts_stats_and_fouls_by_wallclock():
    st = game_states(
        plays(),
        [1, 2, 3],
        [int(pd.Timestamp("2026-01-01T00:20:00Z").timestamp())],
        by="ts",
    )
    s = st.set_index("player_id")
    assert s.at[1, "points_so_far"] == 2
    assert s.at[2, "points_so_far"] == 3 and s.at[2, "threes_so_far"] == 1
    assert s.at[3, "assists_so_far"] == 1
    assert s.at[2, "fouls_so_far"] == 0  # the foul comes later
    assert s.at[1, "home_margin"] == pytest.approx(5.0)
