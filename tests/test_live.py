import pandas as pd
import pytest

from nba_model.features import build_features, model_rows
from nba_model.live import (
    home_spread_from,
    parse_injuries,
    parse_scoreboard,
    upcoming_rows,
)
from tests.synthetic import league


def test_home_spread_from_details():
    assert home_spread_from("DET -1.5", "DET", "BOS") == -1.5
    assert home_spread_from("BOS -4", "DET", "BOS") == 4.0
    assert home_spread_from("EVEN", "DET", "BOS") == 0.0
    assert pd.isna(home_spread_from(None, "DET", "BOS"))


def test_parse_scoreboard_skips_preseason():
    payload = {
        "events": [
            {"id": "1", "season": {"type": 1}, "competitions": []},
            {
                "id": "401909088",
                "season": {"type": 2},
                "date": "2026-10-20T23:00Z",
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "DET"}},
                            {"homeAway": "away", "team": {"abbreviation": "BOS"}},
                        ],
                        "status": {"type": {"completed": False}},
                        "odds": [{"details": "DET -1.5", "overUnder": 221.5}],
                    }
                ],
            },
        ]
    }
    games = parse_scoreboard(payload, 2027)
    assert len(games) == 1
    game = games.iloc[0]
    assert game.game_id == 401909088
    assert (game.home, game.away) == ("DET", "BOS")
    assert game.home_spread == -1.5 and game.total == 221.5
    assert game.tip_utc == pd.Timestamp("2026-10-20 23:00", tz="UTC")


def test_parse_injuries_reads_id_and_status():
    payload = {
        "injuries": [
            {
                "injuries": [
                    {
                        "status": "Out",
                        "athlete": {
                            "displayName": "Player 100",
                            "headshot": {
                                "href": "https://a.espncdn.com/i/headshots/nba/players/full/100.png"
                            },
                            "team": {"abbreviation": "AAA"},
                        },
                    },
                    {
                        "status": "Day-To-Day",
                        "athlete": {
                            "displayName": "Player 101",
                            "links": [
                                {"href": "https://www.espn.com/nba/player/_/id/101/x"}
                            ],
                            "team": {"abbreviation": "AAA"},
                        },
                    },
                ]
            }
        ]
    }
    report = parse_injuries(payload)
    assert list(report.player_id) == [100, 101]
    assert list(report.out) == [True, False]


def test_upcoming_game_uses_injury_report_for_teammates_out():
    players, games = league()
    tip = players.tip_utc.max() + pd.Timedelta(days=2)
    schedule = pd.DataFrame(
        {
            "game_id": [999],
            "season": [2022],
            "season_type": ["regular"],
            "tip_utc": [tip],
            "home": ["AAA"],
            "away": ["BBB"],
            "home_spread": [-2.0],
            "total": [224.0],
            "completed": [False],
        }
    )
    report = pd.DataFrame(
        {"team": ["AAA"], "player_id": [100], "status": ["Out"], "out": [True]}
    )
    listed = pd.DataFrame({"game_id": [999], "team": ["AAA"], "player_id": [101]})
    upcoming = upcoming_rows(players, schedule, report, listed)
    assert set(upcoming.player_id) == set(range(100, 108)) | set(range(200, 208))
    assert upcoming.set_index("player_id").at[100, "did_not_play"]
    rows = build_features(
        pd.concat([players, upcoming], ignore_index=True),
        pd.concat([games, schedule], ignore_index=True),
    )
    priced = model_rows(rows)
    priced = priced[priced.game_id == 999].set_index("player_id")
    assert 100 not in priced.index
    assert priced.at[101, "vacated_share_minutes"] > 0.1
    assert priced.at[200, "vacated_share_minutes"] == 0
    assert priced.at[101, "implied_points"] == pytest.approx(113.0)
