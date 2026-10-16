import numpy as np
import pandas as pd

from kalshi.books import attach, consensus, implied, parse_event
from research.bookline_backtest import difference, find_event

EVENT = {
    "id": "ev1",
    "commence_time": "2026-01-10T00:30:00Z",
    "home_team": "San Antonio Spurs",
    "away_team": "New York Knicks",
    "bookmakers": [
        {
            "key": "draftkings",
            "markets": [
                {
                    "key": "player_points",
                    "last_update": "2026-01-09T19:00:00Z",
                    "outcomes": [
                        {
                            "name": "Over",
                            "description": "Victor Wembanyama",
                            "price": -120,
                            "point": 24.5,
                        },
                        {
                            "name": "Under",
                            "description": "Victor Wembanyama",
                            "price": +100,
                            "point": 24.5,
                        },
                    ],
                },
                {
                    "key": "player_points_alternate",
                    "outcomes": [
                        {
                            "name": "Over",
                            "description": "Victor Wembanyama",
                            "price": +150,
                            "point": 29.5,
                        },
                        {
                            "name": "Over",
                            "description": "Victor Wembanyama",
                            "price": -120,
                            "point": 24.5,
                        },
                    ],
                },
            ],
        },
        {
            "key": "fanduel",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "Over",
                            "description": "Victor Wembanyama",
                            "price": -110,
                            "point": 24.5,
                        },
                        {
                            "name": "Under",
                            "description": "Victor Wembanyama",
                            "price": -110,
                            "point": 24.5,
                        },
                        {
                            "name": "Over",
                            "description": "Keldon Johnson Jr.",
                            "price": -105,
                            "point": 9.5,
                        },
                        {
                            "name": "Under",
                            "description": "Keldon Johnson Jr.",
                            "price": -115,
                            "point": 9.5,
                        },
                    ],
                }
            ],
        },
    ],
}


def test_implied():
    assert np.isclose(implied(-110), 110 / 210)
    assert np.isclose(implied(+150), 100 / 250)


def test_parse_maps_teams_and_stats():
    lines = parse_event(EVENT)
    assert set(lines.home) == {"SAS"} and set(lines.away) == {"NYK"}
    assert set(lines.stat) == {"points"}
    assert lines.player_key.str.contains("keldon johnson").any()
    assert "jr" not in " ".join(lines.player_key)


def test_consensus_devigs_and_averages_only_two_sided_lines():
    c = consensus(parse_event(EVENT)).set_index(["player_key", "stat", "line"])
    dk = (120 / 220) / (120 / 220 + 0.5)
    fd = 0.5
    row = c.loc[("victor wembanyama", "points", 24.5)]
    assert row.book_n == 2
    assert np.isclose(row.p_book, (dk + fd) / 2)
    # The 29.5 alternate is quoted over only: no probability for it.
    assert ("victor wembanyama", "points", 29.5) not in c.index
    kj = c.loc[("keldon johnson", "points", 9.5)]
    assert kj.book_n == 1 and 0.47 < kj.p_book < 0.49


def test_attach_matches_exact_lines():
    priced = pd.DataFrame(
        {
            "player": ["Victor Wembanyama", "Victor Wembanyama", "Someone Else"],
            "stat": ["points", "points", "assists"],
            "line": [24.5, 34.5, 4.5],
        }
    )
    out = attach(priced, consensus(parse_event(EVENT)))
    assert out.p_book.notna().tolist() == [True, False, False]
    assert out.book_n.tolist()[0] == 2
    empty = attach(priced, None)
    assert empty.p_book.isna().all() and (empty.book_n == 0).all()


def test_find_event_by_teams_and_time():
    listed = [
        {
            "id": "a",
            "home_team": "San Antonio Spurs",
            "away_team": "New York Knicks",
            "commence_time": "2026-01-10T00:30:00Z",
        },
        {
            "id": "b",
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "commence_time": "2026-01-10T00:30:00Z",
        },
    ]
    tip = pd.Timestamp("2026-01-10T01:00:00Z")
    assert find_event(listed, "SAS", "NYK", tip)["id"] == "a"
    assert find_event(listed, "SAS", "NYK", tip + pd.Timedelta(hours=9)) is None


def test_difference_clustered():
    a = pd.DataFrame({"pnl_if_filled": [0.1, 0.2, 0.3, 0.2], "game_id": [1, 1, 2, 3]})
    b = pd.DataFrame({"pnl_if_filled": [0.0, -0.1, 0.0, 0.1], "game_id": [1, 2, 4, 4]})
    d = difference(a, b)
    assert np.isclose(d["cents"], 20.0)
    assert d["t"] > 0
    assert difference(a, b.iloc[0:0]) is None
