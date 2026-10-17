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
            "tip_utc": [EVENT["commence_time"]] * 3,
        }
    )
    out = attach(priced, consensus(parse_event(EVENT)))
    assert out.p_book.notna().tolist() == [True, False, False]
    assert out.book_n.tolist()[0] == 2
    empty = attach(priced, None)
    assert empty.p_book.isna().all() and (empty.book_n == 0).all()


def _two_games():
    rows = []
    for event, (start, over, under) in {
        "A": ("2026-10-21T23:00:00Z", -120, 100),
        "B": ("2026-10-22T23:00:00Z", -200, 170),
    }.items():
        for book in ("dk", "fd"):
            for side, odds in (("Over", over), ("Under", under)):
                rows.append(
                    {
                        "event_id": event,
                        "commence_utc": start,
                        "book": book,
                        "player_key": "x y",
                        "stat": "points",
                        "line": 24.5,
                        "side": side,
                        "odds": odds,
                    }
                )
    return pd.DataFrame(rows)


def test_consensus_keeps_games_apart_and_only_observed_rows():
    """A player on a back-to-back: one row per game, no cartesian blow-up."""
    c = consensus(_two_games())
    assert len(c) == 2 and sorted(c.event_id) == ["A", "B"]
    assert (c.book_n == 2).all()
    a, b = c.set_index("event_id").p_book.loc[["A", "B"]]
    assert a < b  # B's over is the shorter price


def test_attach_matches_the_line_to_its_game():
    c = consensus(_two_games())
    priced = pd.DataFrame(
        {
            "player_key": ["x y", "x y"],
            "stat": ["points", "points"],
            "line": [24.5, 24.5],
            "tip_utc": ["2026-10-22T23:30:00Z", "2026-10-21T23:30:00Z"],
        }
    )
    out = attach(priced, c)
    by_day = c.set_index("event_id").p_book
    assert np.isclose(out.p_book.iloc[0], by_day["B"])
    assert np.isclose(out.p_book.iloc[1], by_day["A"])
    # Without a tip time the game cannot be identified: attach nothing.
    blind = attach(priced.drop(columns="tip_utc"), c)
    assert blind.p_book.isna().all() and (blind.book_n == 0).all()


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


def test_nfl_priced_frame_from_open_markets():
    from live.nfl_trade import COLUMNS, priced_frame

    now = pd.Timestamp("2026-10-18T14:00:00Z")
    markets = [
        {
            "ticker": "KXNFLRECYDS-26OCT18CHIATL-ATLDLONDON-75",
            "event_ticker": "KXNFLRECYDS-26OCT18CHIATL",
            "title": "Drake London: 75+ receiving yards",
            "yes_sub_title": "Drake London: 75+",
            "floor_strike": 74.5,
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.44",
            "last_price_dollars": "0.42",
            "occurrence_datetime": "2026-10-18T20:00:00Z",
        },
        {
            "ticker": "KXNFLANYTD-26OCT18CHIATL-ATLBROBINSON-1",
            "event_ticker": "KXNFLANYTD-26OCT18CHIATL",
            "title": "Bijan Robinson: 1+ touchdowns",
            "yes_sub_title": "Bijan Robinson: 1+",
            "floor_strike": None,
            "yes_bid_dollars": "0.60",
            "yes_ask_dollars": "0.63",
            "last_price_dollars": "0.61",
            "occurrence_datetime": "2026-10-18T20:00:00Z",
        },
    ]
    out = priced_frame(markets, {markets[0]["ticker"]: 0.38}, now)
    assert list(out.columns) == COLUMNS
    assert out.hours_to_tip.tolist() == [6.0, 6.0]
    assert (
        out.team.tolist() == ["ATL", "ATL"] and out.game_id.iloc[0] == "26OCT18CHIATL"
    )
    assert out.stat.tolist() == ["receiving_yards", "skill_tds"]
    assert out.line.tolist() == [74.5, 0.5]
    assert out.p_model.round(2).tolist()[0] == 0.38 and pd.isna(out.p_model.iloc[1])


def test_attach_latest_drops_lines_stale_at_the_books(tmp_path, monkeypatch):
    from kalshi.books import attach_latest

    now = pd.Timestamp.now(tz="UTC")
    book = pd.DataFrame(
        {
            "player_key": ["fresh guy", "stale guy"],
            "stat": ["points", "points"],
            "line": [20.5, 20.5],
            "p_book": [0.5, 0.5],
            "book_n": [2, 2],
            "snapshot_utc": [now.isoformat()] * 2,  # downloaded just now...
            "updated_utc": [
                (now - pd.Timedelta(minutes=30)).isoformat(),
                (
                    now - pd.Timedelta(hours=9)
                ).isoformat(),  # ...but this book last moved 9 h ago
            ],
            "commence_utc": [(now + pd.Timedelta(hours=3)).isoformat()] * 2,
        }
    )
    path = tmp_path / "nba_latest.csv"
    book.to_csv(path, index=False)
    priced = pd.DataFrame(
        {
            "player_key": ["fresh guy", "stale guy"],
            "stat": ["points", "points"],
            "line": [20.5, 20.5],
            "tip_utc": [(now + pd.Timedelta(hours=3)).isoformat()] * 2,
        }
    )
    out = attach_latest(priced, path=path, max_age_hours=4)
    assert out.p_book.notna().tolist() == [True, False]


def test_consensus_carries_oldest_book_update_time():
    lines = _two_games()
    lines["snapshot_utc"] = ["2026-10-21T10:00:00Z"] * 4 + ["2026-10-21T12:00:00Z"] * 4
    lines.loc[lines.book == "fd", "snapshot_utc"] = "2026-10-21T13:00:00Z"
    c = consensus(lines)
    assert "updated_utc" in c
    assert str(c.set_index("event_id").updated_utc["A"]).startswith("2026-10-21 10:00")
