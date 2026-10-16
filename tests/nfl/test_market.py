import pandas as pd
import pytest

from markets.nfl.market import compare, summarize
from tests.nfl.test_kalshi import raw_market


def setup(model_p=0.45, yes_bid="0.3000", yes_ask="0.3400", yards=95):
    games = pd.DataFrame(
        {
            "game_id": ["2026_06_BAL_CLE"],
            "gameday": ["2026-10-18"],
            "gametime": ["13:00"],  # 17:00 UTC in October
        }
    )
    snapshots = pd.DataFrame(
        [
            {"snapshot_utc": "2026-10-18T16:30:00+00:00", "series": "KXNFLRECYDS"}
            | raw_market(yes_bid_dollars=yes_bid, yes_ask_dollars=yes_ask),
            # Taken after kickoff: must be ignored.
            {"snapshot_utc": "2026-10-18T18:00:00+00:00", "series": "KXNFLRECYDS"}
            | raw_market(yes_bid_dollars="0.9000", yes_ask_dollars="0.9500"),
        ]
    )
    ladders = pd.DataFrame(
        {
            "made_utc": ["2026-10-18T13:00:00+00:00", "2026-10-18T19:00:00+00:00"],
            "game_id": "2026_06_BAL_CLE",
            "player": "Zay Flowers",
            "player_id": "zf",
            "stat": "receiving_yards",
            "line": 79.5,
            "p_over": [model_p, 0.99],  # the second is after kickoff
        }
    )
    players = pd.DataFrame(
        {
            "game_id": ["2026_06_BAL_CLE"],
            "player_id": ["zf"],
            "passing_yards": 0,
            "passing_tds": 0,
            "receiving_yards": yards,
            "receptions": 6,
            "rushing_yards": 0,
            "rushing_tds": 0,
            "receiving_tds": 1,
        }
    )
    return snapshots, ladders, games, players


def test_uses_only_prices_from_before_kickoff():
    row = compare(*setup()).iloc[0]
    assert row.yes_ask_dollars == pytest.approx(0.34)
    assert row.p_over == pytest.approx(0.45)
    assert row.over == 1.0


def test_paper_trade_buys_yes_when_model_beats_ask_after_fees():
    row = compare(*setup(model_p=0.45)).iloc[0]
    fee = 0.07 * 0.34 * 0.66
    assert row.side == "yes"
    assert row.edge == pytest.approx(0.45 - 0.34 - fee)
    assert row.profit_per_contract == pytest.approx(1 - 0.34 - fee)


def test_no_trade_when_edge_is_small_and_no_side_when_model_is_low():
    assert compare(*setup(model_p=0.37)).iloc[0].side == ""
    row = compare(*setup(model_p=0.10, yards=50)).iloc[0]
    assert row.side == "no"  # no costs 1 - 0.30 = 0.70; model says 0.90
    assert row.profit_per_contract == pytest.approx(1 - 0.70 - 0.07 * 0.70 * 0.30)


def test_summary_scores_model_and_market():
    summary = summarize(compare(*setup())).set_index("stat")
    assert summary.loc["all", "markets"] == 1
    assert summary.loc["all", "brier_model"] == pytest.approx((0.45 - 1) ** 2)
    assert summary.loc["all", "brier_market_mid"] == pytest.approx((0.32 - 1) ** 2)
