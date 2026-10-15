import numpy as np
import pandas as pd
import pytest

from nba_model.paper import empty_ledger, new_positions, scorecard, settle
from nba_model.strategy import (
    historical_trades,
    one_per_player,
    order_fee,
    qualifying,
    simulate,
)

TIP = pd.Timestamp("2026-11-03 00:00", tz="UTC")  # 7 pm New York, Nov 2


def market(**overrides):
    row = {
        "ticker": "T1",
        "game_id": 1,
        "player_id": 10,
        "player": "A",
        "team": "AAA",
        "stat": "points",
        "line": 19.5,
        "p_model": 0.60,
        "yes_bid": 0.49,
        "yes_ask": 0.50,
        "tip_utc": TIP,
        "over": 1.0,
    }
    return row | overrides


def test_qualifying_needs_tight_spread_and_edge_after_fees():
    frame = pd.DataFrame(
        [
            market(),  # yes: 0.60 - 0.50 - 0.0175 = 0.0825
            market(ticker="T2", yes_bid=0.45, yes_ask=0.50),  # spread 5c
            market(ticker="T3", p_model=0.55),  # edge 0.0325
            market(ticker="T4", p_model=0.40, yes_bid=0.49, yes_ask=0.50),  # no side
        ]
    )
    q = qualifying(frame)
    assert list(q.ticker) == ["T1", "T4"]
    assert list(q.side) == ["yes", "no"]
    assert q.edge.iloc[0] == pytest.approx(0.0825)
    assert q.cost.iloc[1] == pytest.approx(0.51)


def test_one_per_player_keeps_largest_edge():
    frame = pd.DataFrame(
        [
            market(edge=0.06, ticker="a"),
            market(edge=0.09, ticker="b"),
            market(stat="assists", edge=0.05, ticker="c"),
        ]
    )
    kept = one_per_player(frame)
    assert sorted(kept.ticker) == ["b", "c"]


def test_order_fee_rounds_up_to_the_cent():
    assert order_fee(1, 0.5) == pytest.approx(0.02)
    assert order_fee(10, 0.5) == pytest.approx(0.18)
    assert order_fee(100, 0.5) == pytest.approx(1.75)


def test_simulate_caps_a_day_and_compounds():
    trades = pd.DataFrame(
        [market(ticker=f"T{i}", side="yes", cost=0.50, over=1.0) for i in range(40)]
    )
    days = simulate(trades, bankroll=500.0)
    # 40 positions at 1% would risk 40%; the 30% cap scales each to $3.75 -> 7 contracts.
    assert days.contracts.iloc[0] == 40 * 7
    assert days.profit.iloc[0] == pytest.approx(
        40 * (7 * 0.5 - 0.13)
    )  # fee 12.25c -> 13c
    flat = simulate(trades, bankroll=500.0, roi_shift=-1.0)  # winnings minus 100%
    assert flat.profit.iloc[0] == pytest.approx(-40 * 0.13)  # only the fees


def test_historical_trades_take_the_earliest_qualifying_hour():
    priced = pd.DataFrame([market()])

    def quotes(hours):
        ask = 0.50 if hours == 5 else 0.40
        return pd.DataFrame(
            {"ticker": ["T1"], "yes_bid": [ask - 0.01], "yes_ask": [ask]}
        )

    t = historical_trades(priced, quotes)
    assert len(t) == 1 and t.hours_before.iloc[0] == 5
    assert t.pnl.iloc[0] == pytest.approx(1 - 0.50 - 0.0175)


def live_row(**overrides):
    row = market(hours_to_tip=3.0, injury_status="")
    row["yes_bid_dollars"], row["yes_ask_dollars"] = (
        row.pop("yes_bid"),
        row.pop("yes_ask"),
    )
    return row | overrides


def test_new_positions_respect_window_injuries_and_earlier_runs():
    now = TIP - pd.Timedelta(hours=3)
    priced = pd.DataFrame(
        [
            live_row(),
            live_row(ticker="T2", player_id=11, hours_to_tip=1.0),  # too late
            live_row(ticker="T3", player_id=12, injury_status="Questionable"),
            live_row(ticker="T4", player_id=13),
        ]
    )
    first = new_positions(priced, empty_ledger(), now)
    assert sorted(first.ticker) == ["T1", "T4"]
    assert (first.contracts == 10).all()  # 1% of $500 at 50 cents; fee on top
    assert first.risked.iloc[0] == pytest.approx(5.18)
    again = new_positions(priced, first, now)
    assert again.empty


def test_settle_handles_yes_no_and_players_who_sat():
    ledger = pd.DataFrame(
        [
            {
                "ticker": "Y",
                "side": "yes",
                "contracts": 10,
                "risked": 5.18,
                "status": "open",
                "game_id": 1,
            },
            {
                "ticker": "N",
                "side": "no",
                "contracts": 10,
                "risked": 5.18,
                "status": "open",
                "game_id": 2,
            },
            {
                "ticker": "S",
                "side": "no",
                "contracts": 10,
                "risked": 5.18,
                "status": "open",
                "game_id": 3,
            },
        ]
    ).reindex(columns=list(empty_ledger().columns))
    results = {
        "Y": {"result": "yes"},
        "N": {"result": "yes"},
        "S": {"result": "scalar", "settlement_value_dollars": "0.3000"},
    }
    out = settle(ledger, results.get)
    assert list(out.profit) == pytest.approx([10 - 5.18, -5.18, 7 - 5.18])
    card = scorecard(out)
    assert card["positions"] == 3 and card["players_who_sat"] == 1
    assert card["bankroll"] == pytest.approx(500 + np.sum([4.82, -5.18, 1.82]))
