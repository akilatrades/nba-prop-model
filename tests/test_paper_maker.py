import pandas as pd
import pytest

from nba_model.paper_maker import (
    candidate_orders,
    check_fills,
    empty_ledger,
    post_orders,
    scorecard,
    settle,
)

TIP = pd.Timestamp("2026-11-03 00:00", tz="UTC")
NOW = TIP - pd.Timedelta(hours=3)


def market(**kw):
    row = {
        "ticker": "T1",
        "game_id": 1,
        "player_id": 10,
        "player": "A",
        "team": "AAA",
        "stat": "points",
        "line": 19.5,
        "p_model": 0.30,
        "yes_bid_dollars": 0.40,
        "yes_ask_dollars": 0.44,
        "tip_utc": TIP,
        "hours_to_tip": 3.0,
        "injury_status": "",
    }
    return row | kw


def test_no_seller_posts_no_inside_the_ask_and_respects_the_veto():
    priced = pd.DataFrame(
        [
            market(),
            market(ticker="T2", line=24.5, p_model=0.60),
            market(ticker="T3", hours_to_tip=1.0),
        ]
    )
    o = candidate_orders(priced, "no_seller")
    assert list(o.ticker) == ["T1"]  # T2 vetoed (model likes yes), T3 too close
    assert o.side.iloc[0] == "no"
    assert o.price.iloc[0] == pytest.approx(0.57)  # no bid 0.56 + 1 cent
    assert o.yes_level.iloc[0] == pytest.approx(0.43)


def test_model_side_needs_three_cents():
    priced = pd.DataFrame(
        [market(p_model=0.43), market(ticker="T2", player_id=11, p_model=0.52)]
    )
    o = candidate_orders(priced, "model_side")
    assert list(o.ticker) == ["T2"]  # yes at 0.41: T2 edge 0.106; T1 edge 0.016


def test_post_size_fill_and_settle():
    priced = pd.DataFrame([market()])
    ledger = post_orders(priced, empty_ledger(), NOW, "no_seller")
    assert ledger.contracts.iloc[0] == 8  # $5 at 57 cents
    assert post_orders(priced, ledger, NOW, "no_seller").empty  # not posted twice

    def trades(ticker, lo, hi):
        return [
            {
                "ticker": "T1",
                "created_time": (NOW + pd.Timedelta(hours=1)).isoformat(),
                "yes_price": 0.43,
                "count": 5.0,
                "taker_outcome_side": "yes",
            },
            {
                "ticker": "T1",
                "created_time": (NOW + pd.Timedelta(hours=1)).isoformat(),
                "yes_price": 0.40,
                "count": 9.0,
                "taker_outcome_side": "no",
            },
        ]

    filled = check_fills(ledger, trades, int(TIP.timestamp()) + 60)
    assert filled.status.iloc[0] == "filled" and filled.filled_contracts.iloc[0] == 5
    done = settle(filled, lambda t: {"result": "no"})
    # 5 no contracts at 0.57 win 0.43 each; maker fee 2.14 cents rounds up to 3
    assert done.profit.iloc[0] == pytest.approx(5 * 0.43 - 0.03)
    card = scorecard(done, "no_seller")
    assert card["fills_settled"] == 1 and card["bankroll"] == pytest.approx(500 + 2.12)


def test_unfilled_orders_close_at_zero():
    ledger = post_orders(pd.DataFrame([market()]), empty_ledger(), NOW, "no_seller")
    out = check_fills(ledger, lambda *a: [], int(TIP.timestamp()) + 60)
    assert out.status.iloc[0] == "unfilled" and out.profit.iloc[0] == 0
