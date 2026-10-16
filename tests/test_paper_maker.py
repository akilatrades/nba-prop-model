import pandas as pd
import pytest

from kalshi.paper_props import (
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


def test_v2_rules_differ_from_v1():
    import pandas as pd

    from kalshi.paper_props import V2_RULES, candidate_orders, stake_fraction

    rows = []
    for i, (hours, p, stat) in enumerate(
        [
            (4.0, 0.30, "points"),
            (1.0, 0.30, "points"),
            (4.0, 0.44, "rebounds"),
            (4.0, 0.44, "points"),
        ]
    ):
        rows.append(
            {
                "ticker": f"T{i}",
                "game_id": 1,
                "player_id": i,
                "player": f"P{i}",
                "team": "AAA",
                "stat": stat,
                "line": 9.5,
                "yes_bid_dollars": 0.40,
                "yes_ask_dollars": 0.45,
                "p_model": p,
                "tip_utc": "2026-10-21T00:00:00Z",
                "injury_status": "",
                "hours_to_tip": hours,
            }
        )
    priced = pd.DataFrame(rows)
    # no at 0.56: edge = 0.44 - 0.56 ... use p=0.30 -> 0.70-0.56-fee ≈ 0.136; p=0.44 -> 0.56-0.56-fee < 0
    v1 = candidate_orders(priced, "no_seller")
    assert set(v1.ticker) == {"T0"}  # T1 outside the 2-6 h window; T2, T3 vetoed
    late = candidate_orders(priced, "no_seller_late")
    assert set(late.ticker) == {"T1"}
    # a marginal order (edge just above zero) passes v1 but not veto2
    priced.loc[3, "p_model"] = 0.43  # edge about +0.0057
    assert "T3" in set(candidate_orders(priced, "no_seller").ticker)
    assert "T3" not in set(candidate_orders(priced, "no_seller_veto2").ticker)
    # the stat veto only bites rebounds
    priced.loc[2, "p_model"] = 0.43
    sv = candidate_orders(priced, "no_seller_stat_veto")
    assert "T3" in set(sv.ticker) and "T2" not in set(sv.ticker)
    assert (
        stake_fraction(0.0, "edge") == 0.005 and stake_fraction(0.05, "edge") == 0.015
    )
    assert (
        stake_fraction(0.10, "edge") == 0.015 and stake_fraction(0.10, "flat") == 0.01
    )
    assert set(V2_RULES) == {
        "no_seller_late",
        "no_seller_edge_sized",
        "no_seller_veto2",
        "no_seller_stat_veto",
    }
