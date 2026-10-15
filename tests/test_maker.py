import pandas as pd
import pytest

from nba_model.maker import fills, maker_fee, orders, settle_orders


def quote(**kw):
    row = {
        "ticker": "T",
        "game_id": 1,
        "player_id": 1,
        "stat": "points",
        "p_model": 0.60,
        "yes_bid": 0.50,
        "yes_ask": 0.53,
        "end_ts": 100,
        "tip_ts": 200,
        "over": 1.0,
    }
    return row | kw


def test_orders_improve_the_bid_when_spread_allows():
    o = orders(pd.DataFrame([quote(), quote(ticker="U", p_model=0.40)]))
    yes = o[o.ticker == "T"].iloc[0]
    assert (yes.side, yes.price, yes.improved) == ("yes", 0.51, True)
    no = o[o.ticker == "U"].iloc[0]
    # no bid = 1 - 0.53 = 0.47, improved to 0.48; as a yes level, 0.52
    assert (no.side, no.price, no.yes_level) == ("no", 0.48, 0.52)


def test_orders_join_the_bid_on_one_cent_spreads_and_need_edge():
    o = orders(pd.DataFrame([quote(yes_ask=0.51), quote(ticker="U", p_model=0.52)]))
    assert list(o.ticker) == ["T"]
    assert o.price.iloc[0] == 0.50 and not o.improved.iloc[0]


def test_fills_follow_price_priority():
    placed = orders(pd.DataFrame([quote(), quote(ticker="J", yes_ask=0.51)]))
    trades = pd.DataFrame(
        [
            # T rests a yes bid at 0.51 (improved): a taker selling yes at 0.51 hits it
            {
                "ticker": "T",
                "ts": 150,
                "yes_price": 0.51,
                "count": 4.0,
                "taker_outcome_side": "no",
            },
            # a taker buying yes does not hit a yes bid
            {
                "ticker": "T",
                "ts": 150,
                "yes_price": 0.53,
                "count": 9.0,
                "taker_outcome_side": "yes",
            },
            # after tip-off: ignored
            {
                "ticker": "T",
                "ts": 250,
                "yes_price": 0.40,
                "count": 9.0,
                "taker_outcome_side": "no",
            },
            # J joined the bid at 0.50: a print at 0.50 is not enough, 0.49 is
            {
                "ticker": "J",
                "ts": 150,
                "yes_price": 0.50,
                "count": 5.0,
                "taker_outcome_side": "no",
            },
            {
                "ticker": "J",
                "ts": 160,
                "yes_price": 0.49,
                "count": 2.0,
                "taker_outcome_side": "no",
            },
        ]
    )
    got = fills(placed, trades)
    assert got["T"] == 4.0 and got["J"] == 2.0


def test_settle_orders_pays_side_minus_price_and_maker_fee():
    placed = orders(pd.DataFrame([quote()]))
    out = settle_orders(placed, pd.Series({"T": 3.0}))
    assert out.filled.iloc[0]
    assert out.pnl_if_filled.iloc[0] == pytest.approx(1 - 0.51 - maker_fee(0.51))
    assert maker_fee(0.5) == pytest.approx(0.004375)
