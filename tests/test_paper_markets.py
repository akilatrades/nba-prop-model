import pandas as pd

from kalshi.paper_markets import (
    check_fills,
    empty_ledger,
    post_orders,
    scorecard,
    settle,
)

NOW = pd.Timestamp("2026-10-16T12:00:00Z")


def market(ticker, bid, ask, hours=6):
    return {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "title": ticker,
        "yes_bid_dollars": bid,
        "yes_ask_dollars": ask,
        "expected_expiration_time": (NOW + pd.Timedelta(hours=hours)).isoformat(),
    }


def test_post_fill_settle():
    markets = [
        market("KXT-26OCT16AB-A", 0.40, 0.45),  # spread 5: improve, no at 0.56
        market("KXT-26OCT16AB-B", 0.60, 0.61),  # spread 1: join at 0.39
        market("KXT-26OCT16CD-C", 0.0, 1.0),  # no quote: skipped
        market("KXT-26OCT17EF-E", 0.5, 0.55, hours=20),  # outside window
    ]
    posted = post_orders("KXT", markets, empty_ledger(), NOW)
    assert list(posted.ticker) == ["KXT-26OCT16AB-A", "KXT-26OCT16AB-B"]
    assert posted.price.tolist() == [0.56, 0.39]
    assert posted.improved.tolist() == [True, False]
    assert posted.contracts.tolist() == [8, 12]  # $5 budget
    # Cancel time: next run in 3 hours (expiration minus 3 hours is at 15:00 too).
    assert posted.cancel_ts.iloc[0] == int((NOW + pd.Timedelta(hours=3)).timestamp())

    def fetch_trades(ticker, min_ts, max_ts):
        when = (NOW + pd.Timedelta(hours=1)).isoformat()
        if ticker.endswith(
            "A"
        ):  # a taker buys yes at our level 0.44 (we improved): fills
            return [
                {
                    "created_time": when,
                    "yes_price": 0.44,
                    "count": 5,
                    "taker_outcome_side": "yes",
                }
            ]
        # Joined the queue at 0.61: a print at 0.61 does not count, only above it.
        return [
            {
                "created_time": when,
                "yes_price": 0.61,
                "count": 50,
                "taker_outcome_side": "yes",
            }
        ]

    later = int((NOW + pd.Timedelta(hours=3)).timestamp())
    ledger = check_fills(posted, fetch_trades, later)
    assert ledger.status.tolist() == ["filled", "unfilled"]
    assert ledger.filled_contracts.iloc[0] == 5
    ledger = settle(ledger, lambda t: {"result": "no"})
    assert ledger.status.iloc[0] == "settled"
    # 5 no contracts at 0.56 win 5 * 0.44 = 2.20; fee 5*0.25*0.07*0.56*0.44 = 2.16 cents, rounded up to 3
    assert ledger.profit.iloc[0] == 2.17
    card = scorecard(ledger, "KXT")
    assert card["bankroll"] == 502.17
    assert card["fills_settled"] == 1


def test_model_veto():
    markets = [
        market(
            "KXT-26OCT16AB-A", 0.40, 0.45
        ),  # no at 0.56; model says yes 0.30: edge 0.14, kept
        market(
            "KXT-26OCT16AB-B", 0.40, 0.45
        ),  # no at 0.56; model says yes 0.50: edge -0.06, vetoed
        market("KXT-26OCT16AB-C", 0.40, 0.45),  # no model price: skipped
    ]
    p_model = {"KXT-26OCT16AB-A": 0.30, "KXT-26OCT16AB-B": 0.50}
    posted = post_orders("KXT+model", markets, empty_ledger(), NOW, p_model=p_model)
    assert posted.ticker.tolist() == ["KXT-26OCT16AB-A", "KXT-26OCT16AB-B"]
    assert posted.status.tolist() == ["resting", "vetoed"]
    assert posted.contracts.tolist() == [8, 0]
    assert posted.reserved.tolist() == [4.48, 0.0]
    later = int((NOW + pd.Timedelta(hours=3)).timestamp())

    def fetch_trades(ticker, min_ts, max_ts):
        when = (NOW + pd.Timedelta(hours=1)).isoformat()
        return [
            {
                "created_time": when,
                "yes_price": 0.44,
                "count": 5,
                "taker_outcome_side": "yes",
            }
        ]

    ledger = check_fills(posted, fetch_trades, later)
    assert ledger.status.tolist() == ["filled", "vetoed_filled"]
    ledger = settle(ledger, lambda t: {"result": "yes"})
    assert ledger.status.tolist() == ["settled", "vetoed_settled"]
    card = scorecard(ledger, "KXT+model")
    assert card["orders"] == 1 and card["vetoed"] == 1
    assert card["bankroll"] == 500 - 5 * 0.56 - 0.03
    # The vetoed order would have lost: no at 0.56, result yes.
    assert card["vetoed_would_have_made_cents"] < -55
