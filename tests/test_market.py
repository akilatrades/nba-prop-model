import numpy as np
import pandas as pd
import pytest

from kalshi.prices import price_at_tip
from kalshi.stats import clustered_logit, clustered_mean_t
from markets.nba.market import match_markets, trades


def test_clustered_t_matches_plain_t_with_singleton_clusters():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 500)
    mean, t = clustered_mean_t(x, np.arange(500))
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(500))
    assert mean == pytest.approx(x.mean())
    assert t == pytest.approx(plain, rel=0.01)


def test_clustered_t_shrinks_when_rows_move_together():
    rng = np.random.default_rng(2)
    shock = np.repeat(rng.normal(0, 1, 100), 5)
    x = 0.1 + shock + rng.normal(0, 0.1, 500)
    _, t_cluster = clustered_mean_t(x, np.repeat(np.arange(100), 5))
    _, t_naive = clustered_mean_t(x, np.arange(500))
    assert abs(t_cluster) < abs(t_naive) / 1.8


def test_clustered_logit_recovers_coefficients():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20000, 2))
    p = 1 / (1 + np.exp(-(0.2 + 1.0 * x[:, 0] + 0.5 * x[:, 1])))
    y = rng.random(20000) < p
    beta, se = clustered_logit(y, x, np.arange(20000) // 4)
    assert beta == pytest.approx([0.2, 1.0, 0.5], abs=0.06)
    assert (se > 0).all()


def test_trades_buy_the_side_with_edge_after_fees():
    frame = pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "p_model": [0.70, 0.30, 0.52],
            "yes_bid": [0.58, 0.40, 0.49],
            "yes_ask": [0.60, 0.42, 0.51],
            "over": [1.0, 0.0, 1.0],
        }
    )
    t = trades(frame, 0.05)
    assert list(t.side) == ["yes", "no"]
    # Yes at 0.60 that wins: 1 - 0.60 - 0.07 x 0.6 x 0.4
    assert t.pnl.iloc[0] == pytest.approx(0.40 - 0.0168)
    # No at 1 - 0.40 = 0.60 that wins (over = 0)
    assert t.pnl.iloc[1] == pytest.approx(0.40 - 0.0168)
    assert t.fee_rounded.iloc[0] == pytest.approx(0.02)


def test_price_at_tip_uses_last_candle_before_tip():
    tip = pd.Timestamp("2026-01-10 00:30", tz="UTC")
    matched = pd.DataFrame({"ticker": ["T"], "tip_utc": [tip]})
    hours = [-3, -2, -1, 0, 1]
    candles = pd.DataFrame(
        {
            "ticker": "T",
            "end_ts": [
                int((tip.floor("h") + pd.Timedelta(hours=h)).timestamp()) for h in hours
            ],
            "yes_bid": [0.40, 0.41, 0.42, 0.43, 0.90],
            "yes_ask": [0.42, 0.43, 0.44, 0.45, 0.95],
            "price": 0.5,
            "volume": 1.0,
        }
    )
    got = price_at_tip(matched, candles)
    assert got.yes_bid.iloc[0] == pytest.approx(0.43)  # candle ending 00:00
    early = price_at_tip(matched, candles, hours_before=2)
    assert early.yes_bid.iloc[0] == pytest.approx(0.41)


def test_match_markets_maps_kalshi_codes_and_names():
    tip = pd.Timestamp("2026-05-31 00:30", tz="UTC")  # May 30 in New York
    box = pd.DataFrame(
        {
            "game_id": [7, 7],
            "tip_utc": [tip, tip],
            "team": ["OKC", "SA"],
            "opp": ["SA", "OKC"],
            "home": [True, False],
            "player_id": [1, 2],
            "player": ["Shai Gilgeous-Alexander", "Victor Wembanyama"],
            "season": 2026,
            "season_type": "playoffs",
        }
    )
    markets = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "game_date": [pd.Timestamp("2026-05-30").date()] * 2,
            "away": ["SAS", "SAS"],
            "home": ["OKC", "OKC"],
            "team": ["SAS", "OKC"],
            "player_key": ["victor wembanyama", "s gilgeous alexander"],
        }
    )
    m = match_markets(markets, box)
    assert list(m.game_id) == [7, 7]
    assert m.player_id.iloc[0] == 2
    assert m.player_id.iloc[1] == 1  # matched on initial + surname
