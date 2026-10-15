import pandas as pd
import pytest

from nba_model import kalshi
from nba_model.kalshi import (
    fee_per_contract,
    normalize_name,
    parse,
    settled_frame,
    snapshot,
)


def raw_market(**overrides):
    market = {
        "ticker": "KXNBAPTS-26MAY30SASOKC-SASVWEMBANYAMA1-25",
        "event_ticker": "KXNBAPTS-26MAY30SASOKC",
        "title": "Victor Wembanyama: 25+ points",
        "yes_sub_title": "Victor Wembanyama: 25+",
        "floor_strike": 24.5,
        "strike_type": "greater",
        "status": "active",
        "yes_bid_dollars": "0.6100",
        "yes_ask_dollars": "0.6300",
        "last_price_dollars": "0.6200",
        "volume_fp": "2010.02",
        "open_interest_fp": "15048.21",
        "open_time": "2026-05-29T04:10:00Z",
        "occurrence_datetime": "2026-05-31T03:30:00Z",
        "close_time": "2026-05-31T03:10:12Z",
        "result": "no",
        "settlement_value_dollars": "0.0000",
    }
    return market | overrides


def test_normalize_name_drops_suffixes_and_punctuation():
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("De'Aaron Fox") == "deaaron fox"
    assert normalize_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert normalize_name("Nikola Jokić") == "nikola jokic"


def test_parse_reads_teams_line_date_and_prices():
    def fetch(series):
        if series != "KXNBAPTS":
            return []
        return [
            raw_market(),
            raw_market(floor_strike=None, yes_sub_title="Victor Wembanyama: 20+"),
        ]

    frame = parse(snapshot(fetch=fetch, now=pd.Timestamp("2026-05-30", tz="UTC")))
    assert list(frame.line) == [24.5, 19.5]
    first = frame.iloc[0]
    assert first.stat == "points"
    assert first.player == "Victor Wembanyama"
    assert first.player_key == "victor wembanyama"
    assert str(first.game_date) == "2026-05-30"
    assert (first.away, first.home, first.team) == ("SAS", "OKC", "SAS")
    assert first.mid == pytest.approx(0.62)


def test_settled_frame_keeps_result_and_settlement():
    frame = parse(
        settled_frame([raw_market(), raw_market(result="scalar")], "KXNBAPTS")
    )
    assert list(frame.result) == ["no", "scalar"]
    assert frame.series.eq("KXNBAPTS").all()


def test_candles_flattens_close_quotes(monkeypatch):
    payload = {
        "candlesticks": [
            {
                "end_period_ts": 1780185600,
                "open_interest": "15048.21",
                "price": {"close": "0.6200"},
                "volume": "37294.80",
                "yes_ask": {"close": "0.6200"},
                "yes_bid": {"close": "0.6100"},
            },
            {
                "end_period_ts": 1780200000,
                "open_interest": "0",
                "price": {"close": None},
                "volume": "0",
                "yes_ask": {"close": "1.0000"},
                "yes_bid": {"close": "0.0000"},
            },
        ]
    }
    calls = []

    def fake_get(path, params):
        calls.append((path, params))
        return payload

    monkeypatch.setattr(kalshi, "get_json", fake_get)
    rows = kalshi.candles("KXNBAPTS-26MAY30SASOKC-SASVWEMBANYAMA1-25", 1, 2)
    assert calls[0][0].startswith("/historical/markets/KXNBAPTS-26MAY30SASOKC")
    assert calls[0][1]["period_interval"] == 60
    assert rows[0]["yes_bid"] == pytest.approx(0.61)
    assert rows[0]["yes_ask"] == pytest.approx(0.62)
    assert pd.isna(rows[1]["price"])


def test_fee_is_largest_at_even_odds():
    assert fee_per_contract(0.5) == pytest.approx(0.0175)
    assert fee_per_contract(0.1) == pytest.approx(0.0063)
    assert fee_per_contract(0.9) == pytest.approx(fee_per_contract(0.1))
