import pandas as pd
import pytest

from markets.nfl.kalshi import fee_per_contract, normalize_name, parse, snapshot


def raw_market(**overrides):
    market = {
        "ticker": "KXNFLRECYDS-26OCT18BALCLE-BALZFLOWERS1-80",
        "event_ticker": "KXNFLRECYDS-26OCT18BALCLE",
        "title": "Zay Flowers: 80+ receiving yards",
        "yes_sub_title": "Zay Flowers: 80+",
        "floor_strike": 79.5,
        "strike_type": "greater",
        "status": "active",
        "yes_bid_dollars": "0.3000",
        "yes_ask_dollars": "0.3400",
        "last_price_dollars": "0.3200",
        "volume_fp": "10.00",
        "open_interest_fp": "4.00",
        "occurrence_datetime": "2026-10-18T20:00:00Z",
        "close_time": "2026-10-25T20:00:00Z",
    }
    return market | overrides


def test_normalize_name_drops_suffixes_and_punctuation():
    assert normalize_name("Deebo Samuel Sr.") == "deebo samuel"
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert normalize_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_parse_reads_line_date_and_prices():
    def fetch(series):
        if series != "KXNFLRECYDS":
            return []
        return [
            raw_market(),
            raw_market(floor_strike=None, yes_sub_title="Zay Flowers: 60+"),
        ]

    frame = parse(snapshot(series=["KXNFLRECYDS", "KXNFLREC"], fetch=fetch))
    assert list(frame.line) == [79.5, 59.5]  # strike, or "60+" -> 59.5
    assert str(frame.game_date.iloc[0]) == "2026-10-18"
    assert frame.stat.iloc[0] == "receiving_yards"
    assert frame["mid"].iloc[0] == pytest.approx(0.32)
    assert frame.player_key.iloc[0] == "zay flowers"


def test_fee_matches_kalshi_formula():
    # 0.07 x P x (1 - P) per contract before rounding: 25 cents -> 1.3125 cents
    assert fee_per_contract(0.25) == pytest.approx(0.013125)
    assert fee_per_contract(0.5) == pytest.approx(0.0175)


def test_snapshot_has_timestamp_and_series():
    frame = snapshot(
        series=["KXNFLRECYDS"],
        fetch=lambda s: [raw_market()],
        now=pd.Timestamp("2026-10-18T15:00:00Z"),
    )
    assert frame.snapshot_utc.iloc[0].startswith("2026-10-18T15:00:00")
    assert frame.series.iloc[0] == "KXNFLRECYDS"
