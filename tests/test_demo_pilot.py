"""Fail-closed controls for the small Kalshi Demo validation pilot."""

import pandas as pd
import pytest

from kalshi.risk import Rejected, RiskLimits, check
from live import nba_trade


def test_pilot_counts_filled_orders_marked_done(monkeypatch, tmp_path):
    """A filled order is often no longer resting; its exposure must remain."""
    monkeypatch.setenv("DEMO_PILOT", "1")
    monkeypatch.setattr(nba_trade, "SETTLEMENTS", tmp_path / "missing.csv")
    ledger = pd.DataFrame(
        [
            {
                "mode": "live", "status": "done", "cost": 9.0,
                "day": "2026-10-18", "game_id": "G", "team": "HOU",
            },
            {
                "mode": "dry", "status": "dry", "cost": 100.0,
                "day": "2026-10-18", "game_id": "G", "team": "HOU",
            },
        ]
    )
    exposure = nba_trade.exposure_from(ledger, "2026-10-18")
    assert exposure.open_total == 9.0
    assert exposure.by_day["2026-10-18"] == 9.0
    limits = RiskLimits(max_order=2.0, max_night=10.0, max_open=10.0)
    with pytest.raises(Rejected, match="night cap"):
        check(2.0, "2026-10-18", "OTHER", limits, exposure, 0)


def test_pilot_counts_other_market_completed_orders(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_PILOT", "1")
    nfl = tmp_path / "nfl"
    nba = tmp_path / "nba"
    nfl.mkdir()
    nba.mkdir()
    pd.DataFrame(
        [
            {"status": "done", "mode": "live", "cost": 7.0},
            {"status": "dry", "mode": "dry", "cost": 90.0},
        ]
    ).to_csv(nfl / "orders.csv", index=False)
    monkeypatch.setattr(nba_trade, "MARKET_DIRS", {"nba": nba, "nfl": nfl})
    assert nba_trade.other_open_exposure("nba") == 7.0


def test_regular_mode_keeps_existing_exposure_semantics(monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_PILOT", raising=False)
    monkeypatch.setattr(nba_trade, "SETTLEMENTS", tmp_path / "missing.csv")
    ledger = pd.DataFrame(
        [
            {
                "status": "done", "cost": 9.0, "day": "2026-10-18",
                "game_id": "G", "team": "HOU",
            }
        ]
    )
    exposure = nba_trade.exposure_from(ledger, "2026-10-18")
    assert exposure.open_total == 0.0
