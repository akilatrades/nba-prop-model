"""End-to-end test of the live trader against a fake exchange and fake quotes."""

import json

import pandas as pd

from live import nba_trade


class FakeClient:
    """Stands in for TradingClient: records orders, returns canned account data."""

    def __init__(self, resting_unknown=False):
        self.env = "demo"
        self.dry_run = False
        self.can_sign = True
        self.placed = []
        self.resting_unknown = resting_unknown

    def balance(self):
        return {"balance": 250.0, "portfolio_value": 250.0}

    def place(self, order):
        self.placed.append(order)
        return {"order_id": f"id-{len(self.placed)}", "status": "resting"}

    def orders(self, status="resting", ticker=None):
        out = [{"order_id": f"id-{i + 1}"} for i in range(len(self.placed))]
        if self.resting_unknown:
            out.append({"order_id": "stranger"})
        return out

    def fills(self, min_ts=None, ticker=None):
        return [
            {
                "trade_id": "t1",
                "ticker": self.placed[0].ticker if self.placed else "X",
                "created_time": "2026-10-21T00:10:00Z",
                "count": 2,
                "yes_price": 55,
            }
        ]

    def settlements(self, min_ts=None):
        return [
            {
                "ticker": "KXNBAPTS-26OCT20AAABBB-OLD-20",
                "revenue": 0,
                "yes_total_cost": 0,
                "no_total_cost": 300,
                "fee_cost": 2,
                "settled_time": "2026-10-21T04:00:00Z",
            }
        ]

    def cancel_all(self, ticker=None):
        return []


def priced_frame(tip):
    rows = []
    for i, (bid, ask, p) in enumerate(
        [(0.40, 0.45, 0.30), (0.60, 0.61, 0.20), (0.50, 0.55, 0.70)]
    ):
        rows.append(
            {
                "ticker": f"KXNBAPTS-26OCT21AAABBB-P{i}-20",
                "game_id": 1,
                "player_id": 10 + i,
                "player": f"Player {i}",
                "team": "AAA",
                "stat": "points",
                "line": 19.5,
                "yes_bid_dollars": bid,
                "yes_ask_dollars": ask,
                "p_model": p,
                "tip_utc": tip.isoformat(),
                "injury_status": "",
            }
        )
    return pd.DataFrame(rows)


def test_trader_places_sized_orders_and_reconciles(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    tip = now + pd.Timedelta(hours=4)
    priced = priced_frame(tip)
    out = tmp_path / "live"
    monkeypatch.setattr(nba_trade, "OUT", out)
    monkeypatch.setattr(nba_trade, "ORDERS", out / "orders.csv")
    monkeypatch.setattr(nba_trade, "FILLS", out / "fills.csv")
    monkeypatch.setattr(nba_trade, "SETTLEMENTS", out / "settlements.csv")
    latest = tmp_path / "latest.csv"
    priced.to_csv(latest, index=False)
    monkeypatch.setattr(nba_trade, "LATEST", latest)
    monkeypatch.setattr(nba_trade, "refresh_quotes", lambda frame, now: frame)
    fake = FakeClient(resting_unknown=True)
    monkeypatch.setattr(nba_trade, "TradingClient", lambda dry_run: fake)
    monkeypatch.setenv("TRADE_MODE", "live")

    class FixedNow:
        @staticmethod
        def now(tz):
            return now.to_pydatetime()

    monkeypatch.setattr(nba_trade, "datetime", FixedNow)
    nba_trade.main()

    ledger = pd.read_csv(out / "orders.csv")
    # Two markets had a positive "no" edge (model yes 0.30 and 0.20); the third
    # (model yes 0.70 against a no price of 0.46) is vetoed by the rule itself.
    assert len(fake.placed) == 2
    assert set(ledger.status) <= {"resting", "done", "rejected"}
    for order in fake.placed:
        assert order.side == "no" and order.expiration_ts == int(tip.timestamp())
        assert (
            order.count * order.price_cents / 100 <= 5.0 + 1e-9
        )  # 1% of $250 or $5 cap
    summary = json.loads((out / "last_run.json").read_text())
    assert summary["placed"] == 2
    assert summary["reconcile"]["on_exchange_not_in_ledger"] == 1  # the stranger
    st = pd.read_csv(out / "settlements.csv")
    assert (
        round(float(st.profit.iloc[0]), 2) == -3.02
    )  # lost $3 of cost plus 2 cents fee
    assert (out / "scorecard.md").exists()


def test_orders_per_game_cap(tmp_path, monkeypatch):
    from kalshi.risk import RiskLimits

    now = pd.Timestamp("2026-10-21T20:00:00Z")
    tip = now + pd.Timedelta(hours=4)
    priced = priced_frame(tip)
    out = tmp_path / "live"
    for name, path in [
        ("OUT", out),
        ("ORDERS", out / "orders.csv"),
        ("FILLS", out / "fills.csv"),
        ("SETTLEMENTS", out / "settlements.csv"),
    ]:
        monkeypatch.setattr(nba_trade, name, path)
    latest = tmp_path / "latest.csv"
    priced.to_csv(latest, index=False)
    monkeypatch.setattr(nba_trade, "LATEST", latest)
    monkeypatch.setattr(nba_trade, "refresh_quotes", lambda frame, now: frame)
    fake = FakeClient()
    monkeypatch.setattr(nba_trade, "TradingClient", lambda dry_run: fake)
    monkeypatch.setattr(
        nba_trade, "RiskLimits", lambda: RiskLimits(max_orders_per_game=1)
    )
    monkeypatch.setenv("TRADE_MODE", "live")

    class FixedNow:
        @staticmethod
        def now(tz):
            return now.to_pydatetime()

    monkeypatch.setattr(nba_trade, "datetime", FixedNow)
    nba_trade.main()
    ledger = pd.read_csv(out / "orders.csv")
    assert len(fake.placed) == 1
    assert (
        ledger.rejected_reason.fillna("").str.contains("orders already in game")
    ).sum() == 1
