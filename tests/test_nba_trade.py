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

    def positions(self):
        return []

    def cancel(self, order_id):
        self.cancelled = getattr(self, "cancelled", []) + [order_id]
        return {"order_id": order_id}


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
    monkeypatch.setattr(nba_trade, "EXITS", out / "exits.csv")
    latest = tmp_path / "latest.csv"
    priced.to_csv(latest, index=False)
    monkeypatch.setattr(nba_trade, "LATEST", latest)
    monkeypatch.setattr(nba_trade, "refresh_quotes", lambda frame, now: frame)
    monkeypatch.setattr(
        nba_trade,
        "injuries",
        lambda: pd.DataFrame(
            columns=[
                "team",
                "player_id",
                "player",
                "status",
                "comment",
                "updated",
                "out",
            ]
        ),
    )
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
    monkeypatch.setattr(
        nba_trade,
        "injuries",
        lambda: pd.DataFrame(
            columns=[
                "team",
                "player_id",
                "player",
                "status",
                "comment",
                "updated",
                "out",
            ]
        ),
    )
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


def test_fresh_injury_news_drops_candidate():
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    cands = pd.DataFrame({"player_id": [10, 11, 12], "edge": [0.1, 0.1, 0.1]})
    report = pd.DataFrame(
        {
            "team": ["AAA", "BBB"],
            "player_id": [10, 12],
            "player": ["a", "c"],
            "status": ["Questionable", "Day-To-Day"],
            "comment": ["", ""],
            "updated": [
                (now - pd.Timedelta(hours=1)).isoformat(),
                (now - pd.Timedelta(days=3)).isoformat(),
            ],
            "out": [False, False],
        }
    )
    nba_trade.injuries = lambda: report
    kept = nba_trade.drop_fresh_injury_news(cands, now)
    assert set(kept.player_id) == {11, 12}  # 10 is fresh news; 12 is old and not Out


def test_injury_cancels_pull_resting_orders_on_changed_entries():
    ledger = pd.DataFrame(
        [
            {
                "ticker": "A",
                "player_id": 1,
                "order_id": "o1",
                "status": "resting",
                "run_utc": "2026-10-21T20:00+00:00",
                "rejected_reason": "",
            },
            {
                "ticker": "B",
                "player_id": 2,
                "order_id": "o2",
                "status": "resting",
                "run_utc": "2026-10-21T20:00+00:00",
                "rejected_reason": "",
            },
            {
                "ticker": "C",
                "player_id": 3,
                "order_id": "o3",
                "status": "resting",
                "run_utc": "2026-10-21T20:00+00:00",
                "rejected_reason": "",
            },
            {
                "ticker": "D",
                "player_id": 1,
                "order_id": "o4",
                "status": "done",
                "run_utc": "2026-10-21T20:00+00:00",
                "rejected_reason": "",
            },
        ]
    )
    report = pd.DataFrame(
        [
            # entry changed after the order: cancel
            {
                "player_id": 1,
                "status": "Questionable",
                "updated": "2026-10-21T21:00Z",
                "out": False,
            },
            # old entry, still questionable: leave it
            {
                "player_id": 2,
                "status": "Questionable",
                "updated": "2026-10-20T12:00Z",
                "out": False,
            },
            # old entry but now Out: cancel
            {
                "player_id": 3,
                "status": "Out",
                "updated": "2026-10-20T12:00Z",
                "out": True,
            },
        ]
    )
    fake = FakeClient()
    out, n = nba_trade.cancel_on_injury(
        ledger, fake, pd.Timestamp("2026-10-21T22:00Z"), report
    )
    assert n == 2
    assert out.set_index("ticker").status.to_dict() == {
        "A": "cancelled",
        "B": "resting",
        "C": "cancelled",
        "D": "done",
    }
    assert set(fake.cancelled) == {"o1", "o3"}
    assert out.set_index("ticker").rejected_reason["C"] == "injury: out"


def test_early_exit_orders():
    tip = pd.Timestamp("2026-10-22T00:00Z")
    quotes = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "yes_bid": [0.30, 0.40, 0.30],
            "yes_ask": [0.33, 0.44, 0.31],
            "tip_utc": [tip, tip, tip],
        }
    )
    # We sold "no" at 0.55 (yes level 0.45) in A, B and C.
    entries = {"A": 0.55, "B": 0.55, "C": 0.55}
    positions = [
        {"market_ticker": "A", "position": -10},  # yes ask 0.33 <= 0.39: exit
        {"market_ticker": "B", "position": -10},  # yes ask 0.44: not yet
        {"market_ticker": "C", "position": -10},  # 1-cent spread: at the bid
        {"market_ticker": "Z", "position": 5},  # a yes position: ignored
    ]
    out = nba_trade.exit_orders(
        positions, quotes, entries, {"A": 4}, pd.Timestamp("2026-10-21T20:00Z")
    )
    by = {o["ticker"]: o for o in out}
    assert set(by) == {"A", "C"}
    assert by["A"]["contracts"] == 6  # 4 already resting to sell
    assert by["A"]["exit_no_price"] == 0.68  # no bid 0.67 + 1 cent
    assert abs(by["A"]["expected_gain"] - 6 * (0.68 - 0.55)) < 1e-9
    assert by["C"]["exit_no_price"] == 0.69  # spread 1 cent: sit at the no bid
    fake = FakeClient()
    placed = nba_trade.place_exits(
        fake, out, pd.Timestamp("2026-10-21T20:00Z").to_pydatetime(), "live"
    )
    assert len(fake.placed) == 2 and all(
        o.action == "sell" and o.side == "no" for o in fake.placed
    )
    assert set(placed.status) == {"placed"}
    scored = nba_trade.score_exits(placed, fetch=lambda t: {"result": "no"})
    assert (scored.hold_profit_per_contract.round(2) == 0.45).all()


def test_entry_prices_average_no_fills():
    fills = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "side": ["no", "no", "yes"],
            "action": ["buy", "buy", "buy"],
            "no_price": [50, 60, 40],
            "count": [1, 3, 2],
        }
    )
    e = nba_trade.entry_prices(fills)
    assert abs(e["A"] - 0.575) < 1e-9 and "B" not in e
