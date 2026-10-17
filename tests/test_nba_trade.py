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
        return getattr(self, "held", [])

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
    monkeypatch.setattr(nba_trade, "projection_age_hours", lambda now, latest=None: 1.0)
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
    assert summary["reconcile"]["on_exchange_not_in_ledger"] == 0
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
        ("EXITS", out / "exits.csv"),
    ]:
        monkeypatch.setattr(nba_trade, name, path)
    latest = tmp_path / "latest.csv"
    priced.to_csv(latest, index=False)
    monkeypatch.setattr(nba_trade, "LATEST", latest)
    monkeypatch.setattr(nba_trade, "refresh_quotes", lambda frame, now: frame)
    monkeypatch.setattr(nba_trade, "projection_age_hours", lambda now, latest=None: 1.0)
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


def test_nfl_market_dry_run(tmp_path, monkeypatch):
    import live.nfl_trade as nfl

    now = pd.Timestamp("2026-10-18T14:00:00Z")
    markets = [
        {
            "ticker": f"KXNFLRECYDS-26OCT18CHIATL-ATLP{i}-75",
            "event_ticker": "KXNFLRECYDS-26OCT18CHIATL",
            "title": f"Player {i}: 75+ receiving yards",
            "yes_sub_title": f"Player {i}: 75+",
            "floor_strike": 74.5,
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.45",
            "last_price_dollars": "0.42",
            "occurrence_datetime": "2026-10-18T18:00:00Z",
        }
        for i in range(3)
    ]
    p_model = {markets[0]["ticker"]: 0.30, markets[1]["ticker"]: 0.70}  # third unpriced
    monkeypatch.setattr(
        nfl,
        "fetch_markets",
        lambda series, status, historical: markets if "RECYDS" in series else [],
    )
    monkeypatch.setattr(nfl, "load_nfl", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(nfl, "model_probabilities", lambda m, g, now: p_model)
    monkeypatch.setattr(
        nfl,
        "attach_latest",
        lambda priced, path=None: priced.assign(p_book=float("nan"), book_n=0),
    )
    monkeypatch.setattr(
        nba_trade, "MARKET_DIRS", {"nba": tmp_path / "nba", "nfl": tmp_path / "nfl"}
    )
    monkeypatch.setenv("TRADE_MARKET", "nfl")
    monkeypatch.setenv("TRADE_MODE", "dry")
    monkeypatch.setenv("DRY_BANKROLL", "250")

    class FixedNow:
        @staticmethod
        def now(tz):
            return now.to_pydatetime()

    monkeypatch.setattr(nba_trade, "datetime", FixedNow)
    nba_trade.main()
    ledger = pd.read_csv(tmp_path / "nfl" / "orders.csv")
    # Only the market the model rates below the "no" price qualifies; the
    # unpriced one is skipped under the v1 rule (no model, no book: no order).
    assert ledger.ticker.tolist() == [markets[0]["ticker"]]
    assert ledger.status.iloc[0] == "dry" and ledger.game_id.iloc[0] == "26OCT18CHIATL"
    summary = json.loads((tmp_path / "nfl" / "last_run.json").read_text())
    assert summary["market"] == "nfl" and summary["placed"] == 1


def _wire(tmp_path, monkeypatch, priced, fake, now, mode="live", report=None):
    out = tmp_path / "live"
    for name, path in [
        ("OUT", out),
        ("ORDERS", out / "orders.csv"),
        ("FILLS", out / "fills.csv"),
        ("SETTLEMENTS", out / "settlements.csv"),
        ("EXITS", out / "exits.csv"),
    ]:
        monkeypatch.setattr(nba_trade, name, path)
    latest = tmp_path / "latest.csv"
    priced.to_csv(latest, index=False)
    monkeypatch.setattr(nba_trade, "LATEST", latest)
    monkeypatch.setattr(nba_trade, "refresh_quotes", lambda frame, now: frame)
    monkeypatch.setattr(nba_trade, "projection_age_hours", lambda now, latest=None: 1.0)
    empty = pd.DataFrame(
        columns=["team", "player_id", "player", "status", "comment", "updated", "out"]
    )
    monkeypatch.setattr(
        nba_trade, "injuries", lambda: empty if report is None else report
    )
    monkeypatch.setattr(nba_trade, "TradingClient", lambda dry_run: fake)
    monkeypatch.setenv("TRADE_MODE", mode)

    class FixedNow:
        @staticmethod
        def now(tz):
            return now.to_pydatetime()

    monkeypatch.setattr(nba_trade, "datetime", FixedNow)
    return out


def test_unknown_exchange_orders_block_new_placements(tmp_path, monkeypatch):
    """A previous run placed and crashed before writing the ledger: the next
    run must see the stranger on the exchange and place nothing."""
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    fake = FakeClient(resting_unknown=True)
    out = _wire(
        tmp_path, monkeypatch, priced_frame(now + pd.Timedelta(hours=4)), fake, now
    )
    nba_trade.main()
    assert fake.placed == []
    summary = json.loads((out / "last_run.json").read_text())
    assert summary["placed"] == 0
    assert summary["reconcile_before"]["on_exchange_not_in_ledger"] == 1


def test_dry_run_rows_do_not_block_live_orders(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    priced = priced_frame(now + pd.Timedelta(hours=4))
    fake = FakeClient()
    out = _wire(tmp_path, monkeypatch, priced, fake, now)
    out.mkdir(parents=True)
    dry = pd.DataFrame(
        [
            {"ticker": t, "status": "dry", "run_utc": "2026-10-21T15:00+00:00"}
            for t in priced.ticker
        ]
    )
    for column in nba_trade.ORDER_COLUMNS:
        if column not in dry:
            dry[column] = pd.NA
    dry[nba_trade.ORDER_COLUMNS].to_csv(out / "orders.csv", index=False)
    nba_trade.main()
    assert len(fake.placed) == 2  # the dry rows did not count as "already in"
    # and in dry mode the same rows DO stop a second dry log of the same markets
    fake2 = FakeClient()
    fake2.can_sign = False
    _wire(tmp_path, monkeypatch, priced, fake2, now, mode="dry")
    nba_trade.main()
    ledger = pd.read_csv(out / "orders.csv")
    assert (ledger.status == "dry").sum() == 3  # the original three, none added


def test_injury_feed_failure_places_nothing(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    fake = FakeClient()
    _wire(tmp_path, monkeypatch, priced_frame(now + pd.Timedelta(hours=4)), fake, now)

    def broken():
        raise OSError("espn down")

    monkeypatch.setattr(nba_trade, "injuries", broken)
    nba_trade.main()
    assert fake.placed == []


def test_stale_projections_place_nothing(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    fake = FakeClient()
    _wire(tmp_path, monkeypatch, priced_frame(now + pd.Timedelta(hours=4)), fake, now)
    monkeypatch.setattr(
        nba_trade, "projection_age_hours", lambda now, latest=None: 12.0
    )
    nba_trade.main()
    assert fake.placed == []


def test_projection_age_from_ladder_names(tmp_path):
    latest = tmp_path / "latest.csv"
    latest.write_text("x\n")
    (tmp_path / "ladders").mkdir()
    (tmp_path / "ladders" / "2026-10-21T1830Z.csv").write_text("x\n")
    (tmp_path / "ladders" / "2026-10-21T1530Z.csv").write_text("x\n")
    age = nba_trade.projection_age_hours(pd.Timestamp("2026-10-21T20:00Z"), latest)
    assert abs(age - 1.5) < 1e-6


def test_teammate_out_drops_candidates_and_cancels_resting():
    now = pd.Timestamp("2026-10-21T22:00Z")
    report = pd.DataFrame(
        [
            {
                "team": "AAA",
                "player_id": 99,
                "player": "star",
                "status": "Out",
                "comment": "",
                "updated": "2026-10-21T21:00Z",
                "out": True,
            }
        ]
    )
    nba_trade.injuries = lambda: report
    cands = pd.DataFrame(
        {"player_id": [10, 20], "team": ["AAA", "BBB"], "edge": [0.1, 0.1]}
    )
    kept = nba_trade.drop_fresh_injury_news(cands, now)
    assert set(kept.team) == {
        "BBB"
    }  # the star's teammate is skipped, not the other game
    ledger = pd.DataFrame(
        [
            {
                "ticker": "A",
                "player_id": 10,
                "team": "AAA",
                "order_id": "o1",
                "status": "resting",
                "run_utc": "2026-10-21T20:00+00:00",
                "rejected_reason": "",
            },
            {
                "ticker": "B",
                "player_id": 20,
                "team": "BBB",
                "order_id": "o2",
                "status": "resting",
                "run_utc": "2026-10-21T20:00+00:00",
                "rejected_reason": "",
            },
            {
                "ticker": "C",
                "player_id": 11,
                "team": "AAA",
                "order_id": "o3",
                "status": "resting",
                "run_utc": "2026-10-21T21:30+00:00",
                "rejected_reason": "",
            },
        ]
    )
    fake = FakeClient()
    out, n = nba_trade.cancel_on_injury(ledger, fake, now, report)
    # A was placed before the star was ruled out: cancelled. C was placed after
    # (already priced with him out): kept. B is another team: kept.
    assert n == 1 and fake.cancelled == ["o1"]
    assert out.set_index("ticker").rejected_reason["A"] == "injury: teammate out"


def test_cancel_all_scoped_to_strategy_prefixes():
    from kalshi.trading import TradingClient

    class Client(TradingClient):
        def __init__(self):
            self.cancelled = []

        def orders(self, status="resting", ticker=None):
            return [
                {"order_id": "a", "ticker": "KXNBAPTS-X"},
                {"order_id": "b", "ticker": "KXNFLRECYDS-X"},
                {"order_id": "c", "ticker": "KXBTC-SOMETHING"},
            ]

        def cancel(self, order_id):
            self.cancelled.append(order_id)
            return {"order_id": order_id}

    c = Client()
    c.cancel_all(prefixes=nba_trade.STRATEGY_PREFIXES)
    assert c.cancelled == ["a", "b"]


def test_filled_unrecorded_position_blocks_new_placements(tmp_path, monkeypatch):
    """A crashed run's order filled completely: nothing is resting, but the
    position exists in a ticker the ledger never ordered. Place nothing."""
    now = pd.Timestamp("2026-10-21T20:00:00Z")
    fake = FakeClient()
    fake.held = [{"ticker": "KXNBAPTS-26OCT21AAABBB-GHOST-20", "position": -3}]
    out = _wire(
        tmp_path, monkeypatch, priced_frame(now + pd.Timedelta(hours=4)), fake, now
    )
    nba_trade.main()
    assert fake.placed == []
    summary = json.loads((out / "last_run.json").read_text())
    assert summary["reconcile_before"]["positions_not_in_ledger"] == 1


def test_projection_age_never_uses_mtime(tmp_path):
    latest = tmp_path / "latest.csv"
    latest.write_text("x\n")  # fresh mtime, no stamp, no ladders
    assert nba_trade.projection_age_hours(
        pd.Timestamp("2026-10-21T20:00Z"), latest
    ) == float("inf")
    (tmp_path / "latest_meta.json").write_text(
        json.dumps({"generated_utc": "2026-10-21T17:30+00:00"})
    )
    age = nba_trade.projection_age_hours(pd.Timestamp("2026-10-21T20:00Z"), latest)
    assert abs(age - 2.5) < 1e-6


def test_kill_switch_spares_orders_the_strategy_did_not_place():
    from kalshi.trading import TradingClient

    class Client(TradingClient):
        def __init__(self):
            self.cancelled = []

        def orders(self, status="resting", ticker=None):
            return [
                {
                    "order_id": "ours-by-ledger",
                    "ticker": "KXNBAPTS-X",
                    "client_order_id": "abc",
                },
                {
                    "order_id": "ours-by-tag",
                    "ticker": "KXNBAREB-X",
                    "client_order_id": "smm-123",
                },
                {
                    "order_id": "manual",
                    "ticker": "KXNBAPTS-Y",
                    "client_order_id": "phone",
                },
                {
                    "order_id": "other-market",
                    "ticker": "KXBTC-Z",
                    "client_order_id": "smm-9",
                },
            ]

        def cancel(self, order_id):
            self.cancelled.append(order_id)
            return {"order_id": order_id}

    c = Client()
    c.cancel_all(prefixes=nba_trade.STRATEGY_PREFIXES, order_ids=["ours-by-ledger"])
    assert c.cancelled == ["ours-by-ledger", "ours-by-tag"]


def test_execution_report_and_queue_log(tmp_path):
    from live.execution import execution_report, log_queue_positions, summary

    now = pd.Timestamp("2026-10-21T20:00Z")
    ledger = pd.DataFrame(
        [
            {
                "order_id": "o1",
                "ticker": "KXNBAPTS-A",
                "day": "2026-10-21",
                "player": "p",
                "stat": "points",
                "line": 19.5,
                "price": 0.55,
                "contracts": 9,
                "edge": 0.03,
                "status": "resting",
                "run_utc": "2026-10-21T19:00+00:00",
            },
            {
                "order_id": "",
                "ticker": "KXNBAPTS-B",
                "day": "2026-10-21",
                "player": "q",
                "stat": "points",
                "line": 19.5,
                "price": 0.55,
                "contracts": 9,
                "edge": 0.03,
                "status": "dry",
                "run_utc": "2026-10-21T19:00+00:00",
            },
        ]
    )

    class Q:
        def queue_position(self, order_id):
            return 40.0

    path = tmp_path / "queue.csv"
    assert log_queue_positions(Q(), ledger, now, path) == 1  # the dry row is skipped
    assert log_queue_positions(Q(), ledger, now + pd.Timedelta(minutes=5), path) == 0
    assert log_queue_positions(Q(), ledger, now + pd.Timedelta(minutes=20), path) == 1
    fills = pd.DataFrame(
        [
            {
                "trade_id": "t",
                "ticker": "KXNBAPTS-A",
                "created_time": "2026-10-21T20:45:00Z",
                "count": 4,
            }
        ]
    )
    rep = execution_report(ledger, fills, pd.read_csv(path))
    assert len(rep) == 1
    r = rep.iloc[0]
    assert r.queue_ahead_at_start == 40.0 and r.filled == 4
    assert abs(r.minutes_to_first_fill - 105) < 1e-6
    s = summary(rep)
    assert s["fill_rate"] == 1.0 and abs(s["mean_fill_share"] - 4 / 9) < 1e-6
