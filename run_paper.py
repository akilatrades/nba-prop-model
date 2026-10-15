"""Settle the paper ledger from Kalshi's results and update the scorecard.

Positions whose games have started at least four hours ago are looked up on
Kalshi; once Kalshi has settled a market (yes, no, or a fixed value when the
player sat), the position's profit is recorded. Writes
outputs/paper/ledger.csv and outputs/paper/scorecard.json and .md.

Run from the repository root: python run_paper.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_model.kalshi import get_json, trades
from nba_model.paper import scorecard, settle
from nba_model.paper_maker import RULES, check_fills
from nba_model.paper_maker import scorecard as maker_scorecard
from nba_model.paper_maker import settle as maker_settle

LEDGER = Path("outputs/paper/ledger.csv")
MAKER_LEDGER = Path("outputs/paper_maker/orders.csv")


def fetch_market(ticker):
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        payload = get_json(path, {})
        if payload and payload.get("market"):
            return payload["market"]
    return None


def fetch_trades(ticker, min_ts, max_ts):
    rows = trades(ticker, min_ts, max_ts, historical=False)
    return rows or trades(ticker, min_ts, max_ts, historical=True)


def resting_orders(now):
    """Fill-check and settle the resting-order paper tests."""
    if not MAKER_LEDGER.exists():
        print("No resting-order ledger yet")
        return
    ledger = pd.read_csv(MAKER_LEDGER)
    ledger = check_fills(ledger, fetch_trades, int(now.timestamp()))
    ledger = maker_settle(ledger, fetch_market)
    ledger.to_csv(MAKER_LEDGER, index=False)
    cards = {rule: maker_scorecard(ledger, rule) for rule in RULES}
    cards["updated_utc"] = now.isoformat(timespec="minutes")
    folder = MAKER_LEDGER.parent
    (folder / "scorecard.json").write_text(json.dumps(cards, indent=2) + "\n")
    lines = [
        "# Resting-order paper tests, 2026-27",
        "",
        (
            "Two rules from RESEARCH_LOG.md (2026-10-15), each from a $500 paper "
            "bankroll. Fills are read from Kalshi's public trade prints."
        ),
        "",
        '| Measure | Model side | "No" seller |',
        "|---|---:|---:|",
    ]
    keys = dict.fromkeys(k for rule in RULES for k in cards[rule])
    for key in keys:
        values = [cards[rule].get(key, "") for rule in RULES]
        lines.append(f"| {key.replace('_', ' ')} | {values[0]} | {values[1]} |")
    (folder / "scorecard.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(cards, indent=2))


def main():
    resting_orders(pd.Timestamp(datetime.now(UTC)))
    if not LEDGER.exists():
        print("No paper ledger yet")
        return
    ledger = pd.read_csv(LEDGER)
    now = pd.Timestamp(datetime.now(UTC))
    started = pd.to_datetime(ledger.tip_utc, utc=True) < now - pd.Timedelta(hours=4)
    due = set(ledger.ticker[ledger.status.eq("open") & started])
    ledger = settle(ledger, lambda t: fetch_market(t) if t in due else None)
    ledger.to_csv(LEDGER, index=False)
    card = scorecard(ledger)
    card["updated_utc"] = now.isoformat(timespec="minutes")
    (LEDGER.parent / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    lines = [
        "# Paper trading scorecard, 2026-27",
        "",
        "The rule in `nba_model/strategy.py`, paper traded from a $500 bankroll.",
        "It goes to real money only after the test in RESEARCH_LOG.md is met.",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in card.items()]
    (LEDGER.parent / "scorecard.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
