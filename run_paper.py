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

from nba_model.kalshi import get_json
from nba_model.paper import scorecard, settle

LEDGER = Path("outputs/paper/ledger.csv")


def fetch_market(ticker):
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        payload = get_json(path, {})
        if payload and payload.get("market"):
            return payload["market"]
    return None


def main():
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
