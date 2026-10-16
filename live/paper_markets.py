"""Model-free "no" seller paper tests in other Kalshi markets (kalshi/paper_markets.py).

Each run: check fills and settle resting orders, then post new ones for every
series. Ledger: outputs/paper_markets/orders.csv; scorecard alongside.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kalshi.api import fetch_market, fetch_markets, fetch_trades
from kalshi.paper_markets import (
    COLUMNS,
    SERIES,
    check_fills,
    empty_ledger,
    post_orders,
    scorecard,
    settle,
)
from markets.nfl.data import load as load_nfl
from markets.nfl.paper import MODEL_SERIES, model_probabilities

LEDGER = Path("outputs/paper_markets/orders.csv")
MODEL_SUFFIX = "+model"


def main():
    now = datetime.now(UTC)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else empty_ledger()
    ledger = check_fills(ledger, fetch_trades, int(now.timestamp()))
    ledger = settle(ledger, fetch_market)
    nfl_games = None
    seen = {}
    for series in SERIES:
        try:
            markets = fetch_markets(series, status="open", historical=False)
        except OSError as error:
            print(f"{series}: could not list markets ({error})")
            continue
        posted = post_orders(series, markets, ledger, now)
        ledger = pd.concat([ledger, posted], ignore_index=True)[COLUMNS]
        print(f"{series}: {len(markets)} open markets, {len(posted)} new orders")
        seen[series] = {"open": len(markets), "posted": int(len(posted))}
        if series in MODEL_SERIES:
            # The same markets with the NFL model as veto (RESEARCH_LOG.md, 2026-10-16).
            if nfl_games is None:
                nfl_games, _ = load_nfl()
            p_model = model_probabilities(markets, nfl_games, now)
            label = series + MODEL_SUFFIX
            posted = post_orders(label, markets, ledger, now, p_model=p_model)
            ledger = pd.concat([ledger, posted], ignore_index=True)[COLUMNS]
            vetoed = int(posted.vetoed.sum()) if len(posted) else 0
            print(
                f"{label}: {len(p_model)} priced by the model, {len(posted) - vetoed} orders, {vetoed} vetoed"
            )
    ledger.to_csv(LEDGER, index=False)
    print("::notice::" + json.dumps(seen))
    labels = list(SERIES) + [s + MODEL_SUFFIX for s in SERIES if s in MODEL_SERIES]
    cards = {s: scorecard(ledger, s) for s in labels}
    cards["updated_utc"] = now.isoformat(timespec="minutes")
    (LEDGER.parent / "scorecard.json").write_text(json.dumps(cards, indent=2) + "\n")
    lines = [
        '# Model-free "no" seller paper tests (other markets)',
        "",
        (
            "One $500 paper bankroll per series; rule in kalshi/paper_markets.py. "
            "Series marked +model use the NFL model as veto (markets/nfl/paper.py)."
        ),
        "",
        "| Series | Orders | Open | Fill rate | Fills settled | ¢/contract | t | Bankroll |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in labels:
        c = cards[s]
        fr = "" if c["fill_rate"] is None else f"{100 * c['fill_rate']:.0f}%"
        lines.append(
            f"| {s} | {c['orders']} | {c['open']} | {fr} | {c['fills_settled']} | "
            f"{c.get('cents_per_contract', '')} | {c.get('t', '')} | ${c['bankroll']:.2f} |"
        )
    (LEDGER.parent / "scorecard.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(cards, indent=2))


if __name__ == "__main__":
    main()
