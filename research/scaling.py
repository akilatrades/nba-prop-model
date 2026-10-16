"""Bankroll replays of the 2025-26 NBA "no" seller (RESEARCH_LOG.md, 2026-10-16).

Replays the season's resting "no" orders (outputs/decompose/orders_no.csv.gz,
every line with the model veto) day by day with the staking rule, so the
question "what does $X become at Y% a line" has a number, with the day
bootstrap giving the spread. Flow share is the fraction of the contracts
that traded through our price that we are assumed to get; half is the
working assumption. In-sample and one season: an upper bound on expectations,
not a forecast.

    python -m research.scaling              # table for several starts and stakes
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.strategy import DAY_CAP, MAX_CONTRACTS, MAX_ORDER_DOLLARS

FILLS = Path("outputs/decompose/orders_no.csv.gz")
OUT = Path("outputs/scaling")


def load_orders():
    o = pd.read_csv(FILLS, low_memory=False)
    o = o[o.model_edge.astype(float) >= 0].copy()  # the rule: model veto on
    o["day"] = (
        pd.to_datetime(o.tip_utc, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    o["price"] = o.price.astype(float)
    o["filled_contracts"] = pd.to_numeric(o.filled_contracts, errors="coerce").fillna(0)
    o["pnl"] = pd.to_numeric(o.pnl_if_filled, errors="coerce").fillna(0)
    return o[["day", "game_id", "price", "edge", "filled_contracts", "pnl"]]


def replay_day(group, bank, stake, flow_share, order_cap, day_cap=DAY_CAP):
    """One day's orders against one bankroll; returns (profit, dollars reserved)."""
    room = day_cap * bank
    budget_each = stake * bank
    if order_cap:
        budget_each = min(budget_each, MAX_ORDER_DOLLARS)
    profit = reserved = 0.0
    for price, filled, pnl in zip(
        group.price.to_numpy(), group.filled_contracts.to_numpy(), group.pnl.to_numpy()
    ):
        budget = min(budget_each, room)
        n = math.floor(budget / price + 1e-9)
        if order_cap:
            n = min(n, MAX_CONTRACTS)
        if n < 1:
            continue
        room -= n * price
        reserved += n * price
        got = min(n, math.floor(flow_share * filled))
        profit += got * pnl
    return profit, reserved


def replay(orders, start, stake, flow_share=0.5, order_cap=True, days=None):
    """Bankroll path over the season (or over a given order of days)."""
    by_day = {
        d: g.sort_values("edge", ascending=False) for d, g in orders.groupby("day")
    }
    days = list(by_day) if days is None else days
    bank, path = float(start), [float(start)]
    for d in days:
        profit, _ = replay_day(by_day[d], bank, stake, flow_share, order_cap)
        bank = max(bank + profit, 0.0)
        path.append(bank)
        if bank <= 0:
            break
    return np.array(path)


def drawdown(path):
    peak = np.maximum.accumulate(path)
    return float(((peak - path) / peak).max()) if len(path) else 0.0


def bootstrap(orders, start, stake, flow_share, order_cap, paths=1000, seed=0):
    rng = np.random.default_rng(seed)
    days = sorted(orders.day.unique())
    ends, dds = [], []
    for _ in range(paths):
        sample = list(rng.choice(days, size=len(days), replace=True))
        p = replay(orders, start, stake, flow_share, order_cap, days=sample)
        ends.append(p[-1])
        dds.append(drawdown(p))
    ends, dds = np.array(ends), np.array(dds)
    return {
        "median_end": round(float(np.median(ends))),
        "p10_end": round(float(np.percentile(ends, 10))),
        "p90_end": round(float(np.percentile(ends, 90))),
        "chance_of_loss": round(float((ends < start).mean()), 3),
        "chance_below_half": round(float((ends < start / 2).mean()), 3),
        "median_max_drawdown": round(float(np.median(dds)), 3),
        "p90_max_drawdown": round(float(np.percentile(dds, 90)), 3),
    }


def table(orders, starts=(250, 500, 1000), stakes=(0.01, 0.02, 0.03, 0.05), flow=0.5):
    rows = []
    for start in starts:
        for stake in stakes:
            path = replay(orders, start, stake, flow, True)
            boot = bootstrap(orders, start, stake, flow, True, paths=400)
            rows.append(
                {
                    "start": start,
                    "stake": stake,
                    "flow_share": flow,
                    "season_end": round(float(path[-1])),
                    "multiple": round(float(path[-1] / start), 2),
                    "max_drawdown": round(drawdown(path), 3),
                    **boot,
                }
            )
            print(json.dumps(rows[-1]))
    return pd.DataFrame(rows)


def main(argv):
    orders = load_orders()
    print(f"{len(orders)} orders over {orders.day.nunique()} days")
    out = table(orders)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "stake_table.csv", index=False)
    lines = [
        "# Bankroll replays: start x stake per line (half the flow, order cap on)",
        "",
        "In-sample 2025-26, every line with the model veto. Bootstrap resamples days (400 paths).",
        "",
        "| Start | Stake | Season end | Multiple | Max DD | Boot median | 10th-90th | P(loss) | P(< half) | Median max DD | 90th max DD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out.itertuples():
        lines.append(
            f"| ${r.start:,} | {r.stake:.0%} | ${r.season_end:,} | {r.multiple}x | {r.max_drawdown:.0%} | "
            f"${r.median_end:,} | ${r.p10_end:,}-${r.p90_end:,} | {r.chance_of_loss:.0%} | {r.chance_below_half:.0%} | "
            f"{r.median_max_drawdown:.0%} | {r.p90_max_drawdown:.0%} |"
        )
    (OUT / "stake_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
