"""Quote placement for the NBA "no" seller (RESEARCH_LOG.md, 2026-10-16,
"Execution studies"): where to rest the order.

Same 2025-26 orders and trade prints as the decomposition; four price
policies, each judged on net profit per 100 eligible orders with one contract
per order (unfilled orders count zero), so a policy cannot win by filling
less. Also reported: fill rate and profit per filled contract.

    join      rest at the "no" bid (yes_ask), behind the queue
    improve1  one cent inside (the current rule)
    improve2  two cents inside when the spread allows, else one
    edge2     one cent inside, only when the model's edge there is >= 2 cents

    python -m research.quote_placement   # offline, writes outputs/execution/placement.*
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.maker import fills, maker_fee
from kalshi.stats import clustered_mean_t

ORDERS = Path("outputs/decompose/orders_no.csv.gz")
TRADES = [
    Path(f"data/kalshi_history/trades_{s}.parquet")
    for s in ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT")
]
OUT = Path("outputs/execution")
FEE_SHARE = 0.25
TICK = 0.01


def load():
    o = pd.read_csv(ORDERS, low_memory=False)
    o = o[o.model_edge.astype(float) >= 0].copy()  # the rule: model veto on
    for c in ("yes_bid", "yes_ask", "over", "p_model"):
        o[c] = o[c].astype(float)
    t = pd.concat([pd.read_parquet(p) for p in TRADES if p.exists()], ignore_index=True)
    stamps = pd.to_datetime(t.created_time, utc=True, format="ISO8601")
    t["ts"] = (
        stamps.dt.tz_convert(None).to_numpy().astype("datetime64[s]").astype("int64")
    )
    t["yes_price"] = t.yes_price.astype(float)
    t["count"] = t["count"].astype(float)
    return o, t


def policy_orders(o, policy):
    spread = (o.yes_ask - o.yes_bid).round(4)
    no_bid = 1 - o.yes_ask
    if policy == "join":
        price, improved = no_bid, pd.Series(False, index=o.index)
    elif policy == "improve1":
        improved = spread >= 2 * TICK - 1e-9
        price = np.where(improved, no_bid + TICK, no_bid)
    elif policy == "improve2":
        two = spread >= 3 * TICK - 1e-9
        one = spread >= 2 * TICK - 1e-9
        price = np.where(two, no_bid + 2 * TICK, np.where(one, no_bid + TICK, no_bid))
        improved = pd.Series(True, index=o.index)
    elif policy == "edge2":
        improved = spread >= 2 * TICK - 1e-9
        price = np.where(improved, no_bid + TICK, no_bid)
    else:
        raise ValueError(policy)
    q = o.copy()
    q["price"] = np.round(price, 4)
    q["improved"] = np.asarray(improved, dtype=bool)
    q["side"] = "no"
    q["yes_level"] = (1 - q.price).round(4)
    q["edge"] = (1 - q.p_model) - q.price - maker_fee(q.price, FEE_SHARE)
    if policy == "edge2":
        q = q[q.edge >= 0.02]
    q = q[(q.price > 0) & (q.price < 1)]
    return q


def evaluate(q, trades, label):
    got = fills(q, trades)
    q = q.copy()
    q["filled_contracts"] = q.ticker.map(got).fillna(0.0)
    q["filled"] = q.filled_contracts >= 1
    fee = maker_fee(q.price, FEE_SHARE)
    q["pnl_fill"] = (1 - q.over) - q.price - fee  # per contract if filled
    q["pnl_order"] = np.where(q.filled, q.pnl_fill, 0.0)  # one contract per order
    mean_o, t_o = clustered_mean_t(q.pnl_order, q.game_id)
    f = q[q.filled]
    mean_f, t_f = (
        clustered_mean_t(f.pnl_fill, f.game_id) if len(f) > 1 else (np.nan, np.nan)
    )
    return {
        "policy": label,
        "eligible_orders": int(len(q)),
        "fill_rate": round(float(q.filled.mean()), 3),
        "profit_per_100_orders_cents": round(100 * 100 * float(mean_o), 1),
        "t_per_order": round(float(t_o), 2) if np.isfinite(t_o) else None,
        "cents_per_filled_contract": round(100 * float(mean_f), 2) if len(f) else None,
        "t_per_fill": round(float(t_f), 2) if np.isfinite(t_f) else None,
        "avg_no_price": round(float(q.price.mean()), 3),
    }


def main():
    o, trades = load()
    rows = [
        evaluate(policy_orders(o, p), trades, p)
        for p in ("join", "improve1", "improve2", "edge2")
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "placement.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [
        "# Quote placement, NBA no-seller 2025-26 (model veto on)",
        "",
        "| Policy | Eligible orders | Fill rate | Profit per 100 orders (¢, 1 contract each) | t | ¢ per filled contract | t | Avg no price |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['eligible_orders']:,} | {r['fill_rate']:.0%} | {r['profit_per_100_orders_cents']:+.1f} | {r['t_per_order']} | {r['cents_per_filled_contract']:+.2f} | {r['t_per_fill']} | {r['avg_no_price']:.3f} |"
        )
    (OUT / "placement.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
