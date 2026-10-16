"""Build outputs/dashboard/summary.json for the dashboard page (dashboard/index.html).

One file, rewritten by every scheduled job, with everything the page shows:
each ledger's bankroll, fills and profit; daily profit by ledger; the live
trader's status; the scan's top markets. Run: python -m live.summarize
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.paper_markets import SERIES as OTHER_SERIES
from kalshi.paper_props import ALL_RULES
from markets.nfl.paper import MODEL_SERIES

OUT = Path("outputs/dashboard/summary.json")
START = 500.0


def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def ledger_rows(frame, group_col, label_fn=None, expected=()):
    """Per-ledger scorecard and per-day profit from a resting-order ledger."""
    out, daily = [], []
    present = set(frame[group_col].astype(str))
    for key in expected:
        if key not in present:
            out.append(
                {
                    "ledger": label_fn(key) if label_fn else str(key),
                    "orders": 0,
                    "open": 0,
                    "fill_rate": None,
                    "fills": 0,
                    "contracts": 0,
                    "profit": 0.0,
                    "cents_per_contract": None,
                    "bankroll": START,
                }
            )
    for key, g in frame.groupby(group_col):
        done = g[g.status.isin(["settled", "unfilled"])]
        settled = g[g.status == "settled"]
        contracts = _num(settled.filled_contracts)
        profit = _num(settled.profit)
        per = (100 * profit.sum() / contracts.sum()) if contracts.sum() else None
        out.append(
            {
                "ledger": label_fn(key) if label_fn else str(key),
                "orders": int(len(g)),
                "open": int(g.status.isin(["resting", "filled"]).sum()),
                "fill_rate": float((done.status == "settled").mean())
                if len(done)
                else None,
                "fills": int(len(settled)),
                "contracts": int(contracts.sum()),
                "profit": round(float(profit.sum()), 2),
                "cents_per_contract": round(per, 2) if per is not None else None,
                "bankroll": round(START + float(_num(done.profit).sum()), 2),
            }
        )
        for day, d in settled.groupby("day"):
            daily.append(
                {
                    "ledger": label_fn(key) if label_fn else str(key),
                    "day": str(day),
                    "profit": round(float(_num(d.profit).sum()), 2),
                    "fills": int(len(d)),
                }
            )
    return out, daily


def live_status(folder):
    """The live trader's state for one market folder (outputs/live or outputs/live/nfl)."""
    live = {}
    p = folder / "last_run.json"
    if p.exists():
        live = json.loads(p.read_text())
    p = folder / "orders.csv"
    if p.exists():
        o = pd.read_csv(p)
        live["orders_by_status"] = o.status.value_counts().to_dict()
        live["dollars_resting"] = round(
            float(_num(o.loc[o.status.isin(["placed", "resting"]), "cost"]).sum()), 2
        )
    p = folder / "settlements.csv"
    if p.exists():
        st = pd.read_csv(p)
        live["realized_by_day"] = {
            str(k): round(float(v), 2)
            for k, v in st.groupby("day").profit.sum().items()
        }
        live["realized_total"] = round(float(_num(st.profit).sum()), 2)
    p = folder / "exits.csv"
    if p.exists():
        ex = pd.read_csv(p)
        scored = ex[ex.result.notna()] if "result" in ex else ex.iloc[0:0]
        live["exits"] = {
            "orders": int(len(ex)),
            "contracts": int(_num(ex.contracts).sum()),
            "expected_gain": round(float(_num(ex.expected_gain).sum()), 2),
            "scored": int(len(scored)),
            "hold_would_have_made": round(
                float(
                    (
                        _num(scored.hold_profit_per_contract) * _num(scored.contracts)
                    ).sum()
                ),
                2,
            )
            if len(scored)
            else None,
        }
    return live


def main():
    now = datetime.now(UTC)
    ledgers, daily = [], []
    other = list(OTHER_SERIES) + [
        x + "+model" for x in OTHER_SERIES if x in MODEL_SERIES
    ]
    p = Path("outputs/paper_markets/orders.csv")
    if p.exists():
        rows, days = ledger_rows(pd.read_csv(p), "series", expected=other)
        for r in rows:
            r["group"] = "other markets (model-free unless +model)"
        ledgers += rows
        daily += days
    p = Path("outputs/paper_maker/orders.csv")
    if p.exists():
        rows, days = ledger_rows(
            pd.read_csv(p), "rule", lambda k: f"NBA {k}", expected=ALL_RULES
        )
    else:
        rows, days = ledger_rows(
            pd.DataFrame(
                columns=["rule", "status", "filled_contracts", "profit", "day"]
            ),
            "rule",
            lambda k: f"NBA {k}",
            expected=ALL_RULES,
        )
        for r in rows:
            r["group"] = "NBA props"
        ledgers += rows
        daily += days
    live = live_status(Path("outputs/live"))
    live_nfl = live_status(Path("outputs/live/nfl"))
    scan = []
    p = Path("outputs/scan/table.csv")
    if p.exists():
        t = pd.read_csv(p)
        t = t[t.no_fills.notna() & (t.no_fills >= 300)].sort_values(
            "no_t", ascending=False
        )
        for r in t.head(15).itertuples():
            scan.append(
                {
                    "series": r.series,
                    "fills": int(r.no_fills),
                    "cents": round(float(r.no_c), 2),
                    "t": round(float(r.no_t), 2),
                    "passes": bool(r.passes)
                    if isinstance(r.passes, (bool, np.bool_))
                    else str(r.passes) == "True",
                }
            )
    research = {}
    p = Path("outputs/bookline/results.json")
    if p.exists():
        b = json.loads(p.read_text())
        rows = {r["label"][:2]: r for r in b.get("rows", []) if r.get("fills")}
        research["bookline"] = {
            "games": b.get("games_sampled"),
            "coverage": b.get("coverage_of_fills"),
            "all": rows.get("1."),
            "veto": rows.get("3."),
            "removed": rows.get("3b"),
            "model": rows.get("5."),
            "gain": b.get("gain_cents_vs_covered"),
            "difference": b.get("veto_minus_covered"),
            "pass": b.get("pass"),
        }
    total_profit = round(sum(r["profit"] for r in ledgers), 2)
    with_fills = [r for r in ledgers if r["fills"] >= 5]
    best = (
        max(with_fills, key=lambda r: r["cents_per_contract"] or -1e9)
        if with_fills
        else None
    )
    worst = (
        min(with_fills, key=lambda r: r["cents_per_contract"] or 1e9)
        if with_fills
        else None
    )
    summary = {
        "updated_utc": now.isoformat(timespec="minutes"),
        "paper": {
            "ledgers": ledgers,
            "daily": daily,
            "total_profit": total_profit,
            "total_fills": int(sum(r["fills"] for r in ledgers)),
            "best": best["ledger"] if best else None,
            "worst": worst["ledger"] if worst else None,
        },
        "live": live,
        "live_nfl": live_nfl,
        "research": research,
        "scan_top": scan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"{len(ledgers)} ledgers, total paper profit ${total_profit}")


if __name__ == "__main__":
    main()
