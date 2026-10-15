"""The live paper-trading ledger for the 2026-27 rule (see strategy.py).

Each projection run adds the positions the rule would take now to
outputs/paper/ledger.csv, sized for the paper bankroll (it starts at $500
and moves with settled results). After games finish, each position is
settled from Kalshi's own settlement of the market, so a player who sits is
settled at whatever value Kalshi set, exactly as a real position would be.
"""

import numpy as np
import pandas as pd

from nba_model.market import clustered_mean_t
from nba_model.strategy import (
    DAY_CAP,
    STAKE,
    WINDOW_HOURS,
    contracts_for,
    one_per_player,
    order_fee,
    qualifying,
)

START_BANKROLL = 500.0
LEDGER_COLUMNS = [
    "placed_utc",
    "day",
    "tip_utc",
    "hours_to_tip",
    "game_id",
    "player_id",
    "player",
    "team",
    "stat",
    "season_type",
    "ticker",
    "line",
    "side",
    "p_model",
    "yes_bid",
    "yes_ask",
    "cost",
    "edge",
    "contracts",
    "fee",
    "risked",
    "status",
    "result",
    "settlement_value",
    "payout",
    "profit",
]


def empty_ledger():
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def eastern_day(times):
    return pd.to_datetime(times, utc=True).dt.tz_convert("America/New_York").dt.date


def bankroll(ledger, start=START_BANKROLL):
    """Starting bankroll plus every settled position's profit."""
    settled = ledger[ledger.status == "settled"]
    return start + float(pd.to_numeric(settled.profit).sum())


def new_positions(priced, ledger, now, start=START_BANKROLL):
    """Positions the rule takes from this run's prices.

    ``priced`` is the projection output: one row per open market with
    p_model, yes_bid_dollars, yes_ask_dollars, tip_utc, hours_to_tip,
    injury_status, game_id, player_id, stat, line, ticker.
    """
    if priced.empty:
        return empty_ledger()
    f = priced.rename(
        columns={"yes_bid_dollars": "yes_bid", "yes_ask_dollars": "yes_ask"}
    )
    lo, hi = WINDOW_HOURS
    f = f[f.hours_to_tip.between(lo, hi) & f.injury_status.fillna("").eq("")]
    t = one_per_player(qualifying(f))
    taken = set(zip(ledger.game_id, ledger.player_id, ledger.stat))
    t = t[
        np.array(
            [k not in taken for k in zip(t.game_id, t.player_id, t.stat)], dtype=bool
        )
    ]
    if t.empty:
        return empty_ledger()
    t = t.sort_values("edge", ascending=False).copy()
    t["day"] = eastern_day(t.tip_utc)
    bank = bankroll(ledger, start)
    rows = []
    for day, group in t.groupby("day"):
        risked_today = float(
            pd.to_numeric(ledger.loc[ledger.day == day, "risked"]).sum()
        )
        room = max(DAY_CAP * bank - risked_today, 0.0)
        for r in group.itertuples():
            budget = min(STAKE * bank, room)
            n = int(contracts_for(budget, r.cost))
            if n < 1:
                continue
            fee = float(order_fee(n, r.cost))
            risked = n * r.cost + fee
            room -= risked
            rows.append(
                {
                    "placed_utc": now.isoformat(timespec="minutes"),
                    "day": day,
                    "tip_utc": r.tip_utc,
                    "hours_to_tip": round(float(r.hours_to_tip), 2),
                    "game_id": int(r.game_id),
                    "player_id": int(r.player_id),
                    "player": r.player,
                    "team": r.team,
                    "stat": r.stat,
                    "season_type": getattr(r, "season_type", ""),
                    "ticker": r.ticker,
                    "line": r.line,
                    "side": r.side,
                    "p_model": round(float(r.p_model), 4),
                    "yes_bid": r.yes_bid,
                    "yes_ask": r.yes_ask,
                    "cost": round(float(r.cost), 4),
                    "edge": round(float(r.edge), 4),
                    "contracts": n,
                    "fee": fee,
                    "risked": round(risked, 2),
                    "status": "open",
                }
            )
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def settle(ledger, fetch_market):
    """Settle open positions whose Kalshi market has a result.

    ``fetch_market(ticker)`` returns Kalshi's market dict (or None).
    """
    ledger = ledger.copy()
    ledger["result"] = ledger.result.astype(object)
    ledger["status"] = ledger.status.astype(object)
    for column in ["settlement_value", "payout", "profit"]:
        ledger[column] = pd.to_numeric(ledger[column]).astype(float)
    for i in ledger.index[ledger.status == "open"]:
        market = fetch_market(ledger.at[i, "ticker"])
        if not market or market.get("result") not in {"yes", "no", "scalar"}:
            continue
        value = float(market.get("settlement_value_dollars") or 0.0)
        if market["result"] == "yes":
            value = 1.0
        elif market["result"] == "no":
            value = 0.0
        side_value = value if ledger.at[i, "side"] == "yes" else 1 - value
        n = float(ledger.at[i, "contracts"])
        payout = n * side_value
        ledger.at[i, "result"] = market["result"]
        ledger.at[i, "settlement_value"] = value
        ledger.at[i, "payout"] = round(payout, 2)
        ledger.at[i, "profit"] = round(payout - float(ledger.at[i, "risked"]), 2)
        ledger.at[i, "status"] = "settled"
    return ledger


def scorecard(ledger, start=START_BANKROLL):
    settled = ledger[ledger.status == "settled"].copy()
    if settled.empty:
        return {"positions": 0, "open": int((ledger.status == "open").sum())}
    for column in ["profit", "risked", "contracts"]:
        settled[column] = pd.to_numeric(settled[column])
    per_contract = settled.profit / settled.contracts
    mean, t = clustered_mean_t(per_contract, settled.game_id)
    return {
        "positions": len(settled),
        "open": int((ledger.status == "open").sum()),
        "games": int(settled.game_id.nunique()),
        "profit_dollars": round(float(settled.profit.sum()), 2),
        "risked_dollars": round(float(settled.risked.sum()), 2),
        "return_per_dollar_pct": round(
            100 * float(settled.profit.sum() / settled.risked.sum()), 2
        ),
        "cents_per_contract": round(100 * float(mean), 2),
        "t": round(float(t), 2) if np.isfinite(t) else None,
        "bankroll": round(bankroll(ledger, start), 2),
        "players_who_sat": int((settled.result == "scalar").sum()),
    }
