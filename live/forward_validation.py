"""Read-only, pre-registered forward execution validation.

This module consumes locally recorded *real* orders and fills. It does not
place, size, cancel, or modify orders. Results are diagnostic, not an
instruction to change capital or trading rules.

python -m live.forward_validation --folder outputs/live
python -m live.forward_validation --folder outputs/live/nfl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINT_FILL_EVENTS = 500
REVIEW_DAYS = 30
MIN_DATA_COVERAGE = 0.95


def _num(frame, column, default=np.nan):
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _timestamp(frame, column):
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def _fill_events(fills, settled_order_ids):
    """Count distinct exchange-confirmed buy fill records with settled results."""
    if fills is None or fills.empty or "order_id" not in fills:
        return 0
    f = fills[fills.order_id.astype(str).isin(set(settled_order_ids))].copy()
    if "action" in f:
        f = f[f.action.fillna("").str.lower().eq("buy")]
    if "side" in f:
        f = f[f.side.fillna("").str.lower().eq("no")]
    for key in ("fill_id", "trade_id"):
        if key in f and f[key].notna().any():
            f = f.drop_duplicates(key)
            break
    return int(len(f))


def _clustered_t(frame, mean_cents):
    """Game-clustered t for contract-weighted profit, rather than per-fill IID."""
    if frame.empty or "game_id" not in frame:
        return None, 0
    d = frame.copy()
    d["game_id"] = d.game_id.fillna(d.ticker).astype(str)
    grouped = d.groupby("game_id").agg(
        pnl=("settled_profit", "sum"), n=("settled_contracts", "sum")
    )
    games = len(grouped)
    n = grouped.n.sum()
    if games < 2 or n <= 0:
        return None, games
    mean = mean_cents / 100
    scores = grouped.pnl - mean * grouped.n
    se = np.sqrt(games / (games - 1) * np.square(scores).sum()) / n
    return (round(float(mean / se), 2) if se > 0 else None), games


def scorecard(report, fills=None, now=None):
    """Build a reproducible checkpoint with explicit missing-data coverage.

    Capture uses only orders whose entire qualifying-volume window has been
    recorded. Exit-traded markets are excluded from hold-to-settlement P&L
    rather than pretending the original buy remained held to settlement.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    base = {
        "as_of_utc": now.isoformat(),
        "checkpoint_fill_events": CHECKPOINT_FILL_EVENTS,
        "review_days": REVIEW_DAYS,
        "status": "awaiting_live_orders",
        "automatic_trade_changes": False,
        "eligible_real_orders": 0,
    }
    if report is None or report.empty:
        return base
    d = report.drop_duplicates("order_id").copy()
    d["order_id"] = d.order_id.astype(str)
    d["accepted"] = _timestamp(d, "accept_utc").fillna(_timestamp(d, "placed_utc"))
    d = d[d.accepted.notna()].copy()
    if d.empty:
        base["status"] = "missing_acceptance_timestamps"
        return base
    started = d.accepted.min()
    elapsed_days = max(0.0, (now - started).total_seconds() / 86400)
    d["expiration"] = _timestamp(d, "expiration_utc")
    matured = d.expiration.notna() & (d.expiration <= now)
    d["filled"] = _num(d, "filled", 0).fillna(0)
    d["settled_contracts"] = _num(d, "settled_contracts")
    d["settled_profit"] = _num(d, "settled_profit")
    d["qualifying_volume"] = _num(d, "qualifying_volume")
    if "qualifying_complete" in d:
        complete = d.qualifying_complete.astype(str).str.lower().eq("true")
    else:
        complete = pd.Series(False, index=d.index)
    eligible_volume = matured & complete & (d.qualifying_volume > 0)
    impossible_volume = eligible_volume & (d.filled > d.qualifying_volume + 1e-9)
    valid_volume = eligible_volume & ~impossible_volume
    capture_denominator = float(d.loc[valid_volume, "qualifying_volume"].sum())
    capture_numerator = float(d.loc[valid_volume, "filled"].sum())
    pooled_capture = (
        round(capture_numerator / capture_denominator, 4)
        if capture_denominator > 0
        else None
    )
    mature_count = int(matured.sum())
    complete_count = int((matured & complete).sum())
    capture_coverage = complete_count / mature_count if mature_count else None

    exited = set()
    if fills is not None and len(fills) and "ticker" in fills and "action" in fills:
        exits = fills[fills.action.fillna("").str.lower().eq("sell")]
        exited = set(exits.ticker.astype(str))
    if "ticker" in d:
        has_exit = d.ticker.astype(str).isin(exited)
    else:
        has_exit = pd.Series(False, index=d.index)
    settled = (
        matured
        & (d.filled > 0)
        & d.settled_profit.notna()
        & (d.settled_contracts > 0)
        & ~has_exit
    )
    settled_ids = d.loc[settled, "order_id"]
    fill_events = _fill_events(fills, settled_ids)
    pnl = d.loc[settled].copy()
    contracts = float(pnl.settled_contracts.sum())
    dollars = float(pnl.settled_profit.sum())
    cents = round(100 * dollars / contracts, 2) if contracts else None
    t_stat, games = _clustered_t(pnl, cents) if cents is not None else (None, 0)
    unknown_outcome = matured & (d.filled > 0) & ~has_exit & ~settled
    pnl_coverage = (
        int((settled | (matured & (d.filled == 0))).sum())
        / int((matured & ~has_exit).sum())
        if (matured & ~has_exit).sum()
        else None
    )
    eligible_known = matured & ~has_exit & ~unknown_outcome
    profit_per_eligible = (
        round(float(d.loc[eligible_known, "settled_profit"].fillna(0).sum())
              / int(eligible_known.sum()), 4)
        if eligible_known.sum()
        else None
    )
    if not pnl.empty and "day" in pnl:
        daily = pnl.groupby("day").settled_profit.sum().sort_index()
        cumulative = daily.cumsum()
        drawdown = float((cumulative.cummax().clip(lower=0) - cumulative).max())
    else:
        drawdown = None
    queue_buckets = {}
    queue = _num(d, "queue_ahead_at_start")
    for name, mask in (
        ("0", queue.eq(0)),
        ("1-20", queue.gt(0) & queue.le(20)),
        ("21-100", queue.gt(20) & queue.le(100)),
        ("100+", queue.gt(100)),
    ):
        scope = matured & mask
        queue_buckets[name] = {
            "orders": int(scope.sum()),
            "fill_probability": (
                round(float((d.loc[scope, "filled"] > 0).mean()), 3)
                if scope.any() else None
            ),
        }

    enough_time = elapsed_days >= REVIEW_DAYS
    enough_fills = fill_events >= CHECKPOINT_FILL_EVENTS
    complete_enough = (
        capture_coverage is not None and capture_coverage >= MIN_DATA_COVERAGE
        and pnl_coverage is not None and pnl_coverage >= MIN_DATA_COVERAGE
        and not impossible_volume.any()
        and not has_exit.any()
    )
    status = (
        "checkpoint_ready_for_human_review" if enough_time and enough_fills
        and complete_enough
        else "data_incomplete" if enough_time and enough_fills
        else "collecting"
    )
    base.update(
        status=status,
        started_utc=started.isoformat(),
        elapsed_days=round(elapsed_days, 1),
        eligible_real_orders=int(len(d)),
        completed_orders=mature_count,
        orders_with_fills=int((d.filled > 0).sum()),
        settled_buy_fill_events=fill_events,
        settled_contracts=round(contracts, 3),
        settled_profit_dollars=round(dollars, 2),
        cents_per_settled_contract=cents,
        game_clustered_t=t_stat,
        independent_games=games,
        max_settlement_day_drawdown_dollars=round(drawdown, 2)
        if drawdown is not None else None,
        profit_per_matured_eligible_order_dollars=profit_per_eligible,
        capture_volume_weighted=pooled_capture,
        capture_contracts=round(capture_numerator, 3),
        qualifying_contracts=round(capture_denominator, 3),
        mature_volume_coverage=round(capture_coverage, 3)
        if capture_coverage is not None else None,
        mature_profit_coverage=round(pnl_coverage, 3)
        if pnl_coverage is not None else None,
        impossible_capture_rows=int(impossible_volume.sum()),
        missing_settlement_orders=int(unknown_outcome.sum()),
        exited_market_orders_excluded=int(has_exit.sum()),
        queue_fill_probability=queue_buckets,
        flow_assumption=0.50,
        flow_below_25pct_flag=(pooled_capture < 0.25)
        if pooled_capture is not None else None,
        notes=(
            "No capital or rule changes are automatic. P&L excludes markets "
            "with sell fills; reconcile exits before treating totals as "
            "strategy-wide realized income. Capture excludes unfinished "
            "windows, invalid denominators, and incomplete observations."
        ),
    )
    return base


def write_outputs(folder, report, fills=None, now=None):
    """Write only derived reports; never interact with the trading exchange."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = scorecard(report, fills, now)
    (folder / "forward_validation.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Forward validation — actual exchange execution",
        "",
        "Diagnostic only. No automated trading or position-size decisions.",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    for key, value in data.items():
        if key in ("queue_fill_probability", "notes"):
            continue
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines += [
        "",
        "## Queue depth versus eventual fill",
        "",
        "| Contracts ahead | Mature orders | Fill probability |",
        "|---|---:|---:|",
    ]
    for label, metrics in data.get("queue_fill_probability", {}).items():
        lines.append(
            f"| {label} | {metrics['orders']} | {metrics['fill_probability']} |"
        )
    lines += ["", data.get("notes", ""), ""]
    (folder / "forward_validation.md").write_text("\n".join(lines))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="outputs/live")
    args = parser.parse_args()
    folder = Path(args.folder)
    report_path = folder / "execution.csv"
    report = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
    fills_path = folder / "fills.csv"
    fills = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()
    print(json.dumps(write_outputs(folder, report, fills), indent=2))


if __name__ == "__main__":
    main()
