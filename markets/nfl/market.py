"""Compare the model's pre-game prices with Kalshi's, then with what happened.

For each Kalshi prop market the comparison uses:

- the last Kalshi snapshot taken before kickoff (the closest thing to a
  closing price in the log),
- the last model ladder made before kickoff,
- the box-score result.

It then scores both sets of prices against the results and records the paper
trades the walkthrough's rule would have made: buy whichever side the model
prices at least ``MIN_EDGE`` above the ask after fees.
"""

import numpy as np
import pandas as pd

from markets.nfl.kalshi import fee_per_contract, normalize_name, parse

MIN_EDGE = 0.05
OUTCOME_COLUMN = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "rushing_yards": "rushing_yards",
    "skill_tds": "skill_tds",
}


def kickoffs(games):
    """Kickoff time in UTC for each game (schedule times are US Eastern)."""
    local = pd.to_datetime(
        games.gameday + " " + games.gametime.fillna("13:00"), errors="coerce"
    )
    utc = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return pd.DataFrame(
        {"game_id": games.game_id, "gameday": games.gameday, "kickoff_utc": utc}
    )


def last_before_kickoff(frame, time_column, keys):
    frame = frame[frame[time_column] < frame.kickoff_utc]
    return frame.sort_values(time_column).drop_duplicates(keys, keep="last")


def outcomes(players):
    frame = players.copy()
    frame["skill_tds"] = frame.rushing_tds + frame.receiving_tds
    columns = list(OUTCOME_COLUMN.values())
    long = frame.melt(
        id_vars=["game_id", "player_id"], value_vars=columns, var_name="stat"
    )
    return long.rename(columns={"value": "actual"})


def compare(snapshots, ladders, games, players):
    """One row per Kalshi market with model price, market price and result."""
    times = kickoffs(games)

    market = parse(snapshots)
    market["snapshot_utc"] = pd.to_datetime(market.snapshot_utc, utc=True)
    market["gameday"] = market.game_date.astype(str)

    model = ladders.copy()
    model["made_utc"] = pd.to_datetime(model.made_utc, utc=True)
    model["player_key"] = model.player.map(normalize_name)
    model = model.merge(times, on="game_id")
    model = last_before_kickoff(
        model, "made_utc", ["game_id", "player_id", "stat", "line"]
    )

    joined = market.merge(
        model[
            [
                "game_id",
                "gameday",
                "kickoff_utc",
                "player_id",
                "player_key",
                "stat",
                "line",
                "p_over",
                "made_utc",
            ]
        ],
        on=["gameday", "player_key", "stat", "line"],
    )
    joined = last_before_kickoff(joined, "snapshot_utc", ["ticker"])
    joined = joined.merge(
        outcomes(players), on=["game_id", "player_id", "stat"], how="left"
    )
    joined["over"] = (
        (joined.actual > joined.line).astype(float).where(joined.actual.notna())
    )
    return add_paper_trades(joined)


def add_paper_trades(frame):
    """Buy yes or no when the model's price beats the ask by MIN_EDGE after fees."""
    frame = frame.copy()
    yes_ask = frame.yes_ask_dollars
    no_ask = 1 - frame.yes_bid_dollars
    yes_edge = frame.p_over - yes_ask - fee_per_contract(yes_ask)
    no_edge = (1 - frame.p_over) - no_ask - fee_per_contract(no_ask)
    frame["side"] = np.select(
        [yes_edge >= MIN_EDGE, no_edge >= MIN_EDGE], ["yes", "no"], default=""
    )
    price = np.where(
        frame.side == "yes", yes_ask, np.where(frame.side == "no", no_ask, np.nan)
    )
    frame["edge"] = np.where(
        frame.side == "yes", yes_edge, np.where(frame.side == "no", no_edge, np.nan)
    )
    won = np.where(frame.side == "yes", frame.over, 1 - frame.over)
    cost = price + fee_per_contract(price)
    frame["profit_per_contract"] = np.where(frame.side != "", won - cost, np.nan)
    return frame


def summarize(compared):
    settled = compared[compared.over.notna() & compared.yes_bid_dollars.notna()]
    if settled.empty:
        return pd.DataFrame()
    rows = []
    for stat, group in [("all", settled), *settled.groupby("stat")]:
        trades = group[group.side != ""]
        rows.append(
            {
                "stat": stat,
                "markets": len(group),
                "brier_model": ((group.p_over - group.over) ** 2).mean(),
                "brier_market_mid": ((group["mid"] - group.over) ** 2).mean(),
                "avg_model_minus_market": (group.p_over - group["mid"]).mean(),
                "paper_trades": len(trades),
                "paper_profit_per_contract": trades.profit_per_contract.mean(),
                "paper_total_profit": trades.profit_per_contract.sum(),
            }
        )
    return pd.DataFrame(rows)
