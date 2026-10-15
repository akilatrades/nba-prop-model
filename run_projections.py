"""Model probabilities for every open Kalshi NBA player prop, before tip-off.

Steps: read the box scores (run fetch_data.py first), take a live snapshot of
Kalshi's open prop markets, read today's and tomorrow's games from ESPN
(tip-off, spread, total) and ESPN's injury report, refit the four stat
models on every completed game with a betting line, and price each market's
line.

Writes outputs/projections/ladders/<UTC timestamp>.csv and
outputs/projections/latest.csv. Edges are for paper trading only until the
market test in RESEARCH_LOG.md has been run and passed.

Run from the repository root: python run_projections.py
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nba_model.evaluate import FIRST_TRAIN_SEASON
from nba_model.features import build_features, model_rows
from nba_model.kalshi import fee_per_contract, normalize_name, parse, snapshot
from nba_model.live import injuries, scoreboard, upcoming_rows
from nba_model.marginals import STATS, StatModel
from nba_model.market import espn_team

OUT = Path("outputs/projections")


def player_ids(players):
    """Normalized name -> most recent player_id with that name."""
    latest = players.sort_values("tip_utc").drop_duplicates("player_id", keep="last")
    latest = latest.assign(player_key=latest.player.map(normalize_name))
    return latest.drop_duplicates("player_key", keep="last").set_index("player_key")


def match_open_markets(markets, schedule, players):
    m = markets.copy()
    m["home_espn"] = m.home.map(espn_team)
    m["away_espn"] = m.away.map(espn_team)
    m["team_espn"] = m.team.map(espn_team)
    games = schedule.assign(
        game_date=schedule.tip_utc.dt.tz_convert("America/New_York").dt.date
    )[["game_id", "game_date", "home", "away", "tip_utc"]].rename(
        columns={"home": "home_espn", "away": "away_espn"}
    )
    m = m.merge(games, on=["game_date", "home_espn", "away_espn"], how="inner")
    ids = player_ids(players)
    m["player_id"] = m.player_key.map(ids.player_id)
    return m


def price_markets(players, games, schedule, markets, report):
    """Model price for each open market on a scheduled game, with edges vs Kalshi."""
    matched = match_open_markets(markets, schedule, players)
    listed = matched.rename(columns={"team_espn": "team"})[
        ["game_id", "team", "player_id"]
    ].drop_duplicates()
    upcoming = upcoming_rows(players, schedule, report, listed)
    all_games = pd.concat(
        [games[~games.game_id.isin(schedule.game_id)], schedule], ignore_index=True
    )
    rows = build_features(pd.concat([players, upcoming], ignore_index=True), all_games)
    eligible = model_rows(rows)
    train = eligible[
        (eligible.season >= FIRST_TRAIN_SEASON)
        & ~eligible.game_id.isin(schedule.game_id)
    ]
    target = eligible[eligible.game_id.isin(schedule.game_id)].set_index(
        ["game_id", "player_id"]
    )
    out = []
    for stat, spec in STATS.items():
        model = StatModel(spec).fit(train)
        ask = matched[(matched.stat == stat) & matched.player_id.notna()].copy()
        ask["player_id"] = ask.player_id.astype("int64")
        keys = pd.MultiIndex.from_frame(ask[["game_id", "player_id"]])
        ask = ask[keys.isin(target.index)]
        if ask.empty:
            continue
        feats = target.loc[pd.MultiIndex.from_frame(ask[["game_id", "player_id"]])]
        feats = feats.reset_index()
        mean = model.mean(feats)
        ask["model_mean"] = mean
        ask["p_model"] = model.prob_over(feats, ask.line.to_numpy(float), mean)
        ask["teammates_out_minutes"] = feats.vacated_share_minutes.to_numpy() * 240
        out.append(ask)
    if not out:
        return pd.DataFrame(columns=COLUMNS), matched
    priced = pd.concat(out, ignore_index=True)
    yes_cost, no_cost = priced.yes_ask_dollars, 1 - priced.yes_bid_dollars
    priced["edge_yes"] = priced.p_model - yes_cost - fee_per_contract(yes_cost)
    priced["edge_no"] = (1 - priced.p_model) - no_cost - fee_per_contract(no_cost)
    priced["paper_side"] = np.select(
        [priced.edge_yes >= 0.05, priced.edge_no >= 0.05], ["yes", "no"], default=""
    )
    priced = priced[COLUMNS].sort_values(["tip_utc", "event_ticker", "player", "stat"])
    return priced, matched


COLUMNS = [
    "snapshot_utc",
    "tip_utc",
    "event_ticker",
    "ticker",
    "player",
    "team",
    "stat",
    "line",
    "model_mean",
    "p_model",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "edge_yes",
    "edge_no",
    "paper_side",
    "teammates_out_minutes",
]


def main():
    now = datetime.now(UTC)
    players = pd.read_parquet("data/player_games.parquet")
    games = pd.read_parquet("data/games.parquet")
    season = now.year + 1 if now.month >= 8 else now.year
    eastern = pd.Timestamp(now).tz_convert("America/New_York")
    days = [eastern.date(), (eastern + timedelta(days=1)).date()]
    schedule = pd.concat(
        [scoreboard(f"{d:%Y%m%d}", season) for d in days], ignore_index=True
    )
    schedule = schedule[~schedule.completed & (schedule.tip_utc > now)]
    markets = parse(snapshot(now=now))
    if schedule.empty or markets.empty:
        print(f"Nothing to price: {len(schedule)} games, {len(markets)} markets")
        return
    priced, matched = price_markets(players, games, schedule, markets, injuries())
    folder = OUT / "ladders"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now:%Y-%m-%dT%H%MZ}.csv"
    priced.round(4).to_csv(path, index=False)
    priced.round(4).to_csv(OUT / "latest.csv", index=False)
    unpriced = matched.player_id.isna().sum()
    print(
        f"Priced {len(priced)} of {len(markets)} open markets for "
        f"{len(schedule)} games ({unpriced} with no player match); saved {path}"
    )


if __name__ == "__main__":
    main()
