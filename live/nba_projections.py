"""Model probabilities for every open Kalshi NBA player prop, before tip-off.

Steps: read the box scores (run fetch_data.py first), take a live snapshot of
Kalshi's open prop markets, read today's and tomorrow's games from ESPN
(tip-off, spread, total) and ESPN's injury report, refit the four stat
models on every completed game with a betting line, and price each market's
line.

Writes outputs/projections/ladders/<UTC timestamp>.csv and
outputs/projections/latest.csv, adds the positions the 2026-27 rule takes
to the paper ledger (outputs/paper/ledger.csv), and writes the day's bet card
(outputs/bets/latest.md). The rule is unproven: it came from looking at
2025-26 after the pre-registered test failed, and is being paper traded.

Run from the repository root: python run_projections.py
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from kalshi.api import fee_per_contract, snapshot
from kalshi.paper_props import COLUMNS as MAKER_COLUMNS
from kalshi.paper_props import RULES, post_orders
from kalshi.paper_props import empty_ledger as maker_ledger
from kalshi.strategy import MONEY_SEASON_TYPES
from markets.nba.evaluate import FIRST_TRAIN_SEASON
from markets.nba.features import build_features, model_rows
from markets.nba.kalshi import SERIES, normalize_name, parse
from markets.nba.live import injuries, scoreboard, upcoming_rows
from markets.nba.marginals import STATS, StatModel
from markets.nba.market import espn_team
from markets.nba.paper import (
    LEDGER_COLUMNS,
    START_BANKROLL,
    bankroll,
    eastern_day,
    empty_ledger,
    new_positions,
)

OUT = Path("outputs/projections")
LEDGER = Path("outputs/paper/ledger.csv")
MAKER_LEDGER = Path("outputs/paper_maker/orders.csv")
BETS = Path("outputs/bets")


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
    )[["game_id", "game_date", "home", "away", "tip_utc", "season_type"]].rename(
        columns={"home": "home_espn", "away": "away_espn"}
    )
    m = m.merge(games, on=["game_date", "home_espn", "away_espn"], how="inner")
    ids = player_ids(players)
    m["player_id"] = m.player_key.map(ids.player_id)
    return m


def price_markets(players, games, schedule, markets, report, now=None):
    """Model price for each open market on a scheduled game, with edges vs Kalshi."""
    matched = match_open_markets(markets, schedule, players)
    listed = (
        matched[["game_id", "team_espn", "player_id"]]
        .rename(columns={"team_espn": "team"})
        .drop_duplicates()
    )
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
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    priced["hours_to_tip"] = (priced.tip_utc - now).dt.total_seconds() / 3600
    status = report.dropna(subset=["player_id"]).drop_duplicates("player_id")
    status = status.set_index(status.player_id.astype("int64")).status
    priced["injury_status"] = priced.player_id.map(status).fillna("")
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
    "hours_to_tip",
    "injury_status",
    "teammates_out_minutes",
    "game_id",
    "player_id",
    "season_type",
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
    markets = parse(snapshot(SERIES, now=now))
    if schedule.empty or markets.empty:
        print(f"Nothing to price: {len(schedule)} games, {len(markets)} markets")
        return
    priced, matched = price_markets(
        players, games, schedule, markets, injuries(), now=now
    )
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
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else empty_ledger()
    ledger["day"] = ledger.day.astype(str)
    added = new_positions(priced, ledger, now)
    added["day"] = added.day.astype(str)
    ledger = pd.concat([ledger, added], ignore_index=True)[LEDGER_COLUMNS]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER, index=False)
    write_bet_card(ledger, now)
    print(f"Paper positions added: {len(added)}")
    makers = pd.read_csv(MAKER_LEDGER) if MAKER_LEDGER.exists() else maker_ledger()
    makers["day"] = makers.day.astype(str)
    for rule in RULES:
        posted = post_orders(priced, makers, now, rule)
        makers = pd.concat([makers, posted], ignore_index=True)[MAKER_COLUMNS]
        print(f"Paper resting orders posted ({rule}): {len(posted)}")
    MAKER_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    makers.to_csv(MAKER_LEDGER, index=False)


HEADER = (
    "| Tip-off (ET) | Player | Bet | Price | Model | Edge | Contracts | Risk | Mode |"
)


def mode(season_type):
    """Regular season: paper. Play-in and playoffs: the 2027 playoff test."""
    return "playoff test" if season_type in MONEY_SEASON_TYPES else "paper"


def write_bet_card(ledger, now):
    """Today's paper positions as a readable page (outputs/bets/latest.md)."""
    today = str(eastern_day(pd.Series([now])).iloc[0])
    day = ledger[ledger.day.astype(str) >= today].copy()
    lines = [
        f"# Bet card, {today}",
        "",
        (
            "**Unproven rule.** In 2025-26 it broke even in the regular season and "
            "made its money in the playoffs, a finding made after the fact (see "
            "RESEARCH_LOG.md). Regular-season positions are paper only. "
            f"Paper bankroll: ${bankroll(ledger, START_BANKROLL):,.2f}."
        ),
        "",
    ]
    if day.empty:
        lines.append("No positions yet today.")
    else:
        lines += [
            HEADER,
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        day["tip"] = pd.to_datetime(day.tip_utc, utc=True).dt.tz_convert(
            "America/New_York"
        )
        for r in day.sort_values(["tip", "player"]).itertuples():
            threshold = int(float(r.line) + 0.5)
            bet = f"{r.side.upper()} {threshold}+ {r.stat}"
            model = float(r.p_model) if r.side == "yes" else 1 - float(r.p_model)
            lines.append(
                f"| {r.tip:%a %-I:%M %p} | {r.player} | {bet} | {100 * float(r.cost):.0f}¢ "
                f"| {100 * model:.0f}% | {100 * float(r.edge):.0f}¢ | {int(r.contracts)} "
                f"| ${float(r.risked):.2f} | {mode(r.season_type)} |"
            )
    BETS.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (BETS / "latest.md").write_text(text)
    (BETS / f"{today}.md").write_text(text)


if __name__ == "__main__":
    main()
