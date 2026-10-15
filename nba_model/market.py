"""The pre-registered market test: the model against Kalshi's price at tip-off.

Steps (rules fixed in RESEARCH_LOG.md on 2026-10-15, before prices were seen):

1. Join each settled Kalshi market to its game (date and teams) and player
   (name and team) in the box scores; keep markets that settled yes or no.
2. Take the last hourly candle ending at or before tip-off; keep two-sided
   quotes (bid above 0, ask below 1, ask above bid).
3. Price each market with the model refit before that month.
4. Test A: logistic regression of the outcome on logit(mid) and logit(model),
   standard errors clustered by game.
5. Test B: buy yes at the ask or no at 1 - bid when the model's edge after
   fees is at least 0.05; P&L per contract with game-clustered t.
"""

import numpy as np
import pandas as pd

from nba_model.evaluate import monthly_models
from nba_model.kalshi import (
    SERIES,
    SETTLED_FIELDS,
    fee_per_contract,
    normalize_name,
    parse,
)

# Kalshi team codes that differ from ESPN's.
KALSHI_TO_ESPN = {
    "SAS": "SA",
    "NYK": "NY",
    "GSW": "GS",
    "NOP": "NO",
    "UTA": "UTAH",
    "WAS": "WSH",
    "PHO": "PHX",
    "BRK": "BKN",
    "CHO": "CHA",
}
MIN_EDGE = 0.05
STRICT_EDGE = 0.08
SPLIT_DATE = pd.Timestamp("2026-02-01", tz="America/New_York")


def espn_team(code):
    return KALSHI_TO_ESPN.get(code, code)


def load_history(folder):
    markets, candles = [], []
    for series in SERIES:
        m, c = (
            folder / f"markets_{series}.parquet",
            folder / f"candles_{series}.parquet",
        )
        if m.exists():
            markets.append(pd.read_parquet(m))
        if c.exists():
            candles.append(pd.read_parquet(c))
    markets = pd.concat(markets, ignore_index=True) if markets else pd.DataFrame()
    if len(markets):  # re-parse the raw fields with the current parser
        markets = parse(markets[["series", *SETTLED_FIELDS]])
    candles = pd.concat(candles, ignore_index=True) if candles else pd.DataFrame()
    return markets, candles


def match_markets(markets, players):
    """Attach game_id, player_id and tip-off to each market (no prices used)."""
    m = markets.copy()
    m["home_espn"] = m.home.map(espn_team)
    m["away_espn"] = m.away.map(espn_team)
    m["team_espn"] = m.team.map(espn_team)
    m["game_date"] = pd.to_datetime(m.game_date).dt.date
    box = players.copy()
    box["local_date"] = box.tip_utc.dt.tz_convert("America/New_York").dt.date
    games = (
        box[box.home]
        .drop_duplicates("game_id")[
            ["game_id", "local_date", "team", "opp", "tip_utc", "season", "season_type"]
        ]
        .rename(
            columns={"team": "home_espn", "opp": "away_espn", "local_date": "game_date"}
        )
    )
    m = m.merge(games, on=["game_date", "home_espn", "away_espn"], how="left")
    roster = box[["game_id", "team", "player_id", "player"]].copy()
    roster["player_key"] = roster.player.map(normalize_name)
    roster = roster.rename(columns={"team": "team_espn"}).drop(columns="player")
    m = m.merge(roster, on=["game_id", "team_espn", "player_key"], how="left")
    # Fall back to surname + first initial within the team for name variants.
    missing = m.player_id.isna() & m.game_id.notna()
    if missing.any():
        roster["short"] = roster.player_key.map(_short)
        short = roster.drop_duplicates(["game_id", "team_espn", "short"], keep=False)
        fix = m.loc[missing, ["game_id", "team_espn", "player_key"]].copy()
        fix["short"] = fix.player_key.map(_short)
        fix = fix.merge(
            short[["game_id", "team_espn", "short", "player_id"]],
            on=["game_id", "team_espn", "short"],
            how="left",
        )
        m.loc[missing, "player_id"] = fix.player_id.to_numpy()
    return m


def _short(key):
    words = str(key).split()
    return f"{words[0][0]} {words[-1]}" if len(words) >= 2 else str(key)


def price_at_tip(matched, candles, hours_before=0):
    """Last candle ending at or before tip-off (minus ``hours_before``)."""
    c = candles.copy()
    m = matched[["ticker", "tip_utc"]].dropna()
    cutoff = m.tip_utc - pd.Timedelta(hours=hours_before)
    m = m.assign(cutoff_ts=cutoff.map(lambda t: int(t.timestamp())))
    c = c.merge(m[["ticker", "cutoff_ts"]], on="ticker")
    c = c[c.end_ts <= c.cutoff_ts].sort_values("end_ts")
    last = c.drop_duplicates("ticker", keep="last")
    return last[["ticker", "end_ts", "yes_bid", "yes_ask", "price", "volume"]]


def two_sided(frame):
    return (frame.yes_bid > 0) & (frame.yes_ask < 1) & (frame.yes_ask > frame.yes_bid)


def model_prices(rows, requests, season, transform=None):
    """Model probability for each requested (game, player, stat, line)."""
    out = []
    for test, models in monthly_models(rows, [season], transform):
        here = requests[requests.game_id.isin(test.game_id)]
        if here.empty:
            continue
        keyed = test.set_index(["game_id", "player_id"])
        for stat, (model, _) in models.items():
            ask = here[here.stat == stat]
            ask = ask[
                pd.MultiIndex.from_frame(ask[["game_id", "player_id"]]).isin(
                    keyed.index
                )
            ]
            if ask.empty:
                continue
            feats = keyed.loc[pd.MultiIndex.from_frame(ask[["game_id", "player_id"]])]
            feats = feats.reset_index()
            mean = model.mean(feats)
            p = model.prob_over(feats, ask.line.to_numpy(float), mean)
            out.append(
                ask[["ticker"]].assign(
                    p_model=p, model_mean=mean, actual=feats[stat].to_numpy()
                )
            )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def trades(frame, min_edge=MIN_EDGE):
    """Test B's rule: one contract on whichever side clears ``min_edge``."""
    f = frame.copy()
    yes_cost = f.yes_ask
    no_cost = 1 - f.yes_bid
    yes_edge = f.p_model - yes_cost - fee_per_contract(yes_cost)
    no_edge = (1 - f.p_model) - no_cost - fee_per_contract(no_cost)
    f["side"] = np.select(
        [yes_edge >= min_edge, no_edge >= min_edge], ["yes", "no"], default=""
    )
    f = f[f.side != ""].copy()
    cost = np.where(f.side == "yes", yes_cost.loc[f.index], no_cost.loc[f.index])
    won = np.where(f.side == "yes", f.over, 1 - f.over)
    f["cost"] = cost
    f["fee"] = fee_per_contract(cost)
    f["fee_rounded"] = np.ceil(np.round(f.fee * 100, 6)) / 100
    f["pnl"] = won - cost - f.fee
    f["pnl_rounded_fee"] = won - cost - f.fee_rounded
    return f


def clustered_mean_t(values, clusters):
    """Mean and its t statistic with standard errors clustered by ``clusters``."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = values.mean()
    sums = pd.Series(values - mean).groupby(np.asarray(clusters)).sum()
    g = len(sums)
    se = np.sqrt((sums**2).sum() * g / (g - 1)) / n
    return mean, mean / se if se > 0 else np.nan


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def clustered_logit(y, x, clusters, iterations=50):
    """Logistic regression (with intercept) and game-clustered standard errors."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ beta))
        w = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    scores = pd.DataFrame(X * (y - p)[:, None]).groupby(np.asarray(clusters)).sum()
    meat = scores.to_numpy().T @ scores.to_numpy()
    g = len(scores)
    cov = bread @ meat @ bread * g / (g - 1)
    se = np.sqrt(np.diag(cov))
    return beta, se


def test_a(frame):
    beta, se = clustered_logit(
        frame.over,
        np.column_stack([logit(frame.mid), logit(frame.p_model)]),
        frame.game_id,
    )
    return {
        "markets": len(frame),
        "games": frame.game_id.nunique(),
        "coef_market": beta[1],
        "t_market": beta[1] / se[1],
        "coef_model": beta[2],
        "t_model": beta[2] / se[2],
        "passes": bool(beta[2] / se[2] >= 2.0),
    }


def test_b_summary(t, column="pnl"):
    mean, t_stat = clustered_mean_t(t[column], t.game_id)
    return {
        "trades": len(t),
        "games": t.game_id.nunique(),
        "pnl_per_contract": mean,
        "t": t_stat,
        "total": t[column].sum(),
        "avg_cost": t.cost.mean(),
        "share_yes": (t.side == "yes").mean(),
    }


def test_b(frame):
    main = trades(frame, MIN_EDGE)
    strict = trades(frame, STRICT_EDGE)
    early = main[main.tip_utc < SPLIT_DATE]
    late = main[main.tip_utc >= SPLIT_DATE]
    result = {
        "all": test_b_summary(main),
        "before_feb": test_b_summary(early),
        "from_feb": test_b_summary(late),
        "strict_0.08": test_b_summary(strict),
        "rounded_fees": test_b_summary(main, "pnl_rounded_fee"),
    }
    passes = (
        result["all"]["pnl_per_contract"] > 0
        and result["all"]["t"] >= 2.0
        and result["before_feb"]["pnl_per_contract"] > 0
        and result["from_feb"]["pnl_per_contract"] > 0
        and result["strict_0.08"]["pnl_per_contract"] > 0
    )
    return result, bool(passes), main
