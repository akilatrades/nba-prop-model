"""Pre-game features built only from games played before the one being predicted.

Player history uses only games the player actually played:

- **Minutes and form:** exponentially weighted averages with a half-life of
  HALF_LIFE games (minutes, points, rebounds, assists, threes, starts).
- **Per-minute rates:** totals over the last RATE_WINDOW games played divided
  by minutes, shrunk toward the position's rate from earlier seasons by
  SHRINK_MINUTES of average play, so a hot week off the bench does not
  dominate.
- **Absence:** how many of his team's games he has missed since he last
  played (players back from injury often play limited minutes).

Team and game context:

- **Betting line:** the team's implied points (closing total and spread) and
  the size of the spread (big favourites rest starters late).
- **Opponent:** what the opponent has allowed per game in each stat, relative
  to the league, weighted toward recent games.
- **Rest:** back-to-backs.
- **Teammates out and back:** the share of the team's recent minutes,
  points, rebounds, assists and threes that belongs to regulars (15+
  minutes a game) who played in the team's last three games but not this
  one, split by the absent player's position group; and the share belonging
  to regulars returning after missing the team's previous game. In the backtest "not playing" is
  read from the box score, which a final injury report shows in most cases
  but not for late scratches.
"""

import numpy as np
import pandas as pd

HALF_LIFE = 5  # games, for minutes and recent form
RATE_WINDOW = 30  # games, for per-minute rates
SHRINK_MINUTES = 150  # minutes of average play added to each rate
TEAM_HALF_LIFE = 10  # team games, for team volume
DEFENSE_HALF_LIFE = 15  # team games, for what an opponent allows
LEAGUE_HALF_LIFE = 300  # team-games, for the league average
LOOKBACK_GAMES = 3  # teammates who played in any of the team's last 3 games
MAX_VACATED = 0.6
REGULAR_MINUTES = 15  # only regulars count as teammates out or back
MAX_GAMES_MISSED = 20
STATS = ["points", "rebounds", "assists", "threes"]
RATE_STATS = STATS + ["fg3a", "fga", "fta"]
FORM = ["minutes"] + STATS + ["starter"]
SHARE_STATS = ["minutes"] + STATS


def _ewm(frame, column, key, half_life, shift):
    grouped = frame.groupby(key, sort=False)[column]
    if shift:
        return grouped.transform(
            lambda s: s.shift().ewm(halflife=half_life, min_periods=1).mean()
        )
    return grouped.transform(lambda s: s.ewm(halflife=half_life, min_periods=1).mean())


def _rolling_sum(frame, column, key, window):
    return frame.groupby(key, sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=1).sum()
    )


def played_rows(players):
    return players[(~players.did_not_play) & (players.minutes > 0)]


def appearing(players):
    """Played, or (for games not yet played) expected to play."""
    mask = (~players.did_not_play) & (players.minutes > 0)
    if "expected" in players:
        mask |= players.expected.fillna(False).astype(bool)
    return mask


def player_history(players):
    """For each game a player played: his history *including* that game.

    Joined to later games with an as-of merge, so every prediction only sees
    games strictly before it.
    """
    p = played_rows(players).sort_values("tip_utc").copy()
    p["starter"] = p.starter.astype(float)
    out = p[["player_id", "tip_utc", "season", "position", "minutes"]].copy()
    out = out.rename(columns={"minutes": "last_minutes"})
    out["last_tip"] = p.tip_utc
    out["n_prior"] = p.groupby("player_id").cumcount() + 1
    for column in FORM:
        out[f"ew_{column}"] = _ewm(p, column, "player_id", HALF_LIFE, shift=False)
    for column in RATE_STATS + ["minutes"]:
        out[f"sum_{column}"] = _rolling_sum(p, column, "player_id", RATE_WINDOW)
    out["sum_threes_att"] = out.sum_fg3a
    return out


def position_rates(players):
    """League per-minute rates by position group from seasons before each season."""
    p = played_rows(players)
    totals = p.groupby(["position", "season"])[RATE_STATS + ["minutes"]].sum()
    rows = []
    for position, by_season in totals.groupby(level="position"):
        by_season = by_season.droplevel(0).sort_index()
        earlier = by_season.cumsum().shift()
        earlier.iloc[0] = by_season.iloc[0]  # first season: its own average
        for season, row in earlier.iterrows():
            rates = {"position": position, "season": season}
            for stat in RATE_STATS:
                rates[f"mu_{stat}"] = row[stat] / row["minutes"]
            rows.append(rates)
    return pd.DataFrame(rows)


def team_games(players, games):
    """One row per team per game: totals, rest, and prior-only team context."""
    p = played_rows(players)
    totals = (
        p.groupby(["game_id", "team"])[STATS + ["fg3a", "fga", "fta", "tov", "oreb"]]
        .sum()
        .reset_index()
    )
    keys = players.drop_duplicates(["game_id", "team"])[
        ["game_id", "team", "opp", "season", "tip_utc", "home"]
    ]
    t = keys.merge(totals, on=["game_id", "team"], how="left")
    t = t.sort_values("tip_utc").reset_index(drop=True)
    t["team_game"] = t.groupby(["team", "season"]).cumcount()
    local_date = t.tip_utc.dt.tz_convert("America/New_York").dt.normalize()
    previous = t.groupby(["team", "season"])["tip_utc"].shift()
    previous_date = previous.dt.tz_convert("America/New_York").dt.normalize()
    t["rest_days"] = ((local_date - previous_date).dt.days).fillna(7).clip(upper=7)
    for stat in STATS:
        t[f"ew_team_{stat}"] = _ewm(t, stat, "team", TEAM_HALF_LIFE, shift=True)
    # What each team allowed: the opponent's totals in the same game.
    opp = t[["game_id", "team"] + STATS].rename(
        columns={"team": "opp", **{s: f"allowed_{s}" for s in STATS}}
    )
    t = t.merge(opp, on=["game_id", "opp"], how="left")
    t = t.sort_values("tip_utc").reset_index(drop=True)
    for stat in STATS:
        t[f"def_{stat}"] = _ewm(t, f"allowed_{stat}", "team", DEFENSE_HALF_LIFE, True)
        league = t[stat].shift().ewm(halflife=LEAGUE_HALF_LIFE, min_periods=1).mean()
        t[f"league_{stat}"] = league
    # Opponent context for each team-game: what the opponent allows.
    defense = t[["game_id", "team"] + [f"def_{s}" for s in STATS]].rename(
        columns={"team": "opp", **{f"def_{s}": f"opp_def_{s}" for s in STATS}}
    )
    t = t.merge(defense, on=["game_id", "opp"], how="left")
    for stat in STATS:
        t[f"opp_factor_{stat}"] = t[f"opp_def_{stat}"] / t[f"league_{stat}"]
    lines = games[["game_id", "home_spread", "total"]]
    t = t.merge(lines, on="game_id", how="left")
    team_spread = np.where(t.home, t.home_spread, -t.home_spread)
    t["team_spread"] = team_spread  # negative: favoured
    t["implied_points"] = (t.total - team_spread) / 2
    keep = [
        "game_id",
        "team",
        "team_game",
        "rest_days",
        "team_spread",
        "implied_points",
        "total",
    ]
    keep += [f"ew_team_{s}" for s in STATS]
    keep += [f"opp_factor_{s}" for s in STATS]
    return t[keep]


def _share_frame(rows, history, team_context):
    """Attach a player's history as of a team-game, and his share of team volume."""
    rows = rows.sort_values("tip_utc")
    hist = history.sort_values("tip_utc")
    columns = ["player_id", "tip_utc", "position"] + [f"ew_{s}" for s in SHARE_STATS]
    merged = pd.merge_asof(
        rows,
        hist[columns].rename(columns={"position": "hist_position"}),
        on="tip_utc",
        by="player_id",
        allow_exact_matches=False,
    )
    team_volume = team_context[["game_id", "team"] + [f"ew_team_{s}" for s in STATS]]
    merged = merged.merge(team_volume, on=["game_id", "team"], how="left")
    merged["share_minutes"] = merged.ew_minutes / 240.0
    for stat in STATS:
        merged[f"share_{stat}"] = merged[f"ew_{stat}"] / merged[f"ew_team_{stat}"]
    return merged


def teammates_out_and_back(players, history, team_context):
    """Vacated shares (same position group / other) and returning shares."""
    p = played_rows(players)
    team_index = team_context[["game_id", "team", "team_game"]]
    played = p[["game_id", "team", "season", "player_id"]].merge(
        team_index, on=["game_id", "team"]
    )
    schedule = (
        players.drop_duplicates(["game_id", "team"])[
            ["game_id", "team", "season", "tip_utc"]
        ]
        .merge(team_index, on=["game_id", "team"])
        .rename(columns={"game_id": "target_game"})
    )
    candidates = []
    for k in range(1, LOOKBACK_GAMES + 1):
        c = played[["team", "season", "player_id", "team_game"]].copy()
        c["team_game"] = c.team_game + k
        candidates.append(c)
    candidates = pd.concat(candidates).drop_duplicates()
    candidates = candidates.merge(schedule, on=["team", "season", "team_game"])
    now = players.loc[appearing(players), ["game_id", "team", "season", "player_id"]]
    now = now.merge(team_index, on=["game_id", "team"])
    today = now[["game_id", "team", "player_id"]].rename(
        columns={"game_id": "target_game"}
    )
    out = candidates.merge(
        today, on=["target_game", "team", "player_id"], how="left", indicator=True
    )
    out = out[out._merge == "left_only"].drop(columns="_merge")
    out = out.rename(columns={"target_game": "game_id"})
    out = _share_frame(out, history, team_context)
    out = out[out.ew_minutes >= REGULAR_MINUTES]
    out["group"] = out.hist_position
    share_columns = [f"share_{s}" for s in SHARE_STATS]
    vacated = out.groupby(["game_id", "team", "group"])[share_columns].sum()
    vacated = vacated.reset_index()

    # Regulars back after missing the team's previous game.
    previous = played[["team", "season", "player_id", "team_game"]].copy()
    previous["team_game"] = previous.team_game + 1
    back = now.merge(
        previous,
        on=["team", "season", "player_id", "team_game"],
        how="left",
        indicator=True,
    )
    back = back[back._merge == "left_only"].drop(columns="_merge")
    back = back.merge(players[["game_id", "player_id", "tip_utc"]], how="left")
    back = _share_frame(back, history, team_context)
    back = back[back.ew_minutes >= REGULAR_MINUTES]
    returning = back[["game_id", "team", "player_id"] + share_columns]
    return vacated, returning


def add_teammate_features(frame, vacated, returning):
    share_columns = [f"share_{s}" for s in SHARE_STATS]
    total = vacated.groupby(["game_id", "team"])[share_columns].sum()
    total = total.add_prefix("vacated_").reset_index()
    same = vacated.rename(columns={"group": "position"})
    same = same.rename(columns={c: f"vacated_same_{c}" for c in share_columns})
    frame = frame.merge(total, on=["game_id", "team"], how="left")
    frame = frame.merge(same, on=["game_id", "team", "position"], how="left")
    for s in SHARE_STATS:
        v, vs = f"vacated_share_{s}", f"vacated_same_share_{s}"
        frame[v] = frame[v].fillna(0.0).clip(upper=MAX_VACATED)
        frame[vs] = frame[vs].fillna(0.0).clip(upper=frame[v])
    team_back = returning.groupby(["game_id", "team"])[share_columns].sum()
    team_back = team_back.add_prefix("returning_").reset_index()
    own = returning.rename(columns={c: f"own_back_{c}" for c in share_columns})
    frame = frame.merge(team_back, on=["game_id", "team"], how="left")
    frame = frame.merge(own, on=["game_id", "team", "player_id"], how="left")
    for s in SHARE_STATS:
        r, o = f"returning_share_{s}", f"own_back_share_{s}"
        frame[r] = (frame[r].fillna(0.0) - frame[o].fillna(0.0)).clip(0, MAX_VACATED)
        frame = frame.drop(columns=o)
    return frame


def _ns(times):
    """UTC times as plain datetime64[ns], which numpy can search reliably."""
    return times.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")


def games_missed(frame, team_context):
    """Team games since the player last played (0 if he played the last one)."""
    tips = team_context[["game_id", "team"]].merge(
        frame[["game_id", "tip_utc"]].drop_duplicates("game_id"), on="game_id"
    )
    by_team = {team: np.sort(_ns(g.tip_utc)) for team, g in tips.groupby("team")}
    out = np.zeros(len(frame))
    team = frame.team.to_numpy()
    tip = _ns(frame.tip_utc)
    last = _ns(frame.last_tip.fillna(frame.tip_utc))
    for t in np.unique(team):
        here = team == t
        series = by_team[t]
        after_last = np.searchsorted(series, last[here], side="right")
        before_now = np.searchsorted(series, tip[here], side="left")
        out[here] = np.maximum(before_now - after_last, 0)
    out = np.minimum(out, MAX_GAMES_MISSED)
    return np.where(pd.isna(frame.last_tip), np.nan, out)


def build_features(players, games):
    """Every player-game row with prior-only features and its outcome columns."""
    players = players.sort_values("tip_utc").reset_index(drop=True)
    history = player_history(players)
    context = team_games(players, games)
    rows = players.copy()
    rows["played"] = appearing(rows)
    hist_cols = [c for c in history.columns if c not in {"season", "position"}]
    rows = pd.merge_asof(
        rows.sort_values("tip_utc"),
        history[hist_cols],
        on="tip_utc",
        by="player_id",
        allow_exact_matches=False,
    )
    rows["n_prior"] = rows.n_prior.fillna(0)
    rates = position_rates(players)
    rows = rows.merge(rates, on=["position", "season"], how="left")
    latest = rates.sort_values("season").groupby("position").last()
    for stat in RATE_STATS:
        mu = f"mu_{stat}"
        rows[mu] = rows[mu].fillna(rows.position.map(latest[mu]))
        numer = rows[f"sum_{stat}"].fillna(0) + SHRINK_MINUTES * rows[mu]
        rows[f"rate_{stat}"] = numer / (rows.sum_minutes.fillna(0) + SHRINK_MINUTES)
    made, att = rows.sum_threes.fillna(0), rows.sum_fg3a.fillna(0)
    rows["three_pct"] = (made + 100 * 0.36) / (att + 100)
    rows = rows.merge(context, on=["game_id", "team"], how="left")
    rows["games_missed"] = games_missed(rows, context)
    vacated, returning = teammates_out_and_back(players, history, context)
    rows = add_teammate_features(rows, vacated, returning)
    rows["playoffs"] = rows.season_type.ne("regular").astype(float)
    return rows.sort_values(["tip_utc", "game_id", "team", "player_id"]).reset_index(
        drop=True
    )


MIN_PRIOR_GAMES = 5
MIN_EW_MINUTES = 10


def model_rows(frame):
    """Rows the model prices: regulars who played, with a betting line."""
    keep = (
        frame.played
        & (frame.n_prior >= MIN_PRIOR_GAMES)
        & (frame.ew_minutes >= MIN_EW_MINUTES)
        & frame.implied_points.notna()
        & frame.opp_factor_points.notna()
        & frame.ew_team_points.notna()
    )
    return frame[keep].copy()
