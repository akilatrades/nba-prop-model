"""Load the saved data and reshape it into team-game and player-game tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from markets.nfl.odds import no_vig

DATA = Path("data/nfl")
# The schedule keeps a franchise's code at the time; box scores use today's.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
SKILL = ["WR", "TE", "RB", "FB"]


# Injury-report statuses treated as "will not play".
RULED_OUT = ["Out", "Doubtful"]


def load(data=DATA):
    games = pd.read_csv(data / "games.csv")
    players = pd.read_csv(data / "player_week.csv.gz")
    return games, players


def load_injuries(data=DATA):
    injuries = pd.read_csv(data / "injuries.csv.gz")
    injuries["team"] = injuries.team.replace(TEAM_ALIASES)
    return injuries.rename(columns={"gsis_id": "player_id"})


def ruled_out(injuries):
    """Players listed Out or Doubtful on the final injury report for a week."""
    out = injuries[injuries.report_status.isin(RULED_OUT)]
    return out[["season", "week", "team", "player_id"]].drop_duplicates()


def team_games(games):
    """One row per team per game with the pre-game market view of that team."""
    rows = []
    for side, other, sign in [("home", "away", 1), ("away", "home", -1)]:
        rows.append(
            pd.DataFrame(
                {
                    "game_id": games.game_id,
                    "season": games.season,
                    "week": games.week,
                    "team": games[f"{side}_team"].replace(TEAM_ALIASES),
                    "opp": games[f"{other}_team"].replace(TEAM_ALIASES),
                    "qb_id": games[f"{side}_qb_id"],
                    "qb_name": games[f"{side}_qb_name"],
                    # spread_line > 0 means the home team is favored.
                    "implied_total": games.total_line / 2
                    + sign * games.spread_line / 2,
                    "p_win": no_vig(
                        games[f"{side}_moneyline"], games[f"{other}_moneyline"]
                    ),
                    "played": games.home_score.notna(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def team_volume(players):
    """Targets, carries and pass attempts each team actually had in each game."""
    columns = ["targets", "carries", "attempts", "passing_yards", "rushing_yards"]
    return (
        players.groupby(["game_id", "team"])[columns]
        .sum()
        .add_prefix("team_")
        .reset_index()
    )


def player_games(games, players):
    """Played player-games with team context and each player's share of volume."""
    teams = team_games(games)
    frame = players.merge(teams, on=["game_id", "season", "week", "team"])
    frame = frame.merge(team_volume(players), on=["game_id", "team"])
    frame["skill_tds"] = frame.rushing_tds + frame.receiving_tds
    for share, count, total in [
        ("target_share", "targets", "team_targets"),
        ("carry_share", "carries", "team_carries"),
        ("att_share", "attempts", "team_attempts"),
    ]:
        frame[share] = frame[count] / frame[total].replace(0, np.nan)
        frame[share] = frame[share].fillna(0.0)
    frame["is_starting_qb"] = frame.player_id == frame.qb_id
    return frame


def upcoming_rows(games, players, season, week, lookback_games=2):
    """Placeholder rows for players expected to play in an upcoming week.

    A player is included if he appeared for the team in one of its last
    ``lookback_games`` games this season. The listed starting QB comes from
    the schedule. Injuries are not checked: see the README.
    """
    teams = team_games(games)
    upcoming = teams[(teams.season == season) & (teams.week == week)]
    played = players[players.season == season]
    rows = []
    for team_row in upcoming.itertuples():
        team_weeks = sorted(played.loc[played.team == team_row.team, "week"].unique())
        recent = team_weeks[-lookback_games:]
        roster = played[(played.team == team_row.team) & played.week.isin(recent)]
        roster = roster.sort_values("week").drop_duplicates("player_id", keep="last")
        # Drop players who have since appeared for another team.
        latest = played.sort_values("week").drop_duplicates("player_id", keep="last")
        moved = latest[latest.team != team_row.team].player_id
        roster = roster[~roster.player_id.isin(moved)]
        listed_qb = team_row.qb_id
        if isinstance(listed_qb, str) and listed_qb not in set(roster.player_id):
            history = players[players.player_id == listed_qb]
            if len(history):
                roster = pd.concat([roster, history.tail(1)])
        for player in roster.itertuples():
            rows.append(
                {
                    "player_id": player.player_id,
                    "player_display_name": player.player_display_name,
                    "position": player.position,
                    "season": season,
                    "week": week,
                    "game_id": team_row.game_id,
                    "team": team_row.team,
                    "opponent_team": team_row.opp,
                }
            )
    frame = pd.DataFrame(rows).merge(
        upcoming, on=["game_id", "season", "week", "team"], how="left"
    )
    frame["is_starting_qb"] = frame.player_id == frame.qb_id
    return frame
