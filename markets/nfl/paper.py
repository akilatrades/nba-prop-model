"""Model probabilities for open Kalshi NFL prop markets (RESEARCH_LOG.md, 2026-10-16).

Maps each open market to the model's over-probability from the latest
ladder made before ``now``, by game day, player name, stat and line. Used by
the "no" seller paper test with the NFL model as veto.
"""

from pathlib import Path

import pandas as pd

from markets.nfl.kalshi import normalize_name, parse
from markets.nfl.market import kickoffs

LADDERS = Path("outputs/nfl/projections/ladders")
MODEL_SERIES = {"KXNFLANYTD", "KXNFLRECYDS"}


def latest_ladder(now, folder=LADDERS):
    """Rows of the most recent ladder made before ``now``, with kickoff times."""
    files = sorted(folder.glob("*.csv.gz"))
    frames = []
    for f in files:
        made = pd.to_datetime(
            f.stem.split("_")[-1].replace(".csv", ""), format="%Y-%m-%dT%H%MZ", utc=True
        )
        if made <= pd.Timestamp(now):
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame(
            columns=[
                "game_id",
                "player",
                "player_id",
                "stat",
                "line",
                "p_over",
                "made_utc",
            ]
        )
    ladder = pd.concat(frames, ignore_index=True)
    ladder["made_utc"] = pd.to_datetime(ladder.made_utc, utc=True)
    return ladder.sort_values("made_utc").drop_duplicates(
        ["game_id", "player_id", "stat", "line"], keep="last"
    )


def model_probabilities(markets, games, now, folder=LADDERS):
    """{ticker: p_over} for the open markets the model can price."""
    m = pd.DataFrame(markets)
    if m.empty:
        return {}
    m["series"] = m.ticker.str.split("-").str[0]
    m = parse(m)
    m["gameday"] = m.game_date.astype(str)
    ladder = latest_ladder(now, folder)
    if ladder.empty:
        return {}
    ladder = ladder.merge(kickoffs(games)[["game_id", "gameday"]], on="game_id")
    ladder["player_key"] = ladder.player.map(normalize_name)
    joined = m.merge(
        ladder[["gameday", "player_key", "stat", "line", "p_over"]],
        on=["gameday", "player_key", "stat", "line"],
    )
    return dict(zip(joined.ticker, joined.p_over))
