"""Kalshi NBA player-prop series and how their titles name players and lines."""

import re
import unicodedata

import pandas as pd

SERIES = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBA3PT": "threes",
}
# Yes/no markets on combinations of stats, used in the correlation test.
COMBO_SERIES = {
    "KXNBA2D": "double_double",
    "KXNBA3D": "triple_double",
}
ALL_SERIES = {**SERIES, **COMBO_SERIES}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name):
    """'Jaren Jackson Jr.' and 'Jaren Jackson' -> 'jaren jackson'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z ]", " ", text.lower().replace("'", "")).split()
    return " ".join(w for w in words if w not in SUFFIXES)


def parse(snapshot_frame):
    """Add player, team, stat, line, game date and prices to raw market rows."""
    frame = snapshot_frame.copy()
    frame["stat"] = frame.series.map(SERIES)
    # Titles were "Anthony Edwards records 40+ points" until January 2026 and
    # "Anthony Edwards: 40+ points" after; the yes subtitle is always
    # "Anthony Edwards: 40+".
    name = frame.yes_sub_title.fillna(frame.title).str.split(":").str[0]
    frame["player"] = name.str.replace(r" records .*$", "", regex=True).str.strip()
    frame["player_key"] = frame.player.map(normalize_name)
    plus = frame.yes_sub_title.str.extract(r"(\d+)\+$")[0].astype(float)
    frame["line"] = pd.to_numeric(frame.floor_strike, errors="coerce").fillna(
        plus - 0.5
    )
    # Event tickers look like KXNBAPTS-26MAY30SASOKC: game date 2026-05-30,
    # San Antonio (away) at Oklahoma City (home). Market tickers add the
    # player's team, e.g. ...-SASVWEMBANYAMA1-25.
    event = frame.event_ticker.str.split("-").str[1]
    frame["game_date"] = pd.to_datetime(
        event.str[:7], format="%y%b%d", errors="coerce"
    ).dt.date
    frame["away"] = event.str[7:10]
    frame["home"] = event.str[10:13]
    frame["team"] = frame.ticker.str.split("-").str[2].str[:3]
    for column in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["mid"] = (frame.yes_bid_dollars + frame.yes_ask_dollars) / 2
    return frame
