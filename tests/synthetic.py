"""A small synthetic league for tests: two teams, eight players each."""

import numpy as np
import pandas as pd

TEAMS = ["AAA", "BBB"]


def season_of(game):
    """Games 0-19 are the 2021 season and 20-39 the 2022 season."""
    return 2021 if game < 20 else 2022


def league(n_games=40, seed=0, sit=None):
    """Player-game rows and games in the formats fetch_data.py writes.

    ``sit`` maps (player_id, game number) pairs that should not play.
    """
    rng = np.random.default_rng(seed)
    sit = sit or set()
    start = pd.Timestamp("2021-10-20 23:00", tz="UTC")
    rows, games = [], []
    for g in range(n_games):
        tip = start + pd.Timedelta(days=2 * g)
        home = TEAMS[g % 2]
        away = TEAMS[1 - g % 2]
        games.append(
            {
                "game_id": g,
                "season": season_of(g),
                "season_type": "regular",
                "tip_utc": tip,
                "home": home,
                "away": away,
                "home_spread": -3.0,
                "total": 220.0,
                "completed": True,
            }
        )
        for t, team in enumerate(TEAMS):
            for k in range(8):
                pid = 100 * (t + 1) + k
                out = (pid, g) in sit
                minutes = 0.0 if out else float(36 - 3 * k + rng.normal(0, 2))
                rate = 0.6 - 0.04 * k
                rows.append(
                    {
                        "game_id": g,
                        "season": season_of(g),
                        "season_type": "regular",
                        "tip_utc": tip,
                        "game_date": tip.date(),
                        "team": team,
                        "opp": TEAMS[1 - t],
                        "home": team == home,
                        "player_id": pid,
                        "player": f"Player {pid}",
                        "position": ["G", "G", "F", "F", "C", "G", "F", "C"][k],
                        "starter": k < 5 and not out,
                        "did_not_play": out,
                        "reason": "INJURY" if out else None,
                        "team_score": 110,
                        "opp_score": 108,
                        "minutes": minutes,
                        "points": float(rng.poisson(rate * minutes)),
                        "rebounds": float(rng.poisson(0.2 * minutes)),
                        "oreb": 1.0,
                        "dreb": 3.0,
                        "assists": float(rng.poisson(0.1 * minutes)),
                        "threes": float(rng.poisson(0.05 * minutes)),
                        "fg3a": float(rng.poisson(0.14 * minutes)),
                        "fgm": 5.0,
                        "fga": float(rng.poisson(0.35 * minutes)),
                        "ftm": 2.0,
                        "fta": float(rng.poisson(0.08 * minutes)),
                        "tov": 1.0,
                        "steals": 1.0,
                        "blocks": 0.0,
                        "fouls": 2.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(games)
