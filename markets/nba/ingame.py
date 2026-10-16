"""Live game state from ESPN play-by-play, and an in-game model for props.

Game state for a player at a moment: points, rebounds, assists and threes
so far; minutes on court so far (lineups rebuilt from substitutions);
personal fouls; whether he is on the court; the score margin; game time
left. The in-game model predicts the stat still to come.

Lineups: ESPN's play-by-play lists substitutions but not who starts each
period. A player is taken to be on the court at the start of a period if his
first appearance in that period is leaving the game, or any other play
before he is substituted in. A player with no appearance in a period is
taken to be off the court for all of it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATS = ["points", "rebounds", "assists", "threes"]
REGULATION = 2880.0
N_BANDS = 10
COLUMNS = [
    "game_id",
    "sequence_number",
    "type_text",
    "scoring_play",
    "score_value",
    "athlete_id_1",
    "athlete_id_2",
    "athlete_id_3",
    "period_number",
    "clock_minutes",
    "clock_seconds",
    "home_score",
    "away_score",
    "wallclock",
]


def load_pbp(path):
    p = pd.read_parquet(path, columns=COLUMNS)
    stamp = pd.to_datetime(p.wallclock, utc=True, errors="coerce")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    p["ts"] = (stamp - epoch) // pd.Timedelta(seconds=1)  # Unix seconds, any unit
    period = p.period_number
    left = p.clock_minutes * 60 + p.clock_seconds
    length = np.where(period <= 4, 720.0, 300.0)
    start = np.where(
        period <= 4, (period - 1) * 720.0, REGULATION + (period - 5) * 300.0
    )
    p["elapsed"] = start + (length - left)
    p = p[p.elapsed.notna() & p.home_score.notna() & p.away_score.notna()]
    return p.sort_values(["game_id", "sequence_number"]).reset_index(drop=True)


def _ids(series):
    return series.fillna(-1).astype("int64").to_numpy()


def player_events(game):
    """One row per (play, player) with the stat increments for that play."""
    a1, a2 = _ids(game.athlete_id_1), _ids(game.athlete_id_2)
    t = game.type_text.fillna("")
    scoring = game.scoring_play.fillna(False).to_numpy()
    value = game.score_value.fillna(0).to_numpy()
    foul = t.str.contains("Foul") & ~t.str.contains("Technical|Turnover")
    rebound = t.str.contains("Rebound").to_numpy()
    base = {"idx": np.arange(len(game))}
    scorer = pd.DataFrame(
        base
        | {
            "player": a1,
            "points": np.where(scoring, value, 0),
            "threes": np.where(scoring & (value == 3), 1, 0),
            "rebounds": np.where(rebound, 1, 0),
            "assists": 0,
            "fouls": foul.to_numpy().astype(int),
        }
    )
    assister = pd.DataFrame(
        base
        | {
            "player": np.where(scoring, a2, -1),
            "points": 0,
            "threes": 0,
            "rebounds": 0,
            "assists": np.where(scoring & (a2 > 0), 1, 0),
            "fouls": 0,
        }
    )
    e = pd.concat([scorer, assister])
    return e[e.player > 0]


def on_court_intervals(game, roster):
    """Elapsed-second intervals on court for each player in ``roster``."""
    sub = game.type_text.eq("Substitution").to_numpy()
    a1, a2, a3 = (
        _ids(game.athlete_id_1),
        _ids(game.athlete_id_2),
        _ids(game.athlete_id_3),
    )
    elapsed = game.elapsed.to_numpy()
    periods = game.period_number.to_numpy()
    intervals = {pid: [] for pid in roster}
    for period in np.unique(periods):
        here = np.where(periods == period)[0]
        if len(here) == 0:
            continue
        p_start = (
            (period - 1) * 720.0 if period <= 4 else REGULATION + (period - 5) * 300.0
        )
        p_end = p_start + (720.0 if period <= 4 else 300.0)
        on = {}
        for pid in roster:
            on[pid] = None
        status = {}
        for i in here:
            if sub[i]:
                for pid, entering in ((a1[i], True), (a2[i], False)):
                    if pid in on and pid not in status:
                        status[pid] = not entering  # leaving first: was on at start
            else:
                for pid in (a1[i], a2[i], a3[i]):
                    if pid in on and pid not in status:
                        status[pid] = True
        current = {pid: (p_start if status.get(pid, False) else None) for pid in roster}
        for i in here:
            if not sub[i]:
                continue
            if a1[i] in current and current[a1[i]] is None:
                current[a1[i]] = elapsed[i]
            if a2[i] in current and current[a2[i]] is not None:
                intervals[a2[i]].append((current[a2[i]], elapsed[i]))
                current[a2[i]] = None
        for pid, began in current.items():
            if began is not None:
                intervals[pid].append((began, p_end))
    return intervals


def minutes_until(intervals, seconds):
    """Minutes on court up to ``seconds`` elapsed, and whether on court then."""
    total, on = 0.0, False
    for start, end in intervals:
        if seconds > start:
            total += max(min(end, seconds) - start, 0.0)
            on = on or (start <= seconds < end)
    return total / 60.0, on


def game_states(game, roster, times, by="elapsed"):
    """State of each roster player at each time (elapsed seconds or wallclock)."""
    events = player_events(game)
    keys = game[by].to_numpy()
    events["key"] = keys[events.idx.to_numpy()]
    intervals = on_court_intervals(game, roster)
    rows = []
    for time in times:
        done = game[game[by] <= time]
        if done.empty:
            continue
        last = done.iloc[-1]
        elapsed = float(last.elapsed)
        so_far = events[events.key <= time].groupby("player")[STATS + ["fouls"]].sum()
        for pid in roster:
            minutes, on = minutes_until(intervals[pid], elapsed)
            row = {
                "player_id": pid,
                "time": time,
                "elapsed": elapsed,
                "minutes_so_far": minutes,
                "on_court": on,
                "home_margin": float(last.home_score - last.away_score),
            }
            for stat in STATS + ["fouls"]:
                row[f"{stat}_so_far"] = (
                    float(so_far.at[pid, stat]) if pid in so_far.index else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def features(rows, stat):
    """Design matrix for the stat still to come."""
    left = np.clip((REGULATION - rows.elapsed) / REGULATION, 0.01, 1.0)
    expected_so_far = rows.ew_minutes * (1 - left)
    margin = rows.home_margin.abs()
    fouls = rows.fouls_so_far
    return np.column_stack(
        [
            np.log(np.maximum(rows.pregame_mean, 0.05)),
            np.log(left),
            np.log1p(rows.minutes_so_far),
            np.log1p(expected_so_far),
            rows.on_court.astype(float),
            fouls,
            (fouls >= 4).astype(float) * left,
            margin,
            margin * (1 - left),
            np.log1p(rows[f"{stat}_so_far"]),
        ]
    )


class InGameModel:
    """Expected stat still to come, with the empirical spread by band."""

    def __init__(self, stat):
        self.stat = stat
        self.glm = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)
        )

    def fit(self, rows):
        y = rows[f"{self.stat}_remaining"].to_numpy(float)
        self.glm.fit(features(rows, self.stat), y)
        mean = self.mean(rows)
        ratio = y / mean
        self.edges = np.quantile(mean, np.linspace(0, 1, N_BANDS + 1)[1:-1])
        band = np.searchsorted(self.edges, mean)
        self.ratios = [np.sort(ratio[band == b]) for b in range(N_BANDS)]
        return self

    def mean(self, rows):
        return np.maximum(self.glm.predict(features(rows, self.stat)), 1e-3)

    def prob_at_least(self, rows, threshold):
        """P(final stat >= threshold) given the stat so far."""
        mean = self.mean(rows)
        need = np.asarray(threshold, float) - rows[f"{self.stat}_so_far"].to_numpy(
            float
        )
        # remaining >= need  <=>  remaining > need - 0.5 (counts)
        cut = (need - 0.5) / mean
        band = np.searchsorted(self.edges, mean)
        out = np.empty(len(mean))
        for b in range(N_BANDS):
            here = band == b
            if here.any():
                above = np.searchsorted(self.ratios[b], cut[here], side="right")
                out[here] = 1 - above / len(self.ratios[b])
        return np.where(need <= 0, 1.0, out)
