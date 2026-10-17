"""Historical test of the sportsbook-line veto (RESEARCH_LOG.md, 2026-10-16).

Takes the 2025-26 "no" seller's fills (outputs/decompose/orders_no.csv.gz),
samples 150 games stratified by month with a fixed seed, pulls the books'
lines from The Odds API as they stood when each order was posted, and scores
the veto variants pre-registered in the log. Raw replies are cached in
data/books/nba_2025-26/ so the scoring can be re-run without credits.

    python -m research.bookline_backtest fetch   # needs ODDS_API_KEY, ~12k credits
    python -m research.bookline_backtest score   # offline

Writes outputs/bookline/results.json, summary.md and book_lines.csv.gz (the
de-vigged consensus only; the raw replies stay in data/books/, untracked).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.books import (
    BooksError,
    TEAMS,
    api_key,
    attach,
    consensus,
    historical_event_odds,
    historical_events,
    load_json,
    parse_event,
    save_json,
)
from kalshi.maker import maker_fee
from kalshi.stats import clustered_mean_t
from markets.nba.kalshi import normalize_name

FILLS = Path("outputs/decompose/orders_no.csv.gz")
CACHE = Path("data/books/nba_2025-26")
OUT = Path("outputs/bookline")
GAMES = 150
SEED = 20261016
FEE_SHARE = 0.25


def load_fills():
    o = pd.read_csv(FILLS, low_memory=False)
    o = o[o.filled.astype(str).str.lower().eq("true")].copy()
    o["tip_utc"] = pd.to_datetime(o.tip_utc, utc=True)
    o["month"] = o.tip_utc.dt.strftime("%Y-%m")
    return o


def sample_games(fills, n=GAMES, seed=SEED):
    """``n`` games spread over the season's months in proportion to their games."""
    # One snapshot per game, requested at the game's EARLIEST posting time, so
    # no order is vetoed with a quote from after it was posted (a reviewer
    # caught the earlier version taking an arbitrary order's posting time).
    earliest = fills.groupby("game_id").end_ts.min()
    games = fills.drop_duplicates("game_id")[
        ["game_id", "tip_utc", "month", "home", "away"]
    ]
    games["end_ts"] = games.game_id.map(earliest)
    games = games.sort_values("game_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    share = games.month.value_counts(normalize=True).sort_index()
    take = (share * n).round().astype(int)
    picked = []
    for month, k in take.items():
        pool = games[games.month == month]
        k = min(int(k), len(pool))
        picked.append(pool.iloc[rng.choice(len(pool), size=k, replace=False)])
    return pd.concat(picked).sort_values("tip_utc").reset_index(drop=True)


def find_event(listed, home, away, tip_utc):
    """The feed's event for this game: same teams, start within three hours."""
    for event in listed:
        h = TEAMS.get(event.get("home_team"), event.get("home_team"))
        a = TEAMS.get(event.get("away_team"), event.get("away_team"))
        start = pd.Timestamp(event["commence_time"])
        if {h, a} == {home, away} and abs(
            (start - tip_utc).total_seconds()
        ) <= 3 * 3600:
            return event
    return None


def fetch(games):
    key = api_key()
    usage = {}
    done = missing = 0
    for g in games.itertuples():
        path = CACHE / f"{int(g.game_id)}.json.gz"
        if path.exists():
            done += 1
            continue
        # Orders were posted at end_ts (5 h before tip for most); the quotes at
        # that moment are the ones the veto would have seen.
        when = pd.Timestamp(int(g.end_ts), unit="s", tz="UTC")
        date_iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            listed, _, usage = historical_events(key, date_iso)
            event = find_event(listed, g.home, g.away, g.tip_utc)
            if event is None:
                save_json({"missing": True, "game_id": int(g.game_id)}, path)
                missing += 1
                continue
            odds, snapshot, usage = historical_event_odds(key, event["id"], date_iso)
            save_json(
                {
                    "game_id": int(g.game_id),
                    "event_id": event["id"],
                    "requested": date_iso,
                    "snapshot": snapshot,
                    "odds": odds,
                },
                path,
            )
            done += 1
        except BooksError as error:
            print(f"::warning::game {g.game_id}: {error}")
            if "401" in str(error) or "422" in str(error):
                break
    print(f"fetched {done} games, {missing} not found; credits {usage}")


BOOK_TABLE = OUT / "book_lines.csv.gz"


def book_table():
    """Consensus book probabilities for every cached game, keyed by game_id.

    Built from the raw replies in data/books/ (kept out of git: the feed's
    terms do not allow redistributing bookmakers' odds) and saved as the
    derived table outputs/bookline/book_lines.csv.gz, which carries only the
    de-vigged consensus per player, stat and line. Without the cache, the
    saved table is used.
    """
    frames = []
    for path in sorted(CACHE.glob("*.json.gz")):
        blob = load_json(path)
        if blob.get("missing"):
            continue
        lines = parse_event(blob["odds"], snapshot_utc=blob.get("snapshot"))
        if lines.empty:
            continue
        c = consensus(lines)
        c["game_id"] = blob["game_id"]
        c["snapshot_utc"] = blob.get("snapshot")
        frames.append(c)
    if not frames:
        if BOOK_TABLE.exists():
            return pd.read_csv(BOOK_TABLE)
        return pd.DataFrame(
            columns=["game_id", "player_key", "stat", "line", "p_book", "book_n"]
        )
    table = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(BOOK_TABLE, index=False)
    return table


def summary(frame, label):
    if frame.empty:
        return {"label": label, "fills": 0}
    per = frame.pnl_if_filled.astype(float)
    _, t = clustered_mean_t(per, frame.game_id)
    return {
        "label": label,
        "fills": int(len(frame)),
        "games": int(frame.game_id.nunique()),
        "cents": round(100 * float(per.mean()), 2),
        "t": round(float(t), 2) if np.isfinite(t) else None,
        "hit": round(float(frame.over.astype(float).mean()), 3),
    }


def difference(a, b):
    """Mean of a minus mean of b, in dollars per contract, with a game-clustered t.

    Each fill's influence on the estimate is +1/n_a or -1/n_b times its
    deviation from its group mean; summing influences within a game and
    taking the root sum of squares gives the clustered standard error.
    """
    if a.empty or b.empty:
        return None
    xa, xb = a.pnl_if_filled.astype(float), b.pnl_if_filled.astype(float)
    est = float(xa.mean() - xb.mean())
    infl = pd.concat(
        [(xa - xa.mean()) / len(xa), -(xb - xb.mean()) / len(xb)], ignore_index=True
    )
    games = pd.concat([a.game_id, b.game_id], ignore_index=True)
    by_game = infl.groupby(games.to_numpy()).sum()
    se = float(np.sqrt((by_game**2).sum())) if len(by_game) > 1 else np.nan
    return {"cents": round(100 * est, 2), "t": round(est / se, 2) if se else None}


def score():
    fills = load_fills()
    games = sample_games(fills)
    sample = fills[fills.game_id.isin(set(games.game_id))].copy()
    sample["player_key"] = sample.player.map(normalize_name)
    book = book_table()
    for column, kind in [("game_id", "int64"), ("line", float), ("p_book", float)]:
        book[column] = book[column].astype(kind)
    book["line"] = book.line.round(1)
    cached = set(book.game_id.unique()) if len(book) else set()
    sample = sample[sample.game_id.isin(cached)]
    merged = attach(sample, None).drop(columns=["p_book", "book_n"])
    merged["game_id"] = merged.game_id.astype("int64")
    merged = merged.merge(
        book[["game_id", "player_key", "stat", "line", "p_book", "book_n"]],
        on=["game_id", "player_key", "stat", "line"],
        how="left",
    )
    merged["book_n"] = merged.book_n.fillna(0).astype(int)
    merged["price"] = merged.price.astype(float)
    merged["book_edge"] = (
        (1 - merged.p_book) - merged.price - maker_fee(merged.price, FEE_SHARE)
    )
    merged["model_edge_c"] = (
        (1 - merged.p_model.astype(float))
        - merged.price
        - maker_fee(merged.price, FEE_SHARE)
    )
    covered = merged[merged.p_book.notna()]
    uncovered = merged[merged.p_book.isna()]
    veto0 = covered[covered.book_edge >= 0]
    cut0 = covered[covered.book_edge < 0]
    veto2 = covered[covered.book_edge >= 0.02]
    model_veto = covered[covered.model_edge_c >= 0]
    both = covered[(covered.book_edge >= 0) & (covered.model_edge_c >= 0)]
    results = {
        "games_sampled": int(len(games)),
        "games_with_book_data": int(len(cached)),
        "coverage_of_fills": round(float(len(covered) / len(merged)), 3)
        if len(merged)
        else None,
        "rows": [
            summary(merged, "1. all fills in the sample"),
            summary(covered, "2. fills with a book quote at the exact line"),
            summary(uncovered, "2b. fills with no book quote"),
            summary(veto0, "3. book veto: book edge >= 0"),
            summary(cut0, "3b. removed by the book veto"),
            summary(veto2, "4. book veto at a 2 cent margin"),
            summary(model_veto, "5. model veto on covered fills"),
            summary(both, "6. both vetoes"),
        ],
        "veto_minus_covered": difference(veto0, cut0),
        "pass": None,
    }
    d = results["veto_minus_covered"]
    r3 = results["rows"][3]
    r2 = results["rows"][1]
    if d and r3.get("fills") and r2.get("fills"):
        gain = (r3.get("cents") or 0) - (r2.get("cents") or 0)
        results["gain_cents_vs_covered"] = round(gain, 2)
        results["pass"] = bool(
            gain >= 1.0 and (d.get("t") or 0) >= 2.0 and r2["fills"] >= 2000
        )
    # Book calibration: how often the over hit, by book probability decile.
    if len(covered):
        bins = pd.cut(covered.p_book, np.linspace(0, 1, 11))
        cal = covered.groupby(bins, observed=True).agg(
            n=("over", "size"),
            hit=("over", "mean"),
            p_book=("p_book", "mean"),
            kalshi_ask=("yes_ask", "mean"),
        )
        results["calibration"] = [
            {
                "bin": str(k),
                "n": int(v.n),
                "hit": round(float(v.hit), 3),
                "p_book": round(float(v.p_book), 3),
                "kalshi_yes_ask": round(float(v.kalshi_ask), 3),
            }
            for k, v in cal.iterrows()
        ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    merged.drop(columns=[c for c in merged.columns if c.startswith("Unnamed")]).to_csv(
        OUT / "fills_with_book.csv.gz", index=False
    )
    write_summary(results)
    print(json.dumps(results["rows"], indent=1))
    print("pass:", results["pass"])


def write_summary(results):
    lines = [
        "# Sportsbook-line veto: historical test on 2025-26 fills",
        "",
        f"{results['games_sampled']} games sampled (seed {SEED}), "
        f"{results['games_with_book_data']} with book lines in the feed; "
        f"book quote at the exact line for {results['coverage_of_fills']:.0%} of fills."
        if results.get("coverage_of_fills") is not None
        else "No book data yet.",
        "",
        "| Fills | Count | Games | Cents per contract | t | Over hit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results["rows"]:
        if r.get("fills"):
            lines.append(
                f"| {r['label']} | {r['fills']:,} | {r['games']} | {r['cents']:+.2f} | {r['t']} | {r['hit']:.3f} |"
            )
        else:
            lines.append(f"| {r['label']} | 0 | | | | |")
    d = results.get("veto_minus_covered")
    if d:
        lines += [
            "",
            f"Kept minus removed by the book veto: {d['cents']:+.2f}¢ per contract, clustered t {d['t']}.",
            f"Veto against all covered fills: {results.get('gain_cents_vs_covered', 0):+.2f}¢.",
            f"**Pre-registered pass (≥ 1¢ gain, t ≥ 2, ≥ 2,000 covered fills): {results['pass']}.**",
        ]
    if results.get("calibration"):
        lines += [
            "",
            "Book probability against what happened (covered fills):",
            "",
            "| Book p(over) | Fills | Over hit | Mean book p | Mean Kalshi yes ask |",
            "|---|---:|---:|---:|---:|",
        ]
        for c in results["calibration"]:
            lines.append(
                f"| {c['bin']} | {c['n']} | {c['hit']:.3f} | {c['p_book']:.3f} | {c['kalshi_yes_ask']:.3f} |"
            )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "score"
    if cmd == "fetch":
        fetch(sample_games(load_fills()))
        score()
    elif cmd == "score":
        score()
    elif cmd == "sample":
        g = sample_games(load_fills())
        print(g.month.value_counts().sort_index().to_string())
        print(len(g), "games")
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
