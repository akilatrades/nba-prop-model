"""Record Kalshi's liquidity and volume reward programs on NBA markets.

Lists every incentive program (all statuses) from Kalshi's public API and
keeps those on NBA markets. Writes outputs/incentives/nba_programs.csv (all
NBA programs seen, deduplicated by id) and summary.json, which says whether
NBA player-prop markets (points, rebounds, assists, threes) carry rewards.

Usage: python scripts/log_incentives.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi.api import get_json
from markets.nba.kalshi import SERIES

OUT = Path("outputs/incentives")


def all_programs():
    rows, cursor = [], ""
    while True:
        params = {"status": "all", "limit": 10000}
        if cursor:
            params["cursor"] = cursor
        page = get_json("/incentive_programs", params) or {}
        rows.extend(page.get("incentive_programs", []))
        cursor = page.get("next_cursor") or ""
        if not cursor or not page.get("incentive_programs"):
            return pd.DataFrame(rows)


def ends(programs):
    return pd.to_datetime(
        programs.end_date, utc=True, format="ISO8601", errors="coerce"
    )


def by_series(programs, top=25):
    """Total period rewards in dollars by series, largest first."""
    if programs.empty:
        return {}
    dollars = pd.to_numeric(programs.period_reward, errors="coerce") / 10000
    series = programs.market_ticker.str.split("-").str[0]
    totals = dollars.groupby(series).sum().sort_values(ascending=False).head(top)
    return {k: round(float(v), 2) for k, v in totals.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    programs = all_programs()
    now = datetime.now(UTC)
    summary = {
        "checked_utc": now.isoformat(timespec="minutes"),
        "programs_total": len(programs),
    }
    if programs.empty:
        summary["note"] = "no programs returned"
    else:
        nba = programs[programs.market_ticker.str.startswith("KXNBA")].copy()
        nba["series"] = nba.market_ticker.str.split("-").str[0]
        nba["reward_dollars"] = (
            pd.to_numeric(nba.period_reward, errors="coerce") / 10000
        )
        path = OUT / "nba_programs.csv"
        if path.exists():
            nba = pd.concat([pd.read_csv(path), nba]).drop_duplicates("id", keep="last")
        nba.to_csv(path, index=False)
        props = nba[nba.series.isin(SERIES)]
        summary.update(
            {
                "nba_programs": len(nba),
                "nba_by_series": nba.series.value_counts().head(25).to_dict(),
                "player_prop_programs": len(props),
                "player_prop_by_status": props.groupby(
                    pd.to_datetime(props.end_date, utc=True) > now
                )
                .size()
                .rename(index={True: "open", False: "ended"})
                .to_dict(),
                "player_prop_reward_dollars_total": float(props.reward_dollars.sum()),
                "player_prop_reward_per_program_median": float(
                    props.reward_dollars.median()
                )
                if len(props)
                else None,
                "types": programs.incentive_type.value_counts().to_dict(),
                "reward_dollars_by_series_open_now": by_series(
                    programs[ends(programs) > pd.Timestamp(now)]
                ),
                "reward_dollars_by_series_all_time": by_series(programs),
                "all_series_with_programs_top": programs.market_ticker.str.split("-")
                .str[0]
                .value_counts()
                .head(30)
                .to_dict(),
            }
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    message = json.dumps(
        {
            k: summary.get(k)
            for k in ["programs_total", "nba_programs", "player_prop_programs"]
        }
    )
    print(json.dumps(summary, indent=2, default=str))
    print("::notice::" + message)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        text = traceback.format_exc()[-3000:]
        print(text)
        print("::error::" + text.replace("%", "%25").replace("\n", "%0A"))
        sys.exit(1)
