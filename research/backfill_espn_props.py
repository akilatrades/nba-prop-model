"""Download the player-prop lines ESPN stores for each 2025-26 game.

For every game in data/games.parquet (run fetch_data.py first) lists ESPN's
odds providers and downloads each provider's stored prop bets, flattened to
one row per prop. Used for idea 2 (RESEARCH_LOG.md). Writes
data/espn_props/props_2026.parquet and coverage.json.

Usage: python -m research.backfill_espn_props
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.probe_espn_props import CORE, get

OUT = Path("data/espn_props")


def ref_id(block, kind):
    ref = (block or {}).get("$ref", "")
    match = re.search(rf"/{kind}/(\d+)", ref)
    return int(match.group(1)) if match else None


def game_props(game_id):
    base = f"{CORE}/events/{game_id}/competitions/{game_id}/odds"
    status, odds = get(base)
    rows, providers = [], []
    for item in (odds or {}).get("items", []):
        provider = (item.get("provider") or {}).get("id")
        name = (item.get("provider") or {}).get("name")
        if not provider or "live" in str(name).lower():
            continue
        page, count = 1, 0
        while True:
            p_status, props = get(f"{base}/{provider}/propBets?limit=1000&page={page}")
            items = (props or {}).get("items", [])
            for prop in items:
                odds_block = prop.get("odds") or {}
                rows.append(
                    {
                        "game_id": game_id,
                        "provider": int(provider),
                        "provider_name": name,
                        "athlete_id": ref_id(prop.get("athlete"), "athletes"),
                        "type_id": (prop.get("type") or {}).get("id"),
                        "type": (prop.get("type") or {}).get("name"),
                        "american": (odds_block.get("american") or {}).get("value"),
                        "american_open": (odds_block.get("american") or {}).get("open"),
                        "total": (odds_block.get("total") or {}).get("value"),
                        "total_open": (odds_block.get("total") or {}).get("open"),
                        "last_updated": prop.get("lastUpdated"),
                        "raw_keys": ",".join(sorted(prop.keys())),
                    }
                )
            count += len(items)
            if not props or page >= (props.get("pageCount") or 1):
                break
            page += 1
        providers.append(
            {"provider": provider, "name": name, "status": p_status, "props": count}
        )
    return game_id, status, providers, rows


def main():
    games = pd.read_parquet("data/games.parquet")
    games = games[(games.season == 2026) & games.completed]
    OUT.mkdir(parents=True, exist_ok=True)
    coverage, rows = [], []
    with ThreadPoolExecutor(12) as pool:
        for game_id, status, providers, props in pool.map(game_props, games.game_id):
            coverage.append(
                {"game_id": int(game_id), "status": status, "providers": providers}
            )
            rows.extend(props)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "props_2026.parquet", index=False, compression="zstd")
    (OUT / "coverage.json").write_text(json.dumps(coverage) + "\n")
    with_props = sum(any(p["props"] for p in c["providers"]) for c in coverage)
    message = f"ESPN props: {len(frame)} rows; {with_props} of {len(coverage)} games have props"
    print(message)
    print("::notice::" + message)


if __name__ == "__main__":
    main()
