"""Probe ESPN's public odds API for stored player-prop lines on past games.

For a few 2025-26 games (and one upcoming game) lists the odds providers and
fetches a sample of each provider's prop bets, recording status codes, counts
and one example. Used to decide whether idea 2 can be tested for free
(RESEARCH_LOG.md). Writes outputs/probes/espn_props.json.

Usage: python -m research.probe_espn_props
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
EVENTS = {
    "2025-11-20 regular season": "401810120",
    "2026-01-26 regular season": "401810518",
    "2026-03-10 regular season": "401810801",
    "2026-06-13 Finals": "401859967",
    "2026-10-20 opening night": "401909088",
}
OUT = Path("outputs/probes/espn_props.json")


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "nba-prop-model"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, None
    except (OSError, ValueError) as error:
        return str(error), None


def main():
    report = {}
    for label, event in EVENTS.items():
        base = f"{CORE}/events/{event}/competitions/{event}/odds"
        status, odds = get(base)
        entry = {"odds_status": status, "providers": []}
        for item in (odds or {}).get("items", [])[:6]:
            ref = item.get("$ref", "")
            provider = (item.get("provider") or {}).get("id") or ref.rstrip("/").split(
                "/"
            )[-1].split("?")[0]
            p_status, props = get(f"{base}/{provider}/propBets?limit=1000")
            sample = None
            items = (props or {}).get("items", [])
            if items:
                first = items[0]
                if "$ref" in first and len(first) == 1:
                    _, first = get(first["$ref"])
                sample = first
            entry["providers"].append(
                {
                    "provider": provider,
                    "name": (item.get("provider") or {}).get("name"),
                    "prop_status": p_status,
                    "prop_count": (props or {}).get("count"),
                    "sample": sample,
                }
            )
        report[label] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2)[:200000] + "\n")
    for label, entry in report.items():
        counts = [
            (p["provider"], p["name"], p["prop_status"], p["prop_count"])
            for p in entry["providers"]
        ]
        print(label, entry["odds_status"], counts)


if __name__ == "__main__":
    main()
