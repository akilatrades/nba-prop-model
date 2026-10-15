"""Probe: which Kalshi series cover Miami temperature, and are the weather data sources reachable."""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_model.kalshi import get_json

OUT = Path("weather/out")


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "kalshi-weather-research"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(600).decode("utf-8", "replace")
    except Exception as error:  # noqa: BLE001
        return str(error), ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    series, cursor = [], ""
    while True:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = get_json("/series", params) or {}
        series.extend(page.get("series", []))
        cursor = page.get("cursor") or ""
        if not cursor or not page.get("series"):
            break
    report["series_total"] = len(series)
    hits = [
        {
            k: s.get(k)
            for k in [
                "ticker",
                "title",
                "category",
                "frequency",
                "settlement_sources",
                "contract_url",
            ]
        }
        for s in series
        if "miami" in str(s.get("title", "")).lower()
        or "MIA" in str(s.get("ticker", ""))
    ]
    report["miami_series"] = hits
    for s in hits:
        ticker = s["ticker"]
        for label, path in [
            ("live_settled", "/markets"),
            ("historical", "/historical/markets"),
        ]:
            params = {"series_ticker": ticker, "limit": 1000}
            if label == "live_settled":
                params["status"] = "settled"
            page = get_json(path, params) or {}
            markets = page.get("markets", [])
            entry = report.setdefault(ticker, {})
            entry[f"{label}_first_page"] = len(markets)
            entry[f"{label}_has_more"] = bool(page.get("cursor"))
            if markets:
                m = markets[0]
                entry[f"{label}_sample"] = {
                    k: m.get(k)
                    for k in [
                        "ticker",
                        "event_ticker",
                        "title",
                        "subtitle",
                        "yes_sub_title",
                        "floor_strike",
                        "cap_strike",
                        "strike_type",
                        "open_time",
                        "close_time",
                        "occurrence_datetime",
                        "result",
                        "volume_fp",
                        "rules_primary",
                    ]
                }
                entry[f"{label}_event_dates"] = (
                    sorted({x["event_ticker"].split("-")[1] for x in markets})[:3]
                    + ["..."]
                    + sorted({x["event_ticker"].split("-")[1] for x in markets})[-3:]
                )
    report["sources"] = {
        "open_meteo_previous_runs": fetch(
            "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=25.79&longitude=-80.29"
            "&hourly=temperature_2m,temperature_2m_previous_day1&start_date=2025-06-01&end_date=2025-06-02"
            "&temperature_unit=fahrenheit"
        ),
        "open_meteo_historical_forecast": fetch(
            "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=25.79&longitude=-80.29"
            "&daily=temperature_2m_max&start_date=2025-06-01&end_date=2025-06-02&temperature_unit=fahrenheit"
        ),
        "iem_asos": fetch(
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=MIA&data=tmpf"
            "&year1=2025&month1=6&day1=1&year2=2025&month2=6&day2=2&tz=Etc/UTC&format=onlycomma"
        ),
        "iem_cli": fetch(
            "https://mesonet.agron.iastate.edu/json/cli.py?station=KMIA&year=2025"
        ),
    }
    (OUT / "probe.json").write_text(
        json.dumps(report, indent=2, default=str)[:300000] + "\n"
    )
    print(
        json.dumps(
            {
                "series_total": report["series_total"],
                "miami": [h["ticker"] for h in hits],
            }
        )
    )
    print("::notice::" + json.dumps({"miami": [h["ticker"] for h in hits]}))


if __name__ == "__main__":
    main()
