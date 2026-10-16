"""Probe: are Kalshi's combo (parlay) markets and their trades public?

Kalshi sells parlays as "multivariate" markets built from legs in other
markets. This lists the multivariate series and collections, counts settled
combo markets on the live and historical endpoints, and saves sample markets
(all fields) and sample trade prints for NBA combos. Prices are only saved as
samples here; nothing is tested.

Writes outputs/probes/kalshi_combos.json.
"""

import json
import sys
import traceback
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi.api import get_json

OUT = Path("outputs/probes")


def annotate(level, message):
    print(message)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"))


def safe(path, params):
    try:
        return get_json(path, params), None
    except (urllib.error.HTTPError, OSError, ValueError) as error:
        body = ""
        if isinstance(error, urllib.error.HTTPError):
            body = error.read(300).decode("utf-8", "replace")
        return None, f"{error} {body}"[:400]


def pages(path, params, key, limit_pages=50):
    out, cursor, errors = [], "", []
    for _ in range(limit_pages):
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        page, error = safe(path, p)
        if error:
            errors.append(error)
            break
        page = page or {}
        out.extend(page.get(key, []))
        cursor = page.get("cursor") or ""
        if not cursor or not page.get(key):
            break
    return out, errors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    series, errors = pages("/series", {"limit": 1000}, "series", 40)
    mve_series = [
        {k: s.get(k) for k in ["ticker", "title", "category", "frequency"]}
        for s in series
        if "MVE" in str(s.get("ticker", ""))
        or "combo" in str(s.get("title", "")).lower()
        or "parlay" in str(s.get("title", "")).lower()
    ]
    report["series_total"] = len(series)
    report["mve_series"] = mve_series
    report["series_errors"] = errors

    collections, errors = pages(
        "/multivariate_event_collections", {"limit": 200}, "multivariate_contracts", 30
    )
    if not collections:
        collections, errors2 = pages(
            "/multivariate_event_collections",
            {"limit": 200},
            "multivariate_event_collections",
            30,
        )
        errors += errors2
    report["collections_count"] = len(collections)
    report["collection_errors"] = errors
    report["collection_samples"] = [
        {k: (v[:3] if isinstance(v, list) else v) for k, v in c.items()}
        for c in collections[:3]
    ]
    report["collection_series"] = Counter(
        str(c.get("series_ticker")) for c in collections
    ).most_common(40)

    samples = {}
    for label, path, params in [
        (
            "live_settled_mve",
            "/markets",
            {"limit": 1000, "status": "settled", "mve_filter": "only"},
        ),
        (
            "historical_mve",
            "/historical/markets",
            {"limit": 1000, "mve_filter": "only"},
        ),
    ]:
        markets, errors = pages(path, params, "markets", 5)
        nba = [
            m
            for m in markets
            if "NBA" in str(m.get("ticker", "")) + str(m.get("title", ""))
        ]
        report[label] = {
            "count_first_pages": len(markets),
            "errors": errors,
            "series_prefixes": Counter(
                str(m.get("ticker", "")).split("-")[0] for m in markets
            ).most_common(30),
            "nba_count": len(nba),
            "sample": markets[:2],
            "nba_sample": nba[:3],
        }
        samples[label] = nba[:5] or markets[:5]

    # NBA combo series by name, if any.
    for s in mve_series:
        if (
            "NBA" in s["ticker"]
            or "BASKETBALL" in str(s["title"]).upper()
            or "SPORTS" in s["ticker"]
        ):
            for label, path, params in [
                (
                    "live",
                    "/markets",
                    {"series_ticker": s["ticker"], "status": "settled", "limit": 1000},
                ),
                (
                    "historical",
                    "/historical/markets",
                    {"series_ticker": s["ticker"], "limit": 1000},
                ),
            ]:
                markets, errors = pages(path, params, "markets", 20)
                report.setdefault("by_series", {})[f"{s['ticker']}:{label}"] = {
                    "count_first_pages": len(markets),
                    "errors": errors,
                    "volume_positive": sum(
                        1 for m in markets if float(m.get("volume_fp") or 0) > 0
                    ),
                    "sample": markets[:2],
                }
                if markets:
                    samples[f"{s['ticker']}:{label}"] = [
                        m for m in markets if float(m.get("volume_fp") or 0) > 0
                    ][:5]

    trades_report = {}
    for label, markets in samples.items():
        for m in markets[:3]:
            ticker = m.get("ticker")
            for tlabel, path in [
                ("live", "/markets/trades"),
                ("historical", "/historical/trades"),
            ]:
                page, error = safe(path, {"ticker": ticker, "limit": 50})
                trades_report[f"{ticker}:{tlabel}"] = {
                    "error": error,
                    "n": len((page or {}).get("trades", [])),
                    "sample": (page or {}).get("trades", [])[:3],
                }
    report["trades"] = trades_report
    (OUT / "kalshi_combos.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )
    annotate(
        "notice",
        json.dumps(
            {
                "mve_series": [s["ticker"] for s in mve_series][:40],
                "collections": report["collections_count"],
                "live_settled_mve": report["live_settled_mve"]["count_first_pages"],
                "live_nba": report["live_settled_mve"]["nba_count"],
                "historical_mve": report["historical_mve"]["count_first_pages"],
                "historical_nba": report["historical_mve"]["nba_count"],
                "trades_with_prints": sum(1 for v in trades_report.values() if v["n"]),
            }
        )[:3000],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
