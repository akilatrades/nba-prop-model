"""Download the data for the Miami high-temperature test (RESEARCH_LOG.md).

Each part runs in its own GitHub Actions job and writes to weather/data:

  markets   every settled market in Kalshi's KXHIGHMIA series ("Highest
            temperature in Miami"), from the historical and the live endpoint.
  candles   hourly candles (yes bid, yes ask, last price, volume) over each
            traded market's whole life.
  trades    every trade print for each traded market.
  forecasts Open-Meteo forecasts saved from earlier model runs (0 to 3 days
            ahead) for the Miami airport, and the National Blend of Models
            (NBM) and GFS MOS text forecasts for KMIA archived by the Iowa
            Environmental Mesonet (IEM).
  obs       the National Weather Service daily climate reports (CLI) for
            KMIA, and every METAR observation from the airport (IEM).

Candles and trades are resumable and stop cleanly after --minutes. Prices
are not looked at here. Prints counts and request errors only.

Usage: python weather/download.py markets|candles|trades|forecasts|obs
"""

import argparse
import io
import json
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_model.kalshi import candles, fetch_markets, trades

SERIES = "KXHIGHMIA"
OUT = Path("weather/data")
# Miami International Airport ASOS (KMIA).
LAT, LON = 25.7881, -80.3169
FIRST_DAY = "2021-01-01"
MARKET_FIELDS = [
    "ticker",
    "event_ticker",
    "title",
    "yes_sub_title",
    "floor_strike",
    "cap_strike",
    "strike_type",
    "open_time",
    "close_time",
    "occurrence_datetime",
    "expiration_time",
    "result",
    "settlement_value_dollars",
    "volume_fp",
    "rules_primary",
]
OPEN_METEO_MODELS = [
    "best_match",
    "ncep_nbm_conus",
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "gem_seamless",
    "ncep_hrrr_conus",
]


def annotate(level, message):
    """Print a GitHub Actions annotation (readable through the API) and the text."""
    print(message, flush=True)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"), flush=True)


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def fetch(url, retries=4, timeout=120):
    """GET a URL; returns the body as text, or raises after retries."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "kalshi-weather-research (github akilatrades)"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            body = error.read(400).decode("utf-8", "replace")
            if error.code in (400, 404) or attempt == retries - 1:
                raise ValueError(f"HTTP {error.code}: {body}") from error
        except (OSError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(5 * (attempt + 1))
    raise ValueError("unreachable")


# --- Kalshi -----------------------------------------------------------------


def to_utc(column):
    return pd.to_datetime(column, utc=True, format="ISO8601", errors="coerce")


def get_markets():
    rows = []
    for source, historical, status in [
        ("historical", True, None),
        ("live", False, "settled"),
    ]:
        got = fetch_markets(SERIES, status=status, historical=historical)
        rows += [{k: m.get(k) for k in MARKET_FIELDS} | {"source": source} for m in got]
        print(f"{source}: {len(got)} markets")
    frame = pd.DataFrame(rows, columns=[*MARKET_FIELDS, "source"])
    # A market can appear in both lists around the cutoff; keep the live copy.
    frame = frame.sort_values("source").drop_duplicates("ticker", keep="last")
    frame["event_date"] = pd.to_datetime(
        frame.event_ticker.str.split("-").str[1], format="%y%b%d", errors="coerce"
    ).dt.date.astype(str)
    frame = frame.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    save(frame, OUT / "kalshi_markets.parquet")
    dates = frame.event_date.dropna()
    annotate(
        "notice",
        f"markets: {len(frame)} ({frame.source.value_counts().to_dict()}), "
        f"{dates.nunique()} days {dates.min()} to {dates.max()}, "
        f"results {frame.result.value_counts().to_dict()}, "
        f"strike types {frame.strike_type.value_counts().to_dict()}",
    )


def traded_markets():
    m = pd.read_parquet(OUT / "kalshi_markets.parquet")
    volume = pd.to_numeric(m.volume_fp, errors="coerce").fillna(0)
    m = m[m.result.isin(["yes", "no"]) & (volume > 0)].copy()
    m["start_ts"] = (to_utc(m.open_time) - pd.Timedelta(hours=1)).map(
        lambda x: int(x.timestamp())
    )
    end = to_utc(m.close_time).fillna(to_utc(m.expiration_time))
    m["end_ts"] = (end + pd.Timedelta(hours=1)).map(lambda x: int(x.timestamp()))
    return m


def run_jobs(function, jobs, workers, deadline):
    """Run function(*args) over {key: args}; returns results, failures."""
    results, failed = {}, {}
    with ThreadPoolExecutor(workers) as pool:
        it = iter(jobs.items())
        pending = {}

        def submit():
            item = next(it, None)
            if item and time.monotonic() < deadline:
                pending[pool.submit(function, *item[1])] = item[0]

        for _ in range(workers * 3):
            submit()
        while pending:
            future = next(as_completed(list(pending)))
            key = pending.pop(future)
            try:
                results[key] = future.result()
            except (OSError, ValueError, KeyError) as error:
                failed[key] = str(error)[:200]
            submit()
            if len(results) % 1000 == 0 and results:
                print(f"  {len(results)} done", flush=True)
    return results, failed


def get_candles(minutes, workers):
    deadline = time.monotonic() + 60 * minutes
    m = traded_markets()
    path = OUT / "kalshi_candles.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    empty_path = OUT / "kalshi_no_candles.txt"
    empty = set(empty_path.read_text().split()) if empty_path.exists() else set()
    finished = (set(done.ticker) if len(done) else set()) | empty
    jobs = {
        r.ticker: (r.ticker, r.start_ts, r.end_ts, 60, r.source == "historical")
        for r in m.itertuples()
        if r.ticker not in finished
    }
    print(f"candles: {len(jobs)} markets to fetch, {len(finished)} done")
    results, failed = run_jobs(candles, jobs, workers, deadline)
    rows = [row for got in results.values() for row in got]
    empty |= {k for k, got in results.items() if not got}
    new = pd.DataFrame(rows)
    combined = pd.concat([done, new], ignore_index=True) if len(done) else new
    if len(combined):
        save(combined.drop_duplicates(["ticker", "end_ts"]), path)
    empty_path.write_text("\n".join(sorted(empty)) + "\n")
    remaining = len(jobs) - len(results)
    annotate(
        "notice" if not failed and not remaining else "warning",
        f"candles: fetched {len(results)}, empty {len(empty)}, failed {len(failed)}, "
        f"remaining {remaining}; {len(combined)} candle rows. "
        f"First failures: {list(failed.items())[:3]}",
    )


def get_trades(minutes, workers):
    deadline = time.monotonic() + 60 * minutes
    m = traded_markets()
    path = OUT / "kalshi_trades.parquet"
    done_path = OUT / "kalshi_trades_done.txt"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    jobs = {
        r.ticker: (r.ticker, r.start_ts, r.end_ts, r.source == "historical")
        for r in m.itertuples()
        if r.ticker not in done
    }
    print(f"trades: {len(jobs)} markets to fetch, {len(done)} done")
    results, failed = run_jobs(trades, jobs, workers, deadline)
    new = pd.DataFrame([row for got in results.values() for row in got])
    if len(new):
        new["ts"] = to_utc(new.created_time).map(lambda x: int(x.timestamp()))
        new = new.drop(columns=["created_time"])
    combined = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(combined):
        save(combined, path)
    done |= set(results)
    done_path.write_text("\n".join(sorted(done)) + "\n")
    remaining = len(jobs) - len(results)
    annotate(
        "notice" if not failed and not remaining else "warning",
        f"trades: fetched {len(results)} markets, failed {len(failed)}, "
        f"remaining {remaining}; {len(combined)} trade rows. "
        f"First failures: {list(failed.items())[:3]}",
    )


# --- Forecasts ---------------------------------------------------------------


def year_chunks(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    while start <= end:
        stop = min(start + pd.DateOffset(months=6) - pd.Timedelta(days=1), end)
        yield start.date().isoformat(), stop.date().isoformat()
        start = stop + pd.Timedelta(days=1)


def open_meteo(end):
    """Hourly 2 m temperature as forecast 0, 1, 2 and 3 days earlier, per model."""
    variables = ["temperature_2m"] + [
        f"temperature_2m_previous_day{d}" for d in (1, 2, 3)
    ]
    frames, notes = [], {}
    for model in OPEN_METEO_MODELS:
        got = 0
        for a, b in year_chunks("2024-01-01", end):
            params = {
                "latitude": LAT,
                "longitude": LON,
                "hourly": ",".join(variables),
                "start_date": a,
                "end_date": b,
                "temperature_unit": "fahrenheit",
                "timezone": "GMT",
            }
            if model != "best_match":
                params["models"] = model
            url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + (
                urllib.parse.urlencode(params)
            )
            try:
                data = json.loads(fetch(url))
            except (OSError, ValueError) as error:
                notes.setdefault(model, []).append(f"{a}: {str(error)[:160]}")
                continue
            hourly = data.get("hourly", {})
            f = pd.DataFrame(hourly)
            if f.empty:
                continue
            suffix = f"_{model}"
            f = f.rename(columns={c: c.replace(suffix, "") for c in f.columns})
            f["model"] = model
            frames.append(f)
            got += len(f)
            time.sleep(1.5)
        notes.setdefault(model, []).append(f"{got} hourly rows")
    frame = pd.concat(frames, ignore_index=True)
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    save(frame, OUT / "open_meteo_previous_runs.parquet")
    return notes


def iem_mos(end):
    """NBM (NBS) and GFS (GFS) MOS text forecasts for KMIA, every run."""
    notes, frames = {}, []
    for model in ["NBS", "GFS"]:
        got = 0
        first_error = None
        start = pd.Timestamp(FIRST_DAY)
        stop_all = pd.Timestamp(end) + pd.Timedelta(days=1)
        while start < stop_all:
            stop = min(start + pd.DateOffset(months=1), stop_all)
            params = {
                "station": "KMIA",
                "model": model,
                "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
                "ets": stop.strftime("%Y-%m-%dT%H:%MZ"),
                "format": "csv",
            }
            url = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?" + (
                urllib.parse.urlencode(params)
            )
            try:
                text = fetch(url, timeout=300)
                f = pd.read_csv(io.StringIO(text))
                if len(f):
                    f["model_name"] = model
                    frames.append(f)
                    got += len(f)
            except (OSError, ValueError) as error:
                first_error = first_error or f"{start.date()}: {str(error)[:200]}"
            start = stop
            time.sleep(1)
        notes[model] = {"rows": got, "first_error": first_error}
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        save(
            frame.astype({c: str for c in frame.columns if frame[c].dtype == object}),
            OUT / "iem_mos.parquet",
        )
        notes["columns"] = list(frame.columns)
    return notes


def get_forecasts():
    end = pd.Timestamp.now(tz="UTC").date().isoformat()
    notes = {"open_meteo": open_meteo(end), "iem_mos": iem_mos(end)}
    (OUT / "forecast_notes.json").write_text(json.dumps(notes, indent=1) + "\n")
    annotate("notice", "forecasts: " + json.dumps(notes)[:3500])


# --- Observations --------------------------------------------------------------


def get_obs():
    end = pd.Timestamp.now(tz="UTC")
    notes = {}
    rows = []
    for year in range(pd.Timestamp(FIRST_DAY).year, end.year + 1):
        url = f"https://mesonet.agron.iastate.edu/json/cli.py?station=KMIA&year={year}"
        try:
            rows += json.loads(fetch(url)).get("results", [])
        except (OSError, ValueError) as error:
            notes[f"cli_{year}"] = str(error)[:200]
        time.sleep(1)
    cli = pd.DataFrame(rows)
    keep = [
        c
        for c in [
            "station",
            "valid",
            "product",
            "high",
            "high_time",
            "low",
            "low_time",
            "high_normal",
            "low_normal",
            "precip",
        ]
        if c in cli
    ]
    cli = cli[keep].astype(str)
    save(cli, OUT / "iem_cli.parquet")
    notes["cli_rows"] = len(cli)

    frames = []
    for a, b in year_chunks(FIRST_DAY, end.date().isoformat()):
        a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)
        params = [
            ("station", "MIA"),
            ("data", "tmpf"),
            ("data", "metar"),
            ("year1", a.year),
            ("month1", a.month),
            ("day1", a.day),
            ("year2", b.year),
            ("month2", b.month),
            ("day2", b.day),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("missing", "M"),
            ("report_type", "3"),
            ("report_type", "4"),
        ]
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + (
            urllib.parse.urlencode(params)
        )
        try:
            text = fetch(url, timeout=600)
            frames.append(pd.read_csv(io.StringIO(text), dtype=str))
        except (OSError, ValueError) as error:
            notes[f"asos_{a.date()}"] = str(error)[:200]
        time.sleep(2)
    asos = pd.concat(frames, ignore_index=True)
    save(asos, OUT / "iem_metars.parquet")
    notes["metar_rows"] = len(asos)
    (OUT / "obs_notes.json").write_text(json.dumps(notes, indent=1) + "\n")
    annotate("notice", "obs: " + json.dumps(notes)[:3500])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "part", choices=["markets", "candles", "trades", "forecasts", "obs"]
    )
    parser.add_argument("--minutes", type=float, default=300)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "markets":
        get_markets()
    elif args.part == "candles":
        get_candles(args.minutes, args.workers)
    elif args.part == "trades":
        get_trades(args.minutes, args.workers)
    elif args.part == "forecasts":
        get_forecasts()
    else:
        get_obs()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
