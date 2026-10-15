"""Shared pieces for the Miami high-temperature test (RESEARCH_LOG.md).

The day that counts: Kalshi's KXHIGHMIA settles on the highest temperature
in the National Weather Service climate report for Miami International
Airport (CLIMIA), whose day runs midnight to midnight local standard time,
05:00 to 05:00 UTC all year (1 a.m. to 1 a.m. during daylight saving time).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("weather/data")
TZ = "America/New_York"
FEE_RATE = 0.07


def climate_day_start(day):
    """UTC start of the climate day for a date (05:00 UTC)."""
    return pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=5)


# --- Markets -----------------------------------------------------------------


def bracket_bounds(markets):
    """Lowest and highest whole-degree high that makes each market settle yes."""
    m = markets.copy()
    floor = pd.to_numeric(m.floor_strike, errors="coerce")
    cap = pd.to_numeric(m.cap_strike, errors="coerce")
    lo = np.where(
        m.strike_type == "between",
        floor,
        np.where(m.strike_type == "greater", np.floor(floor) + 1, -np.inf),
    )
    hi = np.where(
        m.strike_type == "between",
        cap,
        np.where(m.strike_type == "less", np.ceil(cap) - 1, np.inf),
    )
    m["lo"] = lo
    m["hi"] = hi
    return m


def load_markets():
    m = pd.read_parquet(DATA / "kalshi_markets.parquet")
    m = bracket_bounds(m)
    m["date"] = pd.to_datetime(m.event_date).dt.date
    m["yes"] = (m.result == "yes").astype(float)
    m.loc[~m.result.isin(["yes", "no"]), "yes"] = np.nan
    return m


def fee(price, rate=FEE_RATE):
    price = np.asarray(price, dtype=float)
    return rate * price * (1 - price)


# --- Observations ------------------------------------------------------------


def load_cli():
    """Final daily high per date from the NWS climate reports (latest report kept)."""
    c = pd.read_parquet(DATA / "iem_cli.parquet")
    c["high"] = pd.to_numeric(c.high, errors="coerce")
    c = c.dropna(subset=["high"]).sort_values(["valid", "product"])
    c = c.drop_duplicates("valid", keep="last")
    c["date"] = pd.to_datetime(c.valid).dt.date
    return c.set_index("date").high


T_GROUP = re.compile(r"\sT([01])(\d{3})([01])(\d{3})\b")
MAX6 = re.compile(r"\s1([01])(\d{3})(?=\s|$)")
MAX24 = re.compile(r"\s4([01])(\d{3})([01])(\d{3})(?=\s|$)")


def _celsius(sign, digits):
    value = int(digits) / 10
    return -value if sign == "1" else value


def to_f(celsius):
    return np.asarray(celsius, dtype=float) * 9 / 5 + 32


def parse_metar(text):
    """Temperature (T group, tenths of a degree C) and 6- and 24-hour maxima."""
    text = " " + str(text) + " "
    remarks = text.split(" RMK ", 1)[1] if " RMK " in text else ""
    remarks = " " + remarks
    out = {"temp_c": np.nan, "max6_c": np.nan, "max24_c": np.nan}
    t = T_GROUP.search(remarks)
    if t:
        out["temp_c"] = _celsius(t.group(1), t.group(2))
    m6 = MAX6.search(remarks)
    if m6:
        out["max6_c"] = _celsius(m6.group(1), m6.group(2))
    m24 = MAX24.search(remarks)
    if m24:
        out["max24_c"] = _celsius(m24.group(1), m24.group(2))
    return out


def load_metars():
    """Every METAR with its time (UTC), temperature in F and reported maxima in F."""
    raw = pd.read_parquet(DATA / "iem_metars.parquet")
    raw = raw.drop_duplicates(["valid", "metar"])
    parsed = pd.DataFrame([parse_metar(x) for x in raw.metar], index=raw.index)
    out = pd.DataFrame(
        {
            "valid": pd.to_datetime(raw.valid, utc=True),
            "tmpf": pd.to_numeric(raw.tmpf, errors="coerce"),
            "temp_f": to_f(parsed.temp_c),
            "max6_f": to_f(parsed.max6_c),
            "max24_f": to_f(parsed.max24_c),
        }
    )
    # The T group is the precise reading; fall back to IEM's tmpf.
    out["temp_f"] = out.temp_f.fillna(out.tmpf)
    out = out.sort_values("valid").reset_index(drop=True)
    out["date"] = (out.valid - pd.Timedelta(hours=5)).dt.date
    return out
