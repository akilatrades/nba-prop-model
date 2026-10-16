"""Forecast model for the Miami daily high (RESEARCH_LOG.md, weather test).

Inputs are Open-Meteo's saved forecasts from earlier model runs. For a date
D, the "day 1" forecast high of a model is the highest hourly temperature it
forecast for 12:00-23:00 UTC on D (8 a.m.-7 p.m. EDT) in the runs about one
day earlier; "day 2" the same from runs about two days earlier. Both come
from runs started at least 24 (or 48) hours before those hours, so day 1 is
complete by 02:00 UTC on D (10 p.m. EDT the evening before) and day 2 by
02:00 UTC on D - 1.

A model's forecast more than 5 F from the median of the models that day is
dropped (one model's saved forecasts contain some broken values). Each
model's forecast is corrected by its mean error over a trailing window
of past days, the corrected forecasts are averaged, and the high is taken as
normal around that average with 1.2 times the root-mean-square of its past
errors (the factor set on 2024 dates, where the plain spread was too narrow),
rounded to a whole degree. Only days whose official high was already published are used:
for date D, days D - WINDOW - 1 to D - 2.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from wx import DATA

MODELS = [
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "gem_seamless",
    "ncep_nbm_conus",
]
LEAD_COLUMNS = {
    1: "temperature_2m_previous_day1",
    2: "temperature_2m_previous_day2",
}
WINDOW = 60
MIN_DAYS = 20
SIGMA_FLOOR = 0.8
SIGMA_SCALE = 1.2  # past errors understate the spread (development dates, 2024)
OUTLIER = 5.0  # a model more than 5 F from the other models' median is dropped


def daily_forecasts():
    """Forecast high per date, lead (1 or 2 days) and model; dates as index."""
    f = pd.read_parquet(DATA / "open_meteo_previous_runs.parquet")
    f = f[f.model.isin(MODELS)]
    f = f[f.time.dt.hour.between(12, 23)]
    f["date"] = f.time.dt.date
    out = {}
    for lead, column in LEAD_COLUMNS.items():
        g = f.groupby(["date", "model"])[column].agg(["max", "count"])
        g = g[g["count"] == 12]["max"].unstack("model").reindex(columns=MODELS)
        median = g.median(axis=1)
        out[lead] = g.where((g.sub(median, axis=0)).abs() <= OUTLIER)
    return out


def rolling_model(forecasts, highs, window=WINDOW, min_days=MIN_DAYS):
    """Mean and spread of the high for every date, from past days only.

    ``forecasts`` has one column per model (index: date); ``highs`` the
    official highs (index: date). Returns a frame with mu, sigma, n_models.
    """
    dates = pd.Index(sorted(forecasts.index))
    errors = highs.reindex(dates).to_numpy()[:, None] - forecasts.reindex(dates)
    errors.index = dates
    rows = []
    for d in dates:
        lo, hi = d - pd.Timedelta(days=window + 1), d - pd.Timedelta(days=2)
        past = errors.loc[(errors.index >= lo) & (errors.index <= hi)]
        counts = past.notna().sum()
        bias = past.mean()[counts >= min_days]
        today = forecasts.loc[d, bias.index].dropna()
        if today.empty:
            continue
        corrected = today + bias[today.index]
        mu = corrected.mean()
        # Past errors of the same average (of the models available today).
        past_f = forecasts.loc[past.index, today.index]
        past_mean = (past_f + bias[today.index]).mean(axis=1, skipna=False)
        past_err = (highs.reindex(past.index) - past_mean).dropna()
        if len(past_err) < min_days:
            continue
        rmse = float(np.sqrt((past_err**2).mean()))
        sigma = max(SIGMA_SCALE * rmse, SIGMA_FLOOR)
        rows.append({"date": d, "mu": mu, "sigma": sigma, "n_models": len(today)})
    return pd.DataFrame(rows).set_index("date")


def prob_between(mu, sigma, lo, hi):
    """P(lo <= rounded high <= hi) for a normal high; lo/hi may be infinite."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    upper = norm.cdf((np.asarray(hi, float) + 0.5 - mu) / sigma)
    lower = norm.cdf((np.asarray(lo, float) - 0.5 - mu) / sigma)
    return np.clip(upper - lower, 0.0, 1.0)


def log_score(mu, sigma, high):
    p = prob_between(mu, sigma, high, high)
    return -np.log(np.clip(p, 1e-6, 1))


# --- On the day: what has been observed so far -------------------------------

OFFSETS = np.arange(-3, 16)
GAP_BINS = [-np.inf, -1, 0, 1, 2, 3, np.inf]
SMOOTH = 3.0
MIX = 0.02  # weight on the forecast-only probability (development dates, 2024)


def intraday_frame(states, mu):
    """Base B (highest whole degree known so far) and features for each state.

    ``states`` has date, h, mh, m6 (wx.obs_states); ``mu`` the day 1
    forecast mean by date.
    """
    s = states.copy()
    six = s.m6.round()
    hourly = s.mh.round()
    s["base"] = np.fmax(six, hourly)
    s["src"] = np.where(six.notna() & (six >= hourly), "six_hour", "hourly")
    s["mu"] = s.date.map(mu)
    s["gap_bin"] = pd.cut(s.mu - s.base, GAP_BINS, labels=False)
    return s


def offset_table(train):
    """P(high - base = j) by hour, source and gap bin, smoothed toward the hour."""
    t = train.dropna(subset=["base", "high"]).copy()
    t["j"] = np.clip(t.high - t.base, OFFSETS[0], OFFSETS[-1]).astype(int)
    pooled = (
        t.groupby("h")
        .j.value_counts()
        .unstack(fill_value=0)
        .reindex(columns=OFFSETS, fill_value=0)
    )
    pooled = (pooled + 0.01).div((pooled + 0.01).sum(axis=1), axis=0)
    u = t.dropna(subset=["gap_bin"])
    cells = (
        u.groupby(["h", "src", "gap_bin"])
        .j.value_counts()
        .unstack(fill_value=0)
        .reindex(columns=OFFSETS, fill_value=0)
    )
    return pooled, cells


def intraday_probs(rows, pooled, cells, lo, hi):
    """P(lo <= high <= hi) for each state row, from the offset table.

    The final on-the-day probability is (1 - MIX) times this plus MIX times
    the forecast-only probability (see ``blend``); on 2024 dates the table
    alone was too sure at both ends (0.1% came true 0.6% of the time, 99.7%
    came true 98.2%).
    """
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows.itertuples()):
        if np.isnan(r.base) or r.h not in pooled.index:
            continue
        prior = pooled.loc[r.h].to_numpy()
        key = (r.h, r.src, r.gap_bin)
        counts = cells.loc[key].to_numpy() if key in cells.index else 0 * prior
        p = (counts + SMOOTH * prior) / (counts.sum() + SMOOTH)
        k = r.base + OFFSETS
        inside = (k >= lo[i]) & (k <= hi[i])
        out[i] = p[inside].sum()
    return out


def blend(p_intraday, p_forecast, mix=MIX):
    return (1 - mix) * np.asarray(p_intraday, float) + mix * np.asarray(
        p_forecast, float
    )
