"""Model refresh study (RESEARCH_LOG.md, 2026-10-17): does any of the standard
toolkit beat the frozen Poisson model, on the 2025-26 holdout only?

Variants of the mean layer (spread layer unchanged):

    control  the frozen Poisson regression
    gbm      gradient-boosted trees, Poisson loss, settings fixed in the log
    hier     Poisson regression x a Gamma-Poisson player factor (partial pooling);
             prior strength chosen on the last training month, never the test month
    iso      control probabilities through an isotonic calibrator fitted on every
             earlier out-of-sample prediction

Judged on log loss over every player-game-line, and on the "no" seller's veto
replayed on the season's resting orders (profit per 100 eligible orders, one
contract each, unfilled = 0), with a game-clustered t on the difference from
the control. Plus the portfolio ordering test (edge / sd) in the desk replay.

    python -m research.model_refresh   # offline, writes outputs/model_refresh/*
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from kalshi.maker import maker_fee
from kalshi.stats import clustered_mean_t
from markets.nba.evaluate import FIRST_TRAIN_SEASON, log_loss, month_of
from markets.nba.features import build_features, model_rows
from markets.nba.marginals import STATS, StatModel
from research import desk

HOLDOUT = 2026
ORDERS = Path("outputs/decompose/orders_no.csv.gz")
DEV = Path("outputs/backtest/development/predictions.parquet")
OUT = Path("outputs/model_refresh")
FEE_SHARE = 0.25
GBM = {
    "loss": "poisson",
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "random_state": 0,
}
PRIORS = [10, 30, 100, 300]
POOL_SEASONS = 2


class GBMModel(StatModel):
    def __init__(self, spec):
        self.spec = spec
        self.glm = HistGradientBoostingRegressor(**GBM)

    def fit(self, rows):
        rows = rows[rows[self.spec.name].notna()]
        y = rows[self.spec.name].to_numpy(float)
        age = rows.season.max() - rows.season.to_numpy()
        self.glm.fit(
            self.spec.features(rows), np.maximum(y, 0), sample_weight=0.5 ** (age / 3)
        )
        self._fit_spread(rows, y)
        self.n_train = len(rows)
        return self

    def mean(self, rows):
        return np.maximum(self.glm.predict(self.spec.features(rows)), 0.05)


class HierModel(StatModel):
    """Poisson regression times a shrunk per-player factor (Gamma-Poisson)."""

    def __init__(self, spec, prior):
        super().__init__(spec)
        self.prior = prior
        self.factor = {}

    def fit(self, rows):
        rows = rows[rows[self.spec.name].notna()]
        y = rows[self.spec.name].to_numpy(float)
        age = rows.season.max() - rows.season.to_numpy()
        self.glm.fit(
            self.spec.features(rows),
            np.maximum(y, 0),
            poissonregressor__sample_weight=0.5 ** (age / 3),
        )
        recent = (rows.season > rows.season.max() - POOL_SEASONS).to_numpy()
        base = self.glm.predict(self.spec.features(rows))
        s = (
            pd.DataFrame(
                {
                    "pid": rows.player_id.to_numpy()[recent],
                    "y": y[recent],
                    "mu": base[recent],
                }
            )
            .groupby("pid")
            .sum()
        )
        self.factor = ((s.y + self.prior) / (s.mu + self.prior)).to_dict()
        self._fit_spread(rows, y)
        self.n_train = len(rows)
        return self

    def mean(self, rows):
        f = rows.player_id.map(self.factor).fillna(1.0).to_numpy(float)
        return self.glm.predict(self.spec.features(rows)) * f


def choose_prior(train, spec):
    """Prior strength by log loss on the last training month (nested)."""
    last = train.month.max()
    inner, valid = train[train.month < last], train[train.month == last]
    best, best_loss = PRIORS[0], np.inf
    for a in PRIORS:
        m = HierModel(spec, a).fit(inner)
        mean = m.mean(valid)
        y = valid[spec.name].to_numpy(float)
        loss = np.mean(
            [
                log_loss(
                    (y > line).astype(float), m.prob_over(valid, line, mean)
                ).mean()
                for line in spec.lines
            ]
        )
        if loss < best_loss:
            best, best_loss = a, loss
    return best


def load_orders():
    o = pd.read_csv(ORDERS, low_memory=False)
    o["filled"] = o.filled.astype(str).str.lower().eq("true")
    for c in ("price", "pnl_if_filled", "line", "p_model", "model_edge"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o["game_id"] = o.game_id.astype(int)
    o["player_id"] = o.player_id.astype(int)
    o["month"] = month_of(
        pd.DataFrame({"tip_utc": pd.to_datetime(o.tip_utc, utc=True)})
    )
    return o


def walk(rows, orders):
    """Per month: fit each variant, price every line and every order."""
    rows = rows[rows.season >= FIRST_TRAIN_SEASON].copy()
    rows["month"] = month_of(rows)
    rows["player_id"] = rows.player_id.astype(int)
    rows["game_id"] = rows.game_id.astype(int)
    months = sorted(rows.loc[rows.season == HOLDOUT, "month"].unique())
    lines_out, orders_out, priors = [], [], {}
    for month in months:
        test = rows[rows.month == month]
        train = rows[rows.month < month]
        keyed = test.set_index(["game_id", "player_id"])
        ask_month = orders[orders.month == month]
        for name, spec in STATS.items():
            a = choose_prior(train, spec)
            priors[f"{month}/{name}"] = a
            models = {
                "control": StatModel(spec).fit(train),
                "gbm": GBMModel(spec).fit(train),
                "hier": HierModel(spec, a).fit(train),
            }
            y = test[name].to_numpy(float)
            means = {k: m.mean(test) for k, m in models.items()}
            for line in spec.lines:
                rec = {
                    "month": str(month),
                    "stat": name,
                    "line": line,
                    "game_id": test.game_id.to_numpy(),
                    "over": (y > line).astype(float),
                }
                for k, m in models.items():
                    rec[f"p_{k}"] = m.prob_over(test, line, means[k])
                lines_out.append(pd.DataFrame(rec))
            ask = ask_month[ask_month.stat == name]
            idx = pd.MultiIndex.from_frame(ask[["game_id", "player_id"]])
            ask = ask[idx.isin(keyed.index)]
            if ask.empty:
                continue
            feats = keyed.loc[
                pd.MultiIndex.from_frame(ask[["game_id", "player_id"]])
            ].reset_index()
            rec = {"ticker": ask.ticker.to_numpy()}
            for k, m in models.items():
                rec[f"p_{k}"] = m.prob_over(
                    feats, ask.line.to_numpy(float), m.mean(feats)
                )
            orders_out.append(pd.DataFrame(rec))
        print(month, "done", flush=True)
    return (
        pd.concat(lines_out, ignore_index=True),
        pd.concat(orders_out, ignore_index=True),
        priors,
    )


def isotonic(lines, order_p, orders):
    """Calibrate control probabilities on earlier out-of-sample predictions only."""
    dev = pd.read_parquet(DEV)[["stat", "p_model", "over", "month"]]
    lines = lines.copy()
    lines["p_iso"] = lines.p_control
    op = orders.merge(order_p, on="ticker", how="inner")
    op["p_iso"] = op.p_control
    op["month"] = op.month.astype(str)
    for stat in STATS:
        for month in sorted(lines.month.unique()):
            prior_h = lines[(lines.stat == stat) & (lines.month < month)]
            prior = pd.concat(
                [
                    dev[dev.stat == stat][["p_model", "over"]],
                    prior_h[["p_control", "over"]].rename(
                        columns={"p_control": "p_model"}
                    ),
                ]
            )
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                prior.p_model, prior.over
            )
            here = (lines.stat == stat) & (lines.month == month)
            lines.loc[here, "p_iso"] = iso.predict(lines.loc[here, "p_control"])
            h2 = (op.stat == stat) & (op.month == month)
            if h2.any():
                op.loc[h2, "p_iso"] = iso.predict(op.loc[h2, "p_control"])
    return lines, op


VARIANTS = ["control", "gbm", "hier", "iso"]


def score_lines(lines):
    out = []
    for v in VARIANTS:
        ll = log_loss(lines.over, lines[f"p_{v}"])
        d = ll - log_loss(lines.over, lines.p_control)
        _, t = clustered_mean_t(d, lines.game_id)
        out.append(
            {
                "variant": v,
                "log_loss": round(float(ll.mean()), 5),
                "vs_control": round(float(d.mean()), 5),
                "t_vs_control": round(float(t), 2) if np.isfinite(t) else None,
                "brier": round(float(((lines[f"p_{v}"] - lines.over) ** 2).mean()), 5),
            }
        )
    return out


def score_veto(op):
    fee = maker_fee(op.price, FEE_SHARE)
    pnl_fill = np.where(op.filled, op.pnl_if_filled, 0.0)
    out, per = [], {}
    for v in VARIANTS:
        edge = (1 - op[f"p_{v}"]) - op.price - fee
        place = edge >= 0
        per[v] = np.where(place, pnl_fill, 0.0)
        mean, t = clustered_mean_t(per[v], op.game_id)
        placed = op[place]
        out.append(
            {
                "variant": v,
                "orders_quoted": len(op),
                "placed": int(place.sum()),
                "fills": int(placed.filled.sum()),
                "profit_per_100_orders": round(100 * 100 * float(mean), 1),
                "t": round(float(t), 2),
                "cents_per_fill": round(
                    100 * float(placed.pnl_if_filled[placed.filled].mean()), 2
                )
                if placed.filled.any()
                else None,
            }
        )
    for r in out:
        d = per[r["variant"]] - per["control"]
        m, t = clustered_mean_t(d, op.game_id)
        r["diff_per_100_orders"] = round(100 * 100 * float(m), 1)
        r["t_diff"] = round(float(t), 2) if np.isfinite(t) else None
    agree = {
        "control_matches_frozen_rule": round(
            float(
                (
                    ((1 - op.p_control) - op.price - fee >= 0) == (op.model_edge >= 0)
                ).mean()
            ),
            4,
        )
    }
    return out, agree


def portfolio():
    o = desk.load_orders()
    o["p_model"] = pd.to_numeric(o.p_model, errors="coerce")
    o["sharpe"] = o.edge / np.sqrt((o.p_model * (1 - o.p_model)).clip(lower=0.01))
    days = sorted(o.day.unique())
    cut = days[len(days) // 2]
    rows = []
    for label, frame in [
        ("whole season", o),
        ("first half", o[o.day < cut]),
        ("second half", o[o.day >= cut]),
    ]:
        for rank, name in [
            ("edge", "edge-first"),
            ("sharpe", "edge / sd"),
            (None, "random"),
        ]:
            path, d = desk.replay(frame, 500.0, rank)
            rows.append(
                {
                    "period": label,
                    "ordering": name,
                    "end": round(float(path[-1])),
                    "max_drawdown": round(desk.drawdown(path), 3),
                    "orders": int(d.placed.sum()),
                    "fills": int(d.filled.sum()),
                    "profit_per_100_orders": round(
                        100 * float(d.profit.sum() / max(d.placed.sum(), 1)), 2
                    ),
                }
            )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    players = pd.read_parquet("data/player_games.parquet")
    games = pd.read_parquet("data/games.parquet")
    rows = model_rows(build_features(players, games))
    orders = load_orders()
    cache = OUT / "lines.parquet", OUT / "order_p.parquet", OUT / "priors.json"
    if all(p.exists() for p in cache):
        lines, order_p = pd.read_parquet(cache[0]), pd.read_parquet(cache[1])
        priors = json.loads(cache[2].read_text())
    else:
        lines, order_p, priors = walk(rows, orders)
        lines.to_parquet(cache[0], index=False)
        order_p.to_parquet(cache[1], index=False)
        cache[2].write_text(json.dumps(priors, indent=1))
    lines, op = isotonic(lines, order_p, orders)
    res = {
        "holdout": "2025-26",
        "player_game_lines": len(lines),
        "orders_priced": len(op),
        "priors_chosen": priors,
        "log_loss": score_lines(lines),
        "veto": None,
        "portfolio": portfolio(),
    }
    res["veto"], res["agreement"] = score_veto(op)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    md = [
        "# Model refresh study, 2025-26 holdout",
        "",
        f"{len(lines):,} player-game-lines; {len(op):,} resting orders re-priced.",
        "",
        "| Variant | Log loss | vs control | t | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in res["log_loss"]:
        md.append(
            f"| {r['variant']} | {r['log_loss']:.5f} | {r['vs_control']:+.5f} | {r['t_vs_control']} | {r['brier']:.5f} |"
        )
    md += [
        "",
        "| Variant | Placed | Fills | Profit per 100 quoted orders (¢) | t | ¢/fill | Diff vs control | t |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res["veto"]:
        md.append(
            f"| {r['variant']} | {r['placed']:,} | {r['fills']:,} | {r['profit_per_100_orders']:+.1f} | {r['t']} | {r['cents_per_fill']} | {r['diff_per_100_orders']:+.1f} | {r['t_diff']} |"
        )
    md += [
        "",
        "| Period | Ordering | End ($500 start) | Max DD | Orders | Fills | Profit per 100 orders (¢) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in res["portfolio"]:
        md.append(
            f"| {r['period']} | {r['ordering']} | {r['end']:,} | {r['max_drawdown']:.1%} | {r['orders']:,} | {r['fills']:,} | {r['profit_per_100_orders']:+.2f} |"
        )
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(res["agreement"])


if __name__ == "__main__":
    main()
