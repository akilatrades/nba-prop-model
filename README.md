# NBA prop model

[![tests](https://github.com/akilatrades/nba-prop-model/actions/workflows/tests.yml/badge.svg)](https://github.com/akilatrades/nba-prop-model/actions/workflows/tests.yml)

A pricing model for NBA player props (points, rebounds, assists, threes),
tested against a full season of Kalshi prices with the betting rule written
down before any price was downloaded. A sister project to my
[NFL parlay model](https://github.com/akilatrades/nfl-parlay-model).

**Result: the model knows something Kalshi's price doesn't, but not enough
to pay the spread and fees.** On 57,111 Kalshi prop markets from the 2025-26
season, combining the model with the price at tip-off predicts outcomes
better than the price alone (t = 13.8). But the betting rule fixed in
advance (buy whichever side the model rates 5 cents better than the ask,
after fees) lost 0.9 cents per contract over 12,495 trades. It fails the
pre-registered test. The model is a calibrated pricing tool, not a proven
money-maker.

| Phase | What it adds | Status |
|---|---|---|
| 1 | A full range of outcomes for each player stat, from minutes, per-minute rates, the betting line, the opponent and teammates who are out; walk-forward backtest | Done |
| 2 | Test against Kalshi's 2025-26 prices at tip-off, after fees, with the rule fixed in advance | Done |
| 3 | Live: Kalshi price log and model prices for every open prop, from opening night | Running |

## Phase 1: the model

**On the 2025-26 season, run once after the model was frozen, its
probabilities were calibrated to within about half a percentage point in
every band, and it beat each player's own recent average on all four stats.**
The gain is about 2% overall but under 1% for heavy-minute players, the ones
Kalshi lists.

![Calibration](outputs/backtest/holdout/calibration.svg)

| Stat | Player-games | Typical miss: model vs recent average | Brier improvement | Log loss improvement | Players averaging 26+ minutes: Brier improvement |
|---|---:|---:|---:|---:|---:|
| Points | 25,511 | 4.70 vs 4.80 | 2.0% | 2.2% | 0.8% |
| Rebounds | 25,511 | 1.92 vs 1.96 | 2.3% | 2.4% | 0.8% |
| Assists | 25,511 | 1.38 vs 1.41 | 2.0% | 2.3% | 0.5% |
| Threes made | 25,511 | 0.93 vs 0.94 | 1.6% | 1.6% | 0.9% |

"Typical miss" is the mean absolute error of the expected value. Brier score
and log loss score the probabilities at five lines per stat (for points
10+, 15+, 20+, 25+, 30+ and 35+); the improvement is relative to a baseline
that uses each player's own recent average with the same spread of outcomes,
so the comparison isolates what the extra pre-game information adds. The
development seasons (2021-22 to 2024-25) gave the same picture:
[development](outputs/backtest/development/summary.csv),
[holdout](outputs/backtest/holdout/summary.csv),
[by line](outputs/backtest/holdout/by_line.csv).

## How it works

1. **Data** ([`fetch_data.py`](fetch_data.py)): ESPN box scores for every player
   and game since 2017-18, including who did not play and why, and the
   consensus closing spread and total for each game since 2020-21, from the
   sportsdataverse hoopR data repository.
2. **Pre-game features, from earlier games only**
   ([`features.py`](nba_model/features.py)):
   - *Minutes:* recent minutes (half-life of five games), minutes last game,
     share of recent starts, team games missed since he last played,
     back-to-backs, playoffs, and the size of the spread (big favourites rest
     starters late).
   - *Rates:* points, rebounds, assists and three-point attempts per minute
     over his last 30 games, shrunk toward his position's rate from earlier
     seasons by 150 minutes of average play.
   - *Game:* the team's implied points from the closing total and spread, and
     what the opponent has allowed in each stat relative to the league.
   - *Teammates out and back:* the share of the team's recent minutes and of
     each stat that belongs to regulars (15+ minutes) who played in the
     team's last three games but are out tonight, split by position group,
     and the share belonging to regulars returning tonight.
3. **Expected value** ([`marginals.py`](nba_model/marginals.py)): a Poisson
   regression per stat, with seasons weighted so one three years older counts
   half as much.
4. **Range of outcomes:** the distribution of actual ÷ expected in the latest
   three seasons, kept separately for ten bands of expected value within three
   tiers of recent minutes (under 20, 20–30, 30+). This keeps the real shape,
   including zero-three nights and 40-point games.
5. **Backtest** ([`evaluate.py`](nba_model/evaluate.py)): refit before every
   calendar month on all earlier games, scored on the month it has not seen.

## Phase 2: the market test

Kalshi publishes hourly bid and ask history for every settled market, so the
model could be tested against a full season of real prices: every points,
rebounds, assists and threes market from Kalshi's first NBA props
(2025-11-18) through the Finals. The rules were committed to
[`RESEARCH_LOG.md`](RESEARCH_LOG.md) before the prices were downloaded, the
model was frozen (commit `ce26c24`), and the test was run once
([`run_market_test.py`](run_market_test.py)).

**Sample:** markets that settled yes or no (markets on players who sat are
settled at a fixed value and excluded), priced at the last hourly quote
before tip-off, with a two-sided quote: 57,111 markets in 1,072 games.

**Test A, does the model add information? Passed.** In a logistic regression
of the outcome on both the market's mid price and the model's probability
(log-odds; standard errors clustered by game), the model's weight is 0.44
(t = 13.8) next to the market's 0.56 (t = 17.9). Alone, the market is
slightly more accurate (Brier score 0.1805 against the model's 0.1812).

**Test B, does it make money? Failed.**

| Rule | Trades | Profit per contract, after fees | t |
|---|---:|---:|---:|
| Model's side when its edge is 5¢+ after fees (pre-registered) | 12,495 | −0.9¢ | −1.5 |
| same, games before February | 2,395 | −3.4¢ | −2.5 |
| same, February to the Finals | 10,100 | −0.4¢ | −0.5 |
| same, 8¢ threshold | 6,933 | −0.9¢ | −1.1 |
| same, fees rounded up per contract | 12,495 | −1.4¢ | −2.2 |
| same, prices two hours before tip-off | 11,817 | −0.4¢ | −0.6 |
| same, without the teammates-out features | 12,130 | −0.8¢ | −1.2 |
| *For comparison: buy every yes at the ask* | 57,111 | −3.4¢ | |
| *For comparison: buy every no at 1 − bid* | 57,111 | −6.0¢ | |

The model picks better sides than chance: its trades lost 0.9¢ where buying
everything lost 3.4–6.0¢. But a round trip through the median 4¢ spread plus
Kalshi's fee costs more than the model's edge. The trades it was most
confident in did no better than the rest: claimed edges of 6¢, 10¢ and 15¢
all realised about −1¢. A likely reason: when the model disagrees most with
the price at tip-off, the price reflects news the model does not have, such
as starting lineups. [Results](outputs/market_test/results.json) ·
[trades](outputs/market_test/trades.csv.gz) ·
[every priced market](outputs/market_test/priced_markets.csv.gz) ·
[independent recomputation](scripts/verify_market_test.py)

One exploratory check, run after the test and not pre-registered: blending
the model with the market (weights refit each month on earlier months) and
trading the blend lost 2.1¢ per contract. Nothing found here is a basis for
trading money.

## Live use

Three scheduled workflows run during the season:

- [`kalshi.yml`](.github/workflows/kalshi.yml) saves every open Kalshi NBA
  prop market each morning and before each evening's tip-offs
  ([`log_kalshi.py`](log_kalshi.py)), into `data/kalshi/`.
- [`projections.yml`](.github/workflows/projections.yml) refreshes the box
  scores, reads ESPN's schedule, lines and injury report (Out and Doubtful
  players sit), refits the models and prices every open Kalshi prop
  ([`run_projections.py`](run_projections.py)). Output:
  [`outputs/projections/latest.csv`](outputs/projections/) with the model's
  expected value and probability, Kalshi's bid and ask, and the edge on each
  side after fees.
- [`backfill.yml`](.github/workflows/backfill.yml) downloaded the settled
  2025-26 markets and their hourly prices for the market test
  ([`scripts/backfill_kalshi.py`](scripts/backfill_kalshi.py)).

## Reproduce

```bash
pip install -r requirements.txt
python fetch_data.py               # box scores, schedules and lines (about 10 MB)
python run_backtest.py             # development seasons; --holdout for 2025-26
python run_market_test.py          # needs data/kalshi_history/ from the backfill
pytest                             # unit tests on synthetic games, no data needed
```

[`RESEARCH_LOG.md`](RESEARCH_LOG.md) records every decision in order,
including the market test as written before any price was seen.

## Limitations

- In the backtest, "teammates out" uses who actually sat. A final injury
  report shows most of that before tip-off but not late scratches, so the
  backtest has a small head start the live model will not; the market test
  reports results with those features switched off.
- The model does not know tonight's starting lineups or minutes limits
  announced shortly before tip-off. The market does.
- Hourly Kalshi candles give the best bid and ask, not the size behind them.
- One season of Kalshi prices is a short record for a small edge.

## AI use

I used AI tools to help write code and documentation and to review my work.
I reviewed the analytical assumptions, methods and results and remain
responsible for the claims made here.

## License

MIT. Box scores and lines come from the sportsdataverse hoopR data
repository; Kalshi prices from Kalshi's public API.
