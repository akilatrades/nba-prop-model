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

**After the test,** the losses turned out to come from trading cost: the
model's picks beat the mid price by 2.4 cents but crossing the spread and
paying the fee cost 3.3. A rule that trades only tight markets, hours before
tip-off, would have broken even in the 2025-26 regular season and made money
in the playoffs. That was found after the fact, so it is being paper traded
through 2026-27 ([Phase 4](#phase-4-the-2026-27-rule-paper-traded)).

| Phase | What it adds | Status |
|---|---|---|
| 1 | A full range of outcomes for each player stat, from minutes, per-minute rates, the betting line, the opponent and teammates who are out; walk-forward backtest | Done |
| 2 | Test against Kalshi's 2025-26 prices at tip-off, after fees, with the rule fixed in advance | Done |
| 3 | Live: Kalshi price log and model prices for every open prop, from opening night | Running |
| 4 | A cost-aware trading rule, paper traded through 2026-27 with a $500 paper bankroll | Running |
| 5 | Three other edges: limit orders (passed), sportsbook lines (no free data), correlation (failed) | Done |
| 6 | "No" seller and model-side resting orders paper traded; in-game trading tested (passed, speed-dependent); reward programs checked | Paper trading from opening night |

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

1. **Data** ([`live/fetch_nba_data.py`](live/fetch_nba_data.py)): ESPN box scores for every player
   and game since 2017-18, including who did not play and why, and the
   consensus closing spread and total for each game since 2020-21, from the
   sportsdataverse hoopR data repository.
2. **Pre-game features, from earlier games only**
   ([`features.py`](markets/nba/features.py)):
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
3. **Expected value** ([`marginals.py`](markets/nba/marginals.py)): a Poisson
   regression per stat, with seasons weighted so one three years older counts
   half as much.
4. **Range of outcomes:** the distribution of actual ÷ expected in the latest
   three seasons, kept separately for ten bands of expected value within three
   tiers of recent minutes (under 20, 20–30, 30+). This keeps the real shape,
   including zero-three nights and 40-point games.
5. **Backtest** ([`evaluate.py`](markets/nba/evaluate.py)): refit before every
   calendar month on all earlier games, scored on the month it has not seen.

## Phase 2: the market test

Kalshi publishes hourly bid and ask history for every settled market, so the
model could be tested against a full season of real prices: every points,
rebounds, assists and threes market from Kalshi's first NBA props
(2025-11-18) through the Finals. The rules were committed to
[`RESEARCH_LOG.md`](RESEARCH_LOG.md) before the prices were downloaded, the
model was frozen (commit `ce26c24`), and the test was run once
([`research/market_test.py`](research/market_test.py)).

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
[independent recomputation](research/verify_market_test.py)

One exploratory check, run after the test and not pre-registered: blending
the model with the market (weights refit each month on earlier months) and
trading the blend lost 2.1¢ per contract. Nothing found here is a basis for
trading money.

## Phase 4: the 2026-27 rule, paper traded

Everything in this section was found after the market test failed, on the
same season, so it is a hypothesis, not a result
([log](RESEARCH_LOG.md), [code](research/strategy_backtest.py),
[results](outputs/strategy/results.json)).

**Where the money went.** Priced at the mid, the model's picks made 2.4¢
per contract; buying at the ask and paying the fee cost 3.3¢. Markets with a
1¢ spread made money after fees (+2.1¢ per contract); markets with spreads
of 5¢ or more lost 5.5¢. Prices a few hours before tip-off were softer than
prices at tip-off, which already reflect lineup news.

**The rule** ([`strategy.py`](kalshi/strategy.py)): take the price 2–6
hours before tip-off; trade only markets with a spread of 2¢ or less; buy the
side the model rates at least 5¢ better than its cost after the fee; one
line per player, stat and game; skip anyone on the injury report. Stake 1% of
the bankroll per position (about $5 on $500), at most 30% of it per day.

**Replayed on 2025-26:** 3,928 positions, +1.9¢ per contract (t 2.3). But
the regular season broke even (+0.02¢ per contract over 2,493 positions);
all of the profit came in the play-in and playoffs (+5.1¢ per contract,
t 3.6, 1,435 positions in 91 games). Starting from $500:

| Part of season | Edge assumed | Median end | 1 in 10 ends below | Chance of a loss |
|---|---|---:|---:|---:|
| Regular season | as in 2025-26 (none) | $496 | $235 | 50% |
| Play-in and playoffs | as in 2025-26 | $2,907 | $1,494 | 0% |
| Play-in and playoffs | half of 2025-26 | $1,233 | $610 | 5% |
| Play-in and playoffs | none | $513 | $251 | 48% |

Ranges come from resampling 2025-26's trading days 2,000 times. The real
2025-26 path went from $500 to $390 by early January, $509 by April and
$2,618 by the Finals.

**What runs now.** Four times a day the projections workflow settles
yesterday's positions from Kalshi's own results, prices every open prop,
and adds the rule's positions to a paper ledger:
[today's bet card](outputs/bets/latest.md),
[scorecard](outputs/paper/scorecard.md),
[ledger](outputs/paper/ledger.csv). Regular-season positions are marked
paper; play-in and playoff positions are marked as the 2027 playoff test. The
pass criteria were committed before the season (RESEARCH_LOG.md).

## Phase 5: three other ways to make money

The first approach (out-predict the price, then pay the spread) failed, so
three different sources of edge were tested on 2025-26, each with its rules
committed before its data was downloaded
([log](RESEARCH_LOG.md), [code](research/new_ideas.py),
[results](outputs/new_ideas/results.json)).

| Idea | What it does | Result |
|---|---|---|
| **1. Be the maker** | Rest a limit order on the model's side, one cent inside the spread, cancel at tip-off; fills read from Kalshi's trade prints | **Passed.** 4,541 fills, +3.1¢ per contract after a maker fee (t 4.1); regular season +2.6¢ (t 3.0) |
| 2. Copy the sportsbooks | Bet Kalshi back toward the sportsbook line | Not testable for free: ESPN stores book props for only 176 of 1,072 games, without over/under labels |
| 3. Structure and correlation | Ladder arbitrage; double-doubles priced from Kalshi's own markets plus fitted correlations | Failed. Ladders are consistent; Kalshi already prices the correlation |

Idea 1 works because the model's information is worth about 2–4¢ a
contract, which is less than the cost of crossing the spread but more than
nothing once you collect the spread instead. Most fills are on the "no"
side: offering "no" just inside the ask, to traders buying "yes". Orders fill
mostly when the market moves against them (had all of them filled they would
have made 11¢ a contract), but the fills still made money.

Starting from $500 (1% per order, at most 30% of the bankroll resting per
day), resampling 2025-26's days 2,000 times:

| Edge assumed | Median end of season | 1 in 10 ends below | Chance of a loss |
|---|---:|---:|---:|
| As in 2025-26 | $2,010 | $1,203 | 0% |
| Half of 2025-26 | $953 | $550 | 6% |
| None | $445 | $270 | 61% |

It needs a program placing and cancelling about 80 orders a day through
Kalshi's API; the simulation cannot know how other traders would have
reacted to those orders. [Independent recomputation](research/verify_maker.py)

## Phase 6: the "no" seller, trading during games, and rewards

Three follow-ups, each with its rules committed before it was run
([log](RESEARCH_LOG.md)).

**The "no" seller (exploratory, now paper traded).** Posting "no" one cent
inside the ask on every market, with no model, made +2.8¢ per filled
contract on 2025-26 (t 5.3); "yes" everywhere lost. Kalshi's prop traders
mostly buy "yes", and selling them "no" pays. With the model vetoing orders
it rates below zero, on every line: +2.9¢ (t 6.1). This was found after the
fact, so from opening night two resting-order rules are paper traded from
separate $500 paper bankrolls, filled from Kalshi's public trade prints
([scorecard](outputs/paper_maker/scorecard.md),
[orders](outputs/paper_maker/orders.csv)).

**Trading during games (passed, but it is about speed).** An in-game model
built from ESPN play-by-play (points, rebounds, assists and threes so far,
minutes on court rebuilt from substitutions, fouls, score, time left),
fitted on earlier seasons, priced 47,570 live Kalshi quotes in 2025-26.
Trading where it disagreed by 5¢ after fees: **+4.2¢ per contract over 4,978
trades (t 6.3)**, +3.6¢ in the regular season
([results](outputs/ingame/results.json), [code](research/ingame.py)). The profit
depends on seeing plays quickly:

| Game state known | ¢ per contract | t |
|---|---:|---:|
| at the quote | +4.2 | 6.3 |
| 1 minute late | +3.2 | 4.6 |
| 3 minutes late | +2.2 | 3.1 |
| 5 minutes late | +0.3 | 0.5 |

Running it live needs a program that reads plays and Kalshi's order book
within seconds; in-game spreads are wide (median 11¢) and thin.

**Liquidity rewards: not on NBA props.** Of 325,341 Kalshi reward programs,
8 were on NBA player props ($800 in total). The money is on weather,
Rotten Tomatoes, crypto, tennis, elections, gasoline prices and NFL fantasy
points ([summary](outputs/incentives/summary.json), checked daily).

## Live use

Three scheduled workflows run during the season:

- [`kalshi.yml`](.github/workflows/kalshi.yml) saves every open Kalshi NBA
  prop market each morning and before each evening's tip-offs
  ([`live/log_kalshi.py`](live/log_kalshi.py)), into `data/kalshi/`.
- [`projections.yml`](.github/workflows/projections.yml), four times a day:
  refreshes the box scores, settles paper positions
  ([`live/nba_paper.py`](live/nba_paper.py)), reads ESPN's schedule, lines and injury
  report (Out and Doubtful players sit), refits the models from scratch on
  every completed game and prices every open Kalshi prop
  ([`live/nba_projections.py`](live/nba_projections.py)). Output:
  [`outputs/projections/latest.csv`](outputs/projections/) with the model's
  expected value and probability, Kalshi's bid and ask, and the edge on each
  side after fees, plus the bet card and paper ledger.
- [`backfill.yml`](.github/workflows/backfill.yml) downloaded the settled
  2025-26 markets and their hourly prices for the market test
  ([`research/backfill_kalshi.py`](research/backfill_kalshi.py)).

## Reproduce

```bash
pip install -r requirements.txt
python -m live.fetch_nba_data               # box scores, schedules and lines (about 10 MB)
python -m research.backtest             # development seasons; --holdout for 2025-26
python -m research.market_test          # needs data/kalshi_history/ from the backfill
python -m research.strategy_backtest    # Phase 4 rule replayed on 2025-26, $500 bankroll
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
- One season of Kalshi prices is a short record for a small edge, and the
  Phase 4 rule was chosen on that same season.
- Kalshi settles markets on players who sit at a value it sets, not at the
  price paid; about 4% of 2025-26 markets settled that way. The rule skips
  anyone on the injury report, and the paper ledger records the real
  settlement when it happens.

## AI use

I used AI tools to help write code and documentation and to review my work.
I reviewed the analytical assumptions, methods and results and remain
responsible for the claims made here.

## License

MIT. Box scores and lines come from the sportsdataverse hoopR data
repository; Kalshi prices from Kalshi's public API.
