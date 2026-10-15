# Research log

Decisions and results in the order they were made. Each test is written down
here before it is run, and results are added below without editing the plan.

## 2026-10-15: Plan and pre-registered market test

### Question

Can a model built from public box scores and betting lines price Kalshi's NBA
player props (points, rebounds, assists, threes) well enough to beat the price
at tip-off, after Kalshi's fees?

### Data

- **Box scores:** ESPN player and team box scores from the sportsdataverse
  hoopR data repository, 2015-16 to 2025-26, regular season and playoffs,
  including players who did not play and the listed reason.
- **Betting lines:** consensus closing spread and total for each game
  (The Odds API, via the same repository), available from 2020-21.
- **Kalshi prices:** every settled market in the series KXNBAPTS, KXNBAREB,
  KXNBAAST and KXNBA3PT for 2025-26 games, with hourly candles (yes bid,
  yes ask, last price, volume) around each game, from Kalshi's public
  historical API. Downloaded by `scripts/backfill_kalshi.py` in GitHub
  Actions starting 2026-10-15.

### Split

- **Development:** walk-forward over 2021-22 to 2024-25. Before each calendar
  month the model is refit on all earlier games and predicts that month.
  Features, half-lives and shrinkage may be chosen by looking at these
  seasons.
- **Holdout:** 2025-26, regular season, play-in and playoffs. It is run once,
  with the model frozen; the commit hash of the frozen model is recorded here
  before the run.
- **Kalshi prices are not examined before the holdout run.** Until the model
  is frozen, the price files are used only for counts and for matching
  tickers to players and games (names, teams, dates), never for prices,
  results or volumes.

### Market test (fixed now, before any price is seen)

**Sample.** Every settled market in the four series for a 2025-26 regular
season, play-in or playoff game that settled yes or no. Markets settled at a
fixed value ("scalar", which Kalshi uses when the player does not play) are
excluded, as are players who cannot be matched to a box score row.

**Price.** The last hourly candle ending at or before the scheduled tip-off.
Only two-sided quotes count: yes bid above 0, yes ask below $1, and ask above
bid. Buying yes costs the ask; buying no costs one minus the bid.

**Model probability.** P(stat at or above the contract's threshold) from the
frozen model, refit before each month on all earlier games, exactly as in the
development backtest.

**Test A, information.** A logistic regression of the outcome on
logit(market mid) and logit(model probability), with standard errors
clustered by game. The model adds information the market lacks if the
coefficient on the model has t of at least 2.0.

**Test B, trading.** Buy one yes contract at the ask when
p − ask − fee(ask) ≥ 0.05; buy one no contract at 1 − bid when
(1 − p) − (1 − bid) − fee(1 − bid) ≥ 0.05. The fee is Kalshi's
0.07 × P × (1 − P) per contract. P&L is per contract, after fees, with all
four stats pooled. The rule **passes** only if all three hold:

1. mean P&L per contract above zero with t of at least 2.0, standard errors
   clustered by game (several lines for the same player and game move
   together);
2. positive in both halves: games before 2026-02-01, and games from
   2026-02-01 on, including the playoffs;
3. still positive with a stricter 0.08 threshold.

Results by stat are reported but are not used to pick a winner after the fact.

**Reported, not used for the pass/fail decision:**

- fees rounded up to the cent per contract, as for one-contract orders;
- prices from the last candle ending at least two hours before tip-off;
- the model without the teammates-out features. In the backtest those use who
  actually sat, which a final injury report shows in most cases but not for
  late scratches, so this check bounds the head start that gives the model;
- results by stat and by traded volume.

**Known limits.** Hourly candles give the top of the book, not its depth, so
size at those prices is not known. A pass would justify paper trading, not
money.

## 2026-10-15: Development results; model frozen

**Model** (`nba_model/features.py`, `nba_model/marginals.py`): for each stat a
Poisson regression on expected minutes (recent minutes, minutes last game,
share of starts, team games missed, teammates out or back, back-to-back,
playoffs, home, size of the spread), the player's per-minute rate shrunk
toward his position, his recent average, the team's implied points, what the
opponent allows relative to the league, and the share of that stat belonging
to regulars who are out or returning. The spread of outcomes is the
empirical distribution of actual ÷ expected by band of expected value,
separately for players averaging under 20, 20–30 and 30+ minutes.

**Development backtest**, 2021-22 to 2024-25, refit before every month on all
earlier games (99,833 player-games per stat; regulars who played, with 5+
earlier games and 10+ recent minutes):

| Stat | Typical miss, model vs recent average | Brier improvement | Log loss improvement | Players averaging 26+ minutes: Brier improvement |
|---|---|---|---|---|
| Points | 4.72 vs 4.83 | 2.1% | 2.3% | 1.3% |
| Rebounds | 1.94 vs 1.98 | 1.9% | 2.0% | 0.6% |
| Assists | 1.36 vs 1.39 | 2.6% | 2.8% | 1.5% |
| Threes | 0.91 vs 0.93 | 1.9% | 2.0% | 1.4% |

Predicted and actual rates agree to within 0.3 percentage points in every
probability band ([calibration](outputs/backtest/development/calibration.csv)).

Choices made while looking at these seasons: keeping the outcome spread by
minutes tier (it fixed a 1-point underestimate of low lines for heavy-minute
players), counting only regulars (15+ minutes) as teammates out or back,
and a five-game half-life for recent form (three and eight games were tried;
neither was clearly better). A bug in the games-missed count was found by a
unit test and fixed; it barely changed the results.

**Frozen.** The model used for the holdout season and the market test is
commit `ce26c24`. Changes after this point will be logged and the holdout
results reported for the frozen version.

**Clarification to the market test, written before any price is examined:**
the model prices players with at least 5 earlier games and 10+ recent minutes
(as in the backtest), so Kalshi markets on other players (mostly rookies in
their first games) are excluded and counted.

## 2026-10-15: Holdout season (2025-26), box scores only

The frozen model (`ce26c24`), refit before each month, run once on the
2025-26 regular season, play-in and playoffs (25,511 player-games per stat):

| Stat | Typical miss, model vs recent average | Brier improvement | Players averaging 26+ minutes: Brier improvement |
|---|---|---|---|
| Points | 4.70 vs 4.80 | 2.0% | 0.8% |
| Rebounds | 1.92 vs 1.96 | 2.3% | 0.8% |
| Assists | 1.38 vs 1.41 | 2.0% | 0.5% |
| Threes | 0.93 vs 0.94 | 1.6% | 0.9% |

Calibration held: predicted and actual rates agree to within 0.6 percentage
points in every band, with a slight lean high between 20% and 50%
([calibration](outputs/backtest/holdout/calibration.csv)). The gain over a
player's recent average is smaller for heavy-minute players, the ones Kalshi
lists, than for the whole league (about 0.5–0.9% against 1.6–2.3%).

## 2026-10-15: Kalshi download and matching (before the market test)

The backfill finished in one run: 80,582 settled markets for 2025-26 games
(points 23,562, rebounds 22,656, assists 17,745, threes 16,619), from
2025-11-18 (Kalshi's first NBA player props) to the Finals, and hourly candles
for the 67,513 that settled yes or no and traded.

Matching markets to box scores used tickers, titles, dates and teams only.
It found that Kalshi's titles read "Anthony Edwards records 40+ points" until
January 2026 and "Anthony Edwards: 40+ points" after, so the parser now takes
the player's name from the yes subtitle, which never changed format. After
that fix 78,130 of the 80,413 markets on regular-season, play-in and playoff
games match a player in that game's box score, and 77,234 are on players the
model prices. Most of the rest are players who were inactive and missing from
the box score; checking the settlement type (not the result) of the
unmatched markets showed all but 11 were settled at a fixed value, which the
test excludes anyway. No prices, yes/no results or volumes were looked at.

The market test runs next, once, with the frozen model.

## 2026-10-15: Market test results (run once, frozen model `ce26c24`)

**Sample:** 57,111 markets in 1,072 games that settled yes or no, on players
the model prices, with a two-sided quote at the last hourly candle before
tip-off. Kalshi's settlement agreed with the box score in 99.98% of markets.

**Test A: passed.** Logistic regression of the outcome on logit(mid) and
logit(model), standard errors clustered by game: market 0.560 (t 17.9),
model 0.436 (t 13.8). It passed for each stat on its own as well (model t
5.3 for points, 9.0 rebounds, 7.4 assists, 10.3 threes). Brier score: market
mid 0.1805, model 0.1812.

**Test B: failed.** 12,495 trades, −0.93¢ per contract after fees, t −1.49.
Criterion 1 fails (mean below zero), criterion 2 fails (before February
−3.35¢, t −2.50; February on −0.35¢, t −0.50), criterion 3 fails (0.08
threshold: 6,933 trades, −0.93¢, t −1.13).

Reported, not used for the decision:

| Variant | Trades | ¢ per contract | t |
|---|---:|---:|---:|
| Fees rounded up to the cent | 12,495 | −1.35 | −2.17 |
| Prices two hours before tip-off | 11,817 | −0.38 | −0.61 |
| Without the teammates-out features | 12,130 | −0.76 | −1.23 |
| Points | 4,807 | −2.00 | −2.33 |
| Rebounds | 3,195 | −1.41 | −1.20 |
| Assists | 2,960 | −0.25 | −0.21 |
| Threes | 1,533 | +2.15 | +1.44 |
| Markets with volume below the median | 4,183 | −1.06 | −1.20 |
| Markets with volume at or above the median | 8,312 | −0.86 | −1.11 |

The positive threes figure is one of four stats, not significant, and found
after the fact; it is not a result.

**Diagnostics, after the test.** Buying every yes at the ask lost 3.4¢ per
contract and every no 6.0¢ (median spread 4¢, average fee 1.2¢), so the
model's side selection was worth something, but less than the cost of
trading. Realised P&L did not rise with the model's claimed edge: claimed
6.4¢, 9.7¢, 15.0¢ and 25.3¢ realised −0.9¢, −0.9¢, −1.3¢ and +0.4¢. Across
probability bands outcomes happened about 1.3 points more often than the mid
price implied (t 3.9), but not by enough to cover the cost of buying yes.

**Exploratory, not pre-registered:** a blend of model and market, with
weights refit each month on earlier months only (the model's weight fell
from 0.81 in December to 0.43 by the Finals), traded with the same rule,
lost 2.1¢ per contract (6,551 trades, t −2.57). The blend's Brier score
(0.17889) beat the mid's (0.17910) by 0.1%.

**Conclusion.** The model holds real information the tip-off price lacks,
but on Kalshi's NBA props it is worth less than the spread and fee. No
trading rule from this work is recommended for money.

**Forward test for 2026-27 (fixed now).** The same Test A and Test B, run on
2026-27 markets from the logged snapshots and the projections the model
makes before each game, at the end of January 2027 and again after the
Finals; plus the same rule priced at the morning snapshot. Any change to the
model before then is logged here with its date.

## 2026-10-15: Exploratory findings and the rule for 2026-27

Everything in this entry was found after the market test failed, on the same
2025-26 markets. It is in-sample and is recorded so the choices can be
checked; it proves nothing until 2026-27 confirms it
(`run_strategy_backtest.py`, `outputs/strategy/`).

**Where the money went.** Priced at the mid, with no spread or fee, the
model's picks made +2.4¢ per contract (t 3.9). Buying at the ask and paying
the fee cost 3.3¢ on average. By spread at tip-off: 1¢ spread +2.1¢ per
contract (3,884 trades), 2¢ +0.7¢, 3–4¢ −1.6¢, 5¢ or more −5.5¢. With
spreads of 2¢ or less, prices taken 2, 4 and 6 hours before tip-off gave
+2.3¢, +1.9¢ and +2.8¢ (t 2.7, 2.1, 3.1), and +2.1¢ to +2.4¢ with the
teammates-out features switched off
([grid](outputs/strategy/exploratory_grid.csv)).

**The rule** (`nba_model/strategy.py`): price taken 2–6 hours before tip-off;
spread of 2¢ or less; the model's edge at least 5¢ after the fee; one line
per player, stat and game; live, skip anyone on the injury report. Staking:
1% of the bankroll per position, at most $25 or 100 contracts, at most 30%
of the bankroll per day.

**Replayed on 2025-26** with two looks per game (5 and 2 hours before
tip-off, about what the live schedule gives): 3,928 positions, +1.9¢ per
contract (t 2.3), 6.2% per dollar risked.

**But it all came from the playoffs.** Regular season: 2,493 positions,
+0.02¢ per contract (t 0.03), 0.1% per dollar. Play-in and playoffs: 1,435
positions in 91 games, +5.1¢ per contract (t 3.6), 16.7% per dollar, 85% of
them on the "no" side. A $500 bankroll at these stakes went to $390 by early
January, $509 by April 10, and $2,618 by the Finals.

$500 bankroll, 2,000 resampled seasons (whole days resampled with
replacement):

| Part of season | Edge assumed | Median end | 1 in 10 ends below | Chance of a loss | Typical worst drawdown |
|---|---|---:|---:|---:|---:|
| Regular season | as in 2025-26 (none) | $496 | $235 | 50% | 46% |
| Play-in and playoffs | as in 2025-26 | $2,907 | $1,494 | 0% | 17% |
| Play-in and playoffs | half of 2025-26 | $1,233 | $610 | 5% | 28% |
| Play-in and playoffs | none | $513 | $251 | 48% | 44% |

The playoff result is one postseason, found by splitting the season after
seeing it, with a rule also chosen on that season. Half the edge is the
planning case, and no edge is entirely possible.

**Forward test for 2026-27 (fixed now, before any 2026-27 price exists).**
The projections workflow runs four times a day and adds the rule's positions
to `outputs/paper/ledger.csv`, sized from a $500 paper bankroll, settled from
Kalshi's own settlement of each market.

1. *Regular season* (to the end of the 2026-27 regular season): the rule
   passes if its positions return more than zero per dollar with
   game-clustered t of at least 2.0 on profit per contract. Paper only; the
   expectation from 2025-26 is about zero.
2. *Play-in and playoffs 2027*: the same test on those positions alone. This
   is the real test of the playoff finding. Whether to use money is the
   account holder's decision; the evidence for it is the in-sample result
   above and nothing else.
3. The rule and model are not changed during the season. Bug fixes are
   logged here with their date.

The original market test at tip-off (Test A and B) is also rerun on 2026-27
as planned.
