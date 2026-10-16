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

## 2026-10-15: Three new ideas, tests fixed before their data is downloaded

The first approach (out-predict the price and pay the spread) failed. Three
different sources of edge are tested on 2025-26. Each test below is written
before the data it needs is downloaded: Kalshi's trade history (idea 1),
sportsbook prop lines (idea 2) and double-double and triple-double markets
(idea 3). Ideas 1 and 3 reuse 2025-26 prop prices and results that have
already been seen; this is stated, not hidden.

Common pass criteria for a trading test: profit per contract above zero with
game-clustered t of at least 2.0; positive in the regular season on its own;
positive both before and from 2026-02-01. Positions are one per player, stat
and game (the line with the largest edge), placed at the first of two looks:
the last hourly candle ending at least 5 hours before tip-off, then at least
2 hours before.

### Idea 1: be the maker (limit orders)

- **Universe:** the 57,111 markets priced in the market test.
- **Order:** on the side the model favours, rest a buy order one cent above
  that side's best bid when the spread is at least 2 cents (then the order
  is alone at the best price), or at the bid when the spread is 1 cent (then
  it joins a queue of unknown size). Placed only if the model's probability
  for that side, minus the order price, minus the maker fee is at least
  3 cents. Cancelled at tip-off.
- **Fill:** from Kalshi's trade prints between placement and tip-off. A
  resting yes order at price q fills if a taker sells yes (taker outcome side
  "no") at a yes price below q, or at q when the order improved the bid; a
  joined order needs a print strictly through its price. No order fills more
  than the contracts that traded.
- **Fee:** primary, a maker fee of a quarter of the taker fee
  (0.0175 × P × (1 − P)); also reported with no maker fee.
- **Reported:** fill rate, and adverse selection: profit on filled orders
  against what the same orders would have made had they all filled.

### Idea 2: copy the sportsbooks

ESPN's public odds API is probed for stored DraftKings player-prop lines on
2025-26 games. If they exist for most games: the book's over/under is turned
into a no-vig probability at its line, moved to each Kalshi line using the
model's distribution shape (the model's expected value is shifted until it
matches the book at the book's line), and compared with Kalshi's price at
the last candle before tip-off (the book's stored line is taken to be its
closing line, so earlier Kalshi prices would see the future). Buy the side
where the book-based probability beats cost plus fee by 3 cents or more.
If no free history exists, this idea cannot be tested without buying data,
and that decision is the account holder's.

### Idea 3: structure and correlation

- **Ladder arbitrage:** for every player, stat, game and hourly candle,
  check whether buying yes on a lower line and no on a higher line costs
  less than $1 after taker fees. That pair pays at least $1 whatever happens.
  Counted, with the first opportunity per player, stat and game priced.
- **Double-doubles:** a fair price for "at least two of points, rebounds,
  assists of 10 or more", with each probability taken from Kalshi's own
  10+ market for that player and stat at the same hour (its mid price), or
  the model when Kalshi has no such market, and the three combined with a
  Gaussian copula whose correlations are fitted on 2021-22 to 2024-25 model
  predictions and results (before any double-double price is downloaded).
  Buy the side where this fair price beats cost plus taker fee by 5 cents.
  Same pass criteria. Also reported: Brier scores of Kalshi's double-double
  price, the fair price, and the price assuming independence.
- **Triple-doubles:** the same, reported only (they are rare).

## 2026-10-15: Results for the three new ideas

Run once each as written above (`run_new_ideas.py`,
[results](outputs/new_ideas/results.json)). Data downloaded after the tests
were committed: 657,723 Kalshi trade prints in the six hours before tip-off,
4,879 double-double and 1,171 triple-double markets, and ESPN's stored
sportsbook props for every 2025-26 game.

### Idea 1, be the maker: passed

14,516 resting orders; 31% filled (59% of orders on the "no" side, 7% on
the "yes" side). Filled orders, with a maker fee of a quarter of the taker
fee: **+3.1¢ per contract, t 4.1**, 4,541 fills in 854 games. Regular season
+2.6¢ (t 3.0); playoffs +4.5¢; before February +2.3¢; from February +3.3¢.
All criteria pass. With no maker fee: +3.7¢ (t 4.8).

Adverse selection is large: had every order filled it would have made
+11.3¢ per contract; the orders that did not fill would have made +15.0¢.
Orders fill mostly when the market moves against them, but the ones that
fill still made money.

Reported checks: every fill requiring a trade through the price, +3.0¢
(t 3.6); the model without the teammates-out features, +3.8¢ (t 5.0); "no"
orders +3.5¢ (t 4.2), "yes" orders +0.1¢ (t 0.1). Return per dollar on
filled orders 10.8%. Recomputed with separate code
(`scripts/verify_maker.py`): identical.

$500 bankroll (each order reserves its cost while it rests; 1% per order,
day's orders capped at 30% of the bankroll, fills limited to the contracts
that traded through): the 2025-26 replay ends at $2,143 with a 13% worst
drawdown. Resampling days 2,000 times: median $2,010 (1 in 10 below
$1,203); with half the edge, median $953 (1 in 10 below $550, 6% chance of a
loss); with no edge, median $445 (61% chance of a loss).

Limits: the simulation cannot know how other traders would have reacted to
these orders; an order alone at a new best price is assumed first in line;
the maker fee for these series is assumed, not confirmed; and about 80
orders a day must be placed and cancelled at tip-off, which needs a program
using a Kalshi API key.

### Idea 2, copy the sportsbooks: not testable for free

ESPN stores sportsbook props for 176 of the 1,072 games with Kalshi props
(ESPN BET for 76 games in November, DraftKings for 100, mostly playoffs).
The stored rows do not say which price is the over and which the under (in
ESPN BET's data the order is unrelated to the result), and the stored times
are morning lines, not closing lines. The pre-registered test needs most
games, so it was not run. Testing it needs a paid historical prop feed, or
logging sportsbook lines from now on.

### Idea 3, structure and correlation: failed

- **Ladder arbitrage:** 6 of 116,307 hourly snapshots with two or more lines
  had a riskless pair after fees, worth 0.2¢ on average. Kalshi keeps its
  ladders consistent.
- **Double-doubles:** the correlations are real (treating the stats as
  independent underprices double-doubles by about a tenth), but Kalshi
  already prices them: Brier score of Kalshi's mid 0.1331, the correlated
  fair price 0.1327, independence 0.1330 (3,607 markets). The rule made 243
  trades at −1.7¢ per contract (t −0.7). Failed.
- **Triple-doubles (reported only):** Kalshi's mid averaged 10.1% against
  7.2% happening and a fair price of 7.7% (697 markets); its Brier score was
  worse than the fair price's (0.0569 against 0.0527). Only 16 trades
  cleared the 5¢ edge, so this is an observation, not a result.

### What this means

The model's edge is real but small; the way to keep it is to provide
liquidity rather than take it. Resting orders on the model's side, mostly
offering "no" just inside the ask, made money in both halves of the season
and in the regular season on its own. It is the one approach so far that
passed a test fixed before its data was seen.

## 2026-10-15: The "no" seller, and the live paper tests for 2026-27

**Exploratory, after idea 1 passed (same 2025-26 data, in-sample).** Resting
"no" orders with no model at all (one per player, stat and game, on the
lowest line) made +2.8¢ per filled contract (8,409 fills, t 5.3); "yes"
orders everywhere lost 2.0¢ (t −1.8). Kalshi's prop traders mostly buy
"yes", and selling them "no" one cent inside the ask paid. Variants tried:

| No-side orders on | Model veto | Fills | ¢ per contract | t | Regular season ¢ |
|---|---|---:|---:|---:|---:|
| lowest line per player and stat | none | 8,409 | +2.79 | 5.3 | +2.79 |
| every line | none | 33,475 | +1.65 | 4.1 | +1.35 |
| lowest line | skip if model edge < −3¢ | 7,271 | +3.17 | 5.7 | +3.03 |
| every line | skip if model edge < −3¢ | 26,449 | +2.32 | 5.3 | +1.96 |
| lowest line | skip if model edge < 0 | 6,277 | +3.55 | 6.0 | +3.54 |
| every line | skip if model edge < 0 | 21,373 | +2.88 | 6.1 | +2.45 |

Fills in the last 30 minutes before tip-off made the most (+7.2¢); cancelling
30 minutes early cut profit by about a fifth, so orders stay up to tip-off.

**Chosen for 2026-27: every line, skipping any order the model rates below
zero.** It is the most diversified (most fills, smallest worst day) and the
veto is the model's only job. These figures are in-sample and are not
evidence; the 2026-27 season is.

### Live paper tests, fixed before opening night (2026-10-20)

Two resting-order rules are paper traded from separate $500 paper
bankrolls, alongside the taker rule already running:

1. **Model side (idea 1, as tested):** on the side the model favours, one
   cent above that side's bid when the spread is 2¢ or more, else at the
   bid; only if the model's edge after the maker fee is at least 3¢; one
   order per player, stat and game, the largest edge.
2. **"No" seller:** a "no" order on every line, one cent above the "no"
   bid when the spread is 2¢ or more, else at the bid; skipped if the
   model's probability of "no", minus the order price and the maker fee, is
   below zero.

Both: orders are posted by the first projection run 2 to 6 hours before
tip-off and cancelled at tip-off; players on the injury report are skipped;
each order is sized at 1% of that rule's paper bankroll (whole contracts, at
most $25 or 100 contracts), and a day's orders together reserve at most 30%
of it. An order fills with as many contracts as traded through its price
between posting and tip-off, read from Kalshi's public trade prints with the
same rules as the backtest. Filled positions are settled from Kalshi's
settlement. Maker fee assumed at a quarter of the taker fee until Kalshi's
schedule for these series is confirmed.

**Pass criteria for each rule, judged at the end of the 2026-27 regular
season:** profit per filled contract above zero with game-clustered t of at
least 2.0, and positive both before and from 2027-02-01. A checkpoint on
2027-01-31 is reported but decides nothing. No rule changes during the
season; bug fixes are logged here with their dates.

## 2026-10-15: In-game test, fixed before it is run

Kalshi's prop markets keep trading during games. The question: does a model
that updates on the live game state price them better than the market?

- **Game state** from ESPN play-by-play (sportsdataverse hoopR), which
  timestamps every play: each player's points, rebounds, assists and threes
  so far, minutes on court so far (lineups rebuilt from substitutions),
  personal fouls, whether he is on the court, the score margin and the game
  time left.
- **In-game model:** the stat still to come, as a Poisson regression on the
  pre-game expected value (the frozen walk-forward model), game time left,
  minutes so far against his usual share, fouls, being on court, margin and
  the stat so far; the spread of outcomes as before (empirical, by band of
  expected value). Fitted on 2023-24 and 2024-25 games only (states every
  six game minutes). Calibration on 2024-25 is checked with a model fitted on
  2023-24 alone before anything touches Kalshi prices.
- **Prices:** every hourly Kalshi candle that closes while the game is in
  progress (more than five minutes after the first play and before the last),
  for the 57,111 priced 2025-26 markets, with a two-sided quote, the line not
  yet reached and the player not fouled out. The state is taken from plays
  stamped at or before the candle's close.
- **Test A:** logistic regression of the outcome on logit(mid) and
  logit(in-game model), standard errors clustered by game; passes if the
  model's t is at least 2.0.
- **Test B:** buy the side the in-game model rates at least 5¢ better than
  the ask (yes) or 1 − bid (no) after the taker fee; one position per player,
  stat and game, at the first in-game snapshot where one qualifies, largest
  edge. Same pass criteria as the other tests (t ≥ 2.0 clustered by game;
  positive in the regular season alone; positive before and from February).
- **Known limits:** a real trader sees plays seconds to minutes after they
  happen, and broadcast viewers may move prices first; hourly candles show
  the best bid and ask at that instant only.

## 2026-10-15: Results: in-game test, and liquidity rewards

### In-game: passed, but the edge is speed

Run once as written above (`run_ingame.py`,
[results](outputs/ingame/results.json)). The state rebuilt from play-by-play
matches the box scores (stats exactly; minutes within about 0.8), and the
in-game model fitted on 2023-24 alone was calibrated on 2024-25 within half
a point at every tested line ([calibration](outputs/ingame/calibration_2024_25.csv)).

- 47,570 in-game snapshots of 30,529 markets in 985 games.
- **Test A: passed.** Model weight 0.79 (t 36.5) against the market mid's
  0.34; Brier score 0.168 for the model against 0.192 for the mid (in-game
  spreads are wide, median 11¢ on the trades, so the mid is a rough price).
- **Test B: passed.** 4,978 trades, **+4.2¢ per contract (t 6.3)**; regular
  season +3.6¢ (t 4.2); playoffs +5.4¢; before February +8.1¢, from February
  +3.8¢. Recomputed with separate code: identical. Game states spot-checked
  against the raw play-by-play: identical.

Reported checks. The same test with the game state taken from before the
quote, as if the trader's feed were late:

| State known | ¢ per contract | t | Regular season ¢ |
|---|---:|---:|---:|
| at the quote | +4.2 | 6.3 | +3.6 |
| 1 minute late | +3.2 | 4.6 | +2.5 |
| 3 minutes late | +2.2 | 3.1 | +1.1 |
| 5 minutes late | +0.3 | 0.5 | −0.9 |

The model's information advantage (Test A) is the same at every delay; the
profit is not. Most of it comes from quotes that have not yet caught up with
the game, and it is gone if the trader is five minutes behind. By part of
the game: first half +3.8¢, third quarter +4.5¢, fourth quarter +7.0¢ (where
spreads are widest, median 25¢, and volume thinnest). On court +4.3¢, on the
bench +4.2¢.

Limits: hourly candles show the book at one instant, not its depth; the
median hourly volume in these markets was about 360 contracts. Doing this
live needs a program that reads plays and Kalshi's order book within seconds.

### Liquidity rewards: almost none on NBA props

Kalshi's public list of incentive programs (325,341 programs, all statuses)
has only 8 on NBA player props, all on points, all ended, $800 in total.
The rewards go to other markets: daily temperature highs (Miami alone
$866,800 over all programs), Rotten Tomatoes scores, 15-minute crypto
prices, tennis, elections, AAA gasoline prices, and NFL fantasy-points
ladders ($124,800). Open now: YouTube views, Rotten Tomatoes, headlines,
AAA gasoline (monthly and New Jersey weekly), crypto. A daily job keeps
`outputs/incentives/` current.

## 2026-10-15: Parlays, fixed before they are run

Akila's question: does combining two or three high-probability plays into a
parlay make more money? A parlay pays only if every leg hits, and its fair
price is the product of the legs' probabilities (if they are independent),
so it compounds whatever edge or cost each leg has. It can only help if the
legs have an edge or if their correlation is priced wrong.

Caveat: this uses the same 2025-26 prices as the market test, whose taker
results are already known (buying single legs at the ask did not pay), so it
is not a blind test of new data.

**Legs.** The 2025-26 markets of the market test, priced from the last hourly
candle at or before tip-off with a two-sided quote, and the frozen model. A
leg is the side the model favours; q is the model's probability for that
side, c its cost (the ask for yes, 1 - bid for no). A leg qualifies when
q >= 0.70 ("high percentage") and q - c - fee(c) >= 0.03 (value after the
taker fee 0.07 x c x (1 - c)). At most one leg per player and game (the one
with the highest q).

**Test P1, nightly parlays across games.** On each date, keep each game's
best qualifying leg (highest q, then largest edge), rank those by q, and
form one 2-leg parlay from the top two and one 3-leg parlay from the top
three (dates with too few games are skipped). The parlay is priced at the
product of the legs' costs, the best a buyer could do (it equals buying the
legs and rolling the winnings; a real combo price adds a margin), with the
fee 0.07 x P x (1 - P) on the combined price P, and pays $1 if every leg
hits. Return per dollar staked = payout / (P + fee) - 1, one parlay per date,
so dates are independent. Each of the 2- and 3-leg rules passes if its mean
return per dollar is positive with t >= 2.0, positive in both halves (before
and from 2026-02-01) and positive in the regular season alone. Compared with
staking the same legs as single bets.

**Reported, not tested.** Every 2- and 3-leg combination of qualifying legs
from different games on the same date (t clustered by date); hit rates
against the model's and the price's implied rates; and, for pairs of legs in
the same game (same player, teammates, opponents), how often both hit
against the product of their probabilities, which is the correlation a
same-game parlay price would have to include.

**$500.** If a rule passes: 2% of the bankroll per parlay, one a night,
nights resampled 2,000 times; with the full edge, half and none.

### Results, P1 (run once as written)

[Results](outputs/parlays/results.json); recomputed with separate code:
identical.

- 2,430 qualifying legs: the model said they would hit 85% of the time; they
  hit 77.8%, about what they cost (78¢). As singles: −1.7% per dollar (t −1.5).
- **Nightly 2-leg parlay** (151 nights): model 87%, price 75.4%, hit 72.2%;
  **−7.0% per dollar** (t −1.5). The same legs as singles: −2.7%.
- **Nightly 3-leg parlay** (127 nights): model 79%, price 63%, hit 59.8%;
  **−9.8% per dollar** (t −1.4). The same legs as singles: −2.4%.
- Every 2-leg and 3-leg combination: −3.9% and −5.9% per dollar.
- Both fail. The model is overconfident exactly where it disagrees most with
  the market, and a parlay multiplies that. These are best-case prices (no
  combo margin).
- Same-game pairs (reported): two legs of the same player hit together a bit
  more often than their probabilities imply (residual covariance +0.004,
  t 4.5, 16,839 pairs); teammates and opponents show nothing (t 0.6, 0.3).

### Test P3, Kalshi's own NBA combos (fixed before their prices are downloaded)

Kalshi sells same-game parlays as combo markets (series KXMVENBASINGLEGAME;
quoted on request by market makers). The probe found them public, with each
combo's legs, result and trade prints.

- **Sample:** settled combos (yes or no) from the 2025-26 season and playoffs
  whose trades were downloaded (a fixed random sample of the traded ones).
- **Question:** do combo takers overpay? For each trade, the taker's side
  (yes or no) and price; return per dollar = payout / price − 1, before fees.
  Primary measure: dollar-weighted return of takers, t clustered by game.
  Takers overpay if it is negative with t ≤ −2.0, in both halves of the
  season (before and from 2026-02-01).
- **Reported:** by number of legs and by side; and the quoted price against
  the product of the legs' prices (for legs in the four prop series, from the
  hourly candle at or before the trade), which is the margin the quote adds.

## 2026-10-15: Where the resting-order profit comes from (decomposition, in-sample)

Same 2025-26 data as idea 1, no new rules; this is diagnosis, not a test
(`outputs/decompose/`). Resting "no" orders on every line (one cent inside
the ask, up to tip-off), with and without the model veto, and the mirror
"yes" orders.

- **The market mid is not biased.** At tip-off the mid averaged 44.3% and
  "yes" happened 45.6%; by price decile the mid is calibrated within about
  two points everywhere. There is no static "retail overpays for yes" in the
  mid, and no longshot premium to harvest: "no" orders made about the same
  (+1.4 to +2.4¢) at every price level from 10¢ to 90¢, and the lines far
  above a player's average (+2.7¢) were only a little better than the rest.
- **The flow is asymmetric.** Resting "no" orders made +1.65¢ (33,475 fills,
  t 4.1) with no model at all; resting "yes" orders lost 2.0¢ (t −3.2). The
  people who hit a resting "no" (yes buyers paying near the ask) are
  uninformed; the people who hit a resting "yes" (yes sellers) are informed.
  So the edge is spread capture from one side of the book, with adverse
  selection on the other.
- **The model's job is to avoid the informed fills.** Profit rises with the
  model's edge: orders the model rated below −5¢ lost 1-2¢; from 0 to 3¢
  made +1.9¢; 3 to 10¢ made +3.4-3.6¢; above 10¢ +3.1¢. With the veto
  (edge ≥ 0) the result is +2.93¢ in both halves of the season (H1 4,287
  fills, H2 16,354), positive in all four stats (rebounds weakest, +1.6¢) and
  in every price band (+2.0 to +3.4¢). Orders two hours before tip made
  +4.5¢ (2,416 fills) against +1.4¢ at five hours.
- **Capacity is not the limit** at this bankroll: millions of contracts
  traded through these prices over the season.

Conclusion: the strategy already chosen for the paper test (sell "no" on
every line, resting inside the ask, skipping orders the model rates below
zero) is the right shape. It is a liquidity-provision strategy with a model
filter, not a forecasting one. Picking winners is not the edge; neither is
hunting outliers.

## 2026-10-16: Scaling the "no" seller (simulation, same 2025-26 orders)

Akila asked what $25,000 and several sports would do. Replays of the chosen
rule (every line, model veto) over the 184 trading days of 2025-26 with the
staking rule (1% per order, 30% per day), unfilled orders reserving capital.
"Flow share" is the fraction of the contracts that traded through our price
that we are assumed to get (the backtest assumed all of it; half is the
working assumption from here on). Other sports are simulated by gluing
together independently resampled NBA days, i.e. they assume a sport with
the same flow and edge, which is untested.

| Bankroll | Flow share | $25/100-contract order cap | End of season | Max drawdown |
|---:|---:|---|---:|---:|
| $500 | 100% | on | $3,192 | 11% |
| $500 | 50% | on | $2,635 | 12% |
| $5,000 | 50% | on | $18,174 | 9% |
| $5,000 | 50% | off | $24,236 | 10% |
| $25,000 | 50% | on | $46,364 | 5% |
| $25,000 | 50% | off | $73,356 (bootstrap median $70,236; 10th-90th $53k-$91k) | 9% |
| $25,000 | 25% | off | $57,904 | 9% |

With $25,000 the order cap binds (about $600 a night deployed), so it has to
come off, and the return per dollar falls from about 4x at $500 to about
1.9x at $25,000 at half the flow: capital stops being the constraint and
flow does. Adding sports (independent days, half the flow, no cap, $25,000):
1 sport $70k, 2 sports $88k, 3 sports $101k, 5 sports $116k median, with
the drawdown falling from 8% to 2%. Fewer sports with no cap beat more
sports with the cap. All in-sample, one season, assuming the edge holds.

## 2026-10-16: Market scan, fixed before the data is downloaded

Question: in which Kalshi sports markets does the structural edge found in
NBA props (resting "no" orders fill against uninformed "yes" buyers) also
exist? The scan needs no model; it measures the flow asymmetry alone, so a
positive result is the floor the model filter would build on.

**Markets.** Every Kalshi series in the Sports category, excluding combo
(multivariate) series. From each series' settled markets of the last 400
days, a random sample (seed 0) of up to 1,200 that traded and settled yes
or no. Series whose first page of settled markets shows under $20,000 of
volume are skipped. NBA props are included as the control: the scan's
numbers for them should agree in sign with idea 1.

**Generic timing** (there is no tip-off time for most sports): hourly
candles ending between 7 and 4 hours before the market's expected
expiration (a game lasts about three hours, so this is pre-game for the
major sports; markets that expire long after the event, such as season-long
or multi-day ones, are reported but flagged). At each candle with a
two-sided quote, a resting "no" order one cent inside the ask (at the bid
when the spread is one cent) rests for one hour and fills from the trade
prints with the price-priority rules of `nba_model/maker.py`. The mirror
"yes" order is run the same way. One order per market per hour.

**Measures per series:** fills, contracts traded through, profit per filled
contract after the maker fee (25% of 0.07 x P x (1 - P)) and its t with
standard errors clustered by event (game); the same for "yes" orders;
fill rate; median spread; volume of the sample. Plus profit per contract
for the first and second half of each series' sample dates.

**Pass, per series:** "no" orders positive with t ≥ 2.0 and positive in
both halves, and "yes" orders not significantly positive (that would mean
the flow is two-sided and the "no" edge is spread capture a model could
also get on the yes side). Passing series are candidates for a model and a
pre-registered test of their own; nothing is traded from this scan.
