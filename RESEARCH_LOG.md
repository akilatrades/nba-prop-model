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
  historical API. Downloaded by `research/backfill_kalshi.py` in GitHub
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

**Model** (`markets/nba/features.py`, `markets/nba/marginals.py`): for each stat a
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
(`research/strategy_backtest.py`, `outputs/strategy/`).

**Where the money went.** Priced at the mid, with no spread or fee, the
model's picks made +2.4¢ per contract (t 3.9). Buying at the ask and paying
the fee cost 3.3¢ on average. By spread at tip-off: 1¢ spread +2.1¢ per
contract (3,884 trades), 2¢ +0.7¢, 3–4¢ −1.6¢, 5¢ or more −5.5¢. With
spreads of 2¢ or less, prices taken 2, 4 and 6 hours before tip-off gave
+2.3¢, +1.9¢ and +2.8¢ (t 2.7, 2.1, 3.1), and +2.1¢ to +2.4¢ with the
teammates-out features switched off
([grid](outputs/strategy/exploratory_grid.csv)).

**The rule** (`kalshi/strategy.py`): price taken 2–6 hours before tip-off;
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

Run once each as written above (`research/new_ideas.py`,
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
(`research/verify_maker.py`): identical.

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

Run once as written above (`research/ingame.py`,
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
prints with the price-priority rules of `kalshi/maker.py`. The mirror
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

## 2026-10-16: Results: market scan and Kalshi NBA combos

### Market scan (run once as written; [results](outputs/scan/results.json), [table](outputs/scan/table.csv))

The listing found 3,972 sports series, 2,270 with at least $20,000 on their
first page. The six shards scanned 274 of them in the time allowed, taken in
order of volume, so the big markets are covered and the long tail is not.

- **Control:** NBA points (1,200-market sample) gave "no" orders +3.0¢ with
  no model (t 1.6; the full-season figure was +1.65¢ for all four stats);
  "yes" orders −2.6¢. Same sign as idea 1. The other three NBA prop series
  were not reached.
- **Game-winner markets lose.** Of 83 "...GAME" series, "no" orders lost in
  most of the large ones: NFL −4.8¢ (t −2.4), MLB −7.9¢ (t −3.0), NBA −1.7¢,
  Liga MX −3.6¢ (t −3.3). As expected: both sides are a team, the flow is
  two-sided and the sharps are there. The same for NBA spreads (−3.9¢,
  t −2.1) and team totals.
- **Ten series passed** (t ≥ 2, both halves positive, "yes" not positive):
  ITF tennis match winners (+4.6¢, t 2.5, $117M sample volume), NCAA women's
  basketball games (+2.9¢, t 2.1), World Cup first-half spreads (+4.9¢),
  NASCAR race winners (+4.0¢, t 3.6), La Liga spreads (+4.3¢), World Cup
  shots on target (+6.3¢, t 3.1), **NFL anytime touchdown (+3.0¢, t 2.2,
  "yes" −1.1¢)**, WNBA first-half totals (+8.8¢), Liga MX spreads (+4.7¢),
  World Cup assists (+3.7¢). Near misses with the same shape: NFL receiving
  yards (+4.8¢, t 1.7), NCAA men's totals and spreads, WTA matches (+3.9¢,
  t 1.9, $750M volume), Ligue 1 games (+2.8¢, t 2.0 but "yes" also moved).
- **Caution.** 274 series were tested; at the 2.0 threshold about six would
  pass by chance. The ones to believe are those where "yes" orders also lost
  clearly (ITF, NASCAR, shots on target, WNBA 1H totals, Liga MX spreads) and
  the player props, where the mechanism is the same as NBA's (NFL anytime
  TD, receiving yards). Several passing markets are long-shot fields (NASCAR
  winners, ITF upsets): there the "no" edge is the favourite-longshot bias,
  not spread capture, and it comes with rare large losses.

Overall across the 274 series, resting "no" made +0.2¢ at the median and
"yes" lost 1.3¢ (fill-weighted). The asymmetry is general; the size is
market-specific.

### Kalshi NBA same-game combos (P3, run once; [results](outputs/parlays/combos.json))

236,239 combos listed for 2025-26; trades downloaded for a 29,934-combo
sample ($1.43M paid, 598 games). Takers were "yes" buyers 98% of the time.

- Takers paid an average 19¢ per contract and 17.3% of contracts paid out:
  **−9.0% per dollar (t −2.5)**. First half −10.2%, second half +2.1% (the
  playoffs; the pre-registered "both halves" criterion fails).
- By legs: 2 legs −4.5%, 3 legs −9.6%, 4 legs −16.4%, 5+ legs −18.7%
  (t −2.3). The loss grows with the number of legs, like sportsbook parlays.
- Quotes are 1.5x the product of the legs' mids (17¢ against 11¢), but the
  legs are positively correlated and 16% hit, so the quoting margin is
  roughly 1 to 3¢ per contract, not 6¢. The makers' profit is mostly the
  longshot premium on many-leg combos.

Being the combo quoter would be the largest edge per dollar seen in this
project, and it is only available through Kalshi's RFQ market-maker API,
which needs onboarding, quoting software and capital for many simultaneous
positions. Noted, not pursued at $500.

### Paper tests in the scanned markets (live from 2026-10-16)

Five of the passing or near-passing series that are in season now run the
scan's model-free rule as live paper tests, each from its own $500
(`kalshi/paper_markets.py`, every three hours): ITF men's tennis, NASCAR
race winners, Liga MX and La Liga spreads, and NFL anytime touchdown. The
rule is as scanned (orders 4 to 10 hours before expected expiration, one
cent inside the ask, cancelled at the next run or three hours before
expiration). The first run posted 33 ITF orders. These exist to check the
scan out of sample; the NFL anytime-touchdown market is also the first
candidate for a model filter (the NFL model). What $500 would have made on
the scan's own sample (half the flow, in-sample, 4 to 11 months of data):
ITF $740, NASCAR $749, Liga MX spreads $715, La Liga spreads $647, NFL
anytime TD $754, NCAA women's games $1,878; the six together from one $500
bankroll, $4,873 over 11 months (about $13,000 if the real flow is five
times the 1,200-market sample). Expect less: these markets were chosen for
passing.

## 2026-10-16: NFL props with the NFL model as filter, fixed before it runs

The scan's two NFL prop hits (anytime touchdown +3.0¢, receiving yards +4.8¢
model-free) get the same treatment as NBA: the "no" seller with a model veto.
The NFL model is the one from akilatrades/nfl-parlay-model, now
`markets/nfl` (walk-forward 2019-2026, calibrated within about a point over
125,803 predictions); it has never been tuned on Kalshi prices.

**Rule (ledgers `KXNFLANYTD+model` and `KXNFLRECYDS+model`, $500 each).**
Every three hours, for every open market in the series whose expected
expiration is 4 to 10 hours away with a two-sided quote: a "no" order one
cent inside the ask (at the bid when the spread is 1¢), cancelled at the next
run or three hours before expiration. **Veto:** the order is skipped when
(1 − p) − price − maker fee < 0, where p is the model's over-probability
for that player and line from the latest ladder made before the run (the
anytime-touchdown market is the 0.5 line of rushing plus receiving
touchdowns). Markets the model cannot price (no ladder row) are skipped, so
the ledger is a strict subset of the model-free one for the same series,
which keeps running alongside as the control. Sizing, fills and settlement
as in the other paper tests.

**Pass criteria, judged after the Super Bowl (2027-02-08):** profit per
filled contract above zero with game-clustered t ≥ 2.0, and the filtered
ledger's profit per contract above the model-free control's over the same
markets. Reported: the vetoed orders' profit (what the model kept us out of).

## 2026-10-16: Market scan, second pass: every Kalshi market

The first pass reached 274 of the 2,270 sports series with at least $20,000
on their first page. This pass runs the same model-free rule and pass
criteria (resting "no" one cent inside the ask, 7 to 4 hours before expected
expiration, one hour's rest, fills from trade prints; pass if "no" is
positive with t ≥ 2 in both halves and "yes" is not significantly positive)
on **every remaining series in every category** with that volume, sports
and non-sports, with 400-market samples instead of 1,200 so it finishes.
Combo (multivariate) series are excluded as before. Non-sports markets have
no game, so "7 to 4 hours before expiration" is simply a pre-close window;
long-horizon markets (futures, season winners, elections) will show up with
few quotes in that window and are reported with their counts. Results merge
into the same table; a series' newest scan wins. Same caution as before: at
t ≥ 2 about one series in forty passes by chance, so passing series are
candidates for paper tests, not conclusions.

## 2026-10-16: v2 of the "no" seller, fixed before opening night

v1 (2026-10-15) runs unchanged through the season as the reference. Each
v2 change below runs beside it as its own $500 paper ledger from opening
night, in `outputs/paper_maker/`, and is judged against v1 at the
2027-01-31 checkpoint on profit per filled contract (clustered t) and on
profit per dollar staked. A variant replaces v1 for the second half only
if it beats it on both out of sample. Evidence for each is from the
2025-26 decomposition (2026-10-15).

| Ledger | Change from v1 | Why |
|---|---|---|
| `no_seller_late` | Posted only 0.5 to 2 hours before tip-off | Orders 2 h before tip made +4.5¢ against +1.4¢ at 5 h; the last 30 minutes +7.2¢ |
| `no_seller_edge_sized` | Stake 0.5% to 1.5% of the bankroll, rising with the model's edge (full at 5¢) | Profit per contract rose from +1.9¢ (edge 0-3¢) to +3.5¢ (3-10¢) |
| `no_seller_veto2` | Skip unless the model's edge is at least 2¢ | Same gradient; trades fills for quality |
| `no_seller_stat_veto` | Rebounds need 2¢ of edge, other stats 0 | Rebounds were the weakest stat (+1.6¢ against +3-4¢) |

**Live trader (`live/nba_trade.py`), added before any real order:** fresh
exchange quotes before posting; a per-team cap ($15) beside the per-game
cap ($25) and a cap of six lines per game; players whose ESPN injury entry
changed in the last six hours, or who are Out, are skipped (the market
knows the news before the model does); realized P&L for the daily loss stop
comes from the account's settlements; after every run the ledger's resting
orders are reconciled with the exchange's and any unknown order is flagged.

**Market selection.** Winner markets come off the list when their two-week
paper tests end: ITF tennis (2026-10-30) and NASCAR (2026-10-30), unless
they are positive with t ≥ 2 by then. The edge is in prop markets ("does X
happen"); winner markets ("A beats B") lost in the scan and are losing live.

**Not yet done, needs a purchase:** a sportsbook-line veto (live odds feed,
about $40 a month). To be pre-registered when the feed is in hand.

## 2026-10-16: Raising the edge: sportsbook-line veto, early exit, faster quoting

Written before any of it runs. Profit for a resting-order seller is fills x
edge per fill x capital turns, less what informed buyers take back. One
change per term, each judged separately.

### Sportsbook-line veto (edge per fill)

The model is the bouncer that decides which "no" tickets to sell. The
sportsbooks are a better bouncer: they price every player prop with far
more information than four stat models fit on box scores. A "no" sold
where Kalshi's "yes" ask is above the books' probability of the over is
selling an overpriced ticket; one sold below it is selling a cheap ticket to
someone who may know more.

**Data.** The Odds API (paid feed, bought 2026-10-16): US books' player
points, rebounds, assists and threes lines, main and alternate. Book
probability of the over at a line = the average, across books quoting both
sides of that exact line, of the over's implied probability with the vig
removed proportionally. A Kalshi line with no book quote has no book
probability and is left to the model.

**Historical test, run first.** From the 2025-26 "no" seller's 33,475
fills (`outputs/decompose/orders_no.csv.gz`, orders posted 5 h and 2 h
before tip-off), a random sample of 150 games, stratified by month (seed
20261016, chosen before fetching). For each, the books' lines at the moment
the order was posted (the feed's snapshot at or just before `end_ts`).
Reported, in cents per filled contract with game-clustered t:

1. all fills in the sample (the baseline, to compare with the season's 1.65¢);
2. fills with a book quote at the exact line (coverage, and whether covered
   lines differ from uncovered ones);
3. **the veto**: covered fills where (1 - p_book) - no price - maker fee ≥ 0;
   and the fills it removes;
4. the veto at a 2¢ margin;
5. the model veto on the same covered fills, for comparison;
6. both vetoes together.

**Pass:** (3) beats (2) by at least 1¢ per contract with the difference's
clustered t ≥ 2, on a sample of at least 2,000 covered fills. If it passes,
`no_seller_bookline` runs as the live trader's rule from opening night;
if not, the trader keeps v1 and the book line runs as a paper ledger only.
About 12,000 of the feed's 100,000 monthly credits.

**Paper ledger `no_seller_bookline`,** from opening night, $500, beside the
others: v1's rule with the book veto on covered lines and the model veto on
uncovered ones. Book lines are logged every two hours in NBA hours
(`outputs/books/`), so the test can be re-run on this season's quotes.

### Early exit (capital turns)

A "no" held to settlement ties up its cost for hours and carries the
game's variance. Rule, live only: when the exchange's "yes" ask has fallen
at least 6¢ below the level we sold at (so the "no" we hold is worth at
least 6¢ more than we paid), rest a sell order for the position 1¢ inside
the "no" bid, post-only, expiring at tip-off. Judged on the exits ledger
(`outputs/live/exits.csv`): realized exit profit against what the same
contracts would have made held to settlement, over the first 200 exits.
The 6¢ threshold is roughly three times the round-trip maker fee at the
mid; it was not fitted to anything.

### Protecting the edge (adverse selection)

- The server quotes every minute instead of every five; the first order at
  a price level fills first, and a quote that is a minute stale is less
  likely to be the one a sharp sweeps.
- Resting orders on a player whose injury-report entry changed since the
  order was placed, or who is now Out or Doubtful, are cancelled on the
  next run, not just skipped for new orders. Logged with the reason.
- No new order inside 15 minutes of tip-off (v1's window already stops at
  30 minutes; this is a hard floor for every rule).

These are protective, not profit rules; their effect shows up as the gap
between paper fills (which assume we get every print through our price)
and real fills. Reported monthly: real fill rate against the paper ledger's,
and realized cents per contract against the paper ledger's over the same
markets.

### NFL props in the live trader (added 2026-10-16, before any order)

The trader also runs on NFL anytime-touchdown and receiving-yards markets
(`TRADE_MARKET=nfl`, `live/nfl_trade.py`), from Sunday 2026-10-18, dry run.
Same rule and risk limits as NBA; one account, so the open-exposure cap
counts both markets' ledgers. Orders rest 2 to 6 hours before kickoff
(kickoff from the market's occurrence time) and expire at kickoff, the NBA
window rather than the scan's "7 to 4 hours before expiration", since the
scan's expiration for these markets is two days after the game. Vetoes: the
NFL model where its ladder prices the line; the books' receiving-yards
consensus where they quote it; anytime touchdown has no two-sided book
quote and keeps the model veto only. A market with neither probability gets
no order. Evidence is thinner than NBA's (anytime TD +3.0¢, t 2.2, 2,255
scan fills; receiving yards +4.8¢, t 1.7, 769): NFL stays in dry run until
its paper ledgers (`outputs/paper_markets/`, from 2026-10-16) are positive
with t ≥ 2, and any real stake is a minority of the bankroll beside NBA.

## 2026-10-16: Stake size at a small bankroll (replay, same 2025-26 orders)

Akila asked whether 2-3% a line instead of 1% raises the return from $250.
Replays with `research/scaling.py` (every line, model veto, half the flow,
$25/100-contract order cap, bootstrap over days) say the per-line stake is
not the constraint: a night has about 175 orders (median), so 1% each asks
for 175% of the bankroll and the **30% daily cap** is what binds. Raising
the stake to 2-5% at the same daily cap leaves the season's end within
noise ($250 -> about $1,600-2,500 median) and concentrates the same 30% in
fewer lines. The lever is the daily cap, with a per-game cap to spread it:

| Start | Stake | Day cap | Game cap | Boot median | 10th | P(loss) | Median max DD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $250 | 1% | 30% | 10% | $1,643 | $875 | 0% | 24% |
| $250 | 2% | 50% | 15% | $2,106 | $981 | 0% | 38% |
| $250 | 3% | 70% | 20% | $2,998 | $1,097 | 0% | 46% |
| $250 | 3% | 100% | 25% | $7,209 | $2,151 | 0% | 45% |
| $500 | 1% | 30% | 10% | $2,732 | $1,733 | 0% | 21% |
| $500 | 3% | 100% | 25% | $11,122 | $4,433 | 0% | 36% |

Stress, $250, shrinking every fill's profit toward zero edge:

| Edge | 1% / 30% day: median, P(loss), P(< half) | 3% / 100% day: median, P(loss), P(< half) |
|---|---|---|
| As backtested (3.1¢) | $1,777, 0%, 0% | $7,246, 0%, 0% |
| Halved (1.6¢) | $778, 3%, 0% | $1,346, 13%, 5% |
| None | $265, 42%, 8% | $82, 67%, 57% |

Reading: the aggressive setting roughly quadruples the median if the edge
is as backtested, and loses more than half the money in a majority of
seasons if there is no edge. The conservative setting is nearly flat with
no edge. Since the first weeks are exactly the test of whether the edge
survives real fills, the trader keeps 1% / 30% / per-game cap until the
realized cents per contract over the first 500 fills are positive with
t ≥ 2; the daily cap then rises in steps (50%, 70%), never above 70%, and
only while the running realized edge stays above 1.5¢. Written here before
the first fill. The replay's drawdowns (21-46%) are larger than the
2026-10-16 scaling entry's (5-12%), which sized edge-first within a day
without a per-game cap and measured drawdown differently; these are the
numbers to plan around.

### Result: sportsbook-line veto, historical test (run 2026-10-16 19:19 UTC)

151 games sampled, all found in the feed; 4,862 fills; a book quote at the
exact line for 66% of them (3,197). Cents per filled contract, game-clustered t:

| Fills | Count | ¢/contract | t |
|---|---:|---:|---:|
| All in the sample | 4,862 | +2.04 | 1.8 |
| Covered by a book quote | 3,197 | +2.15 | 1.6 |
| **Book veto (book edge ≥ 0)** | 1,647 | **+3.81** | **2.4** |
| Removed by the book veto | 1,550 | +0.38 | 0.3 |
| Book veto at 2¢ | 476 | +2.23 | 1.0 |
| Model veto, same covered fills | 2,074 | +2.69 | 1.6 |
| Both vetoes | 1,149 | +3.74 | 1.9 |

Kept minus removed: +3.44¢, t 2.42. Gain over all covered fills +1.66¢.
**Passes the pre-registered test** (≥ 1¢, t ≥ 2, ≥ 2,000 covered fills).
The books are better calibrated than Kalshi's ask at every decile (table in
`outputs/bookline/summary.md`); the 2¢ margin removes too much. Per the
pre-registration, `no_seller_bookline` is the live trader's default rule
from opening night (`TRADE_RULE`), with the model veto on uncovered lines.
Caveats: 150 games, one season, and the fills the veto removes were still
slightly positive, so the gain is in selection, not in avoiding losses; the
paper ledger beside v1 is the out-of-sample test.

## 2026-10-16: Market scan, second pass: results

All 20 shards finished 22:16 UTC. Both passes together: 5,393 series listed,
1,178 with enough trading in the window to test, **70 pass** (positive "no"
seller with t ≥ 2 in both halves, "yes" not significantly positive). At one
in forty by chance, about 30 of the 70 are noise, and 45 of the 70 have
fewer than 100 fills. Full table: `outputs/scan/table.csv`.

**Non-sports: no edge.** Commodities (10 series, 2,226 fills) −0.6¢;
financials (S&P ranges) −1.1¢; crypto, economics, elections, politics and
entertainment have no series with 100 fills that passes. The exception is
the **Mentions** markets (will X be said): 3 series, 1,398 fills, +3.6¢;
NCAA Mention +6.5¢ (t 2.3, 813 fills). A market with a fan on the "yes"
side; the rest have none. The hope that price markets (crypto, equities)
would fill NBA's dark days is not supported.

**Scorer props: flat.** NHL goalscorer +0.1¢ (180 fills), NFL passing
touchdowns −4.2¢ (334), NHL first goal +4.7¢ and points +6.2¢ on 62 and 143
fills. Not a second NBA.

**Passes with at least 100 fills, prop-shaped, in season or soon:**

| Series | Fills | ¢ | t | Halves | Season |
|---|---:|---:|---:|---|---|
| KXCLUBFBTTS, club soccer both teams to score | 567 | +11.5 | 3.4 | +13.0 / +10.2 | now, daily |
| KXNCAAMB1HSPREAD, college basketball 1H spread | 223 | +16.4 | 4.0 | +20.2 / +13.7 | from Nov |
| KXUCLCORNERS, Champions League corners | 141 | +29.4 | 3.1 | +26.1 / +31.8 | Tue/Wed |
| KXUFCMOF, UFC method of finish | 218 | +8.6 | 2.3 | +6.8 / +10.4 | Saturdays |
| KXNBAFTM, NBA free throws made | 127 | +12.4 | 2.1 | +18.7 / +9.6 | from Oct 20 |
| KXNCAAMENTION, NCAA broadcast mentions | 813 | +6.5 | 2.3 | +2.7 / +10.0 | now |
| KXWNBA1QSPREAD / KXWNBA1HTOTAL | 109 / 675 | +16.4 / +8.8 | 2.7 / 2.6 | both + | summer |

Winner-shaped passes (NWSL spread, NCAAW, AHL, VTB, Rocket League, EWSL,
NASCAR, ITF) are noted and not tested further: winner markets lost in the
first pass and are losing live.

**Pre-registered paper tests, from 2026-10-17:** model-free "no" seller
ledgers ($500 each, same rule, window and fills as the others) for the six
in-season rows above: BTTS, UCL corners, UFC method of finish, NBA free
throws made, NCAA Mention and, from the first game in November, NCAA 1H
spreads. Judged after 2026-12-15 (or 300 fills, whichever is later): kept if
positive with t ≥ 2; otherwise off the list. Expected by chance: about one
of the six passes anyway, so a single survivor is weak evidence; two or more
is not. Grouped reading, written before the data: the mechanism is a fan
backing a person or an event to *happen*; BTTS, corners, free throws and
mentions fit; a 1H spread is half-way to a winner market.

## 2026-10-16: Execution studies (from the research agenda), written before running

Akila's research agenda (strategy_research.pdf) puts execution first: whether
the "no" seller's fills are the ones a sharp sweeps. Two offline studies on
the 2025-26 NBA fills and trade prints; no change to the running ledgers or
the trader.

**Markouts and a toxicity veto (`research/markouts.py`).** For each paper
fill, the first print at or through our level is the fill; markouts are
the last "yes" price 5, 15 and 60 minutes after it and at the last print
before tip, from our side (positive = price moved our way). Features known
before posting: the yes price change in the prior hour, prints and
yes-taker share in the prior 15 minutes, volume so far. Splits at each
feature's median, plus two flags fixed now: **toxic** = prior-hour momentum
above +2¢, or yes-taker share above 80% on at least 3 prints; **late** =
filled within 15 minutes of tip. Endpoint: settlement profit per contract
(clustered t) in each split; markouts are diagnostics, not the endpoint.
Reading, in advance: if the toxic group's settlement profit is below the
rest by ≥ 1.5¢ with the difference's t ≥ 2, a pre-posting veto goes into
the v2 paper ledgers (not the trader) as `no_seller_toxveto`. If the
groups do not differ, the fills are not being picked off in a way these
features see, which is good news for execution.

**Quote placement (`research/quote_placement.py`).** Same orders, same
prints, four price policies: join the no bid; improve 1¢ (the current
rule); improve 2¢ when the spread allows; improve 1¢ only when the
model's edge at that price is at least 2¢. Endpoint: net profit per 100
eligible orders (unfilled orders count zero), not per fill, so a policy
cannot win by filling less. Also reported: fill rate and per-fill profit.
Expected in advance: joining the bid fills less and earns more per fill;
2¢ fills more and earns less; which wins per 100 orders is the question.

### Results: execution studies (run 2026-10-16, offline)

**Markouts.** All 33,475 paper fills matched to the print that filled them.
Settlement profit +1.65¢ (t 4.1); markouts from our side −1.16¢ at 5
minutes, −0.98¢ at 15, −0.91¢ at 60 and −0.94¢ at the last pre-tip print.
So after we are filled the "yes" price drifts about 1¢ *against* us and
stays there, yet the position still settles +1.65¢: the market's "yes"
stays overpriced relative to outcomes all the way to tip. Two readings
follow. (1) There is mild adverse selection in the price, but the edge is
in settlement, not in the mark: an exit rule that sells back on price
moves would give the edge away, and the pre-registered 6¢ early exit will
rarely trigger. The exits ledger stays, judged on realized exits against
hold value as written. (2) None of the pre-posting features separate good
fills from bad: the **toxic** flag (1,167 fills) made +2.96¢ against
+1.61¢ for the rest, the wrong direction for a veto; fills in the last 15
minutes before tip made +3.29¢ (t 3.2) against +1.56¢ earlier; large
prints that filled us made +1.51¢ against +1.80¢ for small ones (not
significant). **The pre-registered veto criterion (toxic group worse by ≥
1.5¢, t ≥ 2) is not met; no toxicity veto.** Most orders have no prints in
the hour before posting (median momentum and 15-minute print count both 0):
these markets are thin before tip, which is itself why resting orders work.

**Quote placement** (35,404 eligible orders, model veto on, one contract
each):

| Policy | Fill rate | ¢ per 100 eligible orders | t | ¢ per filled contract |
|---|---:|---:|---:|---:|
| Join the no bid | 30% | +86 | 4.7 | +2.86 |
| **Improve 1¢ (current rule)** | 58% | **+171** | 6.1 | +2.94 |
| Improve 2¢ when the spread allows | 68% | +168 | 5.5 | +2.47 |
| Improve 1¢ only with model edge ≥ 2¢ | 58% | +191 (on 28,213 orders) | 6.2 | +3.30 |

Joining the queue halves the fills and the profit; 2¢ inside buys 10
points of fill rate and gives it all back in edge; the 1¢ rule is at the
optimum of the three price policies. The edge-2¢ filter earns more per
order on fewer orders (total +540 against +605 for all orders at 1¢), so
it is the better rule when the daily cap, not the opportunity set, binds,
which is the small-bankroll case; it is already running as
`no_seller_veto2`. No change to the trader from this study.

## 2026-10-17: Desk studies, written before running (`research/desk.py`)

Four replays on the 2025-26 NBA "no" seller (model veto on, half the flow,
$500, 1% a line, 30% a night, 10% a game), offline, nothing live changes.

1. **Value per quote.** A logistic fill model from what is known at posting
   (spread, no price, open interest, the hour's volume, hours to tip), fit
   on the first half of the season and scored on the second. On the second
   half, the daily cap is spent by three rankings: model edge (the current
   rule), P(fill) × edge, and P(fill) × edge per dollar-hour reserved, plus a
   random-order control. Endpoint: season end from $500 and profit per 100
   orders placed. Adopted into the v2 ledgers only if the EV ranking beats
   edge-first by ≥ 10% of end bankroll on the holdout half.
2. **Inventory.** Nightly paper profit per fill against that night's
   league-wide scoring surprise (points per game minus the trailing-20-night
   mean). Pre-registered reading: a correlation below −0.3 means a night of
   "no" fills is one short-scoring position and a game-total hedge is worth
   testing; above −0.15, the per-game caps already handle it.
3. **Sizing under uncertainty.** Daily cap = 30% × (shrunk running edge / 3¢),
   clamped to 10–70%, where the running edge is the mean profit per contract
   of fills so far shrunk toward zero by 250, 500 or 1,000 imaginary
   zero-edge fills. Compared with the fixed 30% cap on the backtested edge,
   a halved edge and no edge. Adopted if it ends higher at the full edge and
   loses less with no edge.
4. **Attribution.** On the 3,197 fills with a book quote: profit = premium
   (what selling above the books' probability is worth, after fee) +
   residual. Reading: a residual near zero means the edge is the liquidity
   premium and the levers are breadth and speed; a residual with t ≥ 2 means
   there is selection skill beyond the books.

### Results: desk studies (run 2026-10-17, offline)

**1. Value per quote: not adopted.** The fill model is well calibrated on
the holdout (predicted 0.29→0.88 by decile against actual 0.30→0.77; spread
and price level do the work, open interest and hours a little). Ranking the
daily cap by P(fill) × edge per dollar-hour ended the holdout half at
$1,566 against $1,161 for edge-first, which clears the pre-registered 10%,
but the gain does not survive inspection: most of it is the hours term
preferring the 2-hour postings over the 5-hour ones (the known late-posting
result, already running as `no_seller_late`), and within a single posting
time the holdout half is the one stretch where edge-first ranking did badly
(random order beat it, $1,064 to $885) while over the first half and the
whole season edge-first beat random ($2,680 to $1,656). A ranking that wins
only on the half it was not fit on, and loses on the other, is noise. The
fill model itself is kept (`research/desk.py`) for the capacity study once
real fills exist.

**2. Inventory: no hedge.** Correlation of nightly profit per fill with the
night's scoring surprise is −0.06 (178 nights); high-scoring nights made
$0.76 per fill against $1.57 on low-scoring ones, a tilt but a weak one.
Below the pre-registered −0.15 line: the per-game and per-team caps are
enough. No game-total hedge.

**3. Adaptive sizing: not adopted as the default, kept as the insurance
option.** Fixed 30% cap: $500 → $1,849 at the full edge, $759 at half,
**$190 with no edge** (74% drawdown). Shrunk-Kelly cap (10–70%, prior 500
zero-edge fills): $1,330 at the full edge, $885 at half, **$635 with no
edge** (31% drawdown). It gives up 28% of the upside for three times the
capital preserved if the edge is not there, because it sits at the 10%
floor for the first month while the running estimate is still near zero.
The pre-registered rule (higher at full edge and safer with none) is not
met, so the fixed cap with the written step-up schedule stays; the adaptive
cap is the right choice for anyone who weights "the edge is gone" more
heavily than the backtest does, and is documented for that.

**4. Attribution: the premium is shared with the books.** On the 3,197
fills with a book quote, our price sat on average only 0.14¢ above the
books' probability; the 2.15¢ of profit is 2.01¢ residual (t 1.5): on
these lines the over came in below *both* Kalshi's price and the books'.
Reading: sportsbooks shade player props toward the over as well (they
carry the same public), so "sold above the books" is a selection signal
rather than the source of the edge; the source is the public's
over-optimism, which both venues price in. Consequence: the book veto
works by picking lines, not by arbitraging the books, and the gross edge
against the books is itself positive but not yet significant (t 1.5).

## 2026-10-17: Model refresh study, fixed before it runs

Prompted by the question of which of the standard quant toolkit the model
uses. Stochastic calculus and differential equations are not tested: the
contract is a binary held to settlement with no hedge, so there is no price
process to solve. Four things that could plausibly help are tested offline
against the frozen model (`research/model_refresh.py`), on the 2025-26
holdout only, with the same walk-forward as the backtest (refit before each
month on earlier games). Nothing is tuned on the holdout; every setting
below is written here first.

**Mean-model variants** (the spread layer is unchanged):

1. `control`: the frozen Poisson regression.
2. `gbm`: gradient-boosted trees with a Poisson loss on the same feature
   matrix and the same season weights. Settings fixed now: 300 rounds,
   learning rate 0.05, 15 leaves, 50 rows per leaf, L2 = 1.0. No tuning.
3. `hier`: the Poisson regression plus a player random effect, Gamma-Poisson
   partial pooling: each player's factor is (his actual + a) / (his expected
   + a) over his training games in the latest two seasons, so a player with
   little history stays at the regression and a player with a long
   consistent residual moves off it. The prior strength `a` is chosen from
   {10, 30, 100, 300} by log loss on the last training month only (never
   the test month).
4. `iso`: the control's probabilities passed through an isotonic
   calibrator fitted, per stat, on every earlier out-of-sample prediction
   (the 2021-22 to 2024-25 development predictions plus earlier holdout
   months).

**Judged on**, in order: (a) log loss over every player-game-line in the
holdout; (b) the veto replayed on the season's 53,549 resting "no" orders,
each variant's probability recomputing the edge and each order's
place/skip, scored as profit per 100 eligible orders with one contract per
order (unfilled orders count zero), with the difference from the control
given a game-clustered t. A variant is adopted only if (a) improves and (b)
improves with t ≥ 2; otherwise the control stays.

**Portfolio allocation**: in the bankroll replay from the desk study
(30% daily cap, 1% stake, 10% game cap), order the day's candidates by
edge ÷ standard deviation of the contract (√(p(1−p)), with within-game
correlation taken as the −0.06 measured in the inventory study, i.e. close
to independent) instead of by edge. Judged on the whole season and on each
half separately, against edge-first and random order. Adopted only if it
beats edge-first on both halves by 10% or more, the same bar the EV-ranking
study failed.

Expected: small log-loss gains at most from `gbm` and `hier`; `iso` close
to nothing because the control is already calibrated to within half a
point per decile; the portfolio ordering close to random because the
correlations are near zero.

### Results: model refresh (run 2026-10-17, offline; `outputs/model_refresh/`)

535,731 player-game-lines and all 53,549 resting orders of 2025-26
re-priced by each variant. **Nothing adopted; the frozen model stays.**

| Variant | Log loss (vs control, t) | Veto: profit per 100 quoted orders (diff vs control, t) |
|---|---|---|
| control | 0.30380 | +112.9¢ |
| gbm | 0.30247 (−0.0013, t −5.8) | +107.9¢ (−5.0, t −0.6) |
| hier | 0.30413 (+0.0003, t 3.1) | +122.4¢ (+9.5, t 1.6) |
| iso | 0.30387 (+0.0001, t 1.8) | +113.2¢ (+0.3, t 0.1) |

- **Gradient boosting** is the better forecaster (0.4% lower log loss, a
  clear t of −5.8 over 536k lines) and makes *less* money as a veto. The
  two judgments disagree, so the pre-registered rule (both must improve)
  says no. Reading: the veto's job is to skip the lines where the public
  is right, and a sharper mean does not change which lines those are; the
  trees mostly reshuffle probabilities inside the band the veto already
  keeps.
- **Hierarchical player factor** goes the other way: slightly worse log
  loss (the nested selection picked the strongest prior, 300, in 27 of 36
  stat-months, i.e. the data asked for almost no player effect beyond the
  regression's own shrunk rates) and a +9.5¢ veto gain at t 1.6, short of
  the bar. Noted as the one candidate worth a second look next season with
  fresh orders; not adopted now.
- **Isotonic calibration** does nothing, as expected: the control is
  already calibrated.
- **Portfolio ordering (edge ÷ sd)**: $1,569 against $1,849 for edge-first
  over the season, and it loses the second half ($783 vs $1,161). Fails the
  bar. A side finding: random order, across six seeds, ends anywhere from
  $722 to $3,541 from the same orders, so the $500 compounding replay is
  far too path-dependent to rank orderings on; the per-100-orders metric
  (edge-first +21.8¢, edge/sd +16.9¢) is the one to read, and it also says
  no. Orders in the top edge quintile (cheap "no", average price 9.5¢) fill
  only 36% of the time against 70% in the middle quintile; a fill-aware
  ordering was the desk study's EV-ranking variant and was not robust.

Consequence: the mean model is not the bottleneck. The edge is in which
lines get quoted (the sportsbook veto) and in getting real fills, which is
where the next effort goes. `control_matches_frozen_rule` = 1.0: the
control's recomputed decisions reproduce the frozen ledger exactly.

## 2026-10-17: Targeted scan of the series the second pass ran out of time on

608 sports series (first-page volume $582M) were never scanned because their
shards hit the five-hour budget. `research/scan_markets.py` now takes
`--series` (and the workflow an input) to scan a named list. Two targeted
runs today, same rules as the second pass (400-market sample, resting "no"
one cent inside the ask, pass = t ≥ 2, both halves positive, "yes" side not
also passing):

**Goalscorer family.** EPL Goalscorer: 10 "no" fills in 400 markets
(+6.1¢, t 0.4). UCL Goalscorer: 41 fills, −8.3¢. UEFA Nations League Goal:
no two-sided quotes at all. EPL Corners: 39 fills, +10.9¢, second half
negative. Reading: soccer player markets on Kalshi are one-sided or empty
before kickoff; there is no crowd to rest against. Not a business today;
re-scan if soccer liquidity grows.

**Prop-like series.** 28 scanned. One pass: **KXNBA3D, NBA player
triple-double: 509 fills, +4.47¢, t 2.25, first half +6.2¢, second half
+2.8¢, "yes" side −11.2¢ (t −2.5)**, the thesis shape exactly (fans buy the
triple-double). Added to the paper ledger, judged 2026-12-15 or at 300
fills like the other candidates. Near misses, not added: NBA blocks (216
fills, +2.0¢, t 0.5, first half negative), WNBA assists (+1.8¢, t 0.4),
UCL both-teams-to-score (+1.4¢, t 0.4). Negative or flat: WNBA points
(−3.4¢) and rebounds (−4.0¢), NFL receptions (−1.1¢, 120 fills), MLB
hits+runs+RBIs (−2.0¢), golf head-to-head (−0.1¢), LoL games (−3.2¢). No
two-sided quotes before kickoff (0 fills in 400 markets): NFL rushing
attempts, fantasy points, longest rush, first team touchdown, NHL saves;
the receiving-yards escalators and ladders returned no settled markets.

Reading across both runs: the breadth that exists is NBA (a fifth NBA
market, same players and nights as the current edge). Most NFL player
props on Kalshi are not quoted two-sided pre-game, so a resting order never
fills there, whatever the edge would be.

## 2026-10-17: Outside review of the repo, item by item

A reviewer's five suggestions, checked against what the record already
holds and, where new, tested the same day (offline, no live change).

1. **Tennis: skip cheap "no".** New ledger. On the scan's own 2025-26 ITF
   orders (2,409 "no" orders, 1,180 fills, April–August), "no" at 30¢ or
   more made +6.4¢ per fill (t 2.7) against +1.7¢ (t 0.7) below 30¢; the
   reviewer's 40¢ cut separates less (+5.0¢ vs +4.4¢). The live ledger's
   first 64 fills lost everything below 30¢ (0 of 22). A second tennis
   ledger, `KXITFMATCH+no30` (same rule, "no" price ≥ 0.30), runs from
   today beside the unfiltered control, one $500 paper bankroll each,
   judged on new fills only at the same 2026-10-30 date. Stated plainly:
   the 30¢ line was chosen with both the historical and the first live
   fills in view, which is why it gets a new ledger rather than replacing
   the control. Not applied to NBA.
2. **Sportsbook veto timing and matching.** Already as the reviewer asks:
   `research/bookline_backtest.py` requests the historical odds snapshot at
   the order's posting time (`end_ts`, five hours before tip for most),
   never later, and matches events by both teams with a start within three
   hours. The residual risk is player-name matching across feeds
   (`normalize_name`); 1,647 of 21,373 fills had a book quote, so coverage,
   not timing, is the weakness, and more books in the consensus is the
   lever. Nothing to change.
3. **Execution: cancel after 30 minutes / only post at a bigger edge.**
   Tested now against the 2025-26 fills matched to their prints
   (`outputs/execution/markouts.csv.gz`): the median fill arrives 105
   minutes after posting. Per 100 eligible orders, hold to tip +170.8¢ (t
   6.1); cancel if unfilled after 30 min +42.6¢; 60 min +71.4¢; 120 min
   +105.5¢; 180 min +130.1¢. Early cancellation keeps the slightly better
   fills (+3.6¢ each) and throws away two thirds of them. Hold stays. The
   "only post at a comfortable edge" variant was the placement study's
   `edge2` (edge ≥ 2¢): fewer orders, less profit per 100 eligible. The
   1¢-inside placement remains the baseline, as the reviewer suggests.
4. **Hindsight in the NBA model (who actually played).** Measured on
   2026-10-15 and bounded: the maker test without the teammates-out
   features (the only features that use the box score's "did not play")
   made +3.8¢ (t 5.0) against +3.4¢ with them, so the head start is not
   where the edge comes from. The model-free rule (+1.65¢, t 4.1) uses no
   model at all and so no hindsight. The live trader reads the actual
   injury report (`cancel_on_injury`), not the box score. Nothing to change.
5. **Correlated exposure within a game.** Already in place: per-game cap
   10% of bankroll, per-team cap, daily cap 30%, and the inventory study
   (2026-10-17) measured nightly profit against the night's scoring
   surprise at −0.06 (178 nights), below the pre-registered line for a
   hedge. The staged 1%/30% start and the adaptive (shrunk-Kelly) cap are
   the conservative-while-unproven options already documented.

Net: one new ledger (tennis price floor), four items confirmed already
handled or tested and declined.

### Second review (same day): two points stood up, both fixed

The reviewer re-read the repo and pushed back on items 2, 4 and 5 above.
Checked against the code and data:

**2a. Per-order snapshot timing in the sportsbook backtest: the reviewer
was right.** `sample_games` took one `end_ts` per game from an arbitrary
order, and 634 of the 1,013 games have both 5-hour and 2-hour postings. In
38 of the 151 sampled games the snapshot was requested after the game's
earliest posting (up to 176 minutes after). Per order, 53 of the 3,197
covered fills were vetoed with a quote newer than their own posting.
Re-scored on the 3,144 fills whose snapshot predates their posting: kept
+3.90¢ (t 2.43), removed +0.38¢, kept-minus-removed +3.52¢ — against +3.81
/ +0.38 / +3.44 as reported. The lookahead rows themselves were −1.5¢, so
the error worked against the result, not for it. Fixed:
`sample_games` now requests the snapshot at each game's earliest posting.
The committed `book_lines.csv.gz` is from the old run; a refetch costs
~12,000 credits and is deferred, since the per-order re-score above
settles the question. Coverage numbers corrected: 4,862 sampled fills in
151 games, 3,197 with a book quote (65.8%), 1,647 kept by the veto.

**2b. Live consensus ignored the event: the reviewer was right, and this
one mattered.** `consensus()` grouped by (book, player, stat, line) with no
event key, and `attach()` matched on (player, stat, line) only. The book
log looks 36 hours ahead, so a player on a back-to-back had both games'
lines in the file and the first one won. Fixed: `consensus()` groups by
event (`event_id`, `commence_utc`) as well, `log_books.py` keeps those
columns, and `attach()` matches on the Eastern game day when both sides
carry a time (falls back to the old match otherwise). Unit-checked on a
two-game case. The NBA book log is empty until the season starts, so no
live decision was affected; the NFL log was one game per player.

**4. Closing lines in the model features: correct as stated, left as a
known limit.** `implied_points` and `team_spread` come from the closing
line, which a 5-hours-before order could not see. The move from opening to
close is small for NBA sides and totals and enters only through the model
veto (the model-free rule, +1.65¢, uses no line). Recorded; a point-in-time
line history would be needed to remove it, and we do not have one.

**5. Within-game correlation measured directly, as asked.** On the 20,641
model-veto fills, standardised against the market-implied probability:
intraclass correlation within a game +0.029 (21 fills per game on
average, variance inflation 1.57×), within a team-game +0.032, within a
player-game +0.067 (4.9 fills), within a night +0.018 (113 fills,
inflation 3.05×). Small per pair, material in aggregate, which is exactly
what clustering is for: the headline t is 9.7 if fills were independent,
6.1 clustered by game, 6.5 clustered by night. The −0.06 figure in the
inventory study was a different quantity (nightly profit against the
night's scoring surprise) and is no longer cited as evidence of
independent positions. The live limits ($25 per game, $15 per team, 6
orders per game, $75 per night) already bound the within-game and
within-night exposure; the paper props ledger has only the daily cap and
is a measurement tool, not a sizing tool, so it stays as is. The $25
per-game live cap equals the replay's 10% at the $250 start and binds
harder as the bankroll grows, which is the conservative side.

Net from the two reviews: two code fixes (snapshot timing, event-aware
book matching), two corrections to how results were described (coverage
numbers, the correlation claim), no change to any rule or result.
