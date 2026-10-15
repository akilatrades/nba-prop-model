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
