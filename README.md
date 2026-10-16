# Sports market maker

[![tests](https://github.com/akilatrades/sports-market-maker/actions/workflows/tests.yml/badge.svg)](https://github.com/akilatrades/sports-market-maker/actions/workflows/tests.yml)

A market-making strategy for Kalshi's sports markets, built from two
player-prop pricing models (NBA and NFL) and a year of pre-registered tests
against real prices. Every test's rule is written in
[RESEARCH_LOG.md](RESEARCH_LOG.md) before its data is downloaded, and every
result is reported, including the failures.

## The finding

Buying what the models like does not pay. Selling what retail buys does.

Across 57,111 Kalshi NBA prop markets from 2025-26, the NBA model knew things
the price did not (t = 13.8 alongside the price), yet buying its picks at the
ask lost 0.9¢ a contract after the spread and fee. The market's price was the
better forecast. The same held for Miami temperature markets and for parlays.

What made money was the other side of the book. The people who buy "yes" on
a player prop at the asking price are uninformed and pay the spread; the
people who sell "yes" into a bid are informed. A resting "no" order one cent
inside the ask, on every line, made +1.65¢ a contract with no model at all
(33,475 fills). The model's job is to keep the sharps out: skipping orders it
rates as losers raised that to **+2.9¢ a contract (t 6.1), positive in both
halves of the season, in all four stats and at every price level.** Buying
"yes" the same way lost 2.0¢.

A scan of 274 Kalshi sports series with the same model-free rule found the
pattern in player props and a handful of other markets (NFL anytime
touchdown, ITF tennis, NASCAR, soccer spreads) and its opposite in
game-winner markets, where both sides are a team and the sharps are present.

Kalshi's own same-game combos (parlays) are the extreme case: buyers lost 9%
per dollar on $1.4M of trades, 19% on five or more legs. That edge belongs to
the quoting market makers, and is noted as the destination, not pursued.

## What runs

| Job | What it does | Since |
|---|---|---|
| NBA "no" seller | Resting "no" on every NBA prop line 2-6 h before tip-off, skipped when the NBA model rates it below zero; $500 paper bankroll | 2026-10-20 |
| NBA model side | The model's favoured side when its edge after fees is 3¢ or more; $500 | 2026-10-20 |
| NBA taker (control) | Buy at the ask when the model's edge is 5¢ or more, tight spreads only | 2026-10-20 |
| NFL "no" seller, model veto | Anytime touchdown and receiving yards, NFL model as veto; $500 each | 2026-10-16 |
| Model-free "no" seller | ITF tennis, NASCAR, Liga MX and La Liga spreads, NFL anytime TD; $500 each, as the control | 2026-10-16 |
| Price logs | Kalshi NBA and NFL prop snapshots before games | 2026-10 |

Fills are read from Kalshi's public trade prints with price priority (an
order fills only with contracts that traded through its price), and
positions settle from Kalshi's results. Scorecards:
[`outputs/paper_maker/`](outputs/paper_maker/) (NBA),
[`outputs/paper_markets/`](outputs/paper_markets/) (everything else).
Pass criteria are fixed in the log; nothing is traded with money.

## Simulations, if the edge holds

Replays of the 2025-26 NBA orders with the staking rule (1% of the bankroll
per order, 30% a day, half the traded flow): $500 to about $2,600 over a
season; $5,000 to about $24,000; $25,000 to about $73,000, where flow rather
than capital becomes the limit. Adding sports with the same edge raises the
end and cuts the drawdown (five sports from $25,000: median $116,000, worst
drawdown 2%). All in-sample and one season; the live ledgers are the test.

## Layout

```
kalshi/        API client, resting-order fills and fees, sizing, paper ledgers, statistics
markets/nba/   NBA prop model: minutes and rates, Poisson GLM, empirical spread, in-game model
markets/nfl/   NFL prop model: usage, efficiency, lines, defence, injuries (from nfl-parlay-model)
live/          scheduled jobs: data refreshes, projections, price logs, paper ledgers
research/      every test: market test, maker, parlays, combos, in-game, weather, the series scan
outputs/       results, scorecards, ledgers        data/   prices and box scores (large files ignored)
docs/          the NBA and NFL projects' own READMEs, kept as history
```

## Reproduce

```bash
pip install -r requirements.txt
pytest                                  # 64 unit tests on synthetic data, no downloads
python -m live.fetch_nba_data           # NBA box scores, schedules and lines (hoopR, ~10 MB)
python -m research.backtest             # NBA walk-forward backtest; --holdout for 2025-26
python -m research.market_test          # NBA model vs Kalshi 2025-26 (needs data/kalshi_history/)
python -m research.new_ideas            # maker, parlays, correlation
python -m research.parlays              # 2- and 3-leg parlays from model picks
python -m research.combos               # Kalshi combo takers
python -m live.fetch_nfl_data && python -m live.nfl_projections   # NFL week ladders
```

Kalshi downloads run in GitHub Actions (`research/backfill_*.py`,
`research/scan_markets.py`); the shell here cannot reach Kalshi.

## What failed

- Buying model picks at the ask (NBA, −0.9¢/contract; weather, −2 to −4¢).
- Parlays built from the model's high-probability picks (−7% to −10% per dollar).
- Ladder arbitrage (6 chances in 116,307 snapshots) and double-doubles priced
  with a copula (Kalshi already prices the correlation).
- In-game trading passed (+4.2¢) but only within a minute of the play; it needs
  a real-time feed and is not run.
- Miami daily-high temperature: the market forecasts better than public models.

## Limitations

- One season of NBA prices; the "no" side was found after idea 1's results,
  so 2026-27 is its first out-of-sample test.
- Hourly candles show the best bid and ask, not depth; the fill rule is
  conservative but the flow share is a guess (half, from here on).
- Scanned markets were chosen because they passed; expect less.
- Kalshi's maker fee is assumed at a quarter of the taker fee.

## AI use

I used AI tools to help write code and documentation and to review my work.
I reviewed the analytical assumptions, methods and results and remain
responsible for the claims made here.

## License

MIT. NBA data from the sportsdataverse hoopR repository, NFL data from
nflverse, Kalshi prices from Kalshi's public API.
