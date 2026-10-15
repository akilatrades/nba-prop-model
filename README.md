# NBA prop model

A pricing model for NBA player props (points, rebounds, assists, threes),
tested against a full season of Kalshi prices. A sister project to my
[NFL parlay model](https://github.com/akilatrades/nfl-parlay-model).

**Status: in progress.** The plan and the market test were written down
before any price was downloaded: see [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

| Phase | What it adds | Status |
|---|---|---|
| 1 | A full range of outcomes for each player stat, from minutes, per-minute rates, opponent, pace, the betting line and teammates who are out; walk-forward backtest | Building |
| 2 | Test against Kalshi's 2025-26 prices at tip-off, after fees, with the rule fixed in advance | Prices downloading |
| 3 | Live: Kalshi price log and daily probability ladders from opening night | Price log running |

## Kalshi data

Kalshi lists NBA player props as ladders of yes/no contracts, for example
"Victor Wembanyama: 25+ points". Two scheduled workflows collect them:

- [`backfill.yml`](.github/workflows/backfill.yml) downloads every settled
  2025-26 market and its hourly prices around each game
  ([`scripts/backfill_kalshi.py`](scripts/backfill_kalshi.py)), into
  `data/kalshi_history/`.
- [`kalshi.yml`](.github/workflows/kalshi.yml) saves every open market each
  morning and before each evening's tip-offs
  ([`log_kalshi.py`](log_kalshi.py)), into `data/kalshi/`.

## License

MIT. Box scores and lines come from the sportsdataverse hoopR data
repository; Kalshi prices from Kalshi's public API.
