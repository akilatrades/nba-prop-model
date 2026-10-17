# Sports market maker

**Forward-validation protocol:** [30-day execution evidence and the 500-fill checkpoint](docs/FORWARD_VALIDATION.md). The live selection, sizing and risk rules are unchanged; reports never enable live trading or authorize a capital increase.

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
(33,475 fills). Buying "yes" the same way lost 2.0¢.

![Resting "no" orders made money at every price level; resting "yes" orders lost](docs/figures/no_vs_yes_by_price.svg)

The model's job is to keep the sharps out. Profit rises with the model's
opinion of the order: skipping the ones it rates below zero raised the result
to **+2.9¢ a contract (t 6.1), positive in both halves of the season, in all
four stats and at every price level.**

![Profit per contract by the model's edge on the order](docs/figures/profit_by_model_edge.svg)

## The research, test by test

Every test below was written into [RESEARCH_LOG.md](RESEARCH_LOG.md) before
its data was downloaded, run once, and recomputed with separate code. One
season of Kalshi prices (2025-26) unless stated.

| Test | Question | Result | Verdict |
|---|---|---|---|
| NBA model backtest (2021-2025, holdout 2025-26) | Can a model price NBA props? | Calibrated within 0.6 points on the holdout season; beats a recent-average baseline | Passed |
| Market test A | Does the model know anything the price doesn't? | Yes: t = 13.8 alongside the price, 57,111 markets | Passed |
| Market test B (taker) | Buy the model's picks at the ask, 5¢ edge | 12,495 trades, **−0.9¢** a contract | Failed |
| Phase 4 (taker, tight spreads, 2-6 h before tip) | Cheaper execution? | +1.9¢ overall but +0.02¢ in the regular season; all of it in the playoffs | Not trusted; paper only |
| Idea 1: resting orders on the model's side | Earn the spread instead of paying it | 4,541 fills, **+3.1¢** (t 4.1); regular season +2.6¢ | Passed |
| Idea 2: copy the sportsbooks | Use closing lines as the filter | Not testable then: 176 of 1,072 games had stored lines. Re-run with a paid odds feed, see below | No result |
| Idea 3: ladder arbitrage, double-doubles | Structural mispricing | 6 riskless pairs in 116,307 snapshots; double-doubles −1.7¢ | Failed |
| The "no" seller (exploratory, then pre-registered live) | Sell "yes" to retail on every line, model as veto | 21,373 fills, **+2.9¢** (t 6.1); +1.65¢ with no model | Live paper test 2026-27 |
| In-game | Trade on play-by-play during games | **+4.2¢** (t 6.3) at the quote; +0.3¢ five minutes late | Passed, but needs a real-time feed; not run |
| Parlays from model picks | 2- and 3-leg parlays of high-probability legs | **−7% and −10%** per dollar; the same legs singly −2.5% | Failed |
| Kalshi NBA combos | Do parlay buyers overpay? | **−9% per dollar** for buyers (t −2.5), −19% on 5+ legs; $1.4M sample | Edge belongs to the quoting makers |
| Miami daily high (weather) | Does the method transfer? | Market forecasts better than public models; every rule lost | Failed |
| Market scan, 274 Kalshi sports series | Where else does the "no" seller work? | Props mostly pay; big-league game winners lose; 10 series passed | Candidates, paper-tested |
| Market scan, second pass: every Kalshi category (1,178 testable series) | Does it work outside sports? | No: commodities −0.6¢, financials −1.1¢, nothing in crypto, politics or entertainment. Only "Mentions" markets (+3.6¢) and more sports props (soccer both-teams-to-score +11.5¢, UCL corners, NBA free throws) | Six paper ledgers from 2026-10-17 |
| Liquidity rewards | Does Kalshi pay makers here? | 8 programs on NBA props, $800 total | No |
| Sportsbook-line veto | Are the books a better veto than the model? | 151 sampled games, books' lines at the moment each order was posted: kept fills **+3.8¢** (t 2.4) against +0.4¢ removed; passed its pre-registered test | Live trader's rule |
| Markouts and toxicity | Are our fills the ones a sharp sweeps? | Price drifts 1¢ against us after a fill and stays, yet settles +1.65¢; no pre-posting feature separates good fills from bad | No veto; edge is in settlement, not the mark |
| Quote placement | Join the bid, 1¢ or 2¢ inside? | Per 100 orders: join +86¢, 1¢ +171¢, 2¢ +168¢; 1¢ with a 2¢ model edge +191¢ | Current 1¢ rule kept |

![Resting "no" orders across Kalshi series: props pay, big-league winners lose](docs/figures/scan_props_vs_winners.svg)

The scan is the clearest statement of the mechanism. In a prop market
("does Jokić score 25?") the "yes" side is what fans buy, so selling it pays.
In a winner market ("do the Lakers beat the Celtics?") both sides are a team,
the flow is two-sided and the sharps are present, so selling "no" is just
backing the underdog at full stake. The passing winner markets (ITF tennis,
NASCAR, WTA) are long-shot fields where the edge, if real, is the
favourite-longshot bias rather than spread capture, and the first days of
live paper trading in ITF are going against it.

## Simulations, if the edge holds

Replays of the 2025-26 NBA "no" seller with the staking rule (1% of the
bankroll per order, 30% a day, half the traded flow): **$500 to about
$2,600**, **$5,000 to about $24,000**, **$25,000 to about $73,000**, where
flow rather than capital becomes the limit. Adding sports with the same edge
raises the end and cuts the drawdown (five sports from $25,000: median
$116,000, worst drawdown 2%). All in-sample and one season; the live ledgers
are the test.

![Bankroll replay, $500 and $5,000 starts](docs/figures/bankroll_replay.svg)

## What runs

| Job | What it does | Since |
|---|---|---|
| NBA "no" seller | Resting "no" on every NBA prop line 2-6 h before tip-off, skipped when the NBA model rates it below zero; $500 paper bankroll | 2026-10-20 |
| NBA "no" seller, v2 variants | Five variants beside v1, each $500: last 2 hours only; sized by edge; 2¢ veto; rebounds veto; sportsbook-line veto (the books' de-vigged probability decides on lines they quote) | 2026-10-20 |
| NBA model side | The model's favoured side when its edge after fees is 3¢ or more; $500 | 2026-10-20 |
| NBA taker (control) | Buy at the ask when the model's edge is 5¢ or more, tight spreads only | 2026-10-20 |
| NFL "no" seller, model veto | Anytime touchdown and receiving yards, NFL model as veto; $500 each | 2026-10-16 |
| Model-free "no" seller | ITF tennis, NASCAR, Liga MX and La Liga spreads, NFL anytime TD and receiving yards; $500 each, as the control | 2026-10-16 |
| Second-scan candidates | Club soccer both-teams-to-score, Champions League corners, UFC method of finish, NBA free throws made, NCAA broadcast mentions, college basketball 1H spreads (from November); $500 each, judged 2026-12-15 | 2026-10-17 |
| Price logs | Kalshi NBA and NFL prop snapshots before games | 2026-10 |
| Book lines | Sportsbooks' NBA prop lines (de-vigged consensus only) six times a day | 2026-10-16 |
| Live trader (dry run) | `live/nba_trade.py`: the "no" seller's real orders on NBA props (from opening night) and NFL anytime-TD and receiving-yards props (from 2026-10-18, `TRADE_MARKET=nfl`), sized from the account, through the risk limits; every 15 minutes on GitHub and every minute on a small server. Cancels resting orders on new injury news; rests an exit when a position has moved 6¢ our way. Places nothing until switched to live by hand | 2026-10-16 |

Fills are read from Kalshi's public trade prints with price priority (an
order fills only with contracts that traded through its price), and
positions settle from Kalshi's results. Scorecards:
`outputs/paper_maker/` (NBA, from opening night 2026-10-20),
[`outputs/paper_markets/`](outputs/paper_markets/) (everything else).
Pass criteria are fixed in the log; nothing is traded with money until the
dry run and a demo-account run have shown real fills matching the paper
ledger's assumptions.

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

## Limitations

- One season of NBA prices; the "no" side was found after idea 1's results,
  so 2026-27 is its first out-of-sample test.
- Hourly candles show the best bid and ask, not depth; the fill rule is
  conservative but the flow share is a guess (half, from here on).
- Scanned markets were chosen because they passed; expect less.
- Kalshi's maker fee is assumed at a quarter of the taker fee.
- Paper fills assume every print through our price was ours. Real fills
  face a queue and adverse selection (informed buyers sweep first); the gap
  between paper and real fill rates is reported monthly once the trader runs.

## AI use

I used AI tools to help write code and documentation and to review my work.
I reviewed the analytical assumptions, methods and results and remain
responsible for the claims made here.

## License

MIT. NBA data from the sportsdataverse hoopR repository, NFL data from
nflverse, Kalshi prices from Kalshi's public API.
