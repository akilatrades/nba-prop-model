# Forward validation — frozen research rules

Established 2026-10-17, before this season's NBA forward sample. The baseline
trader/research code is commit `1167400bc246ab324472eefc6a24bdb8e2cf7d52`.
The changes introduced with this plan concern **measurement only**, not
trade selection, quote prices, cancel rules, capital allocation, or live mode.

## Research question

Can the historically simulated NBA resting-NO edge (~+2.9 cents per filled
contract after modeled fees) survive **actual exchange acceptance, fills,
fees, and real account risk constraints** at usable capacity?

The historical scaling study's 50% assumed share of qualifying prints is a
sensitivity assumption, not an observation. It must not be restated as a
measured live fill rate.

## When the clock starts

Start at the first exchange-accepted *real* strategy order. Count 30 calendar
days from that timestamp. Do not treat scheduled jobs, dry runs, or days with
no positions as empirical evidence. Keep the rule set frozen through the
observation window; record any unavoidable fixes as deviations with a new
version and separately labeled subsequent cohort.

## Pre-registered checkpoint

Review when **both** (a) at least 30 days have elapsed since the first accepted
real order and (b) at least 500 **unique, settled exchange buy fill records**
have been observed. This definition uses fill events, **not** contracts,
orders, or hypothetical trade-throughs. The checkpoint authorizes a human
review, not automatic scaling. Fewer observations = inconclusive, not failed.

Report all of the following on the same cohort:

- **Volume-weighted capture** = total actual filled NO contracts divided by
  total qualifying printed contracts for mature, completely observed,
  comparable resting-order windows; also show missing/impossible denominator
  rates. Retain fill share of an individual order as a separate metric.
- **Queue depth**: fill probability for starting-queue bins 0, 1–20, 21–100,
  and >100 contracts; missing queue observations remain missing.
- **Net profit**: dollars and cents per truly settled contract *after actual
  execution fees*. Use game-clustered uncertainty; also show P&L per eligible
  real order, including the nonfills as zeros, and drawdown by settlement day.
- **Capacity and selection**: placed orders, fill events, filled contracts,
  number of independent games, quality of coverage, and any exit-traded
  markets excluded from a hold-to-settlement proxy.

Known limitation: the order-level calculation currently uses the original
NO buy and final market payout. A market with an early sell/exit needs separate
FIFO or equivalent realized-P&L accounting; exclude it from the *hold-to-
settlement* statistic, and explicitly report the excluded count. Neither
missing outcomes nor missing public trade prints may be silently treated as
losses, zero qualifying volume, or successful capture.

## Intervention criteria (analysis, not trading automation)

- If measured comparable flow capture falls **below 25%**, revisit the
  bankroll-scaling sensitivity study with the measured share before even
  considering a capital increase.
- Evaluate net P&L, queue-depth effects, missing-data fraction, game/night
  concentration, and fill-selection effects. A positive backtest or positive
  first week is **not** enough to scale.
- A checkpoint may be declared `data_incomplete` until >=95% of matured
  windows have complete trade-volume observations and >=95% of unexited
  matured orders have known profitability (unfilled orders count as known
  zero P&L). Orders involving early exits require reconciled realized
  accounting before any claim of strategy-wide income.
- A statistically ambiguous or unfavorable forward result means no
  automatic strategy "optimization", bankroll step-up, or live-rule changes.

## Operations

The existing DigitalOcean process keeps trading exactly as configured.
The trader writes:
`outputs/live/{orders.csv,fills.csv,queue.csv,execution.csv,
execution.md,forward_validation.json,forward_validation.md}`
(and corresponding files under `outputs/live/nfl/`).

Read-only manual report regeneration:

```sh
python -m live.forward_validation --folder outputs/live
python -m live.forward_validation --folder outputs/live/nfl
```

This command **does not submit/cancel orders, change secrets or alter strategy
parameters**. Check that execution reporting preserves matured qualifying
volume across older-than-three-day replays and that actual settlement
outcomes are populated. The scorecard is a diagnostic snapshot; source CSVs
are the audit trail. Don't commit personal API secrets or privately sensitive
account detail.

The forward review should cover the whole cohort. Avoid inspecting early
winners and quietly altering thresholds. Changes to production money settings
require an explicit separate decision.
