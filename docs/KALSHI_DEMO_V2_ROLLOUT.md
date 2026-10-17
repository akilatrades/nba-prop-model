# Kalshi V2 demo rollout — operator checklist

The bot remains **dry-run by default**. Updating GitHub does not deploy to the
DigitalOcean VPS or authorize live production-money trades.

## Verified on DigitalOcean (2026-10-17)

- `/etc/market-maker.env`: `KALSHI_ENV=demo`, `TRADE_MODE=dry`.
- A read-only authenticated balance request succeeded.
- A separate manual **V2** demonstration placed one 16-cent YES bid,
  canceled 1.00 resting contract, and then independently verified status
  `canceled`, zero fills, and $2,500 in available demo funds.
- `market-maker.timer` was stopped for the test and should remain paused
  until the rollout checklist passes.
- The manual order test **did not** exercise strategy candidate selection,
  execution reporting, fills, or the automated trading loop.

## Changes made to the execution client

`TradingClient.place` uses POST `/trade-api/v2/portfolio/events/orders`.
`TradingClient.cancel` uses DELETE
`/trade-api/v2/portfolio/events/orders/{order_id}` with the
`market_ticker` query parameter for exchange-shard routing.
The legacy authenticated *read* endpoints are still used for balance,
resting orders, fills, and reconciliation.

The strategy's existing `Order` abstraction is translated to Kalshi's V2
single-YES-book API:

| Strategy action | V2 side | V2 YES limit price |
| --- | --- | --- |
| Buy YES | `bid` | `YES price` |
| Sell YES | `ask` | `YES price` |
| Buy NO | `ask` | `1 - NO price` |
| Sell NO | `bid` | `1 - NO price` |

Order counts are fixed-point strings; prices are dollar strings.
The strategy retains post-only, tip-off expiry, self-trade prevention and
exchange-pause cancellation. The existing limits in `kalshi/risk.py` are
unchanged. Production *placement* requires an additional explicit
`ALLOW_PRODUCTION_TRADING=YES` flag (never use that flag for demo).

## Deployment gates

1. Check the pull-request tests and review V2 request/response conversions,
   particularly the Buy NO and Sell NO cases.
2. On the VPS, confirm the timer is **inactive**, `TRADE_MODE=dry`,
   and `KALSHI_ENV=demo`. Never print full `/etc/market-maker.env` or any
   private keys.
3. Before pulling GitHub changes, back up the deployment configuration and
   the entire local `outputs/live` ledgers safely **on the server**, then
   check for uncommitted changes. The VPS runner auto-commits run summaries;
   do not overwrite or discard them.
4. Deploy the reviewed revision while the timer remains paused.
   Run `.venv/bin/python -m pytest tests/test_trading.py tests/test_nba_trade.py`.
5. Run one manual **dry-mode** pass for NFL and NBA. Check `last_run.json`,
   `orders.csv`, logs and reconciliation diagnostics. Confirm no external
   orders were submitted.
6. Diagnose zero-candidate NFL scans separately. Do **not** lower model-edge
   thresholds just to manufacture an order or claim that a historical
   simulation proves future profitability.
7. Review the complete live risk budget, unmatched-order handling, and the
   `outputs/live/KILL` behavior before enabling demo exchange submissions.
   Start with one guarded, manually observed demo pass and inspect orders,
   fills and positions; don't jump to minute-by-minute unattended demo runs.
8. Only after this should `market-maker.timer` be restarted, with the
   **demo** URL and execution mode verified. Never switch to `KALSHI_ENV=prod`
   as part of this rollout.

The 2026-10-17 manual demo order demonstrated only the order/cancel path;
the strategy's forward profitability and real fill capture remain unverified.

References:
- https://docs.kalshi.com/api-reference/orders/create-order-v2
- https://docs.kalshi.com/api-reference/orders/cancel-order-v2
