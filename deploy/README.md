# Deploying the always-on trader

One small Ubuntu machine runs `live/nba_trade.py` every minute during NBA
hours, once for NBA props and once for NFL props (`TRADE_MARKET=nfl`, log in
`trader-nfl.log`) (the first order at a price level fills first, so a quote that is a
minute stale is less likely to be the one a sharp sweeps). `install.sh` sets it up; credentials go in `/etc/market-maker.env`
(root only, never in the repository). Modes: `TRADE_MODE=dry` logs orders
without sending them; `live` sends them to the exchange named by `KALSHI_ENV`
(`demo` or `prod`); `TRADE_RULE` picks the variant (`no_seller`, or
`no_seller_bookline` once its historical test passes). Sportsbook lines reach
the machine through `git pull`: the `book lines` GitHub job writes
`outputs/books/nba_latest.csv` and `nfl_latest.csv` six times a day. To stop everything: `systemctl disable --now market-maker.timer`,
or create `outputs/live/KILL` in the repository, which also cancels resting orders.
