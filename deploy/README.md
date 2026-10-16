# Deploying the always-on trader

One small Ubuntu machine runs `live/nba_trade.py` every five minutes during
NBA hours. `install.sh` sets it up; credentials go in `/etc/market-maker.env`
(root only, never in the repository). Modes: `TRADE_MODE=dry` logs orders
without sending them; `live` sends them to the exchange named by `KALSHI_ENV`
(`demo` or `prod`). To stop everything: `systemctl disable --now market-maker.timer`,
or create `outputs/live/KILL` in the repository, which also cancels resting orders.
