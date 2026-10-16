#!/usr/bin/env bash
# One-time setup of the always-on trader on a fresh Ubuntu 24.04 machine.
# Run as root:  bash <(curl -fsSL https://raw.githubusercontent.com/akilatrades/sports-market-maker/main/deploy/install.sh)
set -euo pipefail

apt-get update -qq
apt-get install -y -qq git python3-venv python3-pip >/dev/null

id -u trader >/dev/null 2>&1 || useradd --create-home --shell /bin/bash trader
cd /home/trader
if [ ! -d sports-market-maker ]; then
  sudo -u trader git clone -q https://github.com/akilatrades/sports-market-maker.git
fi
cd sports-market-maker
sudo -u trader git pull -q || true
sudo -u trader python3 -m venv .venv
sudo -u trader .venv/bin/pip install -q -r requirements.txt

# Credentials live in one root-only file; the trader reads them as environment variables.
if [ ! -f /etc/market-maker.env ]; then
  cat > /etc/market-maker.env <<'ENV'
TRADE_MODE=dry
KALSHI_ENV=demo
# Which "no" seller variant to trade: no_seller (v1) or no_seller_bookline
# once its historical test passes (RESEARCH_LOG.md, 2026-10-16).
TRADE_RULE=no_seller
KALSHI_KEY_ID=paste-key-id-here
KALSHI_PRIVATE_KEY=paste-private-key-here
ODDS_API_KEY=paste-odds-key-here
DRY_BANKROLL=250
# Optional: a GitHub fine-grained token (Contents: read and write on this repo)
# so the machine can push its trade log to outputs/live for review.
GIT_TOKEN=
ENV
  chmod 600 /etc/market-maker.env
  echo ">> Edit /etc/market-maker.env with your keys:  nano /etc/market-maker.env"
fi

# The runner: pull the latest code and outputs, run the trader, push its log.
cat > /home/trader/run.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd /home/trader/sports-market-maker
# Credentials arrive from systemd's EnvironmentFile; nothing is read here.
git pull -q --rebase origin main || true
mkdir -p outputs/live
.venv/bin/python -m live.nba_trade >> /home/trader/trader.log 2>&1
if [ -n "${GIT_TOKEN:-}" ]; then
  git add -A outputs/live >/dev/null 2>&1 || true
  git diff --cached --quiet && exit 0
  git -c user.name=trader -c user.email=trader@market-maker commit -qm "Trade run $(date -u +%Y-%m-%dT%H:%MZ) (${TRADE_MODE}, ${KALSHI_ENV})" >/dev/null 2>&1 || true
  git push -q "https://x-access-token:${GIT_TOKEN}@github.com/akilatrades/sports-market-maker.git" HEAD:main >/dev/null 2>&1 || true
fi
RUN
chmod 755 /home/trader/run.sh
chown trader:trader /home/trader/run.sh

# Every minute, 11am to midnight ET (15:00-04:59 UTC), as a systemd timer.
# A pass takes a few seconds; systemd does not start one while another runs.
cat > /etc/systemd/system/market-maker.service <<'UNIT'
[Unit]
Description=Sports market maker: one trading pass
[Service]
Type=oneshot
User=trader
EnvironmentFile=/etc/market-maker.env
ExecStart=/home/trader/run.sh
UNIT
cat > /etc/systemd/system/market-maker.timer <<'UNIT'
[Unit]
Description=Run the market maker every minute during NBA hours
[Timer]
OnCalendar=*-*-* 15..23:*:00
OnCalendar=*-*-* 00..04:*:00
AccuracySec=5s
Persistent=true
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now market-maker.timer
echo ">> Installed. Timer status:"; systemctl list-timers market-maker.timer --no-pager | head -3
echo ">> Logs: tail -f /home/trader/trader.log"
