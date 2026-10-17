#!/usr/bin/env bash
# Root-only guarded deployment of a 48-hour Kalshi DEMO order-observation pilot.
# The normal trading timer must be paused and TRADE_MODE=dry at the start.
set -Eeuo pipefail

REPO=/home/trader/sports-market-maker
CONFIG=/etc/market-maker.env
BACKUP=""
if [[ "$EUID" -ne 0 ]]; then
  echo "STOP: run this as root in the DigitalOcean console." >&2
  exit 1
fi
if systemctl is-active --quiet market-maker.timer; then
  echo "STOP: trading timer is already active." >&2
  exit 1
fi
if systemctl is-active --quiet market-maker.service; then
  echo "STOP: trading service is already executing." >&2
  exit 1
fi
if [[ -e "$REPO/outputs/live/KILL" ]]; then
  echo "STOP: emergency KILL file exists; refusing activation." >&2
  exit 1
fi
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "STOP: expected Python virtualenv is missing." >&2
  exit 1
fi

# Do not log secrets. Refuse conflicting or duplicate settings.
python3 - <<'PY'
from pathlib import Path
lines = Path("/etc/market-maker.env").read_text().splitlines()
values = {}
for line in lines:
    if "=" in line and not line.lstrip().startswith("#"):
        name, value = line.split("=", 1)
        values.setdefault(name, []).append(value)
for name, expected in (("KALSHI_ENV", "demo"), ("TRADE_MODE", "dry")):
    if values.get(name) != [expected]:
        raise SystemExit(f"STOP: {name} must be uniquely configured as {expected}")
if values.get("ALLOW_PRODUCTION_TRADING") == ["YES"]:
    raise SystemExit("STOP: production override must not be configured")
for name in ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY"):
    if len(values.get(name, [])) != 1 or not values[name][0]:
        raise SystemExit(f"STOP: {name} must be present exactly once")
if len(values.get("DEMO_PILOT", [])) > 1:
    raise SystemExit("STOP: duplicate DEMO_PILOT settings")
PY

cd "$REPO"
if [[ "$(runuser -u trader -- git branch --show-current)" != "main" ]]; then
  echo "STOP: repository branch is not main." >&2
  exit 1
fi
if ! grep -q 'DEMO_PILOT' "$REPO/live/nba_trade.py"; then
  echo "STOP: pilot risk controls are not installed." >&2
  exit 1
fi
if ! grep -q 'market_data_base_url()' "$REPO/kalshi/api.py"; then
  echo "STOP: venue-matched market data patch is not installed." >&2
  exit 1
fi

echo "Running targeted safety tests (no exchange orders)..."
runuser -u trader -- bash -c '
  cd /home/trader/sports-market-maker
  .venv/bin/python -m pytest -q     tests/test_trading.py tests/test_kalshi.py tests/test_demo_pilot.py
'

# Preserve credentials and on-disk order history before modifying the environment.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
install -d -m 700 /root/market-maker-config-backups
BACKUP="/root/market-maker-config-backups/market-maker.env.$STAMP"
cp -p "$CONFIG" "$BACKUP"
install -d -m 700 -o trader -g trader /home/trader/market-maker-backups
runuser -u trader -- tar -czf   "/home/trader/market-maker-backups/live-pre-pilot-$STAMP.tar.gz"   -C "$REPO" outputs/live
echo "Trading logs and environment backed up (credential contents hidden)."

rollback() {
  result=$?
  systemctl stop market-maker.timer >/dev/null 2>&1 || true
  systemctl stop market-maker-demo-stop.timer >/dev/null 2>&1 || true
  if [[ -n "$BACKUP" && -f "$BACKUP" ]]; then
    cp -p "$BACKUP" "$CONFIG"
  fi
  echo "STOP: Activation did not finish. Original dry-mode config restored." >&2
  exit "$result"
}
trap rollback ERR

# Configure small-budget pilot while still DRY.
python3 - <<'PY'
from pathlib import Path
import os

path = Path("/etc/market-maker.env")
lines = path.read_text().splitlines()
lines = [x for x in lines if not x.startswith("DEMO_PILOT=")]
lines.append("DEMO_PILOT=1")
temp = path.with_name(path.name + ".pilot-tmp")
fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write("\n".join(lines) + "\n")
os.replace(temp, path)
PY

echo "Running one systemd DRY pass on the DEMO market feed..."
BEFORE=$(date +%s)
systemctl start market-maker.service
python3 - "$BEFORE" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(
    "/home/trader/sports-market-maker/outputs/live/nfl/last_run.json"
)
if not summary.exists() or summary.stat().st_mtime < int(sys.argv[1]) - 2:
    raise SystemExit("STOP: NFL service did not produce a fresh dry-run summary")
data = json.loads(summary.read_text())
if (data.get("env"), data.get("mode"), data.get("demo_pilot")) != (
    "demo", "dry", True
):
    raise SystemExit("STOP: service summary lacks expected DEMO/DRY/PILOT settings")
if data.get("bankroll", 0) <= 0:
    raise SystemExit("STOP: no authenticated Demo bankroll")
print("DRY PASS VERIFIED — Model candidates:", data.get("candidates", 0))
print("DRY PASS VERIFIED — Simulated bankroll:", data.get("bankroll"))
PY

# Switch only the same demo exchange service to submitting simulated-money orders.
python3 - <<'PY'
from pathlib import Path
import os

path = Path("/etc/market-maker.env")
lines = path.read_text().splitlines()
if lines.count("TRADE_MODE=dry") != 1 or lines.count("KALSHI_ENV=demo") != 1:
    raise SystemExit("STOP: demo/dry settings changed unexpectedly")
lines = ["TRADE_MODE=live" if x == "TRADE_MODE=dry" else x for x in lines]
temp = path.with_name(path.name + ".live-tmp")
fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write("\n".join(lines) + "\n")
os.replace(temp, path)
PY

# First arm the automatic cutoff, then enable the existing one-minute runner.
systemd-run --quiet --unit market-maker-demo-stop --on-active=48h   /bin/bash "$REPO/ops/stop_demo_pilot.sh"
systemctl start market-maker.timer
systemctl is-active --quiet market-maker.timer

trap - ERR
echo "DEMO PILOT ACTIVATED — simulated funds only, 48-hour automatic stop."
echo "Per order max: \$2 | game day max: \$10 | total committed max: \$10"
echo "No promises of immediate fills: strategy still enforces the 2–6-hour window."
echo "To stop sooner: bash $REPO/ops/stop_demo_pilot.sh"
echo "To review fills: check outputs/live/nfl/fills.csv and forward_validation.json"
