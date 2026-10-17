#!/usr/bin/env bash
# Stops the supervised Demo-only trial and returns the service to dry mode.
set -euo pipefail
if [[ "$EUID" -ne 0 ]]; then
  echo "STOP: run as root for systemd and /etc settings" >&2
  exit 1
fi
systemctl stop market-maker.timer
python3 - <<'PY'
from pathlib import Path
import os

path = Path("/etc/market-maker.env")
lines = path.read_text().splitlines()
if sum(line.startswith("KALSHI_ENV=") for line in lines) != 1 or not any(
    line == "KALSHI_ENV=demo" for line in lines
):
    raise SystemExit("Safety stop: Demo environment configuration is unexpected.")
if sum(line.startswith("TRADE_MODE=") for line in lines) != 1:
    raise SystemExit("Safety stop: ambiguous TRADE_MODE. Timer is already stopped.")
lines = ["TRADE_MODE=dry" if x.startswith("TRADE_MODE=") else x for x in lines]
temp = path.with_name(path.name + ".stop-tmp")
fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write("\n".join(lines) + "\n")
os.replace(temp, path)
PY
echo "DEMO PILOT STOPPED. Timer inactive, TRADE_MODE=dry. Check resting Demo orders separately."
