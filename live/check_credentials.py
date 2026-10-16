"""Read-only check that the Kalshi API credentials work: prints the balance.

Reads KALSHI_KEY_ID, KALSHI_PRIVATE_KEY and KALSHI_ENV from the environment.
Places nothing. Run from the repository root: python -m live.check_credentials
"""

import json
import sys

from kalshi.trading import TradingClient, TradingError


def main():
    client = TradingClient(dry_run=True)
    if not client.can_sign:
        print(
            "::error::credentials missing: KALSHI_KEY_ID or KALSHI_PRIVATE_KEY is empty"
        )
        sys.exit(1)
    try:
        balance = client.balance()
        resting = client.orders("resting")
    except TradingError as error:
        print(f"::error::{error}")
        sys.exit(1)
    out = {"env": client.env, **balance, "resting_orders": len(resting)}
    print(json.dumps(out))
    print("::notice::" + json.dumps(out))


if __name__ == "__main__":
    main()
