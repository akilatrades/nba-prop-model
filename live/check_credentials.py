"""Read-only check that the Kalshi API credentials work: prints the balance.

Reads KALSHI_KEY_ID, KALSHI_PRIVATE_KEY and KALSHI_ENV from the environment.
Places nothing. Run from the repository root: python -m live.check_credentials
"""

import json
import sys

from kalshi.trading import TradingClient, TradingError


def main():
    import os

    key_id = os.environ.get("KALSHI_KEY_ID", "")
    pem = os.environ.get("KALSHI_PRIVATE_KEY", "")
    print(
        f"key id length {len(key_id)}, private key length {len(pem)}, "
        f"starts with BEGIN: {pem.lstrip().startswith('-----BEGIN')}, "
        f"lines: {pem.count(chr(10)) + 1}"
    )
    try:
        client = TradingClient(dry_run=True)
    except Exception as error:  # noqa: BLE001 - report, without the key material
        print(
            f"::error::could not load credentials: {type(error).__name__}: {str(error)[:300]} | "
            f"key id length {len(key_id)}, pem length {len(pem)}, "
            f"BEGIN present: {'-----BEGIN' in pem}, END present: {'-----END' in pem}, "
            f"first 11 chars: {pem.strip()[:11]!r}"
        )
        sys.exit(1)
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
