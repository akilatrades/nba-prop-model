"""Authenticated Kalshi trading client, with a dry-run mode.

Kalshi signs each request with an RSA key pair: the headers carry the key id,
a millisecond timestamp and a base64 RSA-PSS (SHA-256) signature over
``timestamp + METHOD + path`` (the path from ``/trade-api/v2`` on, without
the query string). Credentials come from the environment:

    KALSHI_KEY_ID        the API key id from the Kalshi account page
    KALSHI_PRIVATE_KEY   the private key in PEM form (the whole file)
    KALSHI_ENV           "demo" (default) or "prod"

In dry-run mode (``TradingClient(dry_run=True)``, the default) orders are
built, validated and logged but not sent; reads still go to the exchange.
Nothing in this module decides what to trade; see ``live/nba_trade.py``.
"""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

BASES = {
    "prod": "https://api.elections.kalshi.com",
    "demo": "https://demo-api.kalshi.co",
}
PREFIX = "/trade-api/v2"


class TradingError(RuntimeError):
    pass


def normalize_pem(pem):
    """Repair a PEM pasted into a secret: literal "\\n", lost line breaks, stray spaces."""
    text = pem.strip().replace("\\n", "\n").replace("\r", "")
    match = re.search(r"-----BEGIN ([A-Z ]+)-----(.*?)-----END \1-----", text, re.DOTALL)
    if not match:
        return text
    label, body = match.group(1), re.sub(r"\s+", "", match.group(2))
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return (
        f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n"
    )


def load_private_key(pem):
    return serialization.load_pem_private_key(
        normalize_pem(pem).encode(), password=None
    )


def sign(private_key, timestamp_ms, method, path):
    message = f"{timestamp_ms}{method.upper()}{path}".encode()
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


@dataclass
class Order:
    """One limit order as Kalshi's API takes it; prices in whole cents."""

    ticker: str
    side: str  # "yes" or "no"
    count: int
    price_cents: int  # price of the side bought
    expiration_ts: int | None = None  # Unix seconds; None rests until cancelled
    post_only: bool = True
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def validate(self):
        if self.side not in ("yes", "no"):
            raise TradingError(f"bad side {self.side!r}")
        if not isinstance(self.count, int) or self.count < 1:
            raise TradingError(f"bad count {self.count!r}")
        if not isinstance(self.price_cents, int) or not 1 <= self.price_cents <= 99:
            raise TradingError(f"bad price {self.price_cents!r}")
        if self.expiration_ts is not None and self.expiration_ts <= time.time():
            raise TradingError(f"expiration {self.expiration_ts} is in the past")

    def body(self):
        self.validate()
        body = {
            "ticker": self.ticker,
            "client_order_id": self.client_order_id,
            "action": "buy",
            "side": self.side,
            "type": "limit",
            "count": self.count,
            "post_only": self.post_only,
            f"{self.side}_price": self.price_cents,
        }
        if self.expiration_ts is not None:
            body["expiration_ts"] = int(self.expiration_ts)
        return body


class TradingClient:
    def __init__(
        self, dry_run=True, env=None, key_id=None, private_key_pem=None, timeout=20
    ):
        self.dry_run = dry_run
        self.env = env or os.environ.get("KALSHI_ENV", "demo")
        if self.env not in BASES:
            raise TradingError(f"KALSHI_ENV must be one of {sorted(BASES)}")
        self.base = BASES[self.env]
        self.key_id = key_id or os.environ.get("KALSHI_KEY_ID", "")
        pem = private_key_pem or os.environ.get("KALSHI_PRIVATE_KEY", "")
        self.private_key = load_private_key(pem) if pem else None
        self.timeout = timeout
        self.log = []  # every would-be or actual write, for the ledger

    # --- plumbing -------------------------------------------------------------

    @property
    def can_sign(self):
        return bool(self.key_id and self.private_key)

    def _headers(self, method, path):
        if not self.can_sign:
            raise TradingError(
                "no credentials: set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY"
            )
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sign(self.private_key, ts, method, path),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method, path, params=None, body=None):
        full = PREFIX + path
        url = self.base + full
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url, data=data, method=method, headers=self._headers(method, full)
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            detail = error.read(500).decode("utf-8", "replace")
            raise TradingError(
                f"{method} {path} -> HTTP {error.code}: {detail}"
            ) from error

    # --- reads ---------------------------------------------------------------

    def balance(self):
        """Available balance and portfolio value, in dollars."""
        b = self._request("GET", "/portfolio/balance")
        return {
            "balance": float(b.get("balance", 0)) / 100,
            "portfolio_value": float(b.get("portfolio_value", 0)) / 100,
        }

    def orders(self, status="resting", ticker=None):
        params = {"status": status, "limit": 200}
        if ticker:
            params["ticker"] = ticker
        out, cursor = [], ""
        while True:
            if cursor:
                params["cursor"] = cursor
            page = self._request("GET", "/portfolio/orders", params)
            out.extend(page.get("orders", []))
            cursor = page.get("cursor") or ""
            if not cursor:
                return out

    def fills(self, min_ts=None, ticker=None):
        params = {"limit": 200}
        if min_ts:
            params["min_ts"] = int(min_ts)
        if ticker:
            params["ticker"] = ticker
        out, cursor = [], ""
        while True:
            if cursor:
                params["cursor"] = cursor
            page = self._request("GET", "/portfolio/fills", params)
            out.extend(page.get("fills", []))
            cursor = page.get("cursor") or ""
            if not cursor:
                return out

    def positions(self):
        return self._request("GET", "/portfolio/positions", {"limit": 200}).get(
            "market_positions", []
        )

    # --- writes ----------------------------------------------------------------

    def place(self, order):
        """Place one order. In dry-run mode, validate and log it instead."""
        body = order.body()
        entry = {**body, "op": "place", "dry_run": self.dry_run, "env": self.env}
        if self.dry_run:
            entry["order_id"] = "dry-run"
            self.log.append(entry)
            return entry
        result = self._request("POST", "/portfolio/orders", body=body)
        entry["order_id"] = (result.get("order") or {}).get("order_id")
        entry["status"] = (result.get("order") or {}).get("status")
        self.log.append(entry)
        return entry

    def cancel(self, order_id):
        entry = {
            "op": "cancel",
            "dry_run": self.dry_run,
            "env": self.env,
            "order_id": order_id,
        }
        if self.dry_run or order_id == "dry-run":
            self.log.append(entry)
            return entry
        self._request("DELETE", f"/portfolio/orders/{order_id}")
        self.log.append(entry)
        return entry

    def cancel_all(self, ticker=None):
        cancelled = []
        for o in self.orders("resting", ticker):
            cancelled.append(self.cancel(o["order_id"]))
        return cancelled
